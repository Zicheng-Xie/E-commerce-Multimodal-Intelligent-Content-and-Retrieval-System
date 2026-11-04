"""Label-free development subject proposals and full official SAM2 views.

Grounding is a candidate-view generator, not proof of business relevance.
All inputs remain in the gallery; unsupported/ambiguous products keep whole images.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import run  # noqa: F401
from business_grounding_audit import ROOT, load, once, relative, verify
from multimodal_ecommerce.holdout_data import fingerprints
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.subject_roi import SubjectDetector


class _PolicySpec(TypedDict):
    """Exact fields of the POLICY configuration and provenance record."""

    detector_threshold: float
    min_box_fraction: float
    max_box_fraction: float
    min_edge: int
    padding: float
    max_padded_fraction: float
    max_boxes: int
    selection: str
    mask_min_fraction_inside_crop: float
    mask_max_fraction_inside_crop: float
    mask_choice: str
    background: list[int]
    zoom_long_edge: int
    model_processor: str
    ambiguous_or_unsupported: str
    relevance_used: bool
    stage: str


POLICY: _PolicySpec = {
    "detector_threshold": 0.30,
    "min_box_fraction": 0.02,
    "max_box_fraction": 0.95,
    "min_edge": 8,
    "padding": 0.08,
    "max_padded_fraction": 0.98,
    "max_boxes": 8,
    "selection": "union of all geometrically valid boxes for one public subject noun; never highest-score-only",
    "mask_min_fraction_inside_crop": 0.05,
    "mask_max_fraction_inside_crop": 0.98,
    "mask_choice": "single mask per box; union of masks; no semantic quality claim",
    "background": [255, 255, 255],
    "zoom_long_edge": 768,
    "model_processor": "official pinned SAM2 defaults; no reduced input settings",
    "ambiguous_or_unsupported": "preserve whole image; no synthetic mask",
    "relevance_used": False,
    "stage": "view_generation_before_QA_and_ranking",
}


def noun(product: str) -> str | None:
    """Own public product head only; never scan context objects in a title.

    Args:
        product: Product used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    if any(w in product for w in ("组合", "不明", "无法", "套装", "道具")):
        return None
    groups = [
        (("电脑", "笔记本"), "laptop bag."),
        (("相机内胆",), "camera bag."),
        (("内胆", "包中包"), "bag organizer."),
        (("耳罩", "耳垫"), "ear pads."),
        (("耳机保护套", "耳机盒保护套"), "earphone case."),
        (("耳机升级线", "耳机线"), "earphone cable."),
        (("头戴", "耳麦", "后挂耳机"), "headphones."),
        (("耳机", "耳塞"), "earphones."),
        (("手机壳", "手机背壳"), "phone case."),
        (("手机防水袋", "防水手机袋", "手机包"), "phone pouch."),
        (("枕套", "枕席"), "pillow case."),
        (("枕",), "pillow."),
        (("鞋收纳盒",), "shoe box."),
        (("靴",), "boot."),
        (("鞋",), "shoe."),
        (("马甲", "背心"), "vest."),
        (("羽绒", "外套", "棉服", "大衣"), "jacket."),
        (("杯配件",), "cup lid."),
        (("柜",), "cabinet."),
        (("杯", "水壶", "奶瓶"), "cup. bottle."),
        (("榴莲",), "durian."),
        (("手提包", "手袋"), "handbag."),
        (("收纳包",), "storage bag."),
        (("茶几", "折叠桌"), "table."),
        (("置物架",), "shelf."),
    ]
    return next(
        (text for aliases, text in groups if any(w in product for w in aliases)), None
    )


def union_box(
    detections: list[dict], size: tuple[int, int]
) -> tuple[None, list] | tuple[list, list]:
    """Return union box for this workflow.

    Args:
        detections: Detections used by this operation.
        size: Size used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    width, height = size
    valid = []
    for r in detections:
        if (
            not np.isfinite([r["score"], *r["box"]]).all()
            or r["score"] < POLICY["detector_threshold"]
        ):
            continue
        x1, y1, x2, y2 = [
            max(0.0, min(v, limit))
            for v, limit in zip(r["box"], [width, height, width, height])
        ]
        dx, dy = x2 - x1, y2 - y1
        if (
            min(dx, dy) < POLICY["min_edge"]
            or not POLICY["min_box_fraction"]
            <= dx * dy / (width * height)
            <= POLICY["max_box_fraction"]
        ):
            continue
        valid.append([x1, y1, x2, y2])
    if not valid or len(valid) > POLICY["max_boxes"]:
        return None, valid
    x1, y1 = min(r[0] for r in valid), min(r[1] for r in valid)
    x2, y2 = max(r[2] for r in valid), max(r[3] for r in valid)
    px, py = (x2 - x1) * POLICY["padding"], (y2 - y1) * POLICY["padding"]
    box = [
        max(0, math.floor(x1 - px)),
        max(0, math.floor(y1 - py)),
        min(width, math.ceil(x2 + px)),
        min(height, math.ceil(y2 + py)),
    ]
    if (box[2] - box[0]) * (box[3] - box[1]) / (width * height) > POLICY[
        "max_padded_fraction"
    ]:
        return None, valid
    return box, valid


def font(size: int) -> ImageFont.FreeTypeFont:
    """Return font for this workflow.

    Args:
        size: Size used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)


def wrap(
    draw: ImageDraw.ImageDraw, text: str, max_width: int, f: ImageFont.FreeTypeFont
) -> list:
    """Return wrap for this workflow.

    Args:
        draw: Draw used by this operation.
        text: Text input used by the operation.
        max_width: Max width used by this operation.
        f: F used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    lines, line = [], ""
    for char in text:
        if draw.textlength(line + char, font=f) > max_width:
            lines.append(line)
            line = ""
        line += char
    return [*lines, line]


def qa_sheets(records: list[dict], folder: Path, *, tile_mode: str = "source") -> list:
    """Return qa sheets for this workflow.

    Args:
        records: Records used by this operation.
        folder: Directory containing the operation artifacts.
        tile_mode: Tile mode used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Notes:
        Writes derived artifacts through canvas.save, folder.mkdir.
    """
    folder.mkdir(parents=True, exist_ok=True)
    outputs = []
    for start in range(0, len(records), 12):
        canvas = Image.new("RGB", (2400, 1440), "white")
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(records[start : start + 12]):
            x, y = (i % 4) * 600, (i // 4) * 480
            source = PROJECT_ROOT / row["image_path"]
            with Image.open(source) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
            image.thumbnail((580, 310))
            canvas.paste(image, (x + (600 - image.width) // 2, y))
            label = row.get("opaque", str(start + i + 1))
            text = f"{label} {row['AI_profile']['product_type']} | {row['title']}"
            for j, line in enumerate(wrap(draw, text, 580, font(22))[:6]):
                draw.text((x + 10, y + 315 + j * 25), line, fill="black", font=font(22))
        path = folder / f"{start//12+1:02d}.png"
        canvas.save(path)
        outputs.append(path)
    return outputs


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        ValueError: Same gallery ID has conflicting public profiles.

    Notes:
        Writes derived artifacts through once.
    """
    verify()
    if (ROOT / "views_protocol.json").exists():
        guard()
        return
    audited = load(ROOT / "internal_data_quality_audit.json")
    records: dict = {}
    for r in audited["catalog"]:
        key = r["id"]
        if key in records:
            if records[key]["profile"] != r["AI_profile"]:
                raise ValueError("Same gallery ID has conflicting public profiles")
            records[key]["memberships"].append(r["cohort"])
        else:
            records[key] = {
                "key": key,
                "kind": "gallery",
                "memberships": [r["cohort"]],
                "profile": r["AI_profile"],
                "image_path": r["image_path"],
                "source_sha256": r["fingerprints"]["sha256"],
                "public_title": r["title"],
                "QA_flags": r["QA_flags"],
            }
    for q in audited["query_images"]:
        value = load(
            PROJECT_ROOT
            / "artifacts/module2_primary_subjects/query_captions"
            / q["cohort"]
            / (q["query_id"] + ".json")
        )["value"]
        key = q["cohort"] + "/" + q["query_id"]
        records[key] = {
            "key": key,
            "kind": "query",
            "memberships": [q["cohort"]],
            "profile": {
                "product_type": value["product"],
                "appearance": value["appearance"],
                "primary_unclear": value["primary_unclear"],
            },
            "image_path": q["image_path"],
            "source_sha256": q["fingerprints"]["sha256"],
            "public_title": "",
            "QA_flags": [],
        }
    values = sorted(records.values(), key=lambda r: r["key"])
    for i, r in enumerate(values, 1):
        r["opaque"] = f"S{i:03d}"
        r["prompt"] = (
            None
            if r["profile"].get("primary_unclear", False)
            else noun(r["profile"]["product_type"])
        )
    once(ROOT / "views_inputs.public.json", values)
    flags = []
    unique: dict = {}
    for r in audited["catalog"]:
        if r["QA_flags"]:
            unique.setdefault(r["id"], r)
    for i, r in enumerate(unique.values(), 1):
        flags.append(dict(r, opaque=f"Q{i:02d}"))
    sheets = qa_sheets(flags, ROOT / "QA/source_sheets")
    once(ROOT / "QA/source_inventory.json", flags)
    paths = [
        Path(__file__),
        PROJECT_ROOT / "tests/test_business_subject_views.py",
        ROOT / "audit_receipt.json",
        ROOT / "views_inputs.public.json",
        ROOT / "sam2_model_manifest.json",
        PROJECT_ROOT / "artifacts/module2_subject_roi/detector_manifest.json",
    ]
    paths += [ROOT / "QA/source_inventory.json", *sheets]
    manifest = load(ROOT / "sam2_model_manifest.json")
    paths += [PROJECT_ROOT / manifest["path"] / name for name in manifest["files"]]
    once(
        ROOT / "views_protocol.json",
        {
            "files": {relative(p): sha256_file(p) for p in paths},
            "policy": POLICY,
            "subject_nouns": {
                p: noun(p)
                for p in sorted({r["profile"]["product_type"] for r in values})
            },
            "records": len(values),
            "native_QA_source_records": len(flags),
            "labels_loaded_by_worker": False,
            "whole_images_preserved": True,
            "publication_claim": "complete official DINO detection and complete official SAM2 segmentation; no STIR or retrieval quality result yet",
            "formal_metrics": None,
        },
    )
    expose(sheets)
    print(
        json.dumps(
            {
                "subject_view_protocol_frozen": True,
                "records": len(values),
                "QA_records": len(flags),
            }
        ),
        flush=True,
    )


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    for name, digest in load(ROOT / "views_protocol.json")["files"].items():
        if sha256_file(PROJECT_ROOT / name) != digest:
            raise ValueError("View source/model/input changed: " + name)


def expose(paths: list[Path]) -> None:
    """Apply the expose workflow step.

    Args:
        paths: Paths used by this operation.

    Notes:
        Writes derived artifacts through registry.parent.mkdir.
    """
    registry = PROJECT_ROOT / "data/module2_business_grounding_derived/records.jsonl"
    registry.parent.mkdir(parents=True, exist_ok=True)
    seen = (
        {
            r["image_path"]
            for r in [
                json.loads(s)
                for s in registry.read_text(encoding="utf-8").splitlines()
                if s
            ]
        }
        if registry.exists()
        else set()
    )
    with registry.open("a", encoding="utf-8") as handle:
        for p in paths:
            name = relative(p)
            if name in seen:
                continue
            handle.write(
                json.dumps(
                    {
                        "id": "business_grounding_derived/" + sha256_file(p),
                        "image_path": name,
                        "fingerprints": fingerprints(p),
                        "role": "exposed development derivative; never unseen acceptance",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            seen.add(name)


@contextmanager
def locked() -> Iterator[None]:
    """Apply the locked workflow step.

    Yields:
        Values produced by this operation, in the stated return order.
    """
    lock = ROOT / "views_runtime.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write("business_subject_views worker\n")
    try:
        yield
    finally:
        lock.unlink()


def worker() -> None:
    """Return worker for this workflow.

    Raises:
        ValueError: View original image changed; Incomplete SAM2 checkpoint loading.

    Notes:
        Writes derived artifacts through Image.fromarray(foreground).save,
        Image.fromarray(part.astype(np.uint8) * 255).save, crop.save, folder.mkdir,
        once, panel.save, write_json.
    """
    guard()
    from gpu_admission import gpu_available

    allowed, info = gpu_available()
    if not allowed:
        write_json(ROOT / "views_waiting.json", info)
        print(json.dumps({"waiting": True, "reason": info}), flush=True)
        return
    with locked():
        records = load(ROOT / "views_inputs.public.json")
        detector = SubjectDetector(
            PROJECT_ROOT / "artifacts/module2_subject_roi/detector_manifest.json"
        )
        plans = []
        try:
            for i, r in enumerate(records):
                folder = ROOT / "subject_views" / r["opaque"]
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / "detection.json"
                if path.exists():
                    value = load(path)
                elif not r["prompt"]:
                    value = {
                        "source_sha256": r["source_sha256"],
                        "box": None,
                        "boxes": [],
                        "reason": "public subject unresolved or unsupported; keep whole",
                    }
                    once(path, value)
                else:
                    raw = detector.detect(PROJECT_ROOT / r["image_path"], r["prompt"])
                    box, boxes = union_box(raw["detections"], raw["source_size"])
                    value = {
                        "source_sha256": r["source_sha256"],
                        "box": box,
                        "boxes": boxes,
                        "reason": "public subject union proposal; semantic correctness not yet reviewed",
                        "native_detector": raw,
                    }
                    once(path, value)
                if (
                    sha256_file(PROJECT_ROOT / r["image_path"])
                    != value["source_sha256"]
                ):
                    raise ValueError("View original image changed")
                plans.append(dict(r, detection=relative(path)))
                if (i + 1) % 25 == 0:
                    print(
                        json.dumps(
                            {"detection_completed": i + 1, "total": len(records)}
                        ),
                        flush=True,
                    )
            once(ROOT / "subject_detection_runtime.json", detector.diagnostics())
        finally:
            detector.close()
        once(ROOT / "subject_detection_plan.json", plans)
        import torch
        from transformers import Sam2Model, Sam2Processor

        manifest = load(ROOT / "sam2_model_manifest.json")
        model_path = PROJECT_ROOT / manifest["path"]
        from multimodal_ecommerce.runtime_device import RuntimeDevice

        runtime = RuntimeDevice(torch)
        runtime.reset_peak()
        model, loading = Sam2Model.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32,
            output_loading_info=True,
        )
        if any(
            loading.get(k)
            for k in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError("Incomplete SAM2 checkpoint loading")
        model.eval().to(runtime.device)
        processor = Sam2Processor.from_pretrained(model_path, local_files_only=True)
        calls, results, derivatives = [], [], []
        try:
            for i, r in enumerate(plans):
                folder = ROOT / "subject_views" / r["opaque"]
                output = folder / "result.json"
                if output.exists():
                    results.append(load(output))
                    continue
                detection = load(PROJECT_ROOT / r["detection"])
                result = {
                    "key": r["key"],
                    "opaque": r["opaque"],
                    "source_image": r["image_path"],
                    "source_sha256": r["source_sha256"],
                    "subject_prompt": r["prompt"],
                    "box": detection["box"],
                    "selection": "whole",
                    "semantic_review": "pending",
                    "model_call": False,
                    "paths": {},
                    "relevance_used": False,
                }
                if detection["box"] is not None:
                    with Image.open(PROJECT_ROOT / r["image_path"]) as source_image:
                        image = ImageOps.exif_transpose(source_image).convert("RGB")
                    batch = processor(
                        images=image,
                        input_boxes=[detection["boxes"]],
                        return_tensors="pt",
                    ).to(runtime.device)
                    runtime.synchronize()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        out = model(**batch, multimask_output=False)
                    runtime.synchronize()
                    masks = processor.post_process_masks(
                        out.pred_masks.cpu(), batch["original_sizes"].cpu()
                    )[0]
                    mask = (
                        masks.numpy()
                        .astype(bool)
                        .reshape(-1, image.height, image.width)
                        .any(axis=0)
                    )
                    box = detection["box"]
                    part = mask[box[1] : box[3], box[0] : box[2]]
                    fraction = float(part.mean())
                    crop = image.crop(tuple(box))
                    native = folder / "box_native.png"
                    crop.save(native)
                    paths = [native]
                    result.update(
                        model_call=True,
                        selection="box_proposal",
                        mask_foreground_fraction=fraction,
                        seconds=time.perf_counter() - started,
                        predicted_mask_scores=out.iou_scores.cpu().tolist(),
                    )
                    if (
                        POLICY["mask_min_fraction_inside_crop"]
                        <= fraction
                        <= POLICY["mask_max_fraction_inside_crop"]
                    ):
                        foreground = np.asarray(crop).copy()
                        foreground[~part] = POLICY["background"]
                        masked = folder / "masked_native.png"
                        Image.fromarray(foreground).save(masked)
                        mask_path = folder / "mask.png"
                        Image.fromarray(part.astype(np.uint8) * 255).save(mask_path)
                        paths += [masked, mask_path]
                        result["selection"] = "box_and_mask_proposals"
                    panel = Image.new("RGB", (1536, 600), "white")
                    draw = ImageDraw.Draw(panel)
                    framed = image.copy()
                    ImageDraw.Draw(framed).rectangle(
                        tuple(box), outline="red", width=max(2, image.width // 200)
                    )
                    views = [("A original + union box", framed), ("B box crop", crop)]
                    if (folder / "masked_native.png").exists():
                        views.append(
                            (
                                "C foreground",
                                Image.open(folder / "masked_native.png").convert("RGB"),
                            )
                        )
                    for j, (label, view) in enumerate(views):
                        view.thumbnail((500, 530))
                        panel.paste(view, (j * 512 + (512 - view.width) // 2, 50))
                        draw.text((j * 512 + 8, 8), label, fill="black", font=font(22))
                    preview = folder / "review.png"
                    panel.save(preview)
                    paths.append(preview)
                    result["paths"] = {
                        p.stem: {"path": relative(p), "sha256": sha256_file(p)}
                        for p in paths
                    }
                    derivatives += paths
                    calls.append(
                        {
                            "key": r["key"],
                            "seconds": result["seconds"],
                            "input_shape": list(batch["pixel_values"].shape),
                            "boxes": len(detection["boxes"]),
                        }
                    )
                    expose(paths)
                once(output, result)
                results.append(result)
                if (i + 1) % 25 == 0:
                    print(
                        json.dumps(
                            {
                                "SAM2_completed": i + 1,
                                "total": len(plans),
                                "full_new_calls": len(calls),
                            }
                        ),
                        flush=True,
                    )
            once(ROOT / "subject_views_results.json", results)
            once(
                ROOT / "sam2_runtime.json",
                {
                    "loading": loading,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "dtype": str(next(model.parameters()).dtype),
                    "calls": calls,
                    "peak_allocated_mib": runtime.allocated() / 2**20,
                    "peak_reserved_mib": runtime.reserved() / 2**20,
                    "complete_checkpoint": True,
                },
            )
        finally:
            del model
            runtime.empty_cache()
        evidence = [
            *sorted((ROOT / "subject_views").rglob("*json")),
            ROOT / "subject_views_results.json",
            ROOT / "subject_detection_runtime.json",
            ROOT / "sam2_runtime.json",
        ]
        evidence += [
            PROJECT_ROOT / x["path"] for r in results for x in r["paths"].values()
        ]
        once(
            ROOT / "views_receipt.json",
            {
                "files": {relative(p): sha256_file(p) for p in evidence},
                "completed": True,
                "semantic_QA_completed": False,
                "formal_metrics": None,
            },
        )
        write_json(
            ROOT / "public_status.json",
            {
                "phase": "complete_subject_views_generated_QA_pending",
                "completed": False,
                "completed_stages": ["data_quality_audit", "full_DINO_SAM2_views"],
                "development_queries": 40,
                "AI_relations_unchanged": 12600,
                "quality_metrics": None,
                "formal_metrics": None,
                "goal_met": False,
                "default_strategy_changed": False,
                "new_independent_AI_blind_predictions": 0,
            },
        )
        print(
            json.dumps(
                {
                    "subject_views_generated": True,
                    "records": len(results),
                    "retrieval_quality_results": None,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "worker"])
    prepare() if parser.parse_args().action == "prepare" else worker()
