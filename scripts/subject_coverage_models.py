"""Full GroundingDINO and FG-CLIP 2 additions; all old features reused exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import run  # noqa: F401
from compare_fgclip2_retrieval import ROOT as FGROOT
from compare_fgclip2_retrieval import image_file, relative, text_file, text_key
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.fgclip_retrieval import FGClipEncoder
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from multimodal_ecommerce.subject_roi import (
    SubjectDetector,
    save_subject_crop,
    subject_prompt,
)
from primary_form_coverage import receipt
from subject_coverage_acquisition import PARENT, ROOT, check
from subject_roi_experiment import load, once


def freeze() -> tuple[Any, list, Any]:
    """Freeze the inputs and policy used by this workflow.

    Returns:
        Values produced by this operation, in the stated return order.

    Notes:
        Writes derived artifacts through once.
    """
    check()
    verify_files(ROOT, load(ROOT / "label_receipt.json")["files"])
    files = [
        Path(__file__),
        ROOT / "label_receipt.json",
        PARENT / "inputs.json",
        PARENT / "roi_feature_index.json",
        PARENT / "detector_manifest.json",
        FGROOT / "model_manifest.json",
        *sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")),
    ]
    once(
        ROOT / "auxiliary_protocol.json",
        dict(
            files={relative(p): sha256_file(p) for p in files},
            full_official_models=True,
            reuse_prior_ROI_and_FG_features=True,
            fixed_policy_source="src/multimodal_ecommerce/subject_roi.py",
        ),
    )
    verify_files(PROJECT_ROOT, load(ROOT / "auxiliary_protocol.json")["files"])
    inputs = load(PARENT / "inputs.json")
    fmap = load(PARENT / "roi_feature_index.json")
    additions = []
    profiles = {r["id"]: r for r in read_rows(ROOT / "additions/profiles.public.jsonl")}
    for doc in load(ROOT / "additions/catalog.final.public.json"):
        imkey, titlekey = sha256_file(Path(doc["image_path"])), text_key(doc["title"])
        inputs["images"][imkey] = relative(Path(doc["image_path"]))
        inputs["texts"][titlekey] = doc["title"]
        hint = profiles[doc["id"]]["product_type"]
        identity = dict(image_key=imkey, prompt=subject_prompt(hint))
        key = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()
        inputs["roi_requests"][key] = dict(**identity, hint=hint)
        additions.append(
            dict(id=doc["id"], image_key=imkey, title_key=titlekey, roi_request=key)
        )
    for data in inputs["cohorts"].values():
        data["gallery"] += additions
    once(ROOT / "inputs.json", inputs)
    once(ROOT / "addition_feature_requests.json", additions)
    return inputs, additions, fmap


def detect() -> None:
    """Apply the detect workflow step.

    Raises:
        FileExistsError: Preserve detections.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    inputs, additions, _ = freeze()
    if (ROOT / "detection_receipt.json").exists():
        raise FileExistsError("Preserve detections")
    detector = SubjectDetector(PARENT / "detector_manifest.json")
    try:
        for i, doc in enumerate(additions, 1):
            key = doc["roi_request"]
            row = inputs["roi_requests"][key]
            path = ROOT / "detections" / key / "result.json"
            if path.exists():
                continue
            source = PROJECT_ROOT / inputs["images"][row["image_key"]]
            if row["prompt"]:
                result = detector.detect(source, row["prompt"])
            else:
                result = dict(
                    prompt=None,
                    decision=dict(
                        status="global_fallback",
                        reason="unsupported public subject; retained",
                        selected=None,
                    ),
                    detections=[],
                    seconds=0,
                )
            result.update(request_id=key, source_sha256=row["image_key"])
            if result["decision"]["status"] == "roi":
                result["crop"] = save_subject_crop(
                    source, result["decision"], path.parent
                )
            once(path, result)
            print(
                dict(
                    detected=i,
                    total=len(additions),
                    status=result["decision"]["status"],
                ),
                flush=True,
            )
        runtime = detector.diagnostics()
    finally:
        detector.close()
    once(ROOT / "detector_runtime.json", runtime)
    results = [
        load(ROOT / "detections" / d["roi_request"] / "result.json") for d in additions
    ]
    receipt(
        ROOT,
        "detection_receipt.json",
        [
            ROOT / "detector_runtime.json",
            *[p for p in sorted((ROOT / "detections").rglob("*")) if p.is_file()],
        ],
        actual_full_detector_calls=len(runtime["calls"]),
        accepted=sum(r["decision"]["status"] == "roi" for r in results),
        fallback=sum(r["decision"]["status"] != "roi" for r in results),
        all_additions_retained=True,
    )


def fg_image(encoder: FGClipEncoder, source: Path, path: Path) -> None:
    """Apply the fg image workflow step.

    Args:
        encoder: Encoder used by this operation.
        source: Source used by this operation.
        path: Path of the input or output artifact used by this operation.

    Notes:
        Writes derived artifacts through np.savez_compressed, once, path.parent.mkdir.
    """
    if not path.exists():
        vector, dense, info = encoder.image(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, global_vector=vector, dense=dense)
        once(
            path.with_suffix(".json"),
            dict(
                source_sha256=sha256_file(source),
                detail=info,
                feature_sha256=sha256_file(path),
            ),
        )
    else:
        assert load(path.with_suffix(".json"))["feature_sha256"] == sha256_file(path)


def encode() -> None:
    """Encode the configured inputs into model feature representations.

    Raises:
        FileExistsError: Preserve auxiliary encoding.

    Notes:
        Writes derived artifacts through np.savez_compressed, once,
        path.parent.mkdir, receipt.
    """
    inputs, additions, fmap = freeze()
    verify_files(ROOT, load(ROOT / "detection_receipt.json")["files"])
    if (ROOT / "fg_encoding_receipt.json").exists():
        raise FileExistsError("Preserve auxiliary encoding")
    paths = {
        "images": {
            k: relative(image_file(k)) for k in load(PARENT / "inputs.json")["images"]
        },
        "texts": {
            k: relative(text_file(k)) for k in load(PARENT / "inputs.json")["texts"]
        },
        "box_texts": {k: relative(text_file(k, True)) for k in inputs["box_texts"]},
    }
    encoder = FGClipEncoder(FGROOT / "model_manifest.json")
    try:
        for i, doc in enumerate(additions, 1):
            source = PROJECT_ROOT / inputs["images"][doc["image_key"]]
            path = ROOT / "fg_features/images" / (doc["image_key"] + ".npz")
            fg_image(encoder, source, path)
            paths["images"][doc["image_key"]] = relative(path)
            title_key = doc["title_key"]
            path = ROOT / "fg_features/text" / (title_key + "_long.npz")
            if not path.exists():
                vector, info = encoder.text(inputs["texts"][title_key], regional=False)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, vector=vector)
                once(
                    path.with_suffix(".json"),
                    dict(
                        source_sha256=title_key,
                        detail=info,
                        feature_sha256=sha256_file(path),
                    ),
                )
            paths["texts"][title_key] = relative(path)
            key = doc["roi_request"]
            result = load(ROOT / "detections" / key / "result.json")
            if result["decision"]["status"] == "roi":
                crop = ROOT / "detections" / key / "zoom.png"
                checksum = sha256_file(crop)
                path = ROOT / "fg_features/roi" / (checksum + ".npz")
                fg_image(encoder, crop, path)
                fmap[key] = dict(file=relative(path), crop_sha256=checksum)
            else:
                fmap[key] = dict(
                    file=paths["images"][doc["image_key"]], crop_sha256=None
                )
            print(dict(FG_encoded=i, total=len(additions)), flush=True)
        runtime = encoder.diagnostics()
    finally:
        encoder.close()
    once(ROOT / "feature_paths.json", paths)
    once(ROOT / "roi_feature_index.json", fmap)
    once(ROOT / "fg_runtime.json", runtime)
    receipt(
        ROOT,
        "fg_encoding_receipt.json",
        [
            ROOT / n
            for n in ["feature_paths.json", "roi_feature_index.json", "fg_runtime.json"]
        ]
        + [p for p in sorted((ROOT / "fg_features").rglob("*")) if p.is_file()],
        new_full_model_requests=len(runtime["calls"]),
        actual_forward_calls=(
            sum(2 if r["kind"] == "image" else 1 for r in runtime["calls"])
            if all("kind" in r for r in runtime["calls"])
            else None
        ),
        old_global_and_query_ROI_features_reused_exactly=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["detect", "encode"])
    globals()[parser.parse_args().action]()
