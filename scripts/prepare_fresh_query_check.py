"""Freeze and acquire six new train-source queries, separate from formal heldout."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, TypedDict

import httpx
import ijson
from PIL import Image, ImageDraw, ImageFont, ImageOps

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_acceptance import product_family_keys
from multimodal_ecommerce.business_review_jobs import canonical_hash
from multimodal_ecommerce.data import parse_pv
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.media import fetch_image, inspect_image
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    PROTECTED_ROOT,
    sha256_file,
    write_json,
)

STAGE = PROJECT_ROOT / "artifacts/module2_rank_stability"
ROOT = STAGE / "fresh_query_check"
PARENT = PROJECT_ROOT / "artifacts/module2_focus_coverage"
QUEUE = PROJECT_ROOT / "data/module2_fresh_query_candidates/records.jsonl"
REGISTRY = PROJECT_ROOT / "data/module2_fresh_query_media/records.jsonl"
BUCKETS = ["headphones", "keyboard", "power_bank", "drinking_cup", "pillow", "chair"]


class _PolicySpec(TypedDict):
    """Exact fields of the POLICY configuration and provenance record."""

    role: str
    source_split: str
    scan_after: int
    scan_limit: int
    seed: int
    metadata_per_bucket: int
    image_queries_per_bucket: int
    technical_spares_only: bool
    replace_for_relevance_or_low_scores: bool
    preserve_zero_positive_queries: bool
    duplicate_dhash_distance: int
    review_origin: str
    human_label: bool
    use_reserved_acceptance: bool
    fine_tuning_or_parameter_search: bool


POLICY: _PolicySpec = dict(
    role="new_query_development_stress_check_not_formal_acceptance",
    source_split="train",
    scan_after=1000000,
    scan_limit=1100000,
    seed=2026102319,
    metadata_per_bucket=3,
    image_queries_per_bucket=1,
    technical_spares_only=True,
    replace_for_relevance_or_low_scores=False,
    preserve_zero_positive_queries=True,
    duplicate_dhash_distance=4,
    review_origin="model",
    human_label=False,
    use_reserved_acceptance=False,
    fine_tuning_or_parameter_search=False,
)


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def bucket(title: str) -> str | None:
    """Return bucket for this workflow.

    Args:
        title: Public product title used as input.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    if (
        "蓝牙" in title
        and "耳机" in title
        and not any(
            x in title
            for x in ["耳机套", "耳机壳", "保护套", "配件", "耳机包", "充电仓"]
        )
    ):
        return "headphones"
    if "键盘" in title and not any(
        x in title for x in ["膜", "键帽", "按键", "贴", "套", "配件", "键盘架"]
    ):
        return "keyboard"
    if any(x in title for x in ["充电宝", "移动电源"]) and not any(
        x in title for x in ["壳", "套", "线", "配件", "电源板"]
    ):
        return "power_bank"
    if any(x in title for x in ["保温杯", "玻璃杯", "水杯"]) and not any(
        x in title for x in ["垫", "刷", "套", "杯盖", "杯架", "配件", "吸管"]
    ):
        return "drinking_cup"
    if any(x in title for x in ["枕头", "枕芯"]) and not any(
        x in title for x in ["枕套", "靠枕", "抱枕", "沙发"]
    ):
        return "pillow"
    if any(x in title for x in ["餐椅", "书桌椅", "家用椅", "电脑椅"]) and not any(
        x in title for x in ["套", "垫", "脚", "配件", "凳", "儿童"]
    ):
        return "chair"
    return None


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        FileExistsError: Preserve query check acquisition.

    Notes:
        Writes derived artifacts through ROOT.mkdir, write_json, write_rows.
    """
    if ROOT.exists() or QUEUE.exists() or REGISTRY.exists():
        raise FileExistsError("Preserve query check acquisition")
    stage = load(STAGE / "protocol.json")
    verify_files(PROJECT_ROOT, stage["files"])
    prior = load(PARENT / "post_development_exclusion_snapshot.json")
    verify_files(PROJECT_ROOT, prior["files"])
    source = PROTECTED_ROOT / "多模态电商数据/product1m_product5m_train_id_label.json"
    ROOT.mkdir(parents=True)
    write_json(
        ROOT / "protocol.json",
        dict(
            policy=POLICY,
            implementation_sha256=sha256_file(Path(__file__)),
            stage_protocol_sha256=sha256_file(STAGE / "protocol.json"),
            source=source.relative_to(PROJECT_ROOT).as_posix(),
            source_sha256=sha256_file(source),
            prior_snapshot_sha256=sha256_file(
                PARENT / "post_development_exclusion_snapshot.json"
            ),
            gallery_public_sha256=sha256_file(PARENT / "inputs.public.json"),
            selection_uses_retrieval_results=False,
            acceptance_predictions=0,
            formal_metrics=None,
            goal_met=False,
        ),
    )
    excluded = set(prior["family_keys"])
    groups: dict = {b: [] for b in BUCKETS}
    with source.open("rb") as handle:
        for number, (sid, raw) in enumerate(ijson.kvitems(handle, ""), 1):
            if number > POLICY["scan_after"]:
                name = bucket(str(raw.get("title") or ""))
                if name and raw.get("url"):
                    row = dict(
                        id=f"product/train/{sid}",
                        source_id=str(sid),
                        dataset="product",
                        split="train",
                        text=raw["title"],
                        image_url=raw["url"],
                        attributes=parse_pv(raw.get("pv") or ""),
                        bucket=name,
                        source_position=number,
                        source_record_sha256=canonical_hash(raw),
                        project_role="exposed_new_query_development_metadata_not_acceptance",
                    )
                    keys = product_family_keys(row)
                    if not keys & excluded:
                        row.update(
                            family_keys=sorted(keys),
                            priority=hashlib.sha256(
                                f"{POLICY['seed']}:{row['id']}".encode()
                            ).hexdigest(),
                        )
                        groups[name].append(row)
            if number == POLICY["scan_limit"]:
                break
    selected, seen = [], set(excluded)
    for name in BUCKETS:
        count = 0
        for row in sorted(groups[name], key=lambda r: r["priority"]):
            if seen & set(row["family_keys"]):
                continue
            selected.append(row)
            seen.update(row["family_keys"])
            count += 1
            if count == POLICY["metadata_per_bucket"]:
                break
    write_rows(ROOT / "source_candidates.jsonl", selected)
    write_rows(QUEUE, selected)
    write_json(
        ROOT / "metadata_receipt.json",
        dict(
            scanned=number,
            eligible_before_family_dedup={b: len(g) for b, g in groups.items()},
            metadata=len(selected),
            per_bucket=dict(Counter(r["bucket"] for r in selected)),
            files={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                for p in [QUEUE, ROOT / "source_candidates.jsonl"]
            },
            model_inferences=0,
        ),
    )
    print(
        json.dumps(
            dict(
                metadata=len(selected),
                per_bucket=dict(Counter(r["bucket"] for r in selected)),
            )
        )
    )


def check() -> None:
    """Verify the prerequisites and integrity of the current workflow."""
    protocol = load(ROOT / "protocol.json")
    assert protocol["policy"] == POLICY
    assert sha256_file(Path(__file__)) == protocol["implementation_sha256"]
    assert sha256_file(STAGE / "protocol.json") == protocol["stage_protocol_sha256"]
    assert (
        sha256_file(PARENT / "post_development_exclusion_snapshot.json")
        == protocol["prior_snapshot_sha256"]
    )
    assert (
        sha256_file(PARENT / "inputs.public.json") == protocol["gallery_public_sha256"]
    )
    verify_files(PROJECT_ROOT, load(STAGE / "protocol.json")["files"])
    verify_files(PROJECT_ROOT, load(ROOT / "metadata_receipt.json")["files"])


def fetch_one(row: dict) -> dict:
    """Return fetch one for this workflow.

    Args:
        row: Row used by this operation.

    Returns:
        Mapping with fields: id, status, image_path, final_url, fingerprints, width,
        height, error_type, error.

    Notes:
        Writes derived artifacts through path.parent.mkdir, path.write_bytes.
    """
    try:
        with httpx.Client(timeout=20, trust_env=False) as client:
            content, url = fetch_image(client, row["image_url"])
        info = inspect_image(content)
        path = ROOT / "media" / (info["sha256"] + info["suffix"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
        return dict(
            id=row["id"],
            status="downloaded",
            image_path=str(path),
            final_url=url,
            fingerprints=fingerprints(path),
            width=info["width"],
            height=info["height"],
        )
    except (httpx.HTTPError, OSError, ValueError) as error:
        logging.getLogger(__name__).debug(
            "Handled exception in fetch_one", exc_info=True
        )
        return dict(
            id=row["id"],
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
        )


def font(size: int) -> ImageFont.FreeTypeFont:
    """Return font for this workflow.

    Args:
        size: Size used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)


def sheets(queries: list[dict]) -> None:
    """Apply the sheets workflow step.

    Args:
        queries: Queries used by this operation.

    Notes:
        Writes derived artifacts through canvas.save, folder.mkdir, panel.save,
        write_rows.
    """
    folder = ROOT / "native_review"
    folder.mkdir(exist_ok=True)
    canvas = Image.new("RGB", (1560, 710), "white")
    draw = ImageDraw.Draw(canvas)
    for i, q in enumerate(queries):
        x, y = (i % 3) * 520, (i // 3) * 355
        with Image.open(q["image_path"]) as im:
            thumb = ImageOps.contain(
                ImageOps.exif_transpose(im).convert("RGB"), (500, 285)
            )
        canvas.paste(thumb, (x + (520 - thumb.width) // 2, y))
        draw.text((x + 8, y + 289), q["id"], fill="black", font=font(18))
        draw.text((x + 8, y + 314), q["source_id"], fill="black", font=font(16))
    canvas.save(folder / "queries.jpg", quality=96)
    gallery = load(PARENT / "inputs.public.json")["documents"]
    write_rows(
        folder / "gallery_mapping.jsonl",
        [dict(position=i, **r) for i, r in enumerate(gallery, 1)],
    )
    for start in range(0, len(gallery), 36):
        panel = Image.new("RGB", (1800, 1800), "white")
        drawing = ImageDraw.Draw(panel)
        for offset, row in enumerate(gallery[start : start + 36]):
            x, y = (offset % 6) * 300, (offset // 6) * 300
            with Image.open(row["image_path"]) as im:
                thumb = ImageOps.contain(
                    ImageOps.exif_transpose(im).convert("RGB"), (288, 235)
                )
            panel.paste(thumb, (x + (300 - thumb.width) // 2, y))
            drawing.text(
                (x + 5, y + 237), str(start + offset + 1), fill="black", font=font(20)
            )
            title = row["title"]
            drawing.text((x + 5, y + 260), title[:16], fill="black", font=font(16))
            drawing.text((x + 5, y + 280), title[16:32], fill="black", font=font(14))
        sheet_number = start // 36 + 1
        panel.save(folder / f"gallery_{sheet_number:02d}.jpg", quality=95)


def download() -> None:
    """Apply the download workflow step.

    Raises:
        FileExistsError: Preserve acquired queries.

    Notes:
        Writes derived artifacts through write_json, write_rows.
    """
    check()
    if (ROOT / "acquisition_receipt.json").exists():
        raise FileExistsError("Preserve acquired queries")
    rows = read_rows(QUEUE)
    cache_path = ROOT / "download_cache.json"
    cache = load(cache_path) if cache_path.exists() else {}
    prior = load(PARENT / "post_development_exclusion_snapshot.json")
    signatures = list(prior["image_fingerprints"])
    selected, dispositions = [], {}
    for name in BUCKETS:
        for row in [r for r in rows if r["bucket"] == name]:
            result = cache.get(row["id"])
            if not result or result["status"] == "failed":
                result = fetch_one(row)
                cache[row["id"]] = result
                write_json(cache_path, cache)
            if result["status"] == "failed":
                if result["error_type"] in {"ConnectError", "ConnectTimeout"}:
                    raise ConnectionError(result["error"])
                dispositions[row["id"]] = "download_or_decode_failure"
                continue
            if is_exposed(result["fingerprints"], signatures, 4):
                dispositions[row["id"]] = "near_or_exact_duplicate"
                continue
            signatures.append(result["fingerprints"])
            selected.append(
                dict(
                    id="fresh_" + name,
                    source_id=row["source_id"],
                    source_product_id=row["id"],
                    text=None,
                    image_path=result["image_path"],
                    exclude_id=None,
                    acquisition_bucket=name,
                    image_fingerprints=result["fingerprints"],
                )
            )
            dispositions[row["id"]] = "query_selected_without_relevance_filter"
            break
    write_rows(ROOT / "queries.evaluator.jsonl", selected)
    public = [
        {k: q[k] for k in ["id", "text", "image_path", "exclude_id"]} for q in selected
    ]
    write_json(ROOT / "queries.public.json", public)
    records = [
        dict(
            **r,
            disposition=dispositions.get(r["id"], "fetched_not_selected"),
            project_role="exposed_query_development_media_not_acceptance",
        )
        for r in cache.values()
    ]
    write_rows(REGISTRY, records)
    sheets(selected)
    write_json(
        ROOT / "acquisition_receipt.json",
        dict(
            requested=6,
            selected=len(selected),
            metadata=len(rows),
            fetched=len(cache),
            disposition=dict(Counter(dispositions.values())),
            files={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                for p in [
                    REGISTRY,
                    ROOT / "queries.evaluator.jsonl",
                    ROOT / "queries.public.json",
                    *sorted((ROOT / "native_review").glob("*")),
                ]
            },
            selection_uses_retrieval_results=False,
            model_inferences=0,
            acceptance_predictions=0,
        ),
    )
    print(
        json.dumps(
            dict(
                acquired=len(selected),
                requested=6,
                fetched=len(cache),
                acceptance_predictions=0,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "download"])
    args = parser.parse_args()
    {"prepare": prepare, "download": download}[args.action]()
