"""Freeze a train-only form-specific supplement before downloading or scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
from prepare_focus_coverage import ROOT as STAGE
from prepare_focus_coverage import check as check_first

ROOT = STAGE / "supplement"
QUEUE = PROJECT_ROOT / "data/module2_form_coverage_candidates/records.jsonl"
REGISTRY = PROJECT_ROOT / "data/module2_form_coverage_media/records.jsonl"
PARENT_SNAPSHOT = (
    PROJECT_ROOT
    / "artifacts/module2_visual_coverage/post_preparation_exclusion_snapshot.json"
)


class _PolicySpec(TypedDict):
    """Exact fields of the POLICY configuration and provenance record."""

    seed: int
    source_split: str
    scan_after: int
    scan_limit: int
    metadata_per_bucket: int
    target_per_bucket: int
    workers: int
    duplicate_dhash_distance: int
    role: str
    selection_after_initial_21_visual_review: bool
    selection_after_retrieval_scores: bool
    automatic_relevance_labels: bool
    acceptance_use_allowed: bool


POLICY: _PolicySpec = dict(
    seed=2026102311,
    source_split="train",
    scan_after=200000,
    scan_limit=1000000,
    metadata_per_bucket=24,
    target_per_bucket=8,
    workers=8,
    duplicate_dhash_distance=4,
    role="query_aware_exposed_form_coverage_development_only",
    selection_after_initial_21_visual_review=True,
    selection_after_retrieval_scores=False,
    automatic_relevance_labels=False,
    acceptance_use_allowed=False,
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
    text = title.casefold()
    if (
        any(t in text for t in ["调料", "调味"])
        and "架" in text
        and any(t in text for t in ["双层", "两层", "二层", "2层", "多层"])
        and any(t in text for t in ["壁挂", "挂墙", "墙", "免打孔", "挂式"])
        and not any(t in text for t in ["浴室", "转角", "三角", "角码", "洞洞"])
    ):
        return "tiered_wall_spice_racks"
    if (
        "餐边柜" in text
        and any(t in text for t in ["红木", "花梨", "紫檀", "明清", "仿古"])
        and not any(
            t in text
            for t in [
                "高柜",
                "立柜",
                "顶柜",
                "展示柜",
                "玻璃",
                "鞋柜",
                "博古",
                "多宝",
                "展示架",
                "凳",
            ]
        )
    ):
        return "classical_low_cabinets"
    return None


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        FileExistsError: Preserve form-specific acquisition.

    Notes:
        Writes derived artifacts through QUEUE.parent.mkdir, ROOT.mkdir, write_json,
        write_rows.
    """
    check_first()
    if ROOT.exists() or QUEUE.exists() or REGISTRY.exists():
        raise FileExistsError("Preserve form-specific acquisition")
    source = PROTECTED_ROOT / "多模态电商数据/product1m_product5m_train_id_label.json"
    prior = load(PARENT_SNAPSHOT)
    verify_files(PROJECT_ROOT, prior["files"])
    for row in read_rows(
        PROJECT_ROOT / "data/module2_focus_coverage_media/records.jsonl"
    ):
        if row.get("image_fingerprints"):
            prior["image_fingerprints"].append(row["image_fingerprints"])
    prior["files"]["data/module2_focus_coverage_media/records.jsonl"] = sha256_file(
        PROJECT_ROOT / "data/module2_focus_coverage_media/records.jsonl"
    )
    ROOT.mkdir(parents=True)
    write_json(ROOT / "prior_exclusion.json", prior)
    write_json(
        ROOT / "protocol.json",
        dict(
            policy=POLICY,
            source=source.relative_to(PROJECT_ROOT).as_posix(),
            source_sha256=sha256_file(source),
            implementation_sha256=sha256_file(Path(__file__)),
            prior_exclusion_sha256=sha256_file(ROOT / "prior_exclusion.json"),
            initial_observations_sha256=sha256_file(
                STAGE / "native_audit/observations.tsv"
            ),
            heldout_pixels_opened=0,
            formal_metrics=None,
            goal_met=False,
            limitation="Adaptive development coverage acquisition after inspecting initial candidates; not independent validation or strategy gain",
        ),
    )
    excluded = set(prior["family_keys"])
    groups: dict = {
        b: [] for b in ["tiered_wall_spice_racks", "classical_low_cabinets"]
    }
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
                        label=raw.get("label"),
                        text=raw["title"],
                        image_url=raw["url"],
                        attributes=parse_pv(raw.get("pv") or ""),
                        bucket=name,
                        source_position=number,
                        source_record_sha256=canonical_hash(raw),
                    )
                    keys = product_family_keys(row)
                    if not keys & excluded:
                        row.update(
                            family_keys=sorted(keys),
                            priority=hashlib.sha256(
                                f"{POLICY['seed']}:{row['id']}".encode()
                            ).hexdigest(),
                            project_role="exposed_development_metadata_not_relevance_truth",
                            image_path=None,
                        )
                        groups[name].append(row)
            if number == POLICY["scan_limit"]:
                break
    chosen, seen = [], set(excluded)
    for name, values in groups.items():
        count = 0
        for row in sorted(values, key=lambda r: r["priority"]):
            if set(row["family_keys"]) & seen:
                continue
            chosen.append(row)
            seen.update(row["family_keys"])
            count += 1
            if count == POLICY["metadata_per_bucket"]:
                break
    write_rows(ROOT / "source_candidates.jsonl", chosen)
    QUEUE.parent.mkdir()
    write_rows(QUEUE, chosen)
    write_json(
        ROOT / "metadata_receipt.json",
        dict(
            scanned=number,
            eligible_before_family_dedup={b: len(v) for b, v in groups.items()},
            metadata=len(chosen),
            per_bucket=dict(Counter(r["bucket"] for r in chosen)),
            queue_sha256=sha256_file(QUEUE),
            source_candidates_sha256=sha256_file(ROOT / "source_candidates.jsonl"),
            model_inferences=0,
            labels_created=0,
        ),
    )
    print(
        json.dumps(
            dict(
                scanned=number,
                metadata=len(chosen),
                per_bucket=dict(Counter(r["bucket"] for r in chosen)),
                model_inferences=0,
            )
        )
    )


def check() -> None:
    """Verify the prerequisites and integrity of the current workflow."""
    protocol = load(ROOT / "protocol.json")
    assert (
        protocol["policy"] == POLICY
        and sha256_file(Path(__file__)) == protocol["implementation_sha256"]
    )
    assert (
        sha256_file(ROOT / "prior_exclusion.json") == protocol["prior_exclusion_sha256"]
    )
    assert (
        sha256_file(STAGE / "native_audit/observations.tsv")
        == protocol["initial_observations_sha256"]
    )
    receipt = load(ROOT / "metadata_receipt.json")
    assert (
        sha256_file(QUEUE) == receipt["queue_sha256"]
        and sha256_file(ROOT / "source_candidates.jsonl")
        == receipt["source_candidates_sha256"]
    )
    verify_files(PROJECT_ROOT, load(ROOT / "prior_exclusion.json")["files"])


def fetch_one(row: dict) -> dict:
    """Return fetch one for this workflow.

    Args:
        row: Row used by this operation.

    Returns:
        Mapping with fields: id, status, path, final_url, fingerprints, width,
        height, error_type, error.

    Notes:
        Writes derived artifacts through directory.mkdir, temporary.write_bytes.
    """
    try:
        with httpx.Client(timeout=20, trust_env=False) as client:
            content, url = fetch_image(client, row["image_url"])
        info = inspect_image(content)
        directory = ROOT / "dataset/media"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (info["sha256"] + info["suffix"])
        if not path.exists():
            temporary = path.with_name(
                path.name
                + "."
                + hashlib.sha256(row["id"].encode()).hexdigest()[:12]
                + ".tmp"
            )
            temporary.write_bytes(content)
            temporary.replace(path)
        return dict(
            id=row["id"],
            status="downloaded",
            path=path.relative_to(ROOT).as_posix(),
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


def download() -> None:
    """Apply the download workflow step.

    Raises:
        FileExistsError: Preserve downloaded supplement.

    Notes:
        Writes derived artifacts through REGISTRY.parent.mkdir, write_json, write_rows.
    """
    check()
    if (ROOT / "dataset/manifest.json").exists():
        raise FileExistsError("Preserve downloaded supplement")
    rows = read_rows(QUEUE)
    prior = load(ROOT / "prior_exclusion.json")
    cache_path = ROOT / "download_cache.json"
    cache = load(cache_path) if cache_path.exists() else {}
    if not cache:
        probe = fetch_one(rows[0])
        if probe["status"] == "failed" and probe["error_type"] in {
            "ConnectError",
            "ConnectTimeout",
        }:
            raise ConnectionError(probe["error"])
        cache[rows[0]["id"]] = probe
        write_json(cache_path, cache)
    buckets = list(dict.fromkeys(r["bucket"] for r in rows))
    by_bucket = {b: [r for r in rows if r["bucket"] == b] for b in buckets}
    chosen: dict[str, list[dict]]
    signatures: list
    chosen, offsets, signatures, dispositions = (
        {b: [] for b in buckets},
        dict.fromkeys(buckets, 0),
        [],
        {},
    )
    while any(len(v) < POLICY["target_per_bucket"] for v in chosen.values()):
        pending = []
        for name in buckets:
            n = POLICY["target_per_bucket"] - len(chosen[name])
            start = offsets[name]
            pending.extend(by_bucket[name][start : start + n])
            offsets[name] = min(len(by_bucket[name]), start + n)
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=POLICY["workers"]) as workers:
            for result in workers.map(
                fetch_one, [r for r in pending if r["id"] not in cache]
            ):
                cache[result["id"]] = result
        write_json(cache_path, cache)
        for row in pending:
            result = cache[row["id"]]
            if result["status"] != "downloaded":
                disposition = "download_or_decode_failure"
            elif is_exposed(result["fingerprints"], prior["image_fingerprints"], 4):
                disposition = "prior_or_reserved_duplicate_risk"
            elif is_exposed(result["fingerprints"], signatures, 4):
                disposition = "within_supplement_duplicate_risk"
            else:
                disposition = "selected_for_AI_review"
                signatures.append(result["fingerprints"])
                chosen[row["bucket"]].append(
                    dict(
                        row,
                        image_path=str(ROOT / result["path"]),
                        fingerprints=result["fingerprints"],
                        catalog_QA_status="pending_AI",
                        relevance_grade=None,
                    )
                )
            dispositions[row["id"]] = disposition
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        write_rows(
            REGISTRY,
            [
                dict(
                    r,
                    media=cache[r["id"]],
                    disposition=dispositions[r["id"]],
                    image_path=(
                        str(ROOT / cache[r["id"]]["path"])
                        if cache[r["id"]]["status"] == "downloaded"
                        else None
                    ),
                    image_fingerprints=cache[r["id"]].get("fingerprints"),
                )
                for r in rows
                if r["id"] in dispositions
            ],
        )
        print(
            json.dumps(
                dict(selected=sum(map(len, chosen.values())), attempted=len(cache))
            ),
            flush=True,
        )
    write_rows(
        ROOT / "dataset/products.jsonl",
        [r for values in chosen.values() for r in values],
    )
    write_json(
        ROOT / "dataset/manifest.json",
        dict(
            products=sum(map(len, chosen.values())),
            per_bucket={b: len(v) for b, v in chosen.items()},
            dispositions=dict(Counter(dispositions.values())),
            attempted=len(cache),
            registry_sha256=sha256_file(REGISTRY),
            heldout_pixels_opened=0,
            labels_created=0,
            model_inferences=0,
            files={
                p.relative_to(ROOT / "dataset").as_posix(): sha256_file(p)
                for p in (ROOT / "dataset").rglob("*")
                if p.is_file()
            },
        ),
    )


def sheets() -> None:
    """Apply the sheets workflow step.

    Notes:
        Writes derived artifacts through canvas.save, folder.mkdir, write_json.
    """
    check()
    rows = read_rows(ROOT / "dataset/products.jsonl")
    folder = ROOT / "native_audit"
    folder.mkdir()
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17)
    mapping = []
    for start in range(0, len(rows), 8):
        canvas = Image.new("RGB", (1440, 1000), "white")
        draw = ImageDraw.Draw(canvas)
        for j, row in enumerate(rows[start : start + 8]):
            x, y = (j % 4) * 360, (j // 4) * 500
            with Image.open(row["image_path"]) as source:
                thumb = ImageOps.exif_transpose(source).convert("RGB")
                thumb.thumbnail((348, 345))
                canvas.paste(
                    thumb,
                    (x + (360 - thumb.width) // 2, y + 32 + (345 - thumb.height) // 2),
                )
            draw.text(
                (x + 6, y + 3),
                f"{start+j+1:02} {row['source_id']}",
                font=font,
                fill="black",
            )
            for line in range(5):
                draw.text(
                    (x + 6, y + 382 + line * 22),
                    row["text"][line * 19 : (line + 1) * 19],
                    font=font,
                    fill="black",
                )
            mapping.append(
                dict(
                    number=start + j + 1,
                    id=row["id"],
                    image_sha256=sha256_file(Path(row["image_path"])),
                )
            )
        canvas.save(folder / f"catalog_{start//8+1:02}.jpg", quality=94)
    write_json(folder / "mapping.json", mapping)
    print(json.dumps(dict(products=len(rows), contact_sheets=(len(rows) + 7) // 8)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "download", "sheets"])
    globals()[parser.parse_args().action]()
