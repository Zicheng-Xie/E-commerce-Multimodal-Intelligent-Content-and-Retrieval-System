"""Bounded acquisition from the already frozen development-only image queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import httpx
from PIL import Image, ImageDraw, ImageFont, ImageOps

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.media import fetch_image, inspect_image
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

ROOT = PROJECT_ROOT / "artifacts/module2_focus_coverage"
PARENT = PROJECT_ROOT / "artifacts/module2_visual_coverage"
QUEUE = PROJECT_ROOT / "data/module2_focus_coverage_candidates/records.jsonl"
REGISTRY = PROJECT_ROOT / "data/module2_focus_coverage_media/records.jsonl"
SNAPSHOT = PARENT / "post_preparation_exclusion_snapshot.json"
RESERVED = PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl"


class _PolicySpec(TypedDict):
    """Exact fields of the POLICY configuration and provenance record."""

    role: str
    target_per_bucket: dict
    target_images: int
    original_requested_image_budget: int
    prior_metadata_shortfall: int
    workers: int
    duplicate_dhash_distance: int
    relevance_from_bucket_allowed: bool
    human_labels: int
    review_origin: str
    acceptance_inferences_allowed: bool
    public_quality_metrics_allowed: bool


POLICY: _PolicySpec = dict(
    role="query_aware_exposed_development_acquisition_only",
    target_per_bucket={"wall_bottle_racks": 9, "classical_storage_cabinets": 12},
    target_images=21,
    original_requested_image_budget=24,
    prior_metadata_shortfall=3,
    workers=8,
    duplicate_dhash_distance=4,
    relevance_from_bucket_allowed=False,
    human_labels=0,
    review_origin="model",
    acceptance_inferences_allowed=False,
    public_quality_metrics_allowed=False,
)


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def freeze() -> None:
    """Freeze the inputs and policy used by this workflow.

    Raises:
        FileExistsError: Preserve existing acquisition evidence.

    Notes:
        Writes derived artifacts through ROOT.mkdir, write_json.
    """
    if (ROOT / "acquisition_protocol.json").exists() or REGISTRY.exists():
        raise FileExistsError("Preserve existing acquisition evidence")
    parent = load(PARENT / "completion_receipt.json")
    verify_files(PARENT, parent["files"])
    verify_files(PROJECT_ROOT, parent["project_files"])
    rows = read_rows(QUEUE)
    assert len(rows) == 41 and len({r["id"] for r in rows}) == 41
    assert Counter(r["bucket"] for r in rows) == {
        "wall_bottle_racks": 9,
        "classical_storage_cabinets": 32,
    }
    reserved_keys = {k for r in read_rows(RESERVED) for k in r["family_keys"]}
    assert not {k for r in rows for k in r["family_keys"]} & reserved_keys
    ROOT.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(__file__),
        PROJECT_ROOT / "src/multimodal_ecommerce/media.py",
        PROJECT_ROOT / "src/multimodal_ecommerce/holdout_data.py",
    ]
    write_json(
        ROOT / "acquisition_protocol.json",
        dict(
            frozen_at=datetime.now(timezone.utc).isoformat(),
            policy=POLICY,
            files={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                for p in [
                    QUEUE,
                    SNAPSHOT,
                    RESERVED,
                    PARENT / "completion_receipt.json",
                    *sources,
                ]
            },
            ordered_candidates_sha256=sha256_file(QUEUE),
            selection="First nine wall-rack and twelve cabinet decodable nonduplicate images; ordered technical spares only, preserve any exhaustion and do not replace based on relevance",
            heldout_pixels_opened=0,
            formal_metrics=None,
            goal_met=False,
        ),
    )
    print(json.dumps(dict(frozen=True, candidates=41, target_images=21)))


def check() -> None:
    """Verify the prerequisites and integrity of the current workflow."""
    protocol = load(ROOT / "acquisition_protocol.json")
    assert protocol["policy"] == POLICY
    verify_files(PROJECT_ROOT, protocol["files"])


def fetch_one(row: dict) -> dict:
    """Return fetch one for this workflow.

    Args:
        row: Row used by this operation.

    Returns:
        Mapping with fields: id, status, path, final_url, fingerprints, width,
        height, error_type, error.

    Notes:
        Writes derived artifacts through folder.mkdir, temporary.write_bytes.
    """
    try:
        with httpx.Client(timeout=20, trust_env=False) as client:
            content, url = fetch_image(client, row["image_url"])
        info = inspect_image(content)
        folder = ROOT / "dataset/media"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (info["sha256"] + info["suffix"])
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
    except (httpx.HTTPError, ValueError, OSError) as error:
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
        FileExistsError: Preserve completed acquisition.

    Notes:
        Writes derived artifacts through REGISTRY.parent.mkdir, write_json, write_rows.
    """
    check()
    if (ROOT / "dataset/manifest.json").exists():
        raise FileExistsError("Preserve completed acquisition")
    rows = read_rows(QUEUE)
    snapshot = load(SNAPSHOT)
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
    queue = {b: [r for r in rows if r["bucket"] == b] for b in buckets}
    chosen: dict[str, list[dict]]
    reasons: Counter[str]
    signatures: list
    chosen, offsets, signatures, reasons, dispositions = (
        {b: [] for b in buckets},
        dict.fromkeys(buckets, 0),
        [],
        Counter(),
        {},
    )
    while any(len(v) < POLICY["target_per_bucket"][b] for b, v in chosen.items()):
        pending = []
        for bucket in buckets:
            needed = POLICY["target_per_bucket"][bucket] - len(chosen[bucket])
            start = offsets[bucket]
            pending.extend(queue[bucket][start : start + needed])
            offsets[bucket] = min(len(queue[bucket]), start + needed)
        if not pending:
            # Preserve shortage instead of inventing or relevance-filtering items.
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
            elif is_exposed(
                result["fingerprints"],
                snapshot["image_fingerprints"],
                POLICY["duplicate_dhash_distance"],
            ):
                disposition = "prior_or_reserved_duplicate_risk"
            elif is_exposed(
                result["fingerprints"], signatures, POLICY["duplicate_dhash_distance"]
            ):
                disposition = "new_image_duplicate_risk"
            else:
                assert (
                    sha256_file(ROOT / result["path"])
                    == result["fingerprints"]["sha256"]
                )
                chosen[row["bucket"]].append(
                    dict(
                        row,
                        image_path=str(ROOT / result["path"]),
                        fingerprints=result["fingerprints"],
                        catalog_QA_status="pending_AI",
                        relevance_grade=None,
                    )
                )
                signatures.append(result["fingerprints"])
                disposition = "selected_for_AI_review"
            dispositions[row["id"]] = disposition
            reasons[disposition] += 1
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        write_rows(
            REGISTRY,
            [
                dict(
                    r,
                    project_role="exposed_development_image_not_relevance_truth",
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
                dict(
                    selected=sum(map(len, chosen.values())),
                    target=POLICY["target_images"],
                    attempted=len(cache),
                )
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
            requested_target=POLICY["target_images"],
            shortfall={
                b: POLICY["target_per_bucket"][b] - len(v) for b, v in chosen.items()
            },
            per_bucket={b: len(v) for b, v in chosen.items()},
            dispositions=dict(reasons),
            attempted=len(cache),
            registry_sha256=sha256_file(REGISTRY),
            relevance_labels=0,
            heldout_pixels_opened=0,
            acceptance_predictions=0,
            files={
                p.relative_to(ROOT / "dataset").as_posix(): sha256_file(p)
                for p in (ROOT / "dataset").rglob("*")
                if p.is_file()
            },
        ),
    )


def sheets() -> None:
    """Apply the sheets workflow step.

    Raises:
        FileExistsError: Preserve reviewed pixels.

    Notes:
        Writes derived artifacts through canvas.save, out.mkdir, write_json.
    """
    check()
    rows = read_rows(ROOT / "dataset/products.jsonl")
    out = ROOT / "native_audit"
    out.mkdir(exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17)
    mapping = []
    for start in range(0, len(rows), 8):
        path = out / f"catalog_{start // 8 + 1:02}.jpg"
        if path.exists():
            raise FileExistsError("Preserve reviewed pixels")
        canvas = Image.new("RGB", (1440, 1000), "white")
        draw = ImageDraw.Draw(canvas)
        for j, row in enumerate(rows[start : start + 8]):
            x, y = (j % 4) * 360, (j // 4) * 500
            with Image.open(row["image_path"]) as im:
                thumb = ImageOps.exif_transpose(im).convert("RGB")
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
                    sheet=path.name,
                    image_sha256=sha256_file(Path(row["image_path"])),
                )
            )
        canvas.save(path, quality=94)
    write_json(out / "mapping.json", mapping)
    print(
        json.dumps(dict(contact_sheets=(len(mapping) + 7) // 8, products=len(mapping)))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "download", "sheets"])
    globals()[parser.parse_args().action]()
