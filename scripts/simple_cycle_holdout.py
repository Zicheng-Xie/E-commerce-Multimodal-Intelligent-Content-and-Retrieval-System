"""Reserve new final-acceptance media using metadata and fingerprints only.

No model, relevance label, prediction or metric is read or produced here.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import ijson

import run  # noqa: F401
from multimodal_ecommerce.business_acceptance import FINE_TYPES
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.media import fetch_image, inspect_image
from multimodal_ecommerce.reservation_sampling import sample_reservation_metadata
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    PROTECTED_ROOT,
    sha256_file,
    write_json,
)

ROOT = PROJECT_ROOT / "artifacts/module2_final_acceptance_20261105"
EXPOSURE = (
    PROJECT_ROOT
    / "artifacts/module2_role_pool_development/post_experiment_exclusion_snapshot.json"
)
SOURCE = PROTECTED_ROOT / "多模态电商数据/product1m_product5m_test_id_label.json"
POLICY: dict[str, Any] = dict(
    seed=2026110502,
    source_split="test",
    prefix_scan_limit=400000,
    queue_per_fine_type=80,
    gallery_per_fine_type=10,
    image_queries_per_fine_type=1,
    text_queries_per_fine_type=1,
    fine_types=FINE_TYPES,
    dhash_threshold=4,
    selection_uses="metadata hash order, download validity and prior-exposure fingerprints only",
    model_inference_allowed=False,
    training_allowed=False,
    semantic_preview_allowed=False,
    final_model_freeze_required=True,
    independent_human_queries_and_gold_required=True,
)


def load(path: Path) -> Any:
    """Read metadata without opening any holdout image semantically.

    Args:
        path: Existing local metadata JSON path.

    Returns:
        Decoded JSON metadata at this serialization boundary.
    """
    return json.loads(path.read_text())


def retry_network_failures() -> None:
    """Archive sandbox connection failures before retrying the identical queue.

    Failed URLs and image decoding failures remain cached. No candidate order or
    exclusion rule changes; network availability cannot be a relevance filter.
    """
    path = ROOT / "technical_downloads.json"
    cache = load(path)
    archive = ROOT / "sandbox_network_failures.json"
    if archive.exists():
        raise FileExistsError("The original network-failure receipt already exists")
    shutil.copyfile(path, archive)
    retained = {
        key: value
        for key, value in cache.items()
        if value.get("error_type") != "ConnectError"
    }
    write_json(path, retained)
    write_json(
        ROOT / "network_recovery.json",
        dict(
            archived_sha256=sha256_file(archive),
            archived_entries=len(cache),
            retry_connection_failures=len(cache) - len(retained),
            preserved_other_results=len(retained),
            unchanged_queue_sha256=sha256_file(ROOT / "candidates.reserved.json"),
        ),
    )


def prepare() -> None:
    """Freeze a new hash-sampled metadata queue, excluding every known exposure."""
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "reservation_protocol.json").exists():
        return
    previous = load(EXPOSURE)
    excluded = set(previous["family_keys"]) | {
        "id:" + str(i) for i in previous["product_source_ids"]
    }
    fine = {
        fine: category for category, labels in FINE_TYPES.items() for fine in labels
    }
    with SOURCE.open("rb") as handle:
        queue, counts, scanned = sample_reservation_metadata(
            ijson.kvitems(handle, ""), POLICY, excluded
        )
    write_json(ROOT / "candidates.reserved.json", queue)
    write_json(
        ROOT / "reservation_protocol.json",
        dict(
            policy=POLICY,
            queue_sha256=sha256_file(ROOT / "candidates.reserved.json"),
            source_sha256=sha256_file(SOURCE),
            exclusion_sha256=sha256_file(EXPOSURE),
            scanned=scanned,
            available=dict(counts),
            metadata_shortage={
                label: counts[label] for label in fine if counts[label] < 11
            },
            model_inference_allowed=False,
            training_allowed=False,
            consumed=False,
            current_status="metadata_reserved_not_final_acceptance_ready",
            formal_metrics=None,
        ),
    )
    print(
        json.dumps(dict(queue=len(queue), scanned=scanned, model_predictions=0)),
        flush=True,
    )


def download_one(row: dict) -> dict:
    """Fetch one source URL with bounded decoding and retain technical failures.

    Args:
        row: One record from the frozen reservation queue.

    Returns:
        Technical success or failure; no semantic or relevance judgment.
    """
    try:
        with httpx.Client(timeout=12, trust_env=False) as client:
            content, url = fetch_image(client, row["image_url"])
        info = inspect_image(content)
        path = ROOT / "media" / (info["sha256"] + info["suffix"])
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
        return dict(
            id=row["id"],
            status="downloaded",
            image_path=str(path.relative_to(ROOT)),
            fingerprints=fingerprints(path),
            final_url=url,
        )
    except (OSError, ValueError, httpx.HTTPError) as error:
        logging.getLogger(__name__).debug(
            "Holdout source download failed", exc_info=True
        )
        return dict(
            id=row["id"],
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
        )


def reserve_media() -> None:
    """Freeze complete image identities without viewing, encoding or scoring them."""
    protocol = load(ROOT / "reservation_protocol.json")
    if (ROOT / "frozen_media_manifest.json").exists():
        return
    if (
        sha256_file(ROOT / "candidates.reserved.json") != protocol["queue_sha256"]
        or sha256_file(EXPOSURE) != protocol["exclusion_sha256"]
    ):
        raise ValueError("Reserved metadata or prior exposure changed")
    queue = load(ROOT / "candidates.reserved.json")
    prior = load(EXPOSURE)
    used = set(prior["family_keys"])
    signatures = list(prior["image_fingerprints"])
    cache_path = ROOT / "technical_downloads.json"
    cache = load(cache_path) if cache_path.exists() else {}
    chosen: dict = {label: [] for labels in FINE_TYPES.values() for label in labels}
    positions = dict.fromkeys(chosen, 0)
    reasons: Counter[str] = Counter()
    while True:
        pending = []
        for label in chosen:
            values = [r for r in queue if r["label"] == label]
            needed = 11 - len(chosen[label])
            count = min(max(0, needed), 4)
            batch = values[positions[label] : positions[label] + count]
            positions[label] += len(batch)
            pending.extend(batch)
        if not pending:
            break
        missing = [r for r in pending if r["id"] not in cache]
        with ThreadPoolExecutor(max_workers=8) as workers:
            for result in workers.map(download_one, missing):
                cache[result["id"]] = result
        write_json(cache_path, cache)
        for row in pending:
            result = cache[row["id"]]
            if result["status"] != "downloaded":
                reasons["download_failed"] += 1
                continue
            if set(row["family_keys"]) & used:
                reasons["family_overlap"] += 1
                continue
            if is_exposed(result["fingerprints"], signatures, 4):
                reasons["byte_pixel_or_near_overlap"] += 1
                continue
            used.update(row["family_keys"])
            signatures.append(result["fingerprints"])
            chosen[row["label"]].append(
                dict(row, **{k: v for k, v in result.items() if k != "id"})
            )
        ready = sum(len(rows) for rows in chosen.values())
        write_json(
            ROOT / "status.json",
            dict(
                phase="technical_media_reservation",
                selected=ready,
                target=330,
                failures=dict(reasons),
                model_predictions=0,
                semantic_image_views=0,
                formal_metrics=None,
            ),
        )
        print(
            f"Reserved media {ready}/330; technical attempts {len(cache)}", flush=True
        )
        if all(len(rows) == 11 for rows in chosen.values()):
            break
    shortage = {
        label: 11 - len(rows) for label, rows in chosen.items() if len(rows) < 11
    }
    if shortage:
        write_json(
            ROOT / "shortage.json",
            dict(
                shortage=shortage,
                reasons=dict(reasons),
                final_acceptance_ready=False,
                model_predictions=0,
            ),
        )
        return
    gallery = []
    queries = []
    for label, rows in chosen.items():
        gallery.extend(rows[:10])
        queries.append(rows[10])
    write_json(ROOT / "gallery.reserved.json", gallery)
    write_json(ROOT / "image_queries.reserved.json", queries)
    import csv

    with (ROOT / "text_queries.human_pending.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fine_type",
                "human_author",
                "independent_query_text",
                "approved_at",
            ],
        )
        writer.writeheader()
        writer.writerows(
            dict(
                fine_type=label,
                human_author="",
                independent_query_text="",
                approved_at="",
            )
            for label in chosen
        )
    selected = gallery + queries
    write_json(
        ROOT / "frozen_media_manifest.json",
        dict(
            policy=POLICY,
            files={
                r["image_path"]: sha256_file(ROOT / r["image_path"]) for r in selected
            },
            gallery=len(gallery),
            image_queries=len(queries),
            text_queries_pending=30,
            human_relevance_labels=0,
            model_predictions=0,
            consumed=False,
            model_inference_allowed=False,
            training_allowed=False,
            current_status="media_frozen_pending_final_model_and_human_queries_gold",
            prior_exclusion_sha256=protocol["exclusion_sha256"],
            formal_metrics=None,
        ),
    )
    write_json(
        ROOT / "status.json",
        dict(
            phase="media_frozen",
            gallery=300,
            image_queries=30,
            text_queries_pending=30,
            model_predictions=0,
            formal_acceptance_ready=False,
        ),
    )


def main() -> None:
    """Create the protected metadata queue or reserve its technically valid media."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["prepare", "download", "worker", "retry-network"]
    )
    action = parser.parse_args().action
    if action == "worker":
        prepare()
        reserve_media()
    else:
        {
            "prepare": prepare,
            "download": reserve_media,
            "retry-network": retry_network_failures,
        }[action]()


if __name__ == "__main__":
    main()
