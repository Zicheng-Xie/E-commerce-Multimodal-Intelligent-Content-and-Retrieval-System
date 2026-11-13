"""One identical-URL retry for transient public CDN failures, retaining round one."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

import run  # noqa: F401
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.holdout_data import is_exposed
from multimodal_ecommerce.safety import sha256_file, write_json
from primary_form_coverage import fetch_one, receipt
from subject_coverage_acquisition import PARENT, QUEUE, ROOT, check
from subject_roi_experiment import load, once


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve final acquisition.

    Notes:
        Writes derived artifacts through once, receipt, write_json, write_rows.
    """
    check()
    folder = ROOT / "retry"
    if (folder / "receipt.json").exists():
        raise FileExistsError("Preserve final acquisition")
    once(
        folder / "protocol.json",
        dict(
            code_sha256=sha256_file(Path(__file__)),
            max_additional_attempts=1,
            same_url_headers_and_proxy=True,
            access_workarounds=False,
            preserve_original_acquisition=True,
            filtering="technical decoding and deduplication only",
        ),
    )
    original = load(ROOT / "download_cache.json")
    cache_path = folder / "cache.json"
    retry = load(cache_path) if cache_path.exists() else {}
    rows = read_rows(QUEUE)
    pending = [
        r
        for r in rows
        if original[r["id"]]["status"] != "downloaded" and r["id"] not in retry
    ]
    with httpx.Client(timeout=25, trust_env=True) as client:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for row, result in zip(
                pending, pool.map(lambda r: fetch_one(client, r, ROOT), pending)
            ):
                retry[row["id"]] = result
                write_json(cache_path, retry)
                print(dict(retried=len(retry), status=result["status"]), flush=True)
    signatures = list(
        load(PARENT / "post_experiment_exclusion_snapshot.json")["image_fingerprints"]
    )
    selected, final = [], []
    for row in rows:
        entry = retry.get(row["id"], original[row["id"]])
        duplicate = entry["status"] == "downloaded" and is_exposed(
            entry["fingerprints"], signatures, 4
        )
        accept = entry["status"] == "downloaded" and not duplicate
        final.append(
            dict(
                **row,
                **entry,
                selected_for_gallery=accept,
                technical_duplicate=bool(duplicate),
                attempt_count=1 + int(row["id"] in retry),
            )
        )
        if accept:
            signatures.append(entry["fingerprints"])
            selected.append(dict(**row, **entry))
    write_rows(folder / "all_attempts.final.jsonl", final)
    write_rows(folder / "selected.final.jsonl", selected)
    once(
        ROOT / "additions/catalog.final.public.json",
        [
            dict(id=r["id"], title=r["text"], image_path=r["image_path"])
            for r in selected
        ],
    )
    receipt(
        folder,
        "receipt.json",
        [
            folder / "cache.json",
            folder / "selected.final.jsonl",
            folder / "all_attempts.final.jsonl",
            folder / "protocol.json",
        ],
        candidates=80,
        retry_attempts=len(retry),
        downloaded=sum(r["status"] == "downloaded" for r in final),
        failed=sum(r["status"] != "downloaded" for r in final),
        selected=len(selected),
        final_catalog_sha256=sha256_file(ROOT / "additions/catalog.final.public.json"),
    )
    print(load(folder / "receipt.json"), flush=True)


if __name__ == "__main__":
    main()
