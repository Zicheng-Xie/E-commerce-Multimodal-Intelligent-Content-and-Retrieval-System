"""Acquire every frozen subject-coverage candidate; never select by relevance."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont, ImageOps

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.holdout_data import is_exposed
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from primary_form_coverage import fetch_one, receipt
from subject_roi_experiment import load, once

ROOT = PROJECT_ROOT / "artifacts/module2_subject_coverage_expansion"
PARENT = PROJECT_ROOT / "artifacts/module2_subject_roi"
QUEUE = PROJECT_ROOT / "data/module2_subject_coverage_candidates/records.jsonl"
MEDIA = PROJECT_ROOT / "data/module2_subject_coverage_media/records.jsonl"


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        FileExistsError: Preserve frozen acquisition.

    Notes:
        Writes derived artifacts through once.
    """
    if (ROOT / "acquisition_protocol.json").exists():
        raise FileExistsError("Preserve frozen acquisition")
    verify_files(PARENT, load(PARENT / "completion_receipt.json")["files"])
    assert (
        sha256_file(QUEUE)
        == "3fcb47a2c85faaa2ee38f34a95019ddd8a5bdf81689dd2628724589c4d601d35"
    )
    paths = [
        Path(__file__),
        QUEUE,
        PARENT / "completion_receipt.json",
        PARENT / "post_experiment_exclusion_snapshot.json",
        PARENT / "next_coverage/post_preparation_exclusion_snapshot.json",
    ]
    once(
        ROOT / "acquisition_protocol.json",
        dict(
            role="adaptive_exposed_development_only_not_acceptance",
            scope="Latest explicit user instruction: attempt and review all 80; retain every technical-success nonduplicate. Supersedes previous proposed four-per-bucket quota.",
            metadata_candidates=80,
            technical_quota=None,
            relevance_filter=False,
            preserve_all_attempts=True,
            dhash_distance=4,
            review_origin="model",
            human_labels=0,
            fixed_k=10,
            precision_required=0.6,
            recall_preferred=0.85,
            formal_metrics=None,
            default_promotion_allowed=False,
            files={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in paths
            },
        ),
    )


def check() -> None:
    """Verify the prerequisites and integrity of the current workflow."""
    verify_files(PROJECT_ROOT, load(ROOT / "acquisition_protocol.json")["files"])


def acquire() -> None:
    """Apply the acquire workflow step.

    Raises:
        FileExistsError: Preserve completed acquisition.

    Notes:
        Writes derived artifacts through once, receipt, write_json, write_rows.
    """
    check()
    if (ROOT / "acquisition_receipt.json").exists():
        raise FileExistsError("Preserve completed acquisition")
    rows = read_rows(QUEUE)
    cache_path = ROOT / "download_cache.json"
    cache = load(cache_path) if cache_path.exists() else {}
    pending = [r for r in rows if r["id"] not in cache]
    # Public CDN GET only. Respect the configured network proxy.
    with httpx.Client(timeout=25, trust_env=True) as client:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for row, result in zip(
                pending, pool.map(lambda r: fetch_one(client, r, ROOT), pending)
            ):
                cache[row["id"]] = result
                write_json(cache_path, cache)
                print(
                    dict(
                        attempted=len(cache), total=len(rows), status=result["status"]
                    ),
                    flush=True,
                )
    signatures = list(
        load(PARENT / "post_experiment_exclusion_snapshot.json")["image_fingerprints"]
    )
    chosen, attempts = [], []
    for row in rows:
        entry = cache[row["id"]]
        duplicate = entry["status"] == "downloaded" and is_exposed(
            entry["fingerprints"], signatures, 4
        )
        selected = entry["status"] == "downloaded" and not duplicate
        attempts.append(
            dict(
                **row,
                **entry,
                selected_for_gallery=selected,
                technical_duplicate=bool(duplicate),
                review_origin="model",
                human_label=False,
            )
        )
        if selected:
            chosen.append(dict(**row, **entry))
            signatures.append(entry["fingerprints"])
    write_rows(MEDIA, attempts)
    write_rows(ROOT / "selected.evaluator.jsonl", chosen)
    once(
        ROOT / "additions/catalog.public.json",
        [dict(id=r["id"], title=r["text"], image_path=r["image_path"]) for r in chosen],
    )
    receipt(
        ROOT,
        "acquisition_receipt.json",
        [
            ROOT / "download_cache.json",
            ROOT / "selected.evaluator.jsonl",
            ROOT / "additions/catalog.public.json",
        ],
        candidates=len(rows),
        attempted=len(cache),
        downloaded=sum(r["status"] == "downloaded" for r in attempts),
        failed=sum(r["status"] != "downloaded" for r in attempts),
        technical_duplicates=sum(r["technical_duplicate"] for r in attempts),
        selected=len(chosen),
        selected_by_bucket=dict(Counter(r["bucket"] for r in chosen)),
        media_registry_sha256=sha256_file(MEDIA),
        all_technical_success_nonduplicates_retained=True,
    )
    print(load(ROOT / "acquisition_receipt.json"), flush=True)


def sheets() -> None:
    """Apply the sheets workflow step.

    Notes:
        Writes derived artifacts through canvas.save, folder.mkdir, receipt.
    """
    check()
    verify_files(ROOT, load(ROOT / "acquisition_receipt.json")["files"])
    rows = read_rows(MEDIA)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
    selected = {
        r["id"]: i
        for i, r in enumerate(read_rows(ROOT / "selected.evaluator.jsonl"), 1)
    }
    folder = ROOT / "native"
    folder.mkdir(exist_ok=True)
    for start in range(0, len(rows), 8):
        canvas = Image.new("RGB", (1600, 820), "white")
        draw = ImageDraw.Draw(canvas)
        for off, row in enumerate(rows[start : start + 8]):
            x, y = off % 4 * 400, off // 4 * 410
            if row["status"] == "downloaded":
                with Image.open(row["image_path"]) as im:
                    pic = ImageOps.contain(
                        ImageOps.exif_transpose(im).convert("RGB"), (390, 300)
                    )
                canvas.paste(pic, (x + (400 - pic.width) // 2, y))
            label = f"candidate {start + off + 1}; gallery {selected.get(row['id'], '-')}; {row['status']}"
            draw.text((x + 3, y + 300), label, fill="black", font=font)
            for line in range(4):
                draw.text(
                    (x + 3, y + 325 + 19 * line),
                    row["text"][line * 22 : (line + 1) * 22],
                    fill="black",
                    font=font,
                )
        path = folder / f"candidates_{start // 8 + 1:02d}.jpg"
        if path.exists():
            raise FileExistsError(path)
        canvas.save(path, quality=97)
    receipt(
        ROOT,
        "native_image_receipt.json",
        sorted(folder.glob("candidates_*.jpg")),
        attempted_candidates=80,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "acquire", "sheets"])
    globals()[parser.parse_args().action]()
