"""Finish a consumed heldout's gallery contact sheets after path-base failure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

import run  # noqa: F401
from independent_business_blind import (
    CONSUMED,
    DATASET,
    RESERVED,
    ROOT,
    check,
    font,
)
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through canvas.save, write_json, write_rows.
    """
    check()
    assert (ROOT / "heldout_unsealed.json").exists() and CONSUMED.exists()
    assert not (ROOT / "profile_receipt.json").exists()
    assert not list((ROOT / "native").glob("gallery_*.jpg"))
    lineage = {
        r["opaque_id"]: r for r in read_rows(DATASET / "evaluator/lineage.jsonl")
    }
    old = json.loads((ROOT / "catalog.public.json").read_text(encoding="utf-8"))
    write_json(
        ROOT / "gallery_path_failure_execution.json",
        dict(
            catalog_before_repair=old,
            error="Frozen gallery products image_path is relative to gallery/, not dataset/. First Image.open failed with FileNotFoundError; zero gallery contact sheets were produced.",
            no_model_or_parameter_change=True,
            original_frozen_script_unchanged=True,
            implementation_sha256=sha256_file(
                PROJECT_ROOT / "scripts/independent_business_blind.py"
            ),
            continuation_sha256=sha256_file(Path(__file__)),
            release_sha256=sha256_file(ROOT / "release.json"),
            original_manifest_sha256=sha256_file(DATASET / "manifest.json"),
            reserved_preparation_sha256=sha256_file(
                RESERVED / "preparation_receipt.json"
            ),
            consumed_heldout_reset=False,
        ),
    )
    rows = []
    for row in old:
        record = lineage[row["id"]]
        assert record["role"] == "gallery"
        path = DATASET / record["image_path"]
        assert sha256_file(path) == record["fingerprints"]["sha256"]
        rows.append(
            dict(id=row["id"], title=row["title"], image_path=str(path.resolve()))
        )
    write_json(ROOT / "catalog.public.json", rows)
    write_rows(
        ROOT / "native/gallery_mapping.jsonl",
        [dict(position=i, **r) for i, r in enumerate(rows, 1)],
    )
    for start in range(0, 300, 10):
        canvas = Image.new("RGB", (2000, 800), "white")
        draw = ImageDraw.Draw(canvas)
        first, last = start + 1, start + 10
        draw.text(
            (8, 4),
            f"Gallery {first}–{last}; profiles before query-image viewing",
            fill="black",
            font=font(20),
        )
        for offset, row in enumerate(rows[start : start + 10]):
            x, y = (offset % 5) * 400, 38 + (offset // 5) * 380
            with Image.open(row["image_path"]) as im:
                thumb = ImageOps.contain(
                    ImageOps.exif_transpose(im).convert("RGB"), (388, 290)
                )
            canvas.paste(thumb, (x + (400 - thumb.width) // 2, y))
            draw.text(
                (x + 5, y + 291), str(start + offset + 1), fill="black", font=font(23)
            )
            draw.text((x + 5, y + 322), row["title"][:21], fill="black", font=font(17))
            draw.text(
                (x + 5, y + 347), row["title"][21:42], fill="black", font=font(16)
            )
        number = start // 10 + 1
        canvas.save(ROOT / f"native/gallery_{number:02d}.jpg", quality=96)
    status = json.loads((ROOT / "public_status.json").read_text(encoding="utf-8"))
    status.update(
        state="heldout_consumed_gallery_profile_authoring_no_predictions",
        heldout_unsealed=True,
        technical_gallery_path_repair="gallery_path_failure_execution.json",
    )
    write_json(ROOT / "public_status.json", status)
    print(
        json.dumps(
            dict(
                gallery_sheets=30,
                heldout_consumed=True,
                resampled=False,
                acceptance_predictions=0,
            )
        )
    )


if __name__ == "__main__":
    main()
