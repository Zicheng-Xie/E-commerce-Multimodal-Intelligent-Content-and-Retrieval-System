"""Render existing copied annotations for a small, explicitly limited review."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import run  # noqa: F401
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Use a new review directory.
        ValueError: Source image escapes release.

    Notes:
        Writes derived artifacts through canvas.save, output.mkdir, write_json,
        write_rows.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/fashion_spotcheck_v1")
    )
    args = parser.parse_args()
    dataset = PROJECT_ROOT / "data/stage2_fashion_clean_v1/samples.jsonl"
    output = ensure_output_path(PROJECT_ROOT / args.output, dataset.parent)
    if output.exists():
        raise FileExistsError("Use a new review directory")
    rows = read_rows(dataset)
    selected = []
    for split in ("train", "validation", "test_a"):
        seen = set()
        for row in rows:
            if row["split"] != split:
                continue
            key = row.get("attribute_key") or tuple(
                sorted(i["category_name"] for i in row["instances"])
            )
            if key not in seen:
                selected.append(row)
                seen.add(key)
            if len(seen) == 3:
                break
    output.mkdir(parents=True)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
    index = []
    for number, row in enumerate(selected, 1):
        source = (dataset.parent / row["image_path"]).resolve()
        if not source.is_relative_to(dataset.parent.resolve()):
            raise ValueError("Source image escapes release")
        with Image.open(source) as opened:
            original = opened.convert("RGB")
        overlay = original.convert("RGBA")
        annotation = Image.new("RGBA", overlay.size)
        draw = ImageDraw.Draw(annotation)
        palette = [(255, 30, 30), (20, 210, 50), (10, 90, 255), (255, 180, 0)]
        for position, item in enumerate(row.get("instances", [])):
            color = palette[position % len(palette)]
            for polygon in item["segmentation"]:
                draw.polygon(
                    list(zip(polygon[::2], polygon[1::2])),
                    fill=(*color, 45),
                    outline=(*color, 255),
                    width=2,
                )
            x, y, w, h = item["bbox"]
            draw.rectangle((x, y, x + w, y + h), outline=(*color, 255), width=3)
            draw.text((x, y), str(position + 1), fill=(*color, 255))
        overlay = Image.alpha_composite(overlay, annotation).convert("RGB")
        canvas = Image.new("RGB", (1100, 650), "white")
        for x, picture in ((0, original), (550, overlay)):
            picture.thumbnail((540, 540))
            canvas.paste(picture, (x + (540 - picture.width) // 2, 70))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (10, 5),
            f"{number}: {row['id']} | original / source annotation",
            fill="black",
            font=font,
        )
        labels = (
            "; ".join(
                f"{i + 1} {v['category_name']}"
                for i, v in enumerate(row.get("instances", []))
            )
            or f"{row.get('attribute_key')}: {row.get('attributes')} | code={row.get('label_code')}"
        )
        draw.text((10, 33), labels, fill="black", font=font)
        destination = output / f"{number:02d}.png"
        canvas.save(destination)
        index.append(
            {
                "id": row["id"],
                "render": destination.name,
                "source_image_sha256": sha256_file(source),
                "render_sha256": sha256_file(destination),
                "source_labels": labels,
                "review_origin": None,
                "judgment": None,
            }
        )
    write_rows(output / "review.template.jsonl", index)
    write_json(
        output / "manifest.json",
        {
            "dataset_sha256": sha256_file(dataset),
            "selection": "First three distinct source label/instance-category combinations per original split; purposive, not random or representative",
            "records": len(index),
            "files": {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()},
        },
    )
    print({"rendered": len(index), "output": str(output)})


if __name__ == "__main__":
    main()
