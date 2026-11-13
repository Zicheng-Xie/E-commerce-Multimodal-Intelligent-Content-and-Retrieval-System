"""Fixed full-catalog sheets for direct Codex AI visual review."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

import run  # noqa: F401
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

ROOT = PROJECT_ROOT / "artifacts/module2_ai_review"


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through canvas.save, destination.mkdir, write_json.
    """
    inputs = json.loads((ROOT / "inputs.json").read_text(encoding="utf-8"))
    products = [p for p in inputs["catalog"] if p["catalog_source"] == "original286"]
    destination = ROOT / "native_audit/original"
    destination.mkdir(exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17)
    mapping = []
    for start in range(0, len(products), 8):
        canvas = Image.new("RGB", (1440, 940), "white")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(products[start : start + 8], start):
            x, y = (index % 4) * 360, ((index - start) // 4) * 470
            picture = ImageOps.exif_transpose(Image.open(row["image_path"])).convert(
                "RGB"
            )
            picture.thumbnail((350, 325))
            canvas.paste(picture, (x + (350 - picture.width) // 2, y + 40))
            draw.text(
                (x + 5, y + 5),
                f"{index + 1:03} {row['id'].rsplit('/', 1)[-1]}",
                font=font,
                fill="black",
            )
            for line in range(4):
                draw.text(
                    (x + 5, y + 365 + 24 * line),
                    row["title"][line * 19 : (line + 1) * 19],
                    font=font,
                    fill="black",
                )
            mapping.append(
                dict(
                    position=index + 1,
                    candidate_id=row["id"],
                    title=row["title"],
                    image_sha256=sha256_file(Path(row["image_path"])),
                    sheet=f"catalog_{start // 8 + 1:02}.jpg",
                )
            )
        canvas.save(destination / f"catalog_{start // 8 + 1:02}.jpg", quality=95)
    write_json(
        destination / "manifest.json",
        dict(
            products=mapping,
            files={
                str(p.relative_to(destination)): sha256_file(p)
                for p in destination.glob("*.jpg")
            },
            implementation_sha256=sha256_file(
                PROJECT_ROOT / "scripts/review_native_catalog.py"
            ),
            review_origin="model",
            human_labels=0,
        ),
    )
    print("Prepared 36 full original catalog sheets")


if __name__ == "__main__":
    main()
