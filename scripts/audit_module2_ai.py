"""Create fixed native-vision audit sheets and preserve structured AI observations."""

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
    products = [p for p in inputs["catalog"] if p["catalog_source"] == "expansion64"]
    destination = ROOT / "native_audit"
    destination.mkdir(exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17)
    mapping = []
    for start in range(0, len(products), 8):
        canvas = Image.new("RGB", (1440, 900), "white")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(products[start : start + 8], start):
            x, y = (index % 4) * 360, ((index - start) // 4) * 450
            image = ImageOps.exif_transpose(Image.open(row["image_path"])).convert(
                "RGB"
            )
            image.thumbnail((350, 330))
            canvas.paste(image, (x + (350 - image.width) // 2, y + 35))
            draw.text(
                (x + 5, y + 5),
                f"{index + 1:02} {row['id'].rsplit('/', 1)[-1]}",
                font=font,
                fill="black",
            )
            title = row["title"]
            for line in range(3):
                draw.text(
                    (x + 5, y + 365 + 24 * line),
                    title[line * 19 : (line + 1) * 19],
                    font=font,
                    fill="black",
                )
            mapping.append(
                dict(
                    position=index + 1,
                    candidate_id=row["id"],
                    title=title,
                    source_image_sha256=sha256_file(Path(row["image_path"])),
                )
            )
        canvas.save(destination / f"catalog_{start // 8 + 1:02}.jpg", quality=95)
    write_json(
        destination / "protocol.json",
        dict(
            role="second_AI_native_visual_audit_not_human_review",
            selection="All 64 expansion candidates in original order, selected before reading local model outcomes",
            local_model_scores_shown=False,
            target_bucket_shown=False,
            instructions="Judge sold subject, image-title agreement, usability and explicit requested attribute; unknowns remain unknown.",
            items=mapping,
            image_files={
                str(p.relative_to(destination)): sha256_file(p)
                for p in destination.glob("*.jpg")
            },
            implementation_sha256=sha256_file(
                PROJECT_ROOT / "scripts/audit_module2_ai.py"
            ),
            actual_humans=0,
        ),
    )
    print("Prepared 8 native AI audit sheets / 64 products")


if __name__ == "__main__":
    main()
