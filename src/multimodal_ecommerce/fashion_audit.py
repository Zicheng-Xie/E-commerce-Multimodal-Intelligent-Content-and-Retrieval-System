"""Validate converted clothing annotations without training visual models."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.safety import ensure_output_path, sha256_file, write_json


def audit_fashion(dataset: Path, output: Path) -> dict:
    """Check coordinates, copied images, attribute codes and official pair IDs.

    Args:
        dataset: Dataset used by this operation.
        output: Output used by this operation.

    Returns:
        Mapping with fields: records, verified_copied_files, dataset_sha256, issues,
        issue_counts, category_instances, cross_split_groups, overlays, scope.

    Raises:
        FileExistsError: Use a new audit directory.
        ValueError: Copied fashion asset mismatch; Image escapes copied dataset.
    """
    output = ensure_output_path(output)
    if output.exists():
        raise FileExistsError("Use a new audit directory")
    records = read_rows(dataset)
    issues = []
    categories: Counter = Counter()
    pairs: dict[str, list[dict]] = defaultdict(list)
    pixels: dict[str, list[dict]] = defaultdict(list)
    overlays: list[dict] = []
    copied_manifest = json.loads(
        (dataset.parent / "manifest.json").read_text(encoding="utf-8")
    )
    for entry in copied_manifest["files"]:
        path = (dataset.parent / entry["path"]).resolve()
        if (
            not path.is_relative_to(dataset.parent.resolve())
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError("Copied fashion asset mismatch")
    for row in records:
        path = (dataset.parent / row["image_path"]).resolve()
        if not path.is_relative_to(dataset.parent.resolve()):
            raise ValueError("Image escapes copied dataset")
        with Image.open(path) as opened:
            picture = opened.convert("RGB")
        width, height = picture.size
        if (width, height) != (row["width"], row["height"]):
            issues.append({"id": row["id"], "issue": "dimension_mismatch"})
        digest = hashlib.sha256(
            f"{width}x{height}:".encode() + picture.tobytes()
        ).hexdigest()
        pixels[digest].append(row)
        if row["dataset"] == "fashionai":
            code = row["label_code"]
            valid = (
                len(code) == len(row["label_values"])
                and not (set(code) - set("nym"))
                and code.count("y") == 1
            )
            if (
                not valid
                or row["attributes"].get(row["attribute_key"])
                != row["label_values"][code.index("y")]
            ):
                issues.append({"id": row["id"], "issue": "invalid_attribute_code"})
            continue
        if row.get("pair_id") is not None:
            pairs[str(row["pair_id"])].append(row)
        draw = ImageDraw.Draw(picture)
        for item in row["instances"]:
            categories[f"{row['split']}/{item['category_name']}"] += 1
            x, y, box_w, box_h = item["bbox"]
            if not all(math.isfinite(value) for value in (x, y, box_w, box_h)) or not (
                0 <= x < x + box_w <= width and 0 <= y < y + box_h <= height
            ):
                issues.append(
                    {
                        "id": row["id"],
                        "instance": item["source_instance_id"],
                        "issue": "bbox_out_of_bounds_or_empty",
                    }
                )
            for polygon in item["segmentation"]:
                if (
                    len(polygon) < 6
                    or len(polygon) % 2
                    or not all(math.isfinite(value) for value in polygon)
                    or any(
                        not (0 <= px <= width and 0 <= py <= height)
                        for px, py in zip(polygon[::2], polygon[1::2])
                    )
                ):
                    issues.append(
                        {
                            "id": row["id"],
                            "instance": item["source_instance_id"],
                            "issue": "polygon_invalid_or_out_of_bounds",
                        }
                    )
                else:
                    points = list(zip(polygon[::2], polygon[1::2]))
                    draw.line(points + points[:1], fill="lime", width=3)
            if not (0 < item["area"] <= width * height):
                issues.append({"id": row["id"], "issue": "empty_or_invalid_mask_area"})
            if box_w > 0 and box_h > 0:
                draw.rectangle((x, y, x + box_w, y + box_h), outline="red", width=2)
        if row["split"] == "train" and len(overlays) < 6:
            target = ensure_output_path(output / "overlays" / path.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            picture.thumbnail((600, 600))
            picture.save(target)
            overlays.append({"id": row["id"], "path": str(target)})
    leaks = []
    for kind, groups in (("pair_id", pairs), ("exact_pixels", pixels)):
        for key, group in groups.items():
            if len({row["split"] for row in group}) > 1:
                leaks.append(
                    {
                        "kind": kind,
                        "key": key,
                        "ids": [row["id"] for row in group],
                        "status": (
                            "confirmed_pixel_match"
                            if kind == "exact_pixels"
                            else "identifier_collision_global_scope_unverified"
                        ),
                    }
                )
    invalid = {item["id"] for item in issues}
    write_json(
        output / "eligibility.json",
        [
            {
                "id": row["id"],
                "annotation_valid": row["id"] not in invalid,
                "source_split": row["split"],
                "split_scoped_pair_key": (
                    f"{row['split']}/{row['pair_id']}"
                    if row.get("pair_id") is not None
                    else None
                ),
            }
            for row in records
        ],
    )
    report = {
        "records": len(records),
        "verified_copied_files": len(copied_manifest["files"]),
        "dataset_sha256": sha256_file(dataset),
        "issues": issues,
        "issue_counts": dict(Counter(row["issue"] for row in issues)),
        "category_instances": dict(categories),
        "cross_split_groups": leaks,
        "overlays": overlays,
        "scope": "Preprocessing validation only. Existing source splits retained. Pair ID collisions across source splits do NOT establish shared identity: global ID scope is unverified. Exact pixel checks are separate. Invalid annotations are excluded in eligibility.json, never silently clamped. No segmentation or attribute model was run.",
    }
    write_json(output / "report.json", report)
    return report
