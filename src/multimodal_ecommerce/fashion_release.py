"""Publish eligible annotation copies without silently repairing source geometry."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.safety import ensure_output_path, sha256_file, write_json


def release_fashion(dataset: Path, audit: Path, output: Path) -> dict:
    """Exclude invalid records, copy valid assets, and rebuild consistent COCO joins.

    Args:
        dataset: Dataset used by this operation.
        audit: Audit used by this operation.
        output: Output used by this operation.

    Returns:
        Mapping with fields: source_records, eligible_records, by_dataset, by_split,
        excluded, copied_files, source_dataset_sha256, source_audit_sha256,
        source_eligibility_sha256, scope.

    Raises:
        FileExistsError: Use a new fashion release directory.
        ValueError: Fashion audit belongs to different data; Eligibility IDs disagree;
            Eligibility contradicts audited issues or source split; Fashion asset
            escapes its dataset; Copied fashion source asset changed; Released asset
            differs from source copy; COCO image IDs collide within an official split.
    """
    output = ensure_output_path(output)
    if output.exists():
        raise FileExistsError("Use a new fashion release directory")
    report = json.loads((audit / "report.json").read_text(encoding="utf-8"))
    if report["dataset_sha256"] != sha256_file(dataset):
        raise ValueError("Fashion audit belongs to different data")
    rows = read_rows(dataset)
    invalid = {issue["id"] for issue in report["issues"]}
    eligibility = json.loads((audit / "eligibility.json").read_text(encoding="utf-8"))
    flags = {row["id"]: row for row in eligibility}
    if (
        len(flags) != len(rows)
        or len(flags) != len(eligibility)
        or set(flags) != {row["id"] for row in rows}
    ):
        raise ValueError("Eligibility IDs disagree")
    for row in rows:
        flag = flags[row["id"]]
        if (
            flag["annotation_valid"] is not (row["id"] not in invalid)
            or flag["source_split"] != row["split"]
        ):
            raise ValueError("Eligibility contradicts audited issues or source split")
    copied = json.loads((dataset.parent / "manifest.json").read_text(encoding="utf-8"))
    source_files = {entry["path"]: entry for entry in copied["files"]}
    output.mkdir(parents=True)
    kept, files = [], []
    coco: dict[str, dict] = {}
    category_maps: dict[str, dict] = {}
    attributes = {}
    for original in rows:
        if original["id"] in invalid:
            continue
        row = dict(original)
        row["annotation_eligible"] = True
        row["split_scoped_pair_key"] = (
            f"{row['split']}/{row['pair_id']}"
            if row.get("pair_id") is not None
            else None
        )
        kept.append(row)
        for field in ("image_path", "annotation_path"):
            if not row.get(field):
                continue
            relative = row[field]
            source = (dataset.parent / relative).resolve()
            target = ensure_output_path(output / relative)
            if not source.is_relative_to(
                dataset.parent.resolve()
            ) or not target.is_relative_to(output):
                raise ValueError("Fashion asset escapes its dataset")
            entry = source_files[relative]
            if sha256_file(source) != entry["sha256"]:
                raise ValueError("Copied fashion source asset changed")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha256_file(target) != entry["sha256"]:
                raise ValueError("Released asset differs from source copy")
            files.append(
                {
                    "path": relative,
                    "sha256": entry["sha256"],
                    "copied_from": str(source),
                }
            )
        if row["dataset"] == "fashionai":
            attributes[row["attribute_key"]] = row["label_values"]
            continue
        split = row["split"]
        document = coco.setdefault(
            split, {"images": [], "annotations": [], "categories": []}
        )
        category_map = category_maps.setdefault(split, {})
        image_id = int(row["source_id"])
        document["images"].append(
            {
                "id": image_id,
                "file_name": row["image_path"],
                "width": row["width"],
                "height": row["height"],
                "source_record_id": row["id"],
            }
        )
        for item in row["instances"]:
            category_map[item["category_id"]] = item["category_name"]
            document["annotations"].append(
                {
                    "id": len(document["annotations"]) + 1,
                    "image_id": image_id,
                    **{
                        key: item[key]
                        for key in (
                            "category_id",
                            "bbox",
                            "segmentation",
                            "area",
                            "area_method",
                            "source_instance_id",
                        )
                    },
                    "iscrowd": 0,
                }
            )
    for split, document in coco.items():
        document["categories"] = [
            {"id": key, "name": value, "supercategory": "clothes"}
            for key, value in sorted(category_maps[split].items())
        ]
        if len({row["id"] for row in document["images"]}) != len(document["images"]):
            raise ValueError("COCO image IDs collide within an official split")
        write_json(output / "annotations" / f"deepfashion_{split}.json", document)
    write_json(output / "annotations/fashionai_attribute_values.json", attributes)
    write_rows(output / "samples.jsonl", kept)
    write_json(output / "manifest.json", {"schema_version": 1, "files": files})
    summary = {
        "source_records": len(rows),
        "eligible_records": len(kept),
        "by_dataset": dict(Counter(row["dataset"] for row in kept)),
        "by_split": dict(Counter(row["split"] for row in kept)),
        "excluded": [row for row in report["issues"] if row["id"] in invalid],
        "copied_files": len(files),
        "source_dataset_sha256": sha256_file(dataset),
        "source_audit_sha256": sha256_file(audit / "report.json"),
        "source_eligibility_sha256": sha256_file(audit / "eligibility.json"),
        "scope": "Whole-record exclusion of invalid annotations. Coordinates, labels, source IDs and official splits retained. Pair keys scoped by source split; no claim that numeric collisions prove identity. No training, segmentation inference or silent clamping.",
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "release_manifest.json",
        {
            "files": {
                path.relative_to(output).as_posix(): sha256_file(path)
                for path in output.rglob("*")
                if path.is_file()
            }
        },
    )
    return summary
