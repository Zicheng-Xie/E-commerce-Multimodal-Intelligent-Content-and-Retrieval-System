"""Publish official v2 records in one streaming pass; preserve v1."""

import argparse
import json
from collections import Counter
from pathlib import Path
from multimodal_ecommerce.segmentation import iter_records, sha256_file


def publish(root, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 2, "splits": {}}
    files = {}
    categories = {}
    with (output / "issues.jsonl").open("w") as audit:
        for split in ("train", "validation", "test"):
            counts, groups = Counter(), Counter()
            image_stems = set()
            path = output / f"{split}.jsonl"
            with path.open("w") as handle:
                for record in iter_records(root.resolve(), split):
                    image_stems.add(Path(record.image_path).stem)
                    counts["images"] += 1
                    for number, obj in enumerate(record.objects):
                        counts["objects"] += 1
                        groups[obj["category_name"]] += 1
                        if split == "train":
                            key = str(obj["category_id"])
                            if (
                                key in categories
                                and categories[key] != obj["category_name"]
                            ):
                                raise ValueError("Inconsistent category mapping")
                            categories[key] = obj["category_name"]
                        for issue in obj["issues"]:
                            counts[issue] += 1
                        if obj["issues"]:
                            audit.write(
                                json.dumps(
                                    {
                                        "image_id": record.image_id,
                                        "object_index": number,
                                        "issues": obj["issues"],
                                    }
                                )
                                + "\n"
                            )
                    # Exclude entire image from training when any object is invalid,
                    # so unlabelled clothing cannot silently become background.
                    row = record.to_dict()
                    row["eligible"] = (
                        split != "test"
                        and bool(record.objects)
                        and all(o["eligible"] for o in record.objects)
                    )
                    counts["eligible_images"] += int(row["eligible"])
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            orphans = sorted(
                p.name
                for p in (root / split / "annos").glob("*.json")
                if p.stem not in image_stems
            )
            report["splits"][split] = dict(
                counts, categories=dict(groups), orphan_annotations=orphans
            )
            files[path.name] = sha256_file(path)
    (output / "scan_report.json").write_text(json.dumps(report, indent=2))
    (output / "categories.json").write_text(json.dumps(categories, indent=2))
    for name in ("issues.jsonl", "scan_report.json", "categories.json"):
        files[name] = sha256_file(output / name)
    manifest = dict(
        schema_version=2,
        release_id=output.name,
        split_policy="official_unchanged",
        bbox_format="xywh",
        polygon_policy="preserve_all_source_coordinates",
        training_policy="eligible_images_only",
        files=files,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish(args.root, args.output)
