"""Publish the immutable official train/validation/test segmentation release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_ecommerce.segmentation import iter_records, scan_dataset, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Release directory must be empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    splits = ("train", "validation", "test")
    report = scan_dataset(args.root, splits)
    (args.output / "scan_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    files = {}
    for split in splits:
        files[split] = write_jsonl(
            iter_records(args.root, split), args.output / f"{split}.jsonl"
        )
    manifest = {
        "schema_version": 1,
        "release_id": "module3_segmentation_official_v1",
        "source_root": str(args.root.resolve()),
        "split_policy": "official_source_splits_unchanged",
        "test_policy": "inference_only_unlabeled",
        "files": files,
        "scan_report_sha256": __import__("hashlib").sha256(
            (args.output / "scan_report.json").read_bytes()
        ).hexdigest(),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
