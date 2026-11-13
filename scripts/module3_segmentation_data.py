"""Build module 3 segmentation contracts and scan reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_ecommerce.segmentation import iter_records, scan_dataset, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    splits = tuple(args.split or ("train", "validation", "test"))
    args.output.mkdir(parents=True, exist_ok=True)
    report = scan_dataset(args.root, splits, args.limit)
    (args.output / "scan_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifests = {}
    for split in splits:
        manifests[split] = write_jsonl(iter_records(args.root, split, args.limit), args.output / f"{split}.jsonl")
    (args.output / "manifest.json").write_text(json.dumps({"schema_version": 1, "splits": manifests}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
