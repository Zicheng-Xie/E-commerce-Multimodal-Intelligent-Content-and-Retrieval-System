"""Snapshot source file metadata without writing into the source tree."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal_ecommerce.safety import ensure_output_path  # noqa: E402
from multimodal_ecommerce.safety import write_json  # noqa: E402


def records(source: Path) -> Iterator:
    """Yield sorted names, sizes and modification times; never follow links.

    Args:
        source: Source used by this operation.

    Yields:
        Values produced by this operation, in the stated return order.
    """
    for directory, subdirectories, filenames in os.walk(source, followlinks=False):
        subdirectories[:] = sorted(
            name for name in subdirectories if not Path(directory, name).is_symlink()
        )
        for name in sorted(filenames):
            path = Path(directory, name)
            stat = path.lstat()
            record = {
                "path": path.relative_to(source).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            if path.suffix.lower() == ".pdf" and not path.is_symlink():
                record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            yield record


def main() -> None:
    """Create a metadata baseline or verify it against current source files.

    Raises:
        FileExistsError: Baseline exists; preserve it instead of overwriting.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["snapshot", "verify"])
    args = parser.parse_args()
    source = ROOT / "Alibaba阿里算法实习"
    snapshot = ensure_output_path(ROOT / "artifacts/source_before.jsonl.gz", source)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "snapshot":
        if snapshot.exists():
            raise FileExistsError(
                "Baseline exists; preserve it instead of overwriting."
            )
        count = 0
        with gzip.open(snapshot, "wt", encoding="utf-8") as handle:
            for record in records(source):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        print(json.dumps({"snapshot": str(snapshot), "files": count}))
        return
    with gzip.open(snapshot, "rt", encoding="utf-8") as handle:
        before = {item["path"]: item for item in map(json.loads, handle)}
    count = len(before)
    differences = []
    for record in records(source):
        original = before.pop(record["path"], None)
        if original != record:
            differences.append({"before": original, "after": record})
    differences.extend({"removed": record} for record in before.values())
    report = {
        "passed": not differences,
        "baseline_files": count,
        "method": "All file paths, sizes, mtime_ns; additionally SHA256 for all PDFs",
        "limitation": "Non-PDF source files are not exhaustively content-hashed.",
        "differences": differences,
    }
    write_json(ROOT / "artifacts/source_integrity.json", report, source)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "differences"}
        )
    )
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
