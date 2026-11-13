"""Build a versioned phase-two dataset without altering source assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.dataset_release import (
    DatasetReleaseFreezeReleaseInputs,
    DatasetReleaseSampleCandidatesInputs,
    freeze_release,
    sample_candidates,
)
from multimodal_ecommerce.media import localize_products
from multimodal_ecommerce.safety import PROJECT_ROOT, PROTECTED_ROOT


def main() -> None:
    """Expose independently resumable sampling, downloading and freezing steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["sample", "download", "freeze"])
    parser.add_argument("--version", default="stage2_v1")
    parser.add_argument("--candidates-per-group", type=int, default=100)
    parser.add_argument("--release-per-group", type=int, default=50)
    args = parser.parse_args()
    if Path(args.version).name != args.version or args.version in {".", "..", ""}:
        parser.error("Version must be one directory name")
    base = PROJECT_ROOT / "data" / args.version
    if args.command == "sample":
        result = sample_candidates(
            DatasetReleaseSampleCandidatesInputs(
                PROTECTED_ROOT,
                PROJECT_ROOT / "data/samples/samples.jsonl",
                base / "candidates",
                per_group=args.candidates_per_group,
            )
        )
    elif args.command == "download":
        result = localize_products(
            base / "candidates/samples.jsonl", base / "localized"
        )
        result = {
            key: value
            for key, value in result.items()
            if key not in {"failures", "duplicates"}
        }
    else:
        result = freeze_release(
            DatasetReleaseFreezeReleaseInputs(
                base / "localized/products.jsonl",
                PROJECT_ROOT / "data/samples/samples.jsonl",
                PROJECT_ROOT / "data/localized/products.jsonl",
                base / "release",
                per_group=args.release_per_group,
            )
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
