"""Copy or audit the phase-two fashion preprocessing dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.data import prepare_samples
from multimodal_ecommerce.fashion_audit import audit_fashion
from multimodal_ecommerce.safety import PROJECT_ROOT, PROTECTED_ROOT, ensure_output_path


def main() -> None:
    """Keep preparation and validation separately reproducible.

    Raises:
        FileExistsError: Use a new dataset directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "audit"])
    parser.add_argument("--dataset", type=Path, default=Path("data/stage2_fashion_v1"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/stage2_fashion_v1")
    )
    args = parser.parse_args()
    directory = ensure_output_path(PROJECT_ROOT / args.dataset)
    if args.command == "prepare":
        if directory.exists():
            raise FileExistsError("Use a new dataset directory")
        result = prepare_samples(
            PROTECTED_ROOT,
            directory,
            product_per_category=0,
            fashion_per_attribute=10,
            deepfashion_per_split=50,
        )
    else:
        result = audit_fashion(directory / "samples.jsonl", PROJECT_ROOT / args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
