"""Create a clean, separately hashed clothing preprocessing release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.fashion_release import release_fashion
from multimodal_ecommerce.safety import PROJECT_ROOT


def main() -> None:
    """Run the command-line workflow defined by this module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/stage2_fashion_v1/samples.jsonl")
    )
    parser.add_argument(
        "--audit", type=Path, default=Path("artifacts/stage2_fashion_v2")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/stage2_fashion_clean_v1")
    )
    args = parser.parse_args()
    result = release_fashion(
        PROJECT_ROOT / args.dataset,
        PROJECT_ROOT / args.audit,
        PROJECT_ROOT / args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
