"""Create a separate release using explicit reviewed taxonomy rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.safety import PROJECT_ROOT
from multimodal_ecommerce.taxonomy import release_taxonomy


def main() -> None:
    """Run the command-line workflow defined by this module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/stage2_v1/release/products.jsonl")
    )
    parser.add_argument("--policy", type=Path, default=Path("configs/taxonomy_v2.json"))
    parser.add_argument("--output", type=Path, default=Path("data/stage2_taxonomy_v2"))
    args = parser.parse_args()
    result = release_taxonomy(
        PROJECT_ROOT / args.dataset,
        PROJECT_ROOT / args.policy,
        PROJECT_ROOT / args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
