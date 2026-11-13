"""Copy 1000 genuine DeepFashion images and source labels for phase-two checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.data import prepare_samples
from multimodal_ecommerce.safety import PROJECT_ROOT, PROTECTED_ROOT


def main() -> None:
    """Preserve source splits and hash every copied asset, never write sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/mentor_fashion_1000_v1")
    )
    args = parser.parse_args()
    result = prepare_samples(
        PROTECTED_ROOT,
        PROJECT_ROOT / args.output,
        product_per_category=0,
        fashion_per_attribute=0,
        deepfashion_per_split=500,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
