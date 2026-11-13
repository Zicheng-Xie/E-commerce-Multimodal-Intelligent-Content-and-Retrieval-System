"""Import actual consensus exports and run the strict human training/scoring path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import run  # noqa: F401,E402
from multimodal_ecommerce.human_business_pipeline import (  # noqa: E402
    run_human_pipeline,
)


def main() -> None:
    """Import actual consensus exports and run the strict human training/scoring
    path.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_human_pipeline(args.merge_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
