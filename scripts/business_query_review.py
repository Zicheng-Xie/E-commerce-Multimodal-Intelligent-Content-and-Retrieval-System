"""Prepare human query-authoring forms or import an actual human export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.business_query_review import (
    import_query_authoring,
    prepare_query_authoring,
)


def main() -> None:
    """Prepare human query-authoring forms or import an actual human export."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "import"])
    parser.add_argument("--reviewed", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare_query_authoring(), ensure_ascii=False))
    else:
        if args.reviewed is None:
            parser.error("--reviewed JSON file is required")
        print(json.dumps(import_query_authoring(args.reviewed), ensure_ascii=False))


if __name__ == "__main__":
    main()
