"""Prepare independent human work items or merge returned real human exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.business_review_jobs import build_jobs, merge_jobs


def main() -> None:
    """Prepare independent human work items or merge returned real human exports."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "merge"])
    parser.add_argument("--inbox", type=Path)
    parser.add_argument("--adjudications", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(build_jobs(), ensure_ascii=False))
    else:
        if args.inbox is None or not args.inbox.is_dir():
            parser.error(
                "--inbox must be an existing folder containing only returned review JSON files"
            )
        print(
            json.dumps(merge_jobs(args.inbox, args.adjudications), ensure_ascii=False)
        )


if __name__ == "__main__":
    main()
