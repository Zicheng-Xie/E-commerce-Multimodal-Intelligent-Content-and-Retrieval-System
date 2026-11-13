"""Prepare or merge second-stage human review batches without source edits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.review_workflow import merge_reviews, prepare_packet
from multimodal_ecommerce.safety import PROJECT_ROOT


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileNotFoundError: Reviewed files directory is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "merge"])
    parser.add_argument(
        "--packet", type=Path, default=Path("artifacts/review_stage2_v1")
    )
    parser.add_argument(
        "--kit", type=Path, default=Path("artifacts/evaluation_stage2_v1")
    )
    parser.add_argument(
        "--copy-run", type=Path, default=Path("artifacts/copy_v3_heldout_001")
    )
    parser.add_argument("--reviews-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--predictions", type=Path, default=Path("artifacts/stage2_predictions_v1")
    )
    args = parser.parse_args()
    packet = PROJECT_ROOT / args.packet
    if args.command == "prepare":
        report = prepare_packet(
            PROJECT_ROOT / args.kit, PROJECT_ROOT / args.copy_run, packet
        )
        print(
            json.dumps(
                {
                    "packet": str(packet),
                    "batches": len(report["batches"]),
                    "retrieval": report["expected_retrieval_judgments"],
                    "copy": report["expected_copy_judgments"],
                },
                ensure_ascii=False,
            )
        )
        return
    if not args.output:
        parser.error("merge requires a new --output directory")
    directory = (
        PROJECT_ROOT / args.reviews_dir if args.reviews_dir else packet / "reviewed"
    )
    if not directory.is_dir():
        raise FileNotFoundError("Reviewed files directory is missing")
    inputs = sorted(directory.glob("*.reviewed.jsonl"))
    report = merge_reviews(
        packet, inputs, PROJECT_ROOT / args.output, PROJECT_ROOT / args.predictions
    )
    print(
        json.dumps(
            {key: report[key] for key in ("ready", "completed", "received_batches")},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
