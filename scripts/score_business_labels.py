"""Score complete real human business labels; preserve null scores while pending."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_evaluation import score_business_development
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.precision_labels import validate_precision_labels
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

ROOT = PROJECT_ROOT / "artifacts/module2_precision60/business"


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        ValueError: Business review packet changed since scoring policy freeze.
        FileExistsError: Preserve the imported human label revision.

    Notes:
        Writes derived artifacts through target.mkdir, write_json, write_rows.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviewed", type=Path, default=ROOT / "review/judgments.blank.jsonl"
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Validate label progress without writing or scoring",
    )
    args = parser.parse_args()
    scoring_protocol = json.loads(
        (ROOT / "label_scoring_protocol.json").read_text(encoding="utf-8")
    )
    verify_files(PROJECT_ROOT, scoring_protocol["implementation"])
    packet = ROOT / "review"
    if (
        sha256_file(packet / "manifest.json")
        != scoring_protocol["packet_manifest_sha256"]
    ):
        raise ValueError("Business review packet changed since scoring policy freeze")
    manifest = json.loads((packet / "manifest.json").read_text(encoding="utf-8"))
    verify_files(packet, manifest["files"])
    expected = read_rows(packet / "judgments.blank.jsonl")
    actual = read_rows(args.reviewed)
    progress = validate_precision_labels(expected, actual)
    progress.pop("completed_rows")
    if args.status_only:
        print(
            json.dumps(
                {**progress, "quality_metrics": None, "formal_acceptance": False}
            )
        )
        return
    target = ROOT / "label_revisions" / sha256_file(args.reviewed)[:20]
    if target.exists():
        raise FileExistsError("Preserve the imported human label revision")
    target.mkdir(parents=True)
    write_rows(target / "judgments.jsonl", actual)
    status = {
        **progress,
        "coverage": "Complete allowed development gallery, not a Top-K pool",
        "formal_acceptance": False,
        "formal_metrics": None,
        "source_sha256": sha256_file(args.reviewed),
        "full_gallery_recall_computable": progress["pending"] == 0,
    }
    if progress["pending"]:
        status.update(
            quality_metrics=None,
            goal_met=None,
            state="awaiting_complete_real_human_judgments",
        )
    else:
        receipt = json.loads(
            (ROOT / "prediction_receipt.json").read_text(encoding="utf-8")
        )
        verify_files(ROOT, receipt["files"])
        rankings = json.loads((ROOT / "rankings.json").read_text(encoding="utf-8"))
        allowed = json.loads(
            (ROOT / "allowed_candidates.json").read_text(encoding="utf-8")
        )
        context = json.loads((ROOT / "review_context.json").read_text(encoding="utf-8"))
        modes = {
            q["id"]: "image" if q["image_path"] else "text" for q in context["queries"]
        }
        result = score_business_development(rankings, allowed, actual, modes)
        write_json(target / "query_metrics.json", result.pop("query_metrics"))
        status.update(result, state="complete_development_human_labels_scored")
    write_json(target / "status.json", status)
    print(
        json.dumps(
            {
                k: status[k]
                for k in [
                    "total",
                    "completed",
                    "pending",
                    "quality_metrics",
                    "formal_acceptance",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
