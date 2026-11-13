"""Score the new frozen rankings only from actual complete dual-human consensus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_evaluation import score_business_development
from multimodal_ecommerce.current_target import assess_current_target
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.human_business_pipeline import REVIEW, validate_consensus
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)

ROOT = PROJECT_ROOT / "artifacts/module2_precision_recall"
RANKINGS = ROOT / "adjudicated_ranker"


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def main(merge_dir: Path) -> None:
    """Run the command-line workflow defined by this module.

    Args:
        merge_dir: Merge dir used by this operation.

    Notes:
        Writes derived artifacts through target.mkdir, write_json.
    """
    merge_dir = merge_dir.resolve()
    merge_dir.relative_to((REVIEW / "merges").resolve())
    verify_files(REVIEW, load(REVIEW / "release_receipt.json")["files"])
    verify_files(ROOT, load(ROOT / "analysis_receipt.json")["files"])
    verify_files(RANKINGS, load(RANKINGS / "receipt.json")["files"])
    labels = read_rows(merge_dir / "judgments.jsonl")
    consensus = validate_consensus(
        load(REVIEW / "manifest.json"),
        read_rows(REVIEW / "judgments.blank.jsonl"),
        load(merge_dir / "input_provenance.json"),
        labels,
    )
    contract = load(PROJECT_ROOT / "configs/module2_current_target.json")
    progress = consensus["status"]
    report = dict(
        human_completed=progress["completed"],
        human_pending=progress["pending"],
        quality_metrics=None,
        metrics_by_mode=None,
        target_assessment=assess_current_target(None, None, contract),
        formal_metrics=None,
        formal_acceptance=False,
        goal_met=False,
        acceptance_model_predictions=0,
    )
    if progress["pending"] == 0:
        old = PROJECT_ROOT / "artifacts/module2_precision60/business"
        allowed = load(old / "allowed_candidates.json")
        queries = load(ROOT / "inputs.json")["queries"]
        modes = {q["id"]: "image" if q["image_path"] else "text" for q in queries}
        scoring = score_business_development(
            load(RANKINGS / "rankings.json"), allowed, labels, modes
        )
        summary, by_mode = scoring["quality_metrics"], scoring["metrics_by_mode"]
        report.update(
            quality_metrics=summary,
            metrics_by_mode=by_mode,
            query_metrics=scoring["query_metrics"],
            zero_positive_query_ids=scoring["zero_positive_query_ids"],
            denominator_policy=scoring["denominator_policy"],
            target_assessment=assess_current_target(summary, by_mode, contract),
        )
        capacity = scoring["capacity"]
        report["full_human_capacity"] = dict(
            precision_ceiling=capacity["precision_ceiling"],
            recall_ceiling_on_positive_queries=capacity[
                "recall_ceiling_on_positive_queries"
            ],
            precision60_feasible=capacity["precision_ceiling"] >= 0.6 - 1e-12,
            recall85_feasible=(
                capacity["recall_ceiling_on_positive_queries"] is not None
                and capacity["recall_ceiling_on_positive_queries"] >= 0.85 - 1e-12
            ),
        )
    provenance = dict(
        human_merge=str(merge_dir.relative_to(PROJECT_ROOT)),
        label_sha256=sha256_file(merge_dir / "judgments.jsonl"),
        provenance_sha256=sha256_file(merge_dir / "input_provenance.json"),
        frozen_ranking_sha256=sha256_file(RANKINGS / "rankings.json"),
        target_sha256=sha256_file(PROJECT_ROOT / "configs/module2_current_target.json"),
        implementation_sha256=sha256_file(Path(__file__)),
    )
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[
        :24
    ]
    target = ensure_output_path(ROOT / "human_scores" / key)
    if (target / "receipt.json").exists():
        verify_files(target, load(target / "receipt.json")["files"])
        assert load(target / "report.json") == report
    else:
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "report.json", report)
        write_json(target / "provenance.json", provenance)
        write_json(
            target / "receipt.json",
            dict(
                files={p.name: sha256_file(p) for p in target.iterdir() if p.is_file()}
            ),
        )
    write_json(
        ROOT / "human_scores/current_status.json",
        dict(report, revision=target.relative_to(PROJECT_ROOT).as_posix()),
    )
    print(json.dumps({k: v for k, v in report.items() if k != "query_metrics"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-dir", type=Path, required=True)
    main(parser.parse_args().merge_dir)
