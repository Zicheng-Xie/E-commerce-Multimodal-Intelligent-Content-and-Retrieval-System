"""Read-only AI provenance enum adapter; no grades or metric formulas change."""

from __future__ import annotations

import argparse
from types import ModuleType

import grounded_blind_dataset as dataset
import grounded_blind_execution_recovery as execution
import grounded_blind_report as original
import run  # noqa: F401
from learned_fusion_experiment import load, once, rel
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.module_loading import load_project_module
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from querywise_paired_bounds import paired_querywise

ROOT = dataset.ROOT
ORIGIN = "Codex native full-gallery AI image review"


def evaluator_view(rows: list[dict]) -> list:
    """Return evaluator view for this workflow.

    Args:
        rows: Records processed by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Only verified native Codex AI provenance may map to model enum.
    """
    output = []
    for row in rows:
        if row.get("review_origin") != ORIGIN or row.get("human_label") is not False:
            raise ValueError(
                "Only verified native Codex AI provenance may map to model enum"
            )
        output.append(
            dict(row, review_origin="model", annotation_tool=row["review_origin"])
        )
    assert [
        (r["query_id"], r["candidate_id"], r["grade"], r["reason"]) for r in rows
    ] == [(r["query_id"], r["candidate_id"], r["grade"], r["reason"]) for r in output]
    return output


def calculated() -> tuple[dict, dict]:
    """Return calculated for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    ranks = load(ROOT / "predictions.json")
    labels = evaluator_view(read_rows(ROOT / "judgments.ai.jsonl"))
    modes = {
        q["id"]: "image" if q["image_path"] else "text"
        for q in load(ROOT / "queries.public.json")
    }
    return evaluate_intervals(ranks, labels, modes), paired_querywise(
        ranks, labels, modes, original.COMPARISONS
    )


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    execution.guard()
    p = load(ROOT / "AI_provenance_execution_protocol.json")
    verify_files(PROJECT_ROOT, p["files"])
    verify_files(ROOT / "evaluation_source_snapshots", p["source_files"])


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Notes:
        Writes derived artifacts through once, target.parent.mkdir, target.write_bytes.
    """
    execution.guard()
    dataset.checked("prediction_receipt.json")
    dataset.checked("label_receipt.json")
    assert not (ROOT / "internal_metrics.ai.blind.json").exists()
    assert load(ROOT / "evaluation_recovery_checks.pre.json")["passed"]
    rows = read_rows(ROOT / "judgments.ai.jsonl")
    view = evaluator_view(rows)
    assert len(rows) == 2304 and sum(r["grade"] is None for r in view) == 16
    once(
        ROOT / "evaluation_provenance_failure_evidence.json",
        dict(
            observed_exception="ValueError: Unique AI judgments with honest provenance required",
            cause="Legacy metric schema expects review_origin=model; frozen labels truthfully use a more specific Codex native AI source string",
            original_origin=ORIGIN,
            AI_relations=2304,
            human_labels=0,
            grade_query_candidate_reason_changes=0,
            retrieval_metric_results_produced=0,
        ),
    )
    sources = [
        "scripts/grounded_blind_evaluation_recovery.py",
        "tests/test_grounded_blind_evaluation_recovery.py",
        "scripts/querywise_paired_bounds.py",
        "scripts/finalize_balanced_gallery.py",
    ]
    prior = load(dataset.DEVELOPMENT / "protocol.json")["files"]
    for name in sources[2:]:
        if name in prior:
            assert sha256_file(PROJECT_ROOT / name) == prior[name]
    checksums = {}
    for name in sources:
        p = PROJECT_ROOT / name
        target = ROOT / "evaluation_source_snapshots" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == p.read_bytes()
        else:
            target.write_bytes(p.read_bytes())
        checksums[name] = sha256_file(p)
    paths = [PROJECT_ROOT / s for s in sources] + [
        ROOT / "evaluation_recovery_checks.pre.json",
        ROOT / "evaluation_provenance_failure_evidence.json",
        ROOT / "label_receipt.json",
        ROOT / "judgments.ai.jsonl",
        ROOT / "prediction_receipt.json",
        ROOT / "predictions.json",
        ROOT / "release_protocol.json",
        ROOT / "argument_order_execution_protocol.json",
    ]
    once(
        ROOT / "AI_provenance_execution_protocol.json",
        dict(
            files={rel(p): sha256_file(p) for p in paths},
            source_files=checksums,
            adapter="In-memory copy changes review_origin from the specific verified native Codex AI string to generic model enum and retains original string as annotation_tool. Original file/receipt unchanged. Every grade, reason, query/candidate and rank unchanged.",
            original_metric_and_shared_label_formula_sources_unchanged=True,
            retrieval_metrics_seen_before_adapter_freeze=False,
            formal_metrics=None,
        ),
    )


def runtime() -> ModuleType:
    """Return runtime for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    module = load_project_module(
        "grounded_blind_AI_provenance_report",
        PROJECT_ROOT / "scripts/grounded_blind_report.py",
    )
    module.calculated = calculated
    return module


def full_verify() -> None:
    """Apply the full verify workflow step."""
    guard()
    module = load_project_module(
        "grounded_blind_complete_recovered_verifier",
        PROJECT_ROOT / "scripts/verify_grounded_blind_release.py",
    )
    module.report = runtime()
    module.verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "finish", "verify"])
    a = parser.parse_args().action
    if a == "prepare":
        prepare()
    elif a == "finish":
        guard()
        runtime().finish()
    else:
        full_verify()
