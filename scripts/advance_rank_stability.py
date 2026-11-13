"""One bounded tail repair and a separately frozen fresh-query development check."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run  # noqa: F401
from multimodal_ecommerce.ai_review_metrics import metric_ranges
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_evaluation import METRICS
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.rank_stability import stable_tail
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from report_focus_coverage import paired_differences

ROOT = PROJECT_ROOT / "artifacts/module2_rank_stability"
PARENT = PROJECT_ROOT / "artifacts/module2_focus_coverage"
ANCHOR = "region_title"
CANDIDATE = "region_top2_described_cross_pool20"
NAME = "region_top2_described_cross_pool20_stable_tail"
POLICY: dict = dict(
    role="single_adaptive_development_tail_repair",
    anchor=ANCHOR,
    candidate=CANDIDATE,
    k=10,
    applies_to="image",
    text_strategy="cross_evidence_pool60",
    new_variants=1,
    further_parameter_search=False,
    runtime_uses_labels=False,
    formal_acceptance=False,
)


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def freeze() -> None:
    """Freeze the inputs and policy used by this workflow.

    Raises:
        FileExistsError: Preserve bounded tail stage.

    Notes:
        Writes derived artifacts through ROOT.mkdir, write_json.
    """
    if ROOT.exists():
        raise FileExistsError("Preserve bounded tail stage")
    receipt = load(PARENT / "completion_receipt.json")
    verify_files(PARENT, receipt["files"])
    verify_files(PROJECT_ROOT, receipt["project_files"])
    ROOT.mkdir(parents=True)
    files = [
        Path(__file__),
        PROJECT_ROOT / "src/multimodal_ecommerce/rank_stability.py",
        PROJECT_ROOT / "tests/test_rank_stability.py",
        PROJECT_ROOT / "scripts/prepare_fresh_query_check.py",
        PROJECT_ROOT / "scripts/retrieve_focus_development.py",
        PROJECT_ROOT / "scripts/compare_region_prefix.py",
        PROJECT_ROOT / "scripts/report_focus_coverage.py",
        PROJECT_ROOT / "configs/module2_current_target.json",
        PARENT / "region_guard/rankings.json",
        PARENT / "judgments.ai.jsonl",
        PARENT / "inputs.public.json",
        PARENT / "post_development_exclusion_snapshot.json",
        PARENT / "completion_receipt.json",
        PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl",
    ]
    write_json(
        ROOT / "protocol.json",
        dict(
            frozen_at=datetime.now(timezone.utc).isoformat(),
            policy=POLICY,
            rationale="Precision gains were accompanied by AP uncertainty. Keep the existing displayed top10 exactly, restore the baseline tail without looking at labels.",
            files={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in files
            },
            fresh_query_check="Six new training-source image queries across headphones, keyboards, power banks, cups, pillows and chairs. No family or near-image reuse. Freeze selection before this comparison; include zero-positive and ambiguous cases. Not formal heldout.",
            acceptance_predictions=0,
            formal_metrics=None,
            goal_met=False,
        ),
    )
    print(json.dumps(dict(frozen=True, new_variants=1, acceptance_predictions=0)))


def check() -> None:
    """Verify the prerequisites and integrity of the current workflow."""
    p = load(ROOT / "protocol.json")
    assert p["policy"] == POLICY
    verify_files(PROJECT_ROOT, p["files"])


def evaluate() -> None:
    """Apply the evaluate workflow step.

    Raises:
        FileExistsError: Preserve comparison.

    Notes:
        Writes derived artifacts through write_json.
    """
    check()
    if (ROOT / "comparison_receipt.json").exists():
        raise FileExistsError("Preserve comparison")
    old = load(PARENT / "region_guard/rankings.json")
    queries = load(PARENT / "inputs.public.json")["queries"]
    modes = {q["id"]: "image" if q["image_path"] else "text" for q in queries}
    rankings = {}
    for qid, methods in old.items():
        selected = {
            name: methods[name]
            for name in ["global", ANCHOR, "cross_evidence_pool60", CANDIDATE]
        }
        selected[NAME] = (
            stable_tail(methods[ANCHOR], methods[CANDIDATE])
            if modes[qid] == "image"
            else list(methods["cross_evidence_pool60"])
        )
        assert selected[NAME][:10] == selected[CANDIDATE][:10]
        rankings[qid] = selected
    labels = read_rows(PARENT / "judgments.ai.jsonl")
    metrics = metric_ranges(rankings, labels, modes)
    paired = paired_differences(rankings, labels, modes)
    for s in ["all", "text", "image"]:
        before, after = [metrics["scopes"][s][n]["ranges"] for n in [CANDIDATE, NAME]]
        for m in ["precision@10", "recall@10", "ndcg@10", "hit@1"]:
            assert before[m] == after[m]
    decision = dict(
        precision_60_AI_lower_all_modes=all(
            metrics["scopes"][s][NAME]["precision_60_lower_bound_met"]
            for s in ["all", "text", "image"]
        ),
        nonregression_all_metrics_all_modes=all(
            paired["scopes"][s][NAME][b][m][
                "nonregression_proven_for_shared_unknown_labels"
            ]
            for s in ["all", "text", "image"]
            for b in ["global", ANCHOR]
            for m in METRICS
        ),
        top10_exactly_preserved=True,
        default_promotion_allowed=False,
        goal_met=False,
        formal_metrics=None,
        acceptance_predictions=0,
    )
    write_json(ROOT / "rankings.json", rankings)
    write_json(ROOT / "internal_metrics.ai.development.json", metrics)
    write_json(ROOT / "internal_paired_differences.json", paired)
    write_json(ROOT / "internal_decision.json", decision)
    names = [
        "rankings.json",
        "internal_metrics.ai.development.json",
        "internal_paired_differences.json",
        "internal_decision.json",
    ]
    write_json(
        ROOT / "comparison_receipt.json",
        dict(
            files={n: sha256_file(ROOT / n) for n in names},
            new_methods=1,
            new_full_rankings=18,
            new_model_inferences=0,
            labels_changed=0,
            queries_changed=0,
            formal_acceptance=False,
        ),
    )
    print(
        json.dumps(
            dict(
                compared=True,
                new_full_rankings=18,
                top10_preserved=True,
                default_promoted=False,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "evaluate"])
    args = parser.parse_args()
    {"freeze": freeze, "evaluate": evaluate}[args.action]()
