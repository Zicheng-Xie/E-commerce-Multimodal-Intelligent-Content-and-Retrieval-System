"""Read-only source, source-family, trained-head, ranking and metric replay."""

from __future__ import annotations

import json
from pathlib import Path

import run  # noqa: F401
from business_grounding_audit import COHORTS, DATA
from business_grounding_audit import ROOT as CYCLE
from business_grounding_audit import load, once, rows
from business_joint_experiment import BASELINES, NEW
from business_joint_experiment import ROOT as JOINT
from business_joint_experiment import guard, predictions, replay_heads
from business_metric_experiment import ROOT as METRIC
from business_metric_experiment import guard as metric_guard
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from querywise_paired_bounds import paired_querywise


def check_files(value: dict) -> None:
    """Apply the check files workflow step.

    Args:
        value: Input value to validate or transform.
    """
    for key in ("files", "input_files", "source_files"):
        for name, digest in value.get(key, {}).items():
            if sha256_file(PROJECT_ROOT / name) != digest:
                raise ValueError("Sealed evidence changed: " + name)


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        ValueError: Full ranking replay differs; Earlier consumed exclusions lost;
            Earlier excluded images lost.

    Notes:
        Writes derived artifacts through once.
    """
    guard()
    metric_guard()
    for root, name in [
        (CYCLE, "audit_receipt.json"),
        (CYCLE, "views_receipt.json"),
        (CYCLE, "semantic_receipt.json"),
        (CYCLE, "semantic_assessment_receipt.json"),
        (CYCLE, "QA_and_pairs_receipt.json"),
        (METRIC, "training_receipt.json"),
        (METRIC, "component_evaluation_receipt.json"),
        (JOINT, "completion_receipt.json"),
    ]:
        check_files(load(root / name))
    replay_heads()
    regenerated = predictions()
    schemas = [
        (JOINT, [(m, b) for m in NEW for b in BASELINES] + [(NEW[2], NEW[0])]),
        (
            METRIC,
            [
                (m, b)
                for m in (
                    "whole_Qwen_image_metric_ablation",
                    "business_metric_projection_oof_ablation",
                )
                for b in ("global", "cross20", "grounded_whole_zoom_union")
            ],
        ),
    ]
    for root, comparisons in schemas:
        for c in COHORTS:
            ranks = load(root / c / "predictions.json")
            labels = rows(DATA / c / "judgments.ai.jsonl")
            modes = {
                q["id"]: "image" if q["image_path"] else "text"
                for q in load(DATA / c / "queries.public.json")
            }
            if root == JOINT and ranks != regenerated[c]:
                raise ValueError("Full ranking replay differs")
            actual = evaluate_intervals(ranks, labels, modes)
            paired = paired_querywise(ranks, labels, modes, comparisons)
            if actual != load(
                root / c / "internal_metrics.ai.development.json"
            ) or paired != load(root / c / "internal_paired_querywise.json"):
                raise ValueError(
                    "Metric/paired-unknown replay differs: " + str(root) + "/" + c
                )
    snapshot = load(JOINT / "post_experiment_exclusion_snapshot.json")
    parent = load(CYCLE / "post_views_exclusion_snapshot.json")
    if not set(parent["ids"]) <= set(snapshot["ids"]):
        raise ValueError("Earlier consumed exclusions lost")
    if not {r["sha256"] for r in parent["image_fingerprints"]} <= {
        r["sha256"] for r in snapshot["image_fingerprints"]
    }:
        raise ValueError("Earlier excluded images lost")
    reg = load(PROJECT_ROOT / "configs/evaluation_registry.json")
    # Later experiments may advance current pointers without invalidating history.
    decision = load(JOINT / "decision.json")
    assert (
        decision["formal_metrics"] is None and not decision["default_strategy_changed"]
    )
    current = load(PROJECT_ROOT / reg["current_preparation_exclusion_snapshot"])
    assert set(snapshot["ids"]) <= set(current["ids"])
    assert {r["sha256"] for r in snapshot["image_fingerprints"]} <= {
        r["sha256"] for r in current["image_fingerprints"]
    }
    assert any(
        r["path_prefix"] == "artifacts/module2_grounded_independent_blind/"
        and r.get("development_or_training_use_allowed") is False
        for r in reg["rules"]
    )
    result = {
        "complete_cycle_verified": True,
        "original_AI_relations_unchanged": 12600,
        "new_independent_AI_blind_predictions": 0,
        "trained_fold_checkpoints_replayed": 18,
        "complete_joint_rankings_replayed": 280,
        "metric_scopes_recomputed": 6,
        "formal_metrics": None,
        "default_strategy_changed": False,
        "verification_source_sha256": sha256_file(Path(__file__)),
    }
    once(CYCLE / "cycle_verification.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
