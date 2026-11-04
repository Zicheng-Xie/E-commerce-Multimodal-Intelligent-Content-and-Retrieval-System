"""Current policy distinguishes required Precision from preferred Recall."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from multimodal_ecommerce.current_target import assess_current_target


def inputs(precision: float = 0.6, recall: float = 0.84) -> tuple[dict, dict, Any]:
    """Return inputs for this workflow.

    Args:
        precision: Precision used by this operation.
        recall: Recall used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    contract = json.loads(
        Path("configs/module2_current_target.json").read_text(encoding="utf-8")
    )
    baseline: dict = dict(
        **{"precision@10": 0.5, "recall@10": 0.8, "ndcg@10": 0.7, "hit@1": 0.7},
        ap=0.7,
        mrr=0.7,
    )
    table: dict = dict(
        global_=copy.deepcopy(baseline),
        region_title=copy.deepcopy(baseline),
        candidate={**baseline, "precision@10": precision, "recall@10": recall},
    )
    table["global"] = table.pop("global_")
    return (
        table,
        {"text": copy.deepcopy(table), "image": copy.deepcopy(table)},
        contract,
    )


def test_recall85_is_preferred_and_does_not_override_required_precision() -> None:
    """Verify that recall85 is preferred and does not override required precision."""
    result = assess_current_target(*inputs())["candidates"]["candidate"]
    assert result["hard_gate_passed"] and not result["recall_preferred_reached"]
    assert not result["development_targets_all_reached"]
    result = assess_current_target(*inputs(0.599, 0.99))["candidates"]["candidate"]
    assert not result["hard_gate_passed"] and result["recall_preferred_reached"]


def test_targets_reached_in_development_do_not_imply_formal_acceptance() -> None:
    """Verify that targets reached in development do not imply formal acceptance."""
    result = assess_current_target(*inputs(0.6, 0.85))
    assert result["candidates"]["candidate"]["development_targets_all_reached"]
    assert result["formal_metrics"] is None and result["goal_met"] is False
    assert result["formal_acceptance"] is False


def test_passing_targets_cannot_hide_other_metric_regression_or_weak_mode() -> None:
    """Verify that passing targets cannot hide other metric regression or weak mode."""
    summary, modes, contract = inputs(0.6, 0.85)
    modes["image"]["candidate"]["ap"] = 0.69
    result = assess_current_target(summary, modes, contract)["candidates"]["candidate"]
    assert not result["hard_gate_passed"]
    assert "ap_regressed_vs_global" in result["required_failures"]["image"]
    modes.pop("image")
    with pytest.raises(ValueError, match="overall, text and image"):
        assess_current_target(summary, modes, contract)


def test_no_complete_human_truth_keeps_all_target_observations_pending() -> None:
    """Verify that no complete human truth keeps all target observations pending."""
    contract = inputs()[2]
    result = assess_current_target(None, None, contract)
    assert result["hard_gate_passed"] is None
    assert result["recall_preferred_reached"] is None
    assert result["formal_metrics"] is None and result["goal_met"] is False
