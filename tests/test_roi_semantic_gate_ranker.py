"""Routing, head-control and fail-closed parent-score contracts."""

from __future__ import annotations

import pytest

from roi_semantic_gate_ranker import (
    METHODS,
    PRIMARY,
    RoiSemanticGateRankerGatedOrdersInputs,
    gated_orders,
    text_orders,
)


def fixture_inputs() -> (
    tuple[list[str], list[str], dict[str, float], dict[str, float], list[str]]
):
    """Return fixture inputs for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    ids = [f"c{i:02}" for i in range(15)]
    pool = ids[:12]
    whole: dict = {cid: i / 12 for i, cid in enumerate(pool)}
    crop: dict = {cid: (11 - i) / 12 for i, cid in enumerate(pool)}
    baseline = ids[5:10] + ids[:5] + ids[10:]
    return ids, pool, whole, crop, baseline


def test_query_gate_and_independent_head_control_preserve_full_gallery_and_tail() -> (
    None
):
    """Verify that query gate and independent head control preserve full gallery and
    tail.
    """
    args = fixture_inputs()
    ids, pool, whole, crop, baseline = args
    rejected, details = gated_orders(
        RoiSemanticGateRankerGatedOrdersInputs(*args, False)
    )
    accepted, accepted_details = gated_orders(
        RoiSemanticGateRankerGatedOrdersInputs(*args, True)
    )
    tail = [cid for cid in baseline if cid not in pool]
    assert rejected["semantic_crop_gate"] == pool[::-1] + tail
    assert accepted["semantic_crop_gate"] == pool + tail
    assert rejected[PRIMARY] == rejected["whole_context_head1"]
    assert accepted[PRIMARY] == accepted["unverified_crop_head1"]
    assert rejected[PRIMARY][1 : len(pool)] == [
        cid for cid in pool[::-1] if cid != baseline[0]
    ]
    assert accepted[PRIMARY][1 : len(pool)] == [
        cid for cid in pool if cid != baseline[0]
    ]
    assert rejected["unverified_crop_head1"] == accepted["unverified_crop_head1"]
    for orders in (rejected, accepted):
        for name, order in orders.items():
            assert len(order) == len(set(order)) == len(ids)
            assert set(order) == set(ids) and order[len(pool) :] == tail
            if name != "semantic_crop_gate":
                assert order[0] == baseline[0]
    assert details["scores"]["semantic_crop_gate"] == whole
    assert accepted_details["scores"]["semantic_crop_gate"] == crop
    assert not details["score_is_probability"] and not details["qrels_used"]


def test_missing_or_nonfinite_model_scores_and_nonboolean_gate_are_not_imputed() -> (
    None
):
    """Verify that missing or nonfinite model scores and nonboolean gate are not
    imputed.
    """
    ids, pool, whole, crop, baseline = fixture_inputs()
    for decision in (None, 1, "accept"):
        with pytest.raises(ValueError):
            gated_orders(
                RoiSemanticGateRankerGatedOrdersInputs(
                    ids, pool, whole, crop, baseline, decision  # type: ignore[arg-type]  # Reject non-boolean gate.
                )
            )
    for bad in (
        {**crop, pool[0]: float("nan")},
        {k: v for k, v in crop.items() if k != pool[0]},
    ):
        with pytest.raises(ValueError):
            gated_orders(
                RoiSemanticGateRankerGatedOrdersInputs(
                    ids, pool, whole, bad, baseline, False
                )
            )


def test_text_path_retains_original_ranking_in_every_new_method() -> None:
    """Verify that text path retains original ranking in every new method."""
    baseline = fixture_inputs()[-1]
    assert text_orders(baseline) == {method: baseline for method in METHODS}
    with pytest.raises(ValueError):
        text_orders(baseline + baseline[:1])
