"""Weak supervision must not leak validation cells or turn uncertainty negative."""

from __future__ import annotations

import numpy as np
import pytest

from multimodal_ecommerce.semantic_ranker import (
    FEATURE_NAMES,
    SemanticRankerFitRankerInputs,
    SemanticRankerTrainingMaskInputs,
    assign_folds,
    fit_ranker,
    machine_topk_bounds,
    training_mask,
)


def test_training_masks_exclude_both_roles_of_heldout_image_sources() -> None:
    """Verify that training masks exclude both roles of heldout image sources."""
    mask, evidence = training_mask(
        SemanticRankerTrainingMaskInputs(
            ["qa", "qb", "qc"],
            np.array([0, 1, 1]),
            ["g0", None, "g2"],
            ["g0", "g1", "g2", "g3"],
            np.array([0, 1, 1, 0]),
            0,
            1,
        )
    )
    assert mask.tolist() == [[False] * 4, [False, False, False, True], [False] * 4]
    assert not set(evidence["training_gallery_families"]) & set(
        evidence["excluded_gallery_families"]
    )
    assert evidence["training_query_ids"] == ["qb"]


@pytest.mark.parametrize("pairwise", [False, True])
def test_excluded_features_and_labels_do_not_change_fit(pairwise: bool) -> None:
    """Verify that excluded features and labels do not change fit.

    Args:
        pairwise: Pairwise used by this operation.
    """
    rng = np.random.default_rng(17)
    x = rng.normal(size=(4, 6, len(FEATURE_NAMES)))
    y = np.array([[1, 0, 0, np.nan, 0, 1]] * 4, dtype=float)
    mask = np.ones((4, 6), dtype=bool)
    mask[0] = False
    mask[:, 0] = False
    base = rng.random((4, 6))
    policy: dict = dict(steps=50, l2=0.05, learning_rate=0.1)
    first = fit_ranker(
        SemanticRankerFitRankerInputs(x, y, mask, base, policy, pairwise=pairwise)
    )
    modified = x.copy()
    modified[~mask] = 1e6
    changed = y.copy()
    changed[~mask] = 1 - y[~mask]
    second = fit_ranker(
        SemanticRankerFitRankerInputs(
            modified, changed, mask, base, policy, pairwise=pairwise
        )
    )
    for key in ["mean", "scale", "weights"]:
        assert np.allclose(first[key], second[key])
    assert first["training_cells"] == 12
    assert first["training_positive"] == 3
    assert first["training_negative"] == 9


def test_fold_assignment_keeps_aliases_together() -> None:
    """Verify that fold assignment keeps aliases together."""
    first = assign_folds(["a", "b", "a", "c"], 3, 9)
    assert first == assign_folds(["c", "b", "a"], 3, 9)
    assert len(first) == 3


def test_proxy_bounds_preserve_unknowns_and_have_no_human_metrics() -> None:
    """Verify that proxy bounds preserve unknowns and have no human metrics."""
    ids = [str(i) for i in range(10)]
    grades = [1, 1, 0, None, None, None, None, None, None, None]
    result = machine_topk_bounds(
        {"q": ids}, {("q", cid): g for cid, g in zip(ids, grades)}
    )
    assert result["top10_machine_positive_lower"] == 0.2
    assert result["top10_machine_positive_upper"] == 0.9
    assert result["human_quality_metrics"] is None
    assert result["formal_metrics"] is None
    assert result["formal_acceptance"] is False
    with pytest.raises(ValueError):
        machine_topk_bounds({"q": ids[:5]}, {})
