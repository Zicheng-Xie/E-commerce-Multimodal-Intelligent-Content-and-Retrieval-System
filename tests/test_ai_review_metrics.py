"""Unknown labels must not create optimistic precision or a false Recall denominator."""

from __future__ import annotations

import pytest

from multimodal_ecommerce.ai_review_metrics import metric_ranges, optional_mean_bounds


def test_optional_recall_queries_can_be_excluded() -> None:
    """Verify that optional recall queries can be excluded."""
    assert optional_mean_bounds([(0.8, 0.8)], [(0.2, 0.2)]) == pytest.approx((0.5, 0.8))
    assert optional_mean_bounds([(0.2, 0.2)], [(0.8, 0.8)]) == pytest.approx((0.2, 0.5))
    assert optional_mean_bounds([], []) == (None, None)


def test_full_metric_ranges_preserve_zero_and_optional_queries() -> None:
    """Verify that full metric ranges preserve zero and optional queries."""
    rankings: dict = {
        q: {"baseline": ["a", "b"]} for q in ["known", "optional", "zero"]
    }
    labels = [
        dict(
            query_id=q,
            candidate_id=c,
            grade=(
                1
                if q == "known" and c == "a"
                else None if q == "optional" and c == "b" else 0
            ),
            review_origin="model",
            human_label=False,
        )
        for q in rankings
        for c in ["a", "b"]
    ]
    result = metric_ranges(rankings, labels, dict.fromkeys(rankings, "image"))
    scores = result["scopes"]["all"]["baseline"]
    assert scores["ranges"]["precision@10"] == pytest.approx([1 / 30, 2 / 30])
    assert scores["ranges"]["recall@10"] == [1, 1]
    assert scores["ranges"]["ap"] == pytest.approx([1 / 3, 1 / 2])
    assert (
        scores["recall_denominator_min"] == 1 and scores["recall_denominator_max"] == 2
    )
    assert result["capacity_by_scope"]["all"]["zero_positive_query_ids"] == ["zero"]
    assert result["formal_metrics"] is None
