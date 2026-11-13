"""Prevent incomplete qrels from becoming misleading formal metrics."""

from __future__ import annotations

import pytest

from multimodal_ecommerce.evaluation import score_rankings, validate_qrels


def test_incomplete_or_unreviewed_qrels_rejected() -> None:
    """Missing negatives matter to the Recall denominator, not just positives."""
    queries: list[dict] = [{"id": "q", "kind": "text", "exclude_id": None}]
    judged: dict = {
        "query_id": "q",
        "product_id": "a",
        "grade": 1,
        "reviewer": "reviewer1",
    }
    with pytest.raises(ValueError, match="Incomplete"):
        validate_qrels(queries, {"a", "b"}, [judged])
    for grade, reviewer in [(None, "reviewer1"), (1, ""), (True, "reviewer1")]:
        with pytest.raises(ValueError, match="Unjudged"):
            validate_qrels(
                queries, {"a"}, [{**judged, "grade": grade, "reviewer": reviewer}]
            )
    with pytest.raises(ValueError, match="duplicate"):
        validate_qrels(queries, {"a"}, [judged, judged])


def test_self_exclusion_and_multiple_relevant_denominator() -> None:
    """Compute against all positives and k, explicitly listing empty queries."""
    queries: list[dict] = [
        {"id": "image", "kind": "image", "exclude_id": "self"},
        {"id": "empty", "kind": "text", "exclude_id": None},
    ]
    labels = [
        {
            "query_id": query["id"],
            "product_id": key,
            "grade": int(query["id"] == "image"),
            "reviewer": "r",
        }
        for query in queries
        for key in {"self", "a", "b"}
        if key != query["exclude_id"]
    ]
    relevant = validate_qrels(queries, {"self", "a", "b"}, labels)
    assert relevant["image"] == {"a", "b"}
    scores = score_rankings(queries, relevant, {"image": ["a"], "empty": ["b"]}, k=10)
    metric = scores["macro_over_positive_queries"]["all"]
    assert metric["precision@10"] == 0.1
    assert metric["recall@10"] == 0.5
    assert scores["no_positive_queries"] == ["empty"]


def test_unknown_or_self_judgment_rejected() -> None:
    """Do not accept labels for the query product excluded from retrieval."""
    with pytest.raises(ValueError, match="Unexpected"):
        validate_qrels(
            [{"id": "q", "exclude_id": "a"}],
            {"a"},
            [{"query_id": "q", "product_id": "a", "grade": 1, "reviewer": "r"}],
        )
