"""Overlapping independent intervals can still have exact zero paired loss."""

from __future__ import annotations

from multimodal_ecommerce.ai_review_metrics import metric_ranges
from querywise_paired_bounds import paired_querywise


def test_nonregression_uses_same_unknown_labels_for_both_methods() -> None:
    """Use the canonical paired evaluator; retain every original assertion."""
    ids = ["known", "unknown", *[f"n{i}" for i in range(10)]]
    ranks: dict = {"q": dict(global_=["unknown", "known", *ids[2:]], candidate=ids)}
    ranks["q"]["global"] = ranks["q"].pop("global_")
    labels = [
        dict(
            query_id="q",
            candidate_id=cid,
            grade=1 if cid == "known" else None if cid == "unknown" else 0,
            review_origin="model",
            human_label=False,
        )
        for cid in ids
    ]
    ordinary = metric_ranges(ranks, labels, dict(q="image"))["scopes"]["all"]
    assert ordinary["candidate"]["ranges"]["precision@10"] == [0.1, 0.2]
    assert ordinary["global"]["ranges"]["precision@10"] == [0.1, 0.2]
    result = paired_querywise(ranks, labels, dict(q="image"), [("candidate", "global")])
    metrics = result["scopes"]["all"]["candidate"]["global"]
    assert metrics["precision@10"]["delta_range"] == [0, 0]
    assert (
        metrics["precision@10"]["nonregression_proven_for_shared_unknown_labels"]
        is True
    )
    assert metrics["hit@1"]["delta_range"] == [0, 1]
