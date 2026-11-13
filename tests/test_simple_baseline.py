"""Guard the cheap ablation and actual-human label audit boundaries."""

from __future__ import annotations

import pytest

from multimodal_ecommerce.simple_baseline import (
    human_accuracy,
    rerank_head,
    sample_audit,
)


def test_head_requires_every_actual_score_and_preserves_tail() -> None:
    """Reject missing scores instead of inventing zero logits."""
    order = [str(i) for i in range(30)]
    scores = {cid: float(i) for i, cid in enumerate(order[:20])}
    actual = rerank_head(order, scores)
    assert actual[:20] == list(reversed(order[:20]))
    assert actual[20:] == order[20:]
    with pytest.raises(ValueError):
        rerank_head(order, dict(list(scores.items())[:-1]))
    scores["0"] = float("nan")
    with pytest.raises(ValueError):
        rerank_head(order, scores)


def test_audit_is_exactly_twenty_percent_and_rank_independent() -> None:
    """Preserve null strata and sample exactly 63 of 315 relations per query."""
    rows = [
        dict(
            query_id=q,
            candidate_id=str(i),
            grade=None if i % 5 == 0 else i % 2,
            human_label=False,
        )
        for q in ["a", "b"]
        for i in range(315)
    ]
    selected = sample_audit(rows)
    assert len(selected) == 126
    assert sum(r["query_id"] == "a" for r in selected) == 63
    assert any(r["grade"] is None for r in selected)
    assert selected == sample_audit(list(reversed(rows)))


def test_empty_answers_do_not_become_human_accuracy() -> None:
    """A prepared template has zero humans and no measurable accuracy."""
    tasks = [dict(task_id="one", grade=1)]
    report = human_accuracy(tasks, [dict(task_id="one", human_grade="")])
    assert report["annotation_quality"] == "pending_human_review"
    assert report["binary_sample_agreement"] is None
    assert report["formal_metrics"] is None


def test_low_agreement_blocks_label_quality_and_abstentions_are_separate() -> None:
    """An observed failure cannot turn into a multiplicative score correction."""
    tasks = [dict(task_id=str(i), grade=1 if i < 5 else None) for i in range(6)]
    answers = [
        dict(
            task_id=str(i),
            reviewer_type="human",
            reviewer="actual reviewer",
            reviewed_at="2026-11-05",
            human_grade="0",
            reason="商品主体不同",
        )
        for i in range(6)
    ]
    report = human_accuracy(tasks, answers)
    assert report["annotation_quality"] == "below_80"
    assert report["counts"]["AI_abstentions"] == 1
    assert report["binary_sample_agreement"] == 0
    answers[0]["reviewer_type"] = "model"
    with pytest.raises(ValueError):
        human_accuracy(tasks, answers)
