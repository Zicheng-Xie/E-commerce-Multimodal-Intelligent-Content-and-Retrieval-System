"""Verify rank stability contracts with isolated regression scenarios."""

from __future__ import annotations

import itertools

import pytest

from multimodal_ecommerce.rank_stability import stable_tail
from multimodal_ecommerce.retrieval_strategies import extended_metrics


def test_stable_tail_preserves_top10_metrics_for_every_label_completion() -> None:
    """Verify that stable tail preserves top10 metrics for every label completion."""
    anchor = [str(i) for i in range(12)]
    candidate = anchor[3:] + anchor[:3]
    result = stable_tail(anchor, candidate)
    assert result[:10] == candidate[:10]
    assert result[10:] == [x for x in anchor if x not in result[:10]]
    for bits in itertools.product([False, True], repeat=4):
        relevant = {c for c, keep in zip(["0", "3", "8", "11"], bits) if keep}
        if not relevant:
            continue
        old = extended_metrics(candidate, relevant)
        new = extended_metrics(result, relevant)
        for name in ["precision@10", "recall@10", "ndcg@10", "hit@1"]:
            assert new[name] == old[name]


def test_tail_does_not_claim_unconditional_ap_or_mrr_protection() -> None:
    """Verify that tail does not claim unconditional ap or mrr protection."""
    anchor = [str(i) for i in range(14)]
    candidate = anchor[:10] + list(reversed(anchor[10:]))
    result = stable_tail(anchor, candidate)
    for metric in ["ap", "mrr"]:
        assert (
            extended_metrics(result, {"13"})[metric]
            < extended_metrics(candidate, {"13"})[metric]
        )


@pytest.mark.parametrize(
    "anchor,candidate,k",
    [(["a", "a"], ["a", "b"], 1), (["a"], ["b"], 1), (["a"], ["a"], True)],
)
def test_rejects_invalid_rankings(anchor: object, candidate: object, k: int) -> None:
    """Verify that rejects invalid rankings.

    Args:
        anchor: Anchor used by this operation.
        candidate: Candidate used by this operation.
        k: K used by this operation.
    """
    with pytest.raises(ValueError):
        stable_tail(anchor, candidate, k)  # type: ignore[arg-type]  # Exercise invalid ranking input.
