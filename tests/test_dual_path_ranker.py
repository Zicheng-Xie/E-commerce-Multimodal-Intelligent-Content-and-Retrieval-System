"""Verify dual path ranker contracts with isolated regression scenarios."""

from __future__ import annotations

import numpy as np
import pytest

from dual_path_ranker import (
    METHODS,
    DualPathRankerFuseContextsInputs,
    fuse_contexts,
    text_orders,
)


def test_shared_pool_and_tail_survive_opposite_context_votes() -> None:
    """Verify that shared pool and tail survive opposite context votes."""
    ids = [f"p{i:02}" for i in range(16)]
    pool = ids[3:15]
    baseline = list(reversed(ids))
    orders, detail = fuse_contexts(
        DualPathRankerFuseContextsInputs(
            ids,
            pool,
            np.arange(12),
            -np.arange(12),
            np.arange(16),
            -np.arange(16),
            np.arange(16),
            baseline,
        )
    )
    assert set(orders) == set(METHODS)
    tail = [cid for cid in baseline if cid not in pool]
    for order in orders.values():
        assert len(order) == len(set(order)) == len(ids)
        assert set(order[:12]) == set(pool)
        assert order[12:] == tail
    assert orders["original_context_union"] != orders["new_context_union"]
    assert detail["score_is_probability"] is False


def test_missing_or_nonfinite_context_scores_cannot_be_imputed() -> None:
    """Verify that missing or nonfinite context scores cannot be imputed."""
    ids = [str(i) for i in range(12)]
    good = np.arange(12, dtype=float)
    with pytest.raises(ValueError, match="both contexts"):
        fuse_contexts(
            DualPathRankerFuseContextsInputs(
                ids, ids, good, good[:-1], good, good, good, ids
            )
        )
    invalid = good.copy()
    invalid[4] = np.nan
    with pytest.raises(ValueError, match="both contexts"):
        fuse_contexts(
            DualPathRankerFuseContextsInputs(
                ids, ids, good, invalid, good, good, good, ids
            )
        )


def test_identical_contexts_preserve_single_route_and_text_is_exact() -> None:
    """Verify that identical contexts preserve single route and text is exact."""
    ids = [str(i) for i in range(14)]
    base = np.arange(14)
    logits = np.arange(10)
    orders, _ = fuse_contexts(
        DualPathRankerFuseContextsInputs(
            ids, ids[:10], logits, logits, base, base, base, ids
        )
    )
    assert orders["dual_context_union"] == orders["original_context_union"]
    assert all(order == ids for order in text_orders(ids).values())
