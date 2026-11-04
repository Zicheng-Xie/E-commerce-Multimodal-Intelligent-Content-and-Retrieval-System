"""Regression coverage for observed product/accessory and candidate-pool failures."""

from __future__ import annotations

import numpy as np
import pytest

from multimodal_ecommerce.intent_retrieval import (
    balanced_pool,
    public_evidence,
    recall_channels,
    rerank_complete,
)


def test_real_tail_mismatches_and_accessory_names_are_explicit_conflicts() -> None:
    """Verify that real tail mismatches and accessory names are explicit conflicts."""
    for title, profile in [
        ("双频无线路由器", "商品主体：路由器。可见外观：黑色"),
        ("无线蓝牙耳机保护套", "商品主体：耳机保护套。可见外观：硅胶"),
        ("手机支架", "商品主体：车载手机支架。可见外观：黑色"),
        ("吹风机收纳包", "商品主体：吹风机收纳包。可见外观：灰色"),
    ]:
        row = public_evidence("无线蓝牙耳机", title, profile)
        assert row["offset"] < 0 and row["state"] == "explicit_type_conflict"


def test_attribute_uncertainty_never_counts_as_supported_positive() -> None:
    """Verify that attribute uncertainty never counts as supported positive."""
    row = public_evidence(
        "无线蓝牙耳机", "有线无线耳机", "商品主体：耳机。可见外观：黑色"
    )
    assert row["offset"] <= 0 and row["attributes"]["unknown"] == ["wireless"]
    row = public_evidence("短袖T恤", "无袖T恤", "商品主体：T恤。可见外观：白色")
    assert row["state"] == "explicit_attribute_conflict"
    assert public_evidence(None, "手机支架", "手机支架")["offset"] == 0


def test_balanced_pool_is_nested_and_does_not_turn_zero_literal_ties_into_votes() -> (
    None
):
    """Verify that balanced pool is nested and does not turn zero literal ties into
    votes.
    """
    ids = [f"{i:02d}" for i in range(80)]
    rng = np.random.default_rng(42)
    x = rng.random((80, 13))
    x[:, 5:7] = 0
    _, channels = recall_channels(ids, x)
    assert len(channels) == 5
    small, broad = balanced_pool(channels, 20), balanced_pool(channels, 60)
    assert small == broad[:20] and len(broad) == len(set(broad)) == 60
    assert all(channel[0] in small for channel in channels)


def test_reranking_retains_full_gallery_self_exclusions_and_fixed_top10() -> None:
    """Verify that reranking retains full gallery self exclusions and fixed top10."""
    ids = [f"{i:02d}" for i in range(80)]
    pool = ids[20:80]
    ranked = rerank_complete(
        ids, np.arange(80, dtype=float), pool, np.arange(60, dtype=float)
    )
    assert len(ranked) == len(set(ranked)) == 80 and set(ranked) == set(ids)
    assert len(ranked[:10]) == 10 and ranked[0] == "79"
    with pytest.raises(ValueError):
        rerank_complete(ids, np.arange(80), ["invalid"], np.zeros(1))
