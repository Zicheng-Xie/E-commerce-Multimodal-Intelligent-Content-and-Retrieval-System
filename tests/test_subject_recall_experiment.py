"""Proposal features need no missing CE, and expansion keeps all old candidates."""

from __future__ import annotations

import numpy as np
import pytest

from subject_attribute_ranker import attribute_vector
from subject_recall_experiment import (
    FEATURE_NAMES,
    public_vector,
    stable_union,
)


def test_public_proposal_features_are_separate_from_any_ce_channel() -> None:
    """Verify that public proposal features are separate from any CE channel."""
    q: dict = dict(
        product="手机壳",
        appearance="",
        visible_text="华为 Mate10",
        primary_unclear=False,
    )
    d: dict = dict(
        product="手机壳",
        appearance="",
        visible_text="华为Mate20Pro",
        primary_unclear=False,
    )
    semantic = public_vector(q, d)
    prior, _ = attribute_vector(np.arange(20, dtype=float), q, d)
    assert semantic == prior[:13].tolist()
    assert len(FEATURE_NAMES) == 33 and not any(
        "CE_" in n or "_CE" in n for n in FEATURE_NAMES
    )


def test_shared_pool_is_stable_union_and_never_removes_old_candidates() -> None:
    """Verify that shared pool is stable union and never removes old candidates."""
    old = [f"old-{i}" for i in range(40)]
    new = [f"new-{i}" for i in range(20)]
    pool = stable_union(old, new)
    assert pool == old + new and len(pool) == 60
    overlapping = old[:10] + new[:10]
    assert stable_union(old, overlapping) == old + new[:10]
    with pytest.raises(AssertionError):
        stable_union(old, ["same"] * 20)
