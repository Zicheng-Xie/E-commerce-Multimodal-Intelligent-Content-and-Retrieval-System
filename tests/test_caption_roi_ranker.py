"""Verify caption roi ranker contracts with isolated regression scenarios."""

from __future__ import annotations

import numpy as np
import pytest

from caption_roi_ranker import (
    PRIMARY,
    CaptionRoiRankerFuseCropsInputs,
    fuse_crops,
    text_orders,
)


def test_real_crop_votes_change_head_without_dropping_any_gallery_item() -> None:
    """Verify that real crop votes change head without dropping any gallery item."""
    ids = [f"p{i:02}" for i in range(16)]
    pool = ids[:12]
    logits = np.arange(12)
    base = np.arange(16)
    orders, details = fuse_crops(
        CaptionRoiRankerFuseCropsInputs(
            ids,
            pool,
            logits,
            logits,
            -logits,
            base,
            base,
            base,
            ids,
        )
    )
    tail = ids[12:]
    assert orders["crop_context_union"][:12] != orders["whole_crop_roi_union"][:12]
    for order in orders.values():
        assert len(order) == len(set(order)) == 16
        assert set(order[:12]) == set(pool)
        assert order[12:] == tail
    assert details["score_is_probability"] is False


def test_crop_context_must_not_be_missing_or_nonfinite() -> None:
    """Verify that crop context must not be missing or nonfinite."""
    ids = [str(i) for i in range(12)]
    score = np.arange(12, dtype=float)
    with pytest.raises(ValueError, match="Real whole and crop"):
        fuse_crops(
            CaptionRoiRankerFuseCropsInputs(
                ids, ids, score, score, score[:-1], score, score, score, ids
            )
        )
    invalid = score.copy()
    invalid[0] = np.inf
    with pytest.raises(ValueError, match="Real whole and crop"):
        fuse_crops(
            CaptionRoiRankerFuseCropsInputs(
                ids, ids, score, score, invalid, score, score, score, ids
            )
        )


def test_identical_whole_crop_has_no_new_fusion_effect_and_text_is_unchanged() -> None:
    """Verify that identical whole crop has no new fusion effect and text is
    unchanged.
    """
    ids = [str(i) for i in range(12)]
    score = np.arange(12)
    orders, _ = fuse_crops(
        CaptionRoiRankerFuseCropsInputs(
            ids, ids, score, score, score, score, score, score, ids
        )
    )
    assert orders[PRIMARY] == orders["crop_context_union"]
    assert all(order == ids for order in text_orders(ids).values())
