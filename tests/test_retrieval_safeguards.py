"""Check public-only caption abstention and score-consistent prefix preservation."""

from __future__ import annotations

import pytest

from multimodal_ecommerce.retrieval_safeguards import (
    compound_caption_conflict,
    guarded_top20,
)


@pytest.mark.parametrize(
    "product,appearance,conflict",
    [
        ("枕头包", "棕色皮革手提包，配有浅粉色手柄", True),
        ("枕头", "白色长方形枕头，表面有绗缝", False),
        ("手机包", "皮革手提包，前面有小口袋", False),
        ("相机包", "黑色相机包，顶部有提手", False),
        ("手机壳", "黑色手机保护壳，有开孔", False),
    ],
)
def test_compound_guard_does_not_turn_real_pillows_or_specific_bags_into_conflicts(
    product: str, appearance: str, conflict: bool
) -> None:
    """Distinguish a contradictory compound from ordinary supported product nouns.

    Args:
        product: Query-only product name.
        appearance: Independently parsed appearance clause of the same caption.
        conflict: Whether the two noun interpretations contradict each other.
    """
    assert (
        compound_caption_conflict(dict(product=product, appearance=appearance))
        is conflict
    )


def test_equal_score_front_result_survives_without_changing_the_scored_pool() -> None:
    """Do not discard the original third result when it ties protected results."""
    original = [str(i) for i in range(30)]
    scores = {cid: float(i) for i, cid in enumerate(original[:20])}
    scores.update({"0": 0.0, "1": 0.0, "2": 0.0})
    result = guarded_top20(original, original, scores)
    assert result[:3] == original[:3]
    assert set(result[:20]) == set(original[:20])
    assert result[20:] == original[20:]
    assert len(result) == len(set(result)) == len(original)
    scores["2"] = -0.1
    result = guarded_top20(original, original, scores)
    assert result[:2] == original[:2] and "2" not in result[:10]


def test_guard_rejects_unscored_or_unsupported_galleries() -> None:
    """Reject missing model scores and a pool dropping original Top10 members."""
    original = [str(i) for i in range(30)]
    with pytest.raises(ValueError, match="Complete scored Top20"):
        guarded_top20(original, original, {cid: 0.0 for cid in original[:19]})
    proposed = original[10:] + original[:10]
    with pytest.raises(ValueError, match="original Top10"):
        guarded_top20(original, proposed, {cid: 0.0 for cid in proposed[:20]})
