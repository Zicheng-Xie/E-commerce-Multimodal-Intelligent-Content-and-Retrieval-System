"""Protect full-gallery comparisons and separate semantic selection from visual order."""

from __future__ import annotations

import pytest

from module2_shared_evaluation import confirmed_change_metrics, precision_delta
from module2_shared_primary_recall import _visual_head_order


def test_visual_order_preserves_selected_membership_and_complete_tail() -> None:
    """Keep semantic Top10 decisions while restoring original visual relative order."""
    original = [str(i) for i in range(15)]
    selected = original[:2] + list(reversed(original[3:11])) + ["2"] + original[11:]
    revised = _visual_head_order(original, selected)
    assert set(revised[:10]) == set(selected[:10])
    assert revised[:10] == original[:2] + original[3:11]
    assert revised[10:] == selected[10:]
    assert len(revised) == len(set(revised)) == len(original)


@pytest.mark.parametrize("invalid", ["duplicate", "different_gallery", "changed_head"])
def test_visual_order_rejects_incomparable_or_unprotected_orders(invalid: str) -> None:
    """Reject incomplete galleries and a changed protected prefix before scoring.

    Args:
        invalid: Deliberate ranking defect rejected before evaluation.
    """
    original = [str(i) for i in range(15)]
    selected = original.copy()
    if invalid == "duplicate":
        selected[-1] = selected[-2]
    elif invalid == "different_gallery":
        selected[-1] = "outside"
    else:
        selected[0], selected[1] = selected[1], selected[0]
    with pytest.raises(ValueError, match="same-gallery"):
        _visual_head_order(original, selected)


def test_human_swap_delta_cancels_shared_unknowns_and_uses_all_query_denominator() -> (
    None
):
    """An exact human-reviewed change remains measurable with shared unjudged results."""
    order = [str(i) for i in range(12)]
    before = {"text": order, "image": order}
    after = {"text": order[:9] + ["10", "9", "11"], "image": order}
    queries: dict = {
        "text": dict(
            mode="text",
            expected_pairs=12,
            judgments={"9": dict(grade=0, human=True), "10": dict(grade=1, human=True)},
        ),
        "image": dict(mode="image", expected_pairs=12, judgments={}),
    }
    result = confirmed_change_metrics(before, after, queries)
    assert result["scopes"]["text"]["precision_at_10_delta"] == pytest.approx(0.1)
    assert result["scopes"]["all"]["precision_at_10_delta"] == pytest.approx(0.05)
    assert result["scopes"]["image"]["precision_at_10_delta"] == 0
    assert result["scopes"]["text"]["recall_at_10_delta"] == pytest.approx([1 / 11, 1])
    assert precision_delta(before, after, queries)["all"] == pytest.approx([0.05, 0.05])
    queries["text"]["judgments"]["10"]["human"] = False
    with pytest.raises(ValueError, match="binary human judgment"):
        confirmed_change_metrics(before, after, queries)
