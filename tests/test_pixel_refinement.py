"""Bounded uncertain review preserves old evidence and model provenance."""

from __future__ import annotations

import pytest

from multimodal_ecommerce.pixel_refinement import (
    merge_machine_refinement,
    select_uncertain,
)


def test_selection_is_capped_and_never_uses_known_relations() -> None:
    """Verify that selection is capped and never uses known relations."""
    rows = [
        dict(query_id="q", candidate_id=str(i), grade=None if i else 1)
        for i in range(10)
    ]
    ranks: dict = {
        "q": {
            "a": [str(i) for i in range(10)],
            "b": list(reversed([str(i) for i in range(10)])),
        }
    }
    selected = select_uncertain(rows, ranks, 3)
    assert len(selected) == 3 and all(r["grade"] is None for r in selected)
    assert selected == select_uncertain(list(reversed(rows)), ranks, 3)


def test_refinement_cannot_rewrite_knowns_or_claim_human_origin() -> None:
    """Verify that refinement cannot rewrite knowns or claim human origin."""
    rows: list[dict] = [
        dict(query_id="q", candidate_id="a", grade=None),
        dict(query_id="q", candidate_id="b", grade=1),
    ]
    review: dict = dict(
        query_id="q",
        candidate_id="a",
        grade=0,
        reason="主体明显不同",
        review_origin="model",
        human_label=False,
    )
    out = merge_machine_refinement(rows, [review])
    assert rows[0]["grade"] is None and out[0]["grade"] == 0 and out[1] == rows[1]
    pending: dict = {**review, "grade": None}
    assert merge_machine_refinement(rows, [pending]) == rows
    with pytest.raises(ValueError):
        merge_machine_refinement(rows, [{**review, "candidate_id": "b"}])
    with pytest.raises(ValueError):
        merge_machine_refinement(rows, [{**review, "review_origin": "human"}])
    with pytest.raises(ValueError):
        merge_machine_refinement(rows, [{**review, "grade": True}])
