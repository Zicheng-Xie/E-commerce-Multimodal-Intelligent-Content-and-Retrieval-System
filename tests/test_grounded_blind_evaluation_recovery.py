"""Provenance adaptation preserves original facts, ranks and exact metrics."""

from __future__ import annotations

import copy

import pytest

from grounded_blind_evaluation_recovery import ORIGIN, evaluator_view
from multimodal_ecommerce.interval_retrieval_metrics import (
    evaluate_intervals,
)


def test_ai_source_adapter_preserves_metrics_and_labels() -> None:
    """Verify that AI source enum adapter preserves actual metric values and original
    labels.
    """
    ids = [f"c{i}" for i in range(12)]
    ranks: dict = {"q": {"baseline": ids}}
    modes: dict = {"q": "image"}
    labels = [
        dict(
            query_id="q",
            candidate_id=cid,
            grade=1 if i < 4 else None if i == 5 else 0,
            reason="visible evidence",
            review_origin=ORIGIN,
            human_label=False,
        )
        for i, cid in enumerate(ids)
    ]
    before = copy.deepcopy(labels)
    adapted = evaluator_view(labels)
    legacy = [dict(r, review_origin="model") for r in labels]
    assert evaluate_intervals(ranks, adapted, modes) == evaluate_intervals(
        ranks, legacy, modes
    )
    assert labels == before and all(r["annotation_tool"] == ORIGIN for r in adapted)
    with pytest.raises(ValueError):
        evaluator_view([dict(labels[0], human_label=True)])
