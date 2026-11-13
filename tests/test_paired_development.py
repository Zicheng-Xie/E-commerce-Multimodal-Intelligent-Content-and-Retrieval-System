"""Overlapping independent intervals can still have exact zero paired loss."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from multimodal_ecommerce.ai_review_metrics import metric_ranges


def test_nonregression_uses_same_unknown_labels_for_both_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that nonregression uses same unknown labels for both methods.

    Args:
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("report_focus_coverage")
    ids = ["known", "unknown", *[f"n{i}" for i in range(10)]]
    ranks: dict = {"q": dict(global_=["unknown", "known", *ids[2:]], candidate=ids)}
    ranks["q"]["global"] = ranks["q"].pop("global_")
    labels = [
        dict(
            query_id="q",
            candidate_id=cid,
            grade=1 if cid == "known" else None if cid == "unknown" else 0,
            review_origin="model",
            human_label=False,
        )
        for cid in ids
    ]
    ordinary = metric_ranges(ranks, labels, dict(q="image"))["scopes"]["all"]
    assert ordinary["candidate"]["ranges"]["precision@10"] == [0.1, 0.2]
    assert ordinary["global"]["ranges"]["precision@10"] == [0.1, 0.2]
    result = module.paired_differences(ranks, labels, dict(q="image"))
    metrics = result["scopes"]["all"]["candidate"]["global"]
    assert metrics["precision@10"]["delta_range"] == [0, 0]
    assert (
        metrics["precision@10"]["nonregression_proven_for_shared_unknown_labels"]
        is True
    )
    assert metrics["hit@1"]["delta_range"] == [0, 1]
