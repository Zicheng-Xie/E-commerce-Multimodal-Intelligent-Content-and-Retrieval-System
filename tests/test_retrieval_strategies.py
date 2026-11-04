"""Test full-ranking metrics and label-free feedback/graph behavior."""

from __future__ import annotations

import numpy as np
import pytest

from multimodal_ecommerce.regional_retrieval import RegionalIndex, region_configuration
from multimodal_ecommerce.retrieval_strategies import (
    STRATEGY_NAMES,
    RetrievalStrategiesReciprocalScoresInputs,
    StrategyExplorer,
    alpha_query_expansion,
    extended_metrics,
    ordered_indices,
    reciprocal_rank_fusion,
    reciprocal_scores,
    subset_region_index,
)


def test_metrics_use_all_positives_and_full_list() -> None:
    """Verify that metrics use all positives and full list."""
    metrics = extended_metrics(["b", "a", "c", "d"], {"a", "d"})
    assert metrics["ap"] == pytest.approx(0.5)
    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["precision@10"] == pytest.approx(0.2)
    assert metrics["recall@10"] == 1
    assert metrics["hit@1"] == 0
    expected_dcg = 1 / np.log2(3) + 1 / np.log2(5)
    assert metrics["ndcg@10"] == pytest.approx(expected_dcg / (1 + 1 / np.log2(3)))
    with pytest.raises(ValueError, match="positives"):
        extended_metrics(["a"], {"a", "outside"})
    with pytest.raises(ValueError):
        extended_metrics(["a", "a"], {"a"})


def test_query_expansion_exact_formula_and_negative_feedback() -> None:
    """Verify that query expansion exact formula and negative feedback."""
    gallery = np.array([[1, 0], [0.6, 0.8], [-1, 0]])
    query = np.array([1, 0])
    expanded, ids = alpha_query_expansion(query, gallery, ["a", "b", "c"], 3, 3)
    expected = query + gallery[0] + 0.6**3 * gallery[1]
    expected /= np.linalg.norm(expected)
    assert expanded == pytest.approx(expected)
    assert ids == ["a", "b", "c"]
    assert np.linalg.norm(expanded) == pytest.approx(1)


def test_rrf_rank_evidence_and_deterministic_ties() -> None:
    """Verify that rrf rank evidence and deterministic ties."""
    branches = [np.array([100, 0, -1]), np.array([0, 2, 1])]
    fused = reciprocal_rank_fusion(branches, ["a", "b", "c"], 60)
    assert fused[0] == pytest.approx(1 / 61 + 1 / 63)
    assert fused[1] == pytest.approx(1 / 62 + 1 / 61)
    assert ordered_indices(np.array([1, 1]), ["z", "a"]).tolist() == [1, 0]
    with pytest.raises(ValueError):
        ordered_indices(np.array([np.nan]), ["a"])


def test_reciprocal_original_only_preserves_cosine_order() -> None:
    """Verify that reciprocal original only preserves cosine order."""
    gallery = np.array([[1, 0], [0.6, 0.8], [-1, 0]])
    scores = reciprocal_scores(
        RetrievalStrategiesReciprocalScoresInputs(
            np.array([1, 0]), gallery, ["a", "b", "c"], k1=2, k2=1, original_weight=1
        )
    )
    assert ordered_indices(scores, ["a", "b", "c"]).tolist() == [0, 1, 2]
    context = reciprocal_scores(
        RetrievalStrategiesReciprocalScoresInputs(
            np.array([1, 0]), gallery, ["a", "b", "c"], k1=2, k2=2
        )
    )
    assert np.isfinite(context).all()
    assert np.all(context <= 0) and np.all(context >= -1)
    permutation = np.array([2, 0, 1])
    reordered = reciprocal_scores(
        RetrievalStrategiesReciprocalScoresInputs(
            np.array([1, 0]), gallery[permutation], ["c", "a", "b"], k1=2, k2=2
        )
    )
    assert reordered == pytest.approx(context[permutation])


def test_exclusions_happen_before_feedback_and_neighborhoods() -> None:
    """Verify that exclusions happen before feedback and neighborhoods."""
    records = [
        {
            "id": str(i),
            "category": "",
            "views": [{"kind": "global"}, {"kind": "center"}],
        }
        for i in range(4)
    ]
    global_bank = np.array([[1, 0], [0.8, 0.6], [0, 1], [-1, 0]], dtype=float)
    index = RegionalIndex(
        records, np.stack([global_bank, global_bank], axis=1), region_configuration()
    )
    active = subset_region_index(index, np.array([1, 2, 3]))
    explorer = StrategyExplorer(active)
    scores, evidence = explorer.evaluate(np.array([[1, 0], [0.8, 0.6]]))
    assert set(scores) == set(STRATEGY_NAMES)
    assert "0" not in evidence["feedback_ids"]
    assert "0" not in evidence["graph_seeds"]
    assert explorer.ids == ["1", "2", "3"]
    assert all(
        value.shape == (3,) and np.isfinite(value).all() for value in scores.values()
    )
    assert not evidence["feedback_ground_truth_used"]


def test_selected_paths_match_exploration_and_title_fusion_is_explicit() -> None:
    """Verify that selected paths match exploration and title fusion is explicit."""
    from multimodal_ecommerce.postranking import (
        PostrankingSearchPostrankInputs,
        search_postrank,
        selected_scores,
    )

    records = [
        {
            "id": str(i),
            "category": "",
            "views": [{"kind": "global"}, {"kind": "center"}],
        }
        for i in range(4)
    ]
    bank = np.array([[1, 0], [0.8, 0.6], [0, 1], [-1, 0]], dtype=float)
    index = RegionalIndex(
        records, np.stack([bank, bank], axis=1), region_configuration()
    )
    query = np.array([[1, 0]])
    explored, _ = StrategyExplorer(index).evaluate(query)
    for method in (
        "gallery_attention",
        "alpha_qe_attention",
        "reciprocal_global",
        "reciprocal_attention_fusion",
    ):
        actual, _ = selected_scores(index, query, method)
        assert actual == pytest.approx(explored[method])
    results, evidence = search_postrank(
        PostrankingSearchPostrankInputs(index, query, exclude_ids={"0"}, top_k=3)
    )
    assert {r["id"] for r in results} == {"1", "2", "3"}
    assert evidence["candidate_count"] == 3
    assert "0" not in evidence["feedback_ids"]
    with pytest.raises(ValueError, match="source-title"):
        search_postrank(
            PostrankingSearchPostrankInputs(index, query, title_fusion=True)
        )
    results, evidence = search_postrank(
        PostrankingSearchPostrankInputs(index, query, texts=bank, title_fusion=True)
    )
    assert evidence["title_fusion"] is True
    assert all(r["strategy"].endswith("_title_rrf") for r in results)
