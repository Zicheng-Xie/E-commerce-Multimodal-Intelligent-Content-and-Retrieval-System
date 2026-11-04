"""Selected label-free post-ranking paths for the product retrieval CLI.

The existing exploration policies stay unchanged for saved-result verification.
"""

from __future__ import annotations

import time
from dataclasses import KW_ONLY, dataclass
from typing import Any

import numpy as np

from multimodal_ecommerce.regional_retrieval import RegionalSearchRequest
from multimodal_ecommerce.retrieval import unit_rows
from multimodal_ecommerce.retrieval_strategies import (
    EXPLORATION_POLICY,
    RetrievalStrategiesReciprocalScoresInputs,
    alpha_query_expansion,
    attention_scores,
    ordered_indices,
    reciprocal_rank_fusion,
    reciprocal_scores,
    subset_region_index,
)

POSTRANK_METHODS = (
    "gallery_attention",
    "alpha_qe_attention",
    "reciprocal_global",
    "reciprocal_attention_fusion",
)


def selected_scores(
    index: Any, query: np.ndarray, strategy: str
) -> tuple[np.ndarray, dict]:
    """Execute just one selected policy with the exact explored configuration.

    Args:
        index: Index used by this operation.
        query: Query record or representation used by this operation.
        strategy: Strategy used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Unknown post-ranking strategy.
    """
    if strategy not in POSTRANK_METHODS:
        raise ValueError("Unknown post-ranking strategy")
    p = EXPLORATION_POLICY
    queries = unit_rows(np.atleast_2d(query))
    q = queries[0]
    bank = index.vectors[:, 0]
    ids = [row["id"] for row in index.records]
    feedback: list[str] = []
    if strategy == "gallery_attention":
        scores = attention_scores(index, queries[:1])
    elif strategy == "alpha_qe_attention":
        expanded, feedback = alpha_query_expansion(
            q, bank, ids, p["qe_neighbors"], p["qe_alpha"]
        )
        scores = attention_scores(index, expanded.reshape(1, -1))
    else:
        scores = reciprocal_scores(
            RetrievalStrategiesReciprocalScoresInputs(
                q,
                bank,
                ids,
                k1=p["reciprocal_k1"],
                k2=p["reciprocal_k2"],
                original_weight=p["reciprocal_original_weight"],
            )
        )
        if strategy == "reciprocal_attention_fusion":
            regional = attention_scores(index, queries[:1])
            mix = p["reciprocal_region_fusion"]
            scores = mix * scores - (1 - mix) * (1 - regional) / 2
    return scores, {
        "policy": dict(p),
        "feedback_ids": feedback,
        "source_identity_or_category_used": False,
        "query_mode": "First query view only; explicit user ROI may define that view",
    }


@dataclass(frozen=True)
class PostrankingSearchPostrankInputs:
    """Input values for search postrank.

    Attributes:
        index: Index used by this operation.
        query: Query record or representation used by this operation.
        strategy: Strategy used by this operation.
        top_k: Top k used by this operation.
        exclude_ids: Exclude ids used by this operation.
        texts: Texts used by this operation.
        title_fusion: Title fusion used by this operation.
    """

    index: Any
    query: np.ndarray
    _: KW_ONLY
    strategy: str = "reciprocal_attention_fusion"
    top_k: int = 10
    exclude_ids: set[str] | None = None
    texts: np.ndarray | None = None
    title_fusion: bool = False


def search_postrank(
    options: PostrankingSearchPostrankInputs,
) -> tuple[list[dict], dict]:
    """Apply exclusions before all feedback; optionally fuse title rank evidence.

    Focus previews and region weights explain the original region branch, not
    the graph-context component or a learned spatial attention layer.

    Args:
        options: Validated input fields described by PostrankingSearchPostrankInputs.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: top_k must be positive; Title fusion requires genuine source-title
            vectors; Title vectors and region bank disagree.
    """
    index = options.index
    query = options.query
    strategy = options.strategy
    top_k = options.top_k
    exclude_ids = options.exclude_ids
    texts = options.texts
    title_fusion = options.title_fusion
    if top_k < 1:
        raise ValueError("top_k must be positive")
    started = time.perf_counter()
    include = np.array(
        [
            i
            for i, row in enumerate(index.records)
            if row["id"] not in (exclude_ids or set())
        ],
        dtype=int,
    )
    if not len(include):
        return [], {
            "candidate_count": 0,
            "scoring_seconds": time.perf_counter() - started,
        }
    active = (
        index
        if len(include) == len(index.records)
        else subset_region_index(index, include)
    )
    queries = unit_rows(np.atleast_2d(query))
    scores, evidence = selected_scores(active, queries, strategy)
    visual_scores = scores.copy()
    ids = [row["id"] for row in active.records]
    title_scores = np.zeros(len(ids))
    if title_fusion:
        if texts is None:
            raise ValueError("Title fusion requires genuine source-title vectors")
        bank = unit_rows(texts)
        if bank.shape != (len(index.records), queries.shape[1]):
            raise ValueError("Title vectors and region bank disagree")
        title_scores = bank[include] @ queries[0]
        scores = reciprocal_rank_fusion(
            [scores, title_scores], ids, EXPLORATION_POLICY["rrf_constant"]
        )
    explanations = {
        row["id"]: row
        for row in active.search(
            RegionalSearchRequest(queries[:1], len(ids), strategy="attention")
        )
    }
    order = ordered_indices(scores, ids)[:top_k]
    results = []
    for rank, position in enumerate(order, 1):
        results.append(
            {
                **explanations[ids[position]],
                "rank": rank,
                "score": float(scores[position]),
                "postrank_visual_score": float(visual_scores[position]),
                "text_score": float(title_scores[position]),
                "strategy": strategy + ("_title_rrf" if title_fusion else ""),
                "region_explanation_scope": "Original region branch only; graph contribution uses neighbor context",
            }
        )
    evidence.update(
        {
            "candidate_count": len(ids),
            "excluded_ids": sorted(exclude_ids or set()),
            "title_fusion": title_fusion,
            "scoring_seconds": time.perf_counter() - started,
        }
    )
    return results, evidence
