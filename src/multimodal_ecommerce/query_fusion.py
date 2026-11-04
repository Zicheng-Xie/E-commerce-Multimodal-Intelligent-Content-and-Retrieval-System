"""Public-query multi-view and description channels, with explicit abstention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multimodal_ecommerce.intent_retrieval import (
    balanced_pool,
    normalized_ranks,
    recall_channels,
)
from multimodal_ecommerce.model_interfaces import SearchableIndex
from multimodal_ecommerce.regional_retrieval import RegionalSearchRequest
from multimodal_ecommerce.semantic_ranker import ranked_ids
from multimodal_ecommerce.visual_query import VISUAL_QUERY_POLICY

CAPTION_WEIGHTS: dict[str, float] = dict(regions=0.5, title=0.25, profile=0.25)


@dataclass(frozen=True)
class QueryFusionVisualChannelsInputs:
    """Input values for visual channels.

    Attributes:
        index: Index used by this operation.
        title_bank: Title bank used by this operation.
        profile_bank: Profile bank used by this operation.
        base_features: Base features used by this operation.
        query_views: Query views used by this operation.
        caption_vector: Caption vector used by this operation.
        description: Description used by this operation.
        positions: Positions used by this operation.
    """

    index: SearchableIndex
    title_bank: np.ndarray
    profile_bank: np.ndarray
    base_features: np.ndarray
    query_views: np.ndarray
    caption_vector: np.ndarray
    description: str
    positions: np.ndarray


def visual_channels(options: QueryFusionVisualChannelsInputs) -> dict:
    """Fuse only runtime fields. Empty descriptions contribute no arbitrary ID ranks.

    Args:
        options: Validated input fields described by QueryFusionVisualChannelsInputs.

    Returns:
        Mapping with fields: ids, multi_scores, caption_scores, fused_scores, pool,
        description, description_channel_active.

    Raises:
        ValueError: Query channels need aligned unique document IDs; Described queries
            require an actual nonzero caption vector.
    """
    index = options.index
    title_bank = options.title_bank
    profile_bank = options.profile_bank
    base_features = options.base_features
    query_views = options.query_views
    caption_vector = options.caption_vector
    description = options.description
    positions = options.positions
    ids = [index.records[i]["id"] for i in positions]
    if len(set(ids)) != len(ids) or base_features.shape[0] != len(ids):
        raise ValueError("Query channels need aligned unique document IDs")

    def score_map(vector: np.ndarray) -> np.ndarray:
        """Compute map.

        Args:
            vector: Vector used by this operation.

        Returns:
            Computed array in the order of the supplied records.
        """
        result = {
            r["id"]: r["score"]
            for r in index.search(
                RegionalSearchRequest(
                    vector, top_k=len(index.records), strategy="attention"
                )
            )
        }
        return np.array([result[cid] for cid in ids])

    multi = score_map(query_views)
    rrf, channels = recall_channels(ids, base_features)
    weights = VISUAL_QUERY_POLICY
    fused = weights["global_weight"] * normalized_ranks(ids, rrf) + weights[
        "multi_view_weight"
    ] * normalized_ranks(ids, multi)
    channels.append(ranked_ids(ids, multi))
    caption = np.zeros(len(ids), dtype=float)
    if description:
        if (
            caption_vector.shape != (title_bank.shape[1],)
            or not np.isfinite(caption_vector).all()
            or np.linalg.norm(caption_vector) < 0.9
        ):
            raise ValueError(
                "Described queries require an actual nonzero caption vector"
            )
        caption = (
            CAPTION_WEIGHTS["regions"] * score_map(caption_vector)
            + CAPTION_WEIGHTS["title"] * (title_bank[positions] @ caption_vector)
            + CAPTION_WEIGHTS["profile"] * (profile_bank[positions] @ caption_vector)
        )
        fused += weights["caption_weight"] * normalized_ranks(ids, caption)
        channels.append(ranked_ids(ids, caption))
    channels.append(ranked_ids(ids, fused))
    return dict(
        ids=ids,
        multi_scores=multi.tolist(),
        caption_scores=caption.tolist(),
        fused_scores=fused.tolist(),
        pool=balanced_pool(channels, min(int(weights["rerank_pool"]), len(ids))),
        description=description,
        description_channel_active=bool(description),
    )
