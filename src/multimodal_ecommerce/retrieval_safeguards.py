"""Conservative public-input safeguards for the reviewed Top20 retrieval policy."""

from __future__ import annotations

import math

from multimodal_ecommerce.intent_retrieval import main_type, public_entity
from multimodal_ecommerce.simple_baseline import rerank_head

BAG_TYPES = frozenset({"bag", "phone_bag", "computer_bag", "camera_bag", "storage_bag"})


def compound_caption_conflict(caption: dict) -> bool:
    """Abstain when a bag compound is parsed as another object despite bag appearance.

    Args:
        caption: Frozen query-only product and appearance observations.

    Returns:
        Whether the noun-head interpretation contradicts its own appearance clause.
    """
    product = caption["product"].strip()
    return (
        product.endswith("包")
        and public_entity(product) not in BAG_TYPES
        and main_type("商品主体：" + caption["appearance"]) in BAG_TYPES
    )


def guarded_top20(original: list[str], proposed: list[str], scores: dict) -> list[str]:
    """Preserve a third original result when its score supports the protected prefix.

    Args:
        original: Full original-image ranking, with the first ten reserved in the pool.
        proposed: Full ranking whose first twenty have complete model scores.
        scores: Actual query/candidate logits for exactly those twenty products.

    Returns:
        Full same-gallery ranking with semantic selection and visual Top10 ordering.
    """
    if (
        len(original) < 20
        or len(original) != len(set(original))
        or len(proposed) != len(set(proposed))
        or set(original) != set(proposed)
        or not set(original[:10]) <= set(proposed[:20])
        or set(scores) != set(proposed[:20])
        or not all(math.isfinite(v) for v in scores.values())
    ):
        raise ValueError(
            "Complete scored Top20, original Top10, and equal galleries required"
        )
    keep = 3 if scores[original[2]] >= min(scores[cid] for cid in original[:2]) else 2
    prefix = original[:keep]
    ranked = prefix + [
        cid for cid in rerank_head(proposed, scores, 20) if cid not in set(prefix)
    ]
    selected = set(ranked[:10])
    return [cid for cid in original if cid in selected] + ranked[10:]
