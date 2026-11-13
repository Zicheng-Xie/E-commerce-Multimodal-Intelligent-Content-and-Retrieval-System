"""Dataset-independent selection for the frozen public-conflict retrieval policy."""

from __future__ import annotations

import math

from multimodal_ecommerce.head_guard import protected_prefix
from multimodal_ecommerce.intent_retrieval import public_entity
from multimodal_ecommerce.public_conflict_guard import reorder_public_conflicts
from multimodal_ecommerce.retrieval_safeguards import (
    compound_caption_conflict,
    guarded_top20,
)
from multimodal_ecommerce.simple_baseline import rerank_head
from multimodal_ecommerce.visual_query import caption_text


def select_pool(
    base: list[str], query: dict, observations: dict
) -> tuple[list[str], bool]:
    """Construct the existing Top20 from model-only coarse/title/type observations.

    Args:
        base: Complete coarse ranking after explicit query exclusions.
        query: Public text/image query.
        observations: Query-only caption, title retrieval and candidate-only types.

    Returns:
        Full proposal order and whether image semantic reranking is active.
    """
    if len(base) < 20 or len(base) != len(set(base)):
        raise ValueError("Complete unique eligible gallery of at least twenty required")
    if query["mode"] == "text":
        tail = observations["title_order"]
        if len(tail) != len(set(tail)) or not set(tail) <= set(base):
            raise ValueError("Title retrieval must contain unique eligible candidates")
        extra = [cid for cid in tail if cid not in set(base[:10])][:10]
        active = True
    else:
        caption = observations["caption"]
        kind = public_entity(caption["product"]) if caption_text(caption) else None
        extra = [
            cid
            for cid in base[10:]
            if kind and observations["supported_types"].get(cid) == kind
        ][:10]
        active = bool(extra)
    head = base[:10] + extra
    head += [cid for cid in base if cid not in set(head)][: 20 - len(head)]
    return head + [cid for cid in base if cid not in set(head)], active


def finish_ranking(
    base: list[str], query: dict, observations: dict, titles: dict
) -> tuple[list[str], dict]:
    """Apply exact semantic protection and public conflicts without relevance inputs.

    Args:
        base: Complete current coarse order.
        query: Public query modality and optional text.
        observations: Public caption/types/title order and actual proposal logits.
        titles: Original titles for exactly the eligible candidates.

    Returns:
        Full result ranking and trace of pool selection and public conflicts.
    """
    proposal, active = select_pool(base, query, observations)
    conflict = query["mode"] == "image" and compound_caption_conflict(
        observations["caption"]
    )
    if query["mode"] == "text" or active and not conflict:
        scores = observations["scores"]
        if set(scores) != set(proposal[:20]) or any(
            not math.isfinite(v) for v in scores.values()
        ):
            raise ValueError(
                "Exactly twenty finite, real cross-encoder scores required"
            )
        ranked = (
            protected_prefix(base, rerank_head(proposal, scores, 20), 2)
            if query["mode"] == "text"
            else guarded_top20(base, proposal, scores)
        )
    else:
        ranked = list(base)
    ranked, trace = reorder_public_conflicts(
        ranked,
        titles,
        text=query.get("text"),
        caption=observations.get("caption"),
        options=dict(protect=0, broad=False, matched_only=True),
    )
    return ranked, dict(
        pool=proposal[:20],
        semantic_active=active,
        caption_abstention=conflict,
        conflicts=trace,
    )
