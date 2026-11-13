"""Fuse existing retrieval evidence inside a fully scored candidate union."""

from __future__ import annotations

import numpy as np

from multimodal_ecommerce.simple_baseline import stable_order


def fuse_paper_candidates(
    cross: list[str],
    dual: list[str],
    regional: list[str],
    lexical: np.ndarray,
    weight: float,
) -> list[str]:
    """Combine full-model ranks and optional attribute BM25 in the scored head.

    Args:
        cross: Complete original Top20 cross-encoder ranking.
        dual: Complete whole/crop cross-encoder ranking.
        regional: Complete regional/title ranking.
        lexical: Nonnegative BM25 scores aligned to the cross ranking.
        weight: Fraction of the normalized head score allocated to BM25.

    Returns:
        Full ranking with the original first result and unselected tail preserved.
    """
    if len(cross) < 20 or len(set(cross)) != len(cross):
        raise ValueError("Expected a unique full gallery of at least twenty products")
    if any(
        len(order) != len(cross) or set(order) != set(cross)
        for order in [dual, regional]
    ):
        raise ValueError("Fusion ranks must refer to the same complete gallery")
    if (
        lexical.shape != (len(cross),)
        or not np.isfinite(lexical).all()
        or (lexical < 0).any()
        or not 0 <= weight <= 1
    ):
        raise ValueError("Invalid lexical evidence or weight")
    union = set(cross[:20]) | set(dual[:20])
    pool = [
        candidate for candidate in cross if candidate in union and candidate != cross[0]
    ]
    positions = [
        {cid: i for i, cid in enumerate(order)} for order in [cross, dual, regional]
    ]
    scores = np.array(
        [
            sum(
                vote / (20 + ranks[cid])
                for vote, ranks in zip([3, 2, 1], positions, strict=True)
            )
            for cid in pool
        ]
    )
    spread = float(np.ptp(scores))
    if spread:
        scores = (scores - scores.min()) / spread
    evidence = np.array([lexical[positions[0][cid]] for cid in pool])
    if evidence.max() > 0:
        scores = (1 - weight) * scores + weight * evidence / evidence.max()
    return (
        [cross[0]]
        + stable_order(pool, scores)
        + [cid for cid in cross if cid not in union]
    )
