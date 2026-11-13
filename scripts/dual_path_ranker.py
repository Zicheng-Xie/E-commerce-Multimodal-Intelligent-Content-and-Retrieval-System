"""Fuse complete, identity-matched Qwen contexts on one shared candidate union."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from multimodal_ecommerce.intent_retrieval import normalized_ranks
from multimodal_ecommerce.semantic_ranker import ranked_ids

METHODS = (
    "original_context_union",
    "new_context_union",
    "dual_context_union",
    "dual_context_roi_union",
)


class _PolicySpec(TypedDict):
    """Exact fields of the POLICY configuration and provenance record."""

    cross_encoder_weight: float
    original_context_weight: float
    new_context_weight: float
    auxiliary_cross_vote_weight: float
    whole_and_roi_vote: float
    pool: str
    tail: str
    missing_context_score: str
    public_type_penalty: bool
    text_route: str
    fixed_k: int


POLICY: _PolicySpec = dict(
    cross_encoder_weight=0.65,
    original_context_weight=0.5,
    new_context_weight=0.5,
    auxiliary_cross_vote_weight=0.25,
    whole_and_roi_vote=0.5,
    pool="stable union of frozen original20 and new-caption20; no label filter",
    tail="same frozen original cross20 tail with all union IDs removed",
    missing_context_score="error; never fill from another context or embedding",
    public_type_penalty=False,
    text_route="frozen original cross20 exactly",
    fixed_k=10,
)


@dataclass(frozen=True)
class DualPathRankerFuseContextsInputs:
    """Input values for fuse contexts.

    Attributes:
        ids: Identifiers in the order expected by the associated arrays.
        pool: Pool used by this operation.
        old_logits: Old logits used by this operation.
        new_logits: New logits used by this operation.
        old_base: Old base used by this operation.
        new_base: New base used by this operation.
        auxiliary: Auxiliary used by this operation.
        baseline: Baseline used by this operation.
    """

    ids: list[str]
    pool: list[str]
    old_logits: npt.ArrayLike
    new_logits: npt.ArrayLike
    old_base: npt.ArrayLike
    new_base: npt.ArrayLike
    auxiliary: npt.ArrayLike
    baseline: list[str]


def fuse_contexts(options: DualPathRankerFuseContextsInputs) -> tuple[dict, dict]:
    """Return fuse contexts for this workflow.

    Args:
        options: Validated input fields described by DualPathRankerFuseContextsInputs.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Need a complete gallery, baseline and unique shared pool; Every
            candidate requires real scores in both contexts.
    """
    ids = options.ids
    pool = options.pool
    old_logits = options.old_logits
    new_logits = options.new_logits
    old_base = options.old_base
    new_base = options.new_base
    auxiliary = options.auxiliary
    baseline = options.baseline
    if (
        len(ids) != len(set(ids))
        or len(pool) != len(set(pool))
        or not set(pool) <= set(ids)
        or len(baseline) != len(ids)
        or set(baseline) != set(ids)
        or len(pool) < 10
    ):
        raise ValueError("Need a complete gallery, baseline and unique shared pool")
    arrays = [
        np.asarray(v, dtype=float)
        for v in (old_logits, new_logits, old_base, new_base, auxiliary)
    ]
    expected = [len(pool), len(pool), len(ids), len(ids), len(ids)]
    if any(
        a.shape != (size,) or not np.isfinite(a).all()
        for a, size in zip(arrays, expected)
    ):
        raise ValueError("Every candidate requires real scores in both contexts")
    old_logits, new_logits, old_base, new_base, auxiliary = arrays
    positions = {cid: i for i, cid in enumerate(ids)}
    selected = [positions[cid] for cid in pool]
    old_ce, new_ce = normalized_ranks(pool, old_logits), normalized_ranks(
        pool, new_logits
    )
    old_rank = normalized_ranks(ids, old_base)[selected]
    new_rank = normalized_ranks(ids, new_base)[selected]
    ce = 0.5 * old_ce + 0.5 * new_ce
    base = 0.5 * old_rank + 0.5 * new_rank
    scores = {
        "original_context_union": 0.65 * old_ce + 0.35 * old_rank,
        "new_context_union": 0.65 * new_ce + 0.35 * new_rank,
        "dual_context_union": 0.65 * ce + 0.35 * base,
        "dual_context_roi_union": 0.65
        * (0.75 * ce + 0.25 * normalized_ranks(pool, auxiliary[selected]))
        + 0.35 * base,
    }
    tail = [cid for cid in baseline if cid not in set(pool)]
    return (
        {method: ranked_ids(pool, values) + tail for method, values in scores.items()},
        dict(
            scores={
                method: dict(zip(pool, values.tolist()))
                for method, values in scores.items()
            },
            shared_pool=list(pool),
            unchanged_common_tail=tail,
            score_is_probability=False,
            qrels_used=False,
        ),
    )


def text_orders(baseline: list[str]) -> dict:
    """Return text orders for this workflow.

    Args:
        baseline: Baseline used by this operation.

    Returns:
        Result mapping produced by the workflow.

    Raises:
        ValueError: Unique full text ranking required.
    """
    if len(baseline) != len(set(baseline)):
        raise ValueError("Unique full text ranking required")
    return {method: list(baseline) for method in METHODS}
