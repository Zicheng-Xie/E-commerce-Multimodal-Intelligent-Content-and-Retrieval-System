"""Full-image and actual subject-crop Qwen interactions on one frozen union."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from multimodal_ecommerce.intent_retrieval import normalized_ranks
from multimodal_ecommerce.semantic_ranker import ranked_ids

METHODS = ("crop_context_union", "whole_crop_context_union", "whole_crop_roi_union")
PRIMARY = "whole_crop_context_union"


class _PolicySpec(TypedDict):
    """Exact fields of the POLICY configuration and provenance record."""

    cross_encoder_weight: float
    whole_old_new_vote: float
    whole_crop_vote: float
    optional_original_FG_ROI_vote: float
    common_pool: str
    common_tail: str
    crop_context: str
    text_route: str
    full_model_configuration_unchanged: bool
    no_score_imputation: bool
    no_public_type_penalties: bool
    fixed_k: int


POLICY: _PolicySpec = dict(
    cross_encoder_weight=0.65,
    whole_old_new_vote=0.5,
    whole_crop_vote=0.5,
    optional_original_FG_ROI_vote=0.25,
    common_pool="unchanged stage24 original20/new20 union",
    common_tail="original cross20 relative order outside shared union",
    crop_context="frozen new-caption query text; actual query/candidate zoom or full fallback",
    text_route="original cross20 exactly",
    full_model_configuration_unchanged=True,
    no_score_imputation=True,
    no_public_type_penalties=True,
    fixed_k=10,
)


@dataclass(frozen=True)
class CaptionRoiRankerFuseCropsInputs:
    """Input values for fuse crops.

    Attributes:
        ids: Identifiers in the order expected by the associated arrays.
        pool: Pool used by this operation.
        old_logits: Old logits used by this operation.
        new_logits: New logits used by this operation.
        crop_logits: Crop logits used by this operation.
        old_base: Old base used by this operation.
        new_base: New base used by this operation.
        auxiliary: Auxiliary used by this operation.
        baseline: Baseline used by this operation.
    """

    ids: list[str]
    pool: list[str]
    old_logits: npt.ArrayLike
    new_logits: npt.ArrayLike
    crop_logits: npt.ArrayLike
    old_base: npt.ArrayLike
    new_base: npt.ArrayLike
    auxiliary: npt.ArrayLike
    baseline: list[str]


def fuse_crops(options: CaptionRoiRankerFuseCropsInputs) -> tuple[dict, dict]:
    """Return fuse crops for this workflow.

    Args:
        options: Validated input fields described by CaptionRoiRankerFuseCropsInputs.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Full unique gallery and aligned shared union required; Real whole
            and crop context scores required for every candidate.
    """
    ids = options.ids
    pool = options.pool
    old_logits = options.old_logits
    new_logits = options.new_logits
    crop_logits = options.crop_logits
    old_base = options.old_base
    new_base = options.new_base
    auxiliary = options.auxiliary
    baseline = options.baseline
    if (
        len(ids) != len(set(ids))
        or len(pool) != len(set(pool))
        or not set(pool) <= set(ids)
        or len(pool) < 10
        or len(baseline) != len(ids)
        or set(baseline) != set(ids)
    ):
        raise ValueError("Full unique gallery and aligned shared union required")
    arrays = [
        np.asarray(value, dtype=float)
        for value in (
            old_logits,
            new_logits,
            crop_logits,
            old_base,
            new_base,
            auxiliary,
        )
    ]
    lengths = [len(pool)] * 3 + [len(ids)] * 3
    if any(
        value.shape != (size,) or not np.isfinite(value).all()
        for value, size in zip(arrays, lengths)
    ):
        raise ValueError(
            "Real whole and crop context scores required for every candidate"
        )
    old_logits, new_logits, crop_logits, old_base, new_base, auxiliary = arrays
    positions = {cid: i for i, cid in enumerate(ids)}
    selected = [positions[cid] for cid in pool]
    whole = 0.5 * normalized_ranks(pool, old_logits) + 0.5 * normalized_ranks(
        pool, new_logits
    )
    crop = normalized_ranks(pool, crop_logits)
    combined = 0.5 * whole + 0.5 * crop
    base = (
        0.5 * normalized_ranks(ids, old_base) + 0.5 * normalized_ranks(ids, new_base)
    )[selected]
    scores = {
        "crop_context_union": 0.65 * crop + 0.35 * base,
        PRIMARY: 0.65 * combined + 0.35 * base,
        "whole_crop_roi_union": 0.65
        * (0.75 * combined + 0.25 * normalized_ranks(pool, auxiliary[selected]))
        + 0.35 * base,
    }
    tail = [cid for cid in baseline if cid not in set(pool)]
    return (
        {name: ranked_ids(pool, values) + tail for name, values in scores.items()},
        dict(
            scores={
                name: dict(zip(pool, values.tolist()))
                for name, values in scores.items()
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
        ValueError: Full unique text ranking required.
    """
    if len(baseline) != len(set(baseline)):
        raise ValueError("Full unique text ranking required")
    return {name: list(baseline) for name in METHODS}
