"""Query-only semantic crop routing with an independent original-head control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

from multimodal_ecommerce.semantic_ranker import ranked_ids

METHODS = (
    "semantic_crop_gate",
    "semantic_crop_gate_head1",
    "unverified_crop_head1",
    "whole_context_head1",
)
PRIMARY = "semantic_crop_gate_head1"


class _PolicySpec(TypedDict):
    """Exact fields of the POLICY configuration and provenance record."""

    accepted_route: str
    rejected_or_uncertain_route: str
    acceptance: str
    head_control: str
    common_pool: str
    common_tail: str
    text_route: str
    no_new_retrieval_model_or_imputed_scores: bool
    head1_is_not_a_general_MRR_or_nonregression_guarantee: bool
    no_query_ID_exceptions_or_qrels: bool
    fixed_k: int


POLICY: _PolicySpec = dict(
    accepted_route="stage25 whole_crop_context_union actual frozen full-model scores",
    rejected_or_uncertain_route="stage24 dual_context_union actual frozen full-model scores",
    acceptance="completed valid query-only decision accept and all four predicates exactly true; valid reject/uncertain or inherited whole fallback is false, engineering errors remain unfinished",
    head_control="retain original cross20 first item; other shared-pool items keep selected-context order",
    common_pool="unchanged original20/new20 union, 504 query-candidate appearances",
    common_tail="original cross20 relative order outside shared union",
    text_route="original cross20 exactly",
    no_new_retrieval_model_or_imputed_scores=True,
    head1_is_not_a_general_MRR_or_nonregression_guarantee=True,
    no_query_ID_exceptions_or_qrels=True,
    fixed_k=10,
)


@dataclass(frozen=True)
class RoiSemanticGateRankerGatedOrdersInputs:
    """Input values for gated orders.

    Attributes:
        ids: Identifiers in the order expected by the associated arrays.
        pool: Pool used by this operation.
        whole_scores: Whole scores used by this operation.
        crop_scores: Crop scores used by this operation.
        baseline: Baseline used by this operation.
        accepted: Accepted used by this operation.
    """

    ids: list[str]
    pool: list[str]
    whole_scores: dict[str, float]
    crop_scores: dict[str, float]
    baseline: list[str]
    accepted: bool


def gated_orders(options: RoiSemanticGateRankerGatedOrdersInputs) -> tuple[dict, dict]:
    """Route only on a strict query-only decision; never inspect relevance labels.

    Args:
        options: Validated input fields described by
            RoiSemanticGateRankerGatedOrdersInputs.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: An explicit boolean semantic decision is required; Full gallery,
            original head and unchanged union required; Both real parent composite score
            maps are required.
    """
    ids = options.ids
    pool = options.pool
    whole_scores = options.whole_scores
    crop_scores = options.crop_scores
    baseline = options.baseline
    accepted = options.accepted
    if type(accepted) is not bool:
        raise ValueError("An explicit boolean semantic decision is required")
    if (
        len(ids) != len(set(ids))
        or len(pool) != len(set(pool))
        or len(pool) < 10
        or not set(pool) <= set(ids)
        or len(baseline) != len(ids)
        or set(baseline) != set(ids)
        or baseline[0] not in pool
    ):
        raise ValueError("Full gallery, original head and unchanged union required")
    for values in (whole_scores, crop_scores):
        if set(values) != set(pool) or any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()
        ):
            raise ValueError("Both real parent composite score maps are required")

    def protect_head(values: dict) -> dict[str, float]:
        """Return protect head for this workflow.

        Args:
            values: Values used by this operation.

        Returns:
            Result mapping produced by the workflow.
        """
        protected = dict(values)
        protected[baseline[0]] = max(values.values()) + 1.0
        return protected

    selected = crop_scores if accepted else whole_scores
    scores = {
        "semantic_crop_gate": dict(selected),
        PRIMARY: protect_head(selected),
        "unverified_crop_head1": protect_head(crop_scores),
        "whole_context_head1": protect_head(whole_scores),
    }
    tail = [cid for cid in baseline if cid not in set(pool)]
    return (
        {
            name: ranked_ids(pool, [values[cid] for cid in pool]) + tail
            for name, values in scores.items()
        },
        dict(
            scores=scores,
            shared_pool=list(pool),
            unchanged_common_tail=tail,
            accepted_crop=accepted,
            route="whole_and_crop" if accepted else "whole_context_only",
            protected_original_head=baseline[0],
            score_kind="parent composite rank score; protected head receives max+1, not a raw model logit",
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
