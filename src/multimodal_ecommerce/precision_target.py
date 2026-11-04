"""Fixed-denominator target feasibility; dataset capacity is not model precision."""

from __future__ import annotations

import math


def precision_capacity(
    positive_counts: list[int],
    *,
    k: int = 10,
    precision_target: float = 0.6,
    recall_target: float = 0.7,
) -> dict:
    """Compute exact oracle bounds from complete relevance counts, keeping zeros.

    Args:
        positive_counts: Positive counts used by this operation.
        k: K used by this operation.
        precision_target: Precision target used by this operation.
        recall_target: Recall target used by this operation.

    Returns:
        Mapping with fields: queries, k, fixed_top_k_slots,
        positive_relations_total, oracle_top_k_true_positives,
        required_top_k_true_positives, precision_target, precision_ceiling,
        precision_target_feasible, unavoidable_shortfall_in_true_positives,
        recall_target, recall_ceiling_on_positive_queries, positive_query_count,
        zero_positive_queries, note, per_query_precision_ceilings.

    Raises:
        ValueError: Need a nonempty cohort and positive integer k; Complete positive
            counts must be nonnegative integers; Targets must be proportions.
    """
    if not positive_counts or type(k) is not int or k < 1:
        raise ValueError("Need a nonempty cohort and positive integer k")
    if any(type(n) is not int or n < 0 for n in positive_counts):
        raise ValueError("Complete positive counts must be nonnegative integers")
    if not 0 <= precision_target <= 1 or not 0 <= recall_target <= 1:
        raise ValueError("Targets must be proportions")
    top_hits = [min(k, n) for n in positive_counts]
    slots = len(positive_counts) * k
    oracle_hits = sum(top_hits)
    needed_hits = math.ceil(precision_target * slots - 1e-12)
    positive_queries = [n for n in positive_counts if n > 0]
    recall_bound = (
        sum(min(k, n) / n for n in positive_queries) / len(positive_queries)
        if positive_queries
        else None
    )
    return {
        "queries": len(positive_counts),
        "k": k,
        "fixed_top_k_slots": slots,
        "positive_relations_total": sum(positive_counts),
        "oracle_top_k_true_positives": oracle_hits,
        "required_top_k_true_positives": needed_hits,
        "precision_target": precision_target,
        "precision_ceiling": oracle_hits / slots,
        "precision_target_feasible": oracle_hits >= needed_hits,
        "unavoidable_shortfall_in_true_positives": max(0, needed_hits - oracle_hits),
        "recall_target": recall_target,
        "recall_ceiling_on_positive_queries": recall_bound,
        "positive_query_count": len(positive_queries),
        "zero_positive_queries": len(positive_counts) - len(positive_queries),
        "note": "Precision includes all queries with fixed k. Zero-positive queries have undefined Recall and are explicitly counted. Capacity is not an observed quality score.",
        "per_query_precision_ceilings": [n / k for n in top_hits],
    }


def partial_precision_bounds(
    grades_by_query: list[list[int | None]], k: int = 10
) -> dict:
    """For incomplete labels, preserve uncertainty instead of filling in negatives.

    Args:
        grades_by_query: Grades by query used by this operation.
        k: K used by this operation.

    Returns:
        Mapping with fields: queries, pending_relations, capacity_lower_bound,
        capacity_upper_bound, quality_metrics, acceptance_ready, scope.

    Raises:
        ValueError: Need a cohort and positive integer k; Grades must be integer 0/1 or
            null.
    """
    if not grades_by_query or type(k) is not int or k < 1:
        raise ValueError("Need a cohort and positive integer k")
    if any(
        g is not None and (type(g) is not int or g not in (0, 1))
        for rows in grades_by_query
        for g in rows
    ):
        raise ValueError("Grades must be integer 0/1 or null")
    positives = [sum(g == 1 for g in rows) for rows in grades_by_query]
    pending = [sum(g is None for g in rows) for rows in grades_by_query]
    return {
        "queries": len(grades_by_query),
        "pending_relations": sum(pending),
        "capacity_lower_bound": sum(min(k, p) for p in positives)
        / (k * len(positives)),
        "capacity_upper_bound": sum(min(k, p + u) for p, u in zip(positives, pending))
        / (k * len(positives)),
        "quality_metrics": None,
        "acceptance_ready": False,
        "scope": "Bounds describe only the supplied candidate relationships; incomplete gallery coverage cannot establish full-gallery capacity or Recall",
    }
