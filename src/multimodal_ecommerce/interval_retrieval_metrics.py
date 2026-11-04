"""Exact individual retrieval bounds without choosing unresolved AI labels."""

from __future__ import annotations

import math

import numpy as np

from multimodal_ecommerce.ai_review_metrics import optional_mean_bounds
from multimodal_ecommerce.business_evaluation import METRICS


def query_bounds(order: list[str], grades: dict[str, int | None]) -> dict:
    """Return query bounds for this workflow.

    Args:
        order: Order used by this operation.
        grades: Grades used by this operation.

    Returns:
        Mapping with fields: known_positive, unknown, ranges.

    Raises:
        ValueError: Full unique ranking and complete judgments required; Binary integers
            or unknown required.
    """
    if len(order) != len(set(order)) or set(order) != set(grades) or not order:
        raise ValueError("Full unique ranking and complete judgments required")
    if any(
        g is not None and (type(g) is not int or g not in (0, 1))
        for g in grades.values()
    ):
        raise ValueError("Binary integers or unknown required")
    values = [grades[c] for c in order]
    positive = sum(g == 1 for g in values)
    unknown = sum(g is None for g in values)
    top_positive = sum(g == 1 for g in values[:10])
    top_unknown = sum(g is None for g in values[:10])
    low = np.full(unknown + 1, np.inf)
    high = np.full(unknown + 1, -np.inf)
    low[0] = high[0] = 0.0
    p, u = 0, 0
    for rank, grade in enumerate(values, 1):
        if grade == 1:
            amount = (p + 1 + np.arange(u + 1)) / rank
            low[: u + 1] += amount
            high[: u + 1] += amount
            p += 1
        elif grade is None:
            # State is the number of selected unknowns; all known positives stay fixed.
            selected_counts = np.arange(1, u + 2)
            take_low = low[: u + 1] + (p + selected_counts) / rank
            take_high = high[: u + 1] + (p + selected_counts) / rank
            low[1 : u + 2] = np.minimum(low[1 : u + 2], take_low)
            high[1 : u + 2] = np.maximum(high[1 : u + 2], take_high)
            u += 1
    ap, recall, ndcg = [], [], []
    weights = [1 / math.log2(i + 2) for i in range(min(10, len(values)))]
    fixed_dcg = sum(w for w, g in zip(weights, values[:10]) if g == 1)
    uncertain_weights = sorted(w for w, g in zip(weights, values[:10]) if g is None)
    for selected in range(unknown + 1):
        total = positive + selected
        if not total:
            ap.append((0.0, 0.0))
            ndcg.append((0.0, 0.0))
            continue  # Recall is undefined for this completion.
        ap.append((float(low[selected] / total), float(high[selected] / total)))
        minimum_top = max(0, selected - (unknown - top_unknown))
        maximum_top = min(selected, top_unknown)
        recall.append(
            ((top_positive + minimum_top) / total, (top_positive + maximum_top) / total)
        )
        ideal = sum(weights[: min(total, 10)])
        ndcg.append(
            (
                (fixed_dcg + sum(uncertain_weights[:minimum_top])) / ideal,
                (
                    fixed_dcg
                    + sum(uncertain_weights[len(uncertain_weights) - maximum_top :])
                )
                / ideal,
            )
        )
    known_first = next((i for i, g in enumerate(values, 1) if g == 1), None)
    possible_first = next((i for i, g in enumerate(values, 1) if g != 0), None)

    def extrema(rows: list[tuple[float, float]]) -> list[float | None]:
        """Return extrema for this workflow.

        Args:
            rows: Records processed by this operation.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        return (
            [min(r[0] for r in rows), max(r[1] for r in rows)] if rows else [None, None]
        )

    return dict(
        known_positive=positive,
        unknown=unknown,
        ranges={
            "precision@10": [top_positive / 10, (top_positive + top_unknown) / 10],
            "recall@10": extrema(recall),
            "ap": extrema(ap),
            "ndcg@10": extrema(ndcg),
            "mrr": [
                1 / known_first if known_first else 0.0,
                1 / possible_first if possible_first else 0.0,
            ],
            "hit@1": [float(values[0] == 1), float(values[0] != 0)],
        },
    )


def evaluate_intervals(
    rankings: dict, labels: list[dict], modes: dict[str, str]
) -> dict:
    """Evaluate intervals.

    Args:
        rankings: Candidate rankings keyed by query and retrieval method.
        labels: Judgments used for the stated development or evaluation protocol.
        modes: Modes used by this operation.

    Returns:
        Mapping with fields: scopes, query_metrics, query_capacity, ranges_are,
        review_origin, human_labels.

    Raises:
        ValueError: Unique AI judgments with honest provenance required; Complete
            query-gallery labels required; Methods differ between queries.
    """
    lookup = {}
    for row in labels:
        key = row["query_id"], row["candidate_id"]
        if (
            key in lookup
            or row.get("review_origin") != "model"
            or row.get("human_label") is not False
        ):
            raise ValueError("Unique AI judgments with honest provenance required")
        lookup[key] = row["grade"]
    expected = {
        (q, c) for q, methods in rankings.items() for c in next(iter(methods.values()))
    }
    if set(lookup) != expected or set(modes) != set(rankings):
        raise ValueError("Complete query-gallery labels required")
    rows, capacities = [], []
    names = set(next(iter(rankings.values())))
    for qid, methods in rankings.items():
        if set(methods) != names:
            raise ValueError("Methods differ between queries")
        grades = {c: g for (q, c), g in lookup.items() if q == qid}
        for name, order in methods.items():
            value = query_bounds(order, grades)
            rows.append(dict(query_id=qid, mode=modes[qid], method=name, **value))
        p, u = value["known_positive"], value["unknown"]
        capacities.append(
            dict(
                query_id=qid,
                known_positive=p,
                unknown=u,
                precision_ceiling_range=[min(10, p) / 10, min(10, p + u) / 10],
                recall_status=(
                    "positive"
                    if p
                    else "possibly_zero_positive" if u else "zero_positive"
                ),
            )
        )
    scopes: dict = {}
    for scope in ["all", *sorted(set(modes.values()))]:
        scopes[scope] = {}
        for name in sorted(names):
            current = [
                r
                for r in rows
                if r["method"] == name and (scope == "all" or r["mode"] == scope)
            ]
            result: dict = {}
            for metric in METRICS:
                if metric == "recall@10":
                    mandatory = [
                        tuple(r["ranges"][metric])
                        for r in current
                        if r["known_positive"]
                    ]
                    optional = [
                        tuple(r["ranges"][metric])
                        for r in current
                        if not r["known_positive"] and r["unknown"]
                    ]
                    result[metric] = list(optional_mean_bounds(mandatory, optional))
                else:
                    result[metric] = [
                        sum(r["ranges"][metric][i] for r in current) / len(current)
                        for i in [0, 1]
                    ]
            scopes[scope][name] = dict(
                ranges=result,
                queries=len(current),
                precision_60_lower_bound_met=result["precision@10"][0] >= 0.6 - 1e-12,
                recall_85_lower_bound_met=result["recall@10"][0] is not None
                and result["recall@10"][0] >= 0.85 - 1e-12,
                recall_denominator_min=sum(bool(r["known_positive"]) for r in current),
                recall_denominator_max=sum(
                    bool(r["known_positive"] or r["unknown"]) for r in current
                ),
            )
    return dict(
        scopes=scopes,
        query_metrics=rows,
        query_capacity=capacities,
        ranges_are="exact label-completion bounds, not confidence intervals",
        review_origin="model",
        human_labels=0,
    )
