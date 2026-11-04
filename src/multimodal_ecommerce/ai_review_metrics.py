"""Exact metric ranges over unresolved AI labels, including variable Recall cohorts."""

from __future__ import annotations

import itertools

from multimodal_ecommerce.business_evaluation import METRICS
from multimodal_ecommerce.retrieval_strategies import extended_metrics


def optional_mean_bounds(
    mandatory: list[tuple[float, float]], optional: list[tuple[float, float]]
) -> tuple[float | None, float | None]:
    """Zero-positive queries may be omitted, rather than forced into Recall averages.

    Args:
        mandatory: Mandatory used by this operation.
        optional: Optional used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    if not mandatory and not optional:
        return None, None
    bounds = []
    for side, reverse in [(0, False), (1, True)]:
        total = sum(row[side] for row in mandatory)
        count = len(mandatory)
        values = [total / count] if count else []
        for row in sorted(optional, key=lambda x: x[side], reverse=reverse):
            total += row[side]
            count += 1
            values.append(total / count)
        bounds.append((min if side == 0 else max)(values))
    return bounds[0], bounds[1]


def metric_ranges(rankings: dict, labels: list[dict], modes: dict) -> dict:
    """Enumerate each query's unresolved cells; never use scores to choose labels.

    Args:
        rankings: Candidate rankings keyed by query and retrieval method.
        labels: Judgments used for the stated development or evaluation protocol.
        modes: Modes used by this operation.

    Returns:
        Mapping with fields: role, ranges_are, scopes, capacity_by_scope,
        query_capacity, query_metrics, review_origin, human_labels, formal_metrics,
        formal_acceptance, goal_met.

    Raises:
        ValueError: Duplicate or foreign AI relation; AI provenance required; Binary
            integer or unknown required; Full relation/query coverage required; Exact
            enumeration budget exceeded; no approximate point score allowed; Method
            coverage differs; Full unique candidate ranking required.
    """
    expected = {
        (q, c) for q, methods in rankings.items() for c in next(iter(methods.values()))
    }
    judgments = {}
    for row in labels:
        key = row["query_id"], row["candidate_id"]
        grade = row["grade"]
        if key in judgments or key not in expected:
            raise ValueError("Duplicate or foreign AI relation")
        if row.get("review_origin") != "model" or row.get("human_label") is not False:
            raise ValueError("AI provenance required")
        if grade is not None and (type(grade) is not int or grade not in (0, 1)):
            raise ValueError("Binary integer or unknown required")
        judgments[key] = grade
    if set(judgments) != expected or set(modes) != set(rankings):
        raise ValueError("Full relation/query coverage required")
    methods = set(next(iter(rankings.values())))
    rows, capacity = [], []
    for qid, current in rankings.items():
        candidates = {c for q, c in expected if q == qid}
        positive = {c for c in candidates if judgments[qid, c] == 1}
        unknown = sorted(c for c in candidates if judgments[qid, c] is None)
        if len(unknown) > 16:
            raise ValueError(
                "Exact enumeration budget exceeded; no approximate point score allowed"
            )
        completions = [
            positive | {c for c, present in zip(unknown, bits) if present}
            for bits in itertools.product([False, True], repeat=len(unknown))
        ]
        capacity.append(
            dict(
                query_id=qid,
                known_positive=len(positive),
                unknown=len(unknown),
                precision_ceiling_lower=min(10, len(positive)) / 10,
                precision_ceiling_upper=min(10, len(positive) + len(unknown)) / 10,
                recall_status=(
                    "positive"
                    if positive
                    else "possibly_zero_positive" if unknown else "zero_positive"
                ),
            )
        )
        if set(current) != methods:
            raise ValueError("Method coverage differs")
        for name, order in current.items():
            if len(order) != len(candidates) or set(order) != candidates:
                raise ValueError("Full unique candidate ranking required")
            observed: list[dict] = []
            for relevant in completions:
                observed.append(
                    extended_metrics(order, relevant)
                    if relevant
                    else {**dict.fromkeys(METRICS, 0.0), "recall@10": None}
                )
            bounds = {}
            for metric in METRICS:
                finite_values = [
                    float(r[metric]) for r in observed if r[metric] is not None
                ]
                bounds[metric] = (
                    [min(finite_values), max(finite_values)]
                    if finite_values
                    else [None, None]
                )
            rows.append(
                dict(
                    query_id=qid,
                    mode=modes[qid],
                    method=name,
                    ranges=bounds,
                    known_positive=len(positive),
                    unknown=len(unknown),
                )
            )
    tables = {}
    for scope in ["all", *sorted(set(modes.values()))]:
        table: dict = {}
        for name in sorted(methods):
            selected = [
                r
                for r in rows
                if r["method"] == name and (scope == "all" or r["mode"] == scope)
            ]
            values: dict = {}
            for metric in METRICS:
                if metric == "recall@10":
                    mandatory = [
                        tuple(r["ranges"][metric])
                        for r in selected
                        if r["known_positive"]
                    ]
                    optional = [
                        tuple(r["ranges"][metric])
                        for r in selected
                        if not r["known_positive"] and r["unknown"]
                    ]
                    values[metric] = list(optional_mean_bounds(mandatory, optional))
                else:
                    values[metric] = [
                        sum(r["ranges"][metric][i] for r in selected) / len(selected)
                        for i in [0, 1]
                    ]
            table[name] = dict(
                ranges=values,
                precision_60_lower_bound_met=values["precision@10"][0] >= 0.6 - 1e-12,
                recall_85_lower_bound_met=values["recall@10"][0] is not None
                and values["recall@10"][0] >= 0.85 - 1e-12,
                queries=len(selected),
                recall_denominator_min=sum(bool(r["known_positive"]) for r in selected),
                recall_denominator_max=sum(
                    bool(r["known_positive"] or r["unknown"]) for r in selected
                ),
            )
        tables[scope] = table
    capacities = {}
    for scope in tables:
        selected = [
            r for r in capacity if scope == "all" or modes[r["query_id"]] == scope
        ]
        capacities[scope] = dict(
            queries=len(selected),
            precision_ceiling_range=[
                sum(r[f"precision_ceiling_{side}"] for r in selected) / len(selected)
                for side in ["lower", "upper"]
            ],
            zero_positive_query_ids=[
                r["query_id"] for r in selected if r["recall_status"] == "zero_positive"
            ],
            possibly_zero_positive_query_ids=[
                r["query_id"]
                for r in selected
                if r["recall_status"] == "possibly_zero_positive"
            ],
        )
    return dict(
        role="AI_assessed_development_only",
        ranges_are="Exact ranges over unresolved AI labels; not confidence intervals, not independent truth",
        scopes=tables,
        capacity_by_scope=capacities,
        query_capacity=capacity,
        query_metrics=rows,
        review_origin="model",
        human_labels=0,
        formal_metrics=None,
        formal_acceptance=False,
        goal_met=False,
    )
