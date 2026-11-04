"""Complete-label business development metrics, with explicit denominators."""

from __future__ import annotations

import numpy as np

from multimodal_ecommerce.precision_target import precision_capacity
from multimodal_ecommerce.retrieval_strategies import extended_metrics

METRICS = ("precision@10", "recall@10", "ap", "ndcg@10", "mrr", "hit@1")


def score_business_development(
    rankings: dict, allowed: dict, labels: list[dict], modes: dict[str, str]
) -> dict:
    """Score a full candidate relation matrix; never turn unknown labels negative.

    Args:
        rankings: Candidate rankings keyed by query and retrieval method.
        allowed: Allowed used by this operation.
        labels: Judgments used for the stated development or evaluation protocol.
        modes: Modes used by this operation.

    Returns:
        Mapping with fields: quality_metrics, metrics_by_mode, query_metrics,
        development_gates, capacity, zero_positive_query_ids, recall_effective_queries,
        precision_effective_queries, denominator_policy, formal_acceptance,
        formal_metrics, goal_met, goal_status.

    Raises:
        ValueError: Query coverage mismatch; Need unique complete integer judgments;
            Full allowed gallery must be judged; Both global and region/title baselines
            are required; Inconsistent method or gallery definitions; Ranking must cover
            the unchanged complete gallery.
    """
    if not allowed or set(rankings) != set(allowed) or set(modes) != set(allowed):
        raise ValueError("Query coverage mismatch")
    expected = {(q, c) for q, ids in allowed.items() for c in ids}
    actual = {}
    for row in labels:
        pair = (row["query_id"], row["candidate_id"])
        if (
            pair in actual
            or pair not in expected
            or type(row["grade"]) is not int
            or row["grade"] not in (0, 1)
        ):
            raise ValueError("Need unique complete integer judgments")
        actual[pair] = row["grade"]
    if set(actual) != expected:
        raise ValueError("Full allowed gallery must be judged")
    methods = set(next(iter(rankings.values())))
    if not {"global", "region_title"} <= methods:
        raise ValueError("Both global and region/title baselines are required")
    relevant = {
        q: {c for c in ids if actual[(q, c)] == 1} for q, ids in allowed.items()
    }
    rows = []
    for query, current in rankings.items():
        if set(current) != methods or len(allowed[query]) != len(set(allowed[query])):
            raise ValueError("Inconsistent method or gallery definitions")
        for method, ranked in current.items():
            if len(ranked) != len(allowed[query]) or set(ranked) != set(allowed[query]):
                raise ValueError("Ranking must cover the unchanged complete gallery")
            values: dict[str, float | None]
            if relevant[query]:
                scores = extended_metrics(ranked, relevant[query])
                values = {k: scores[k] for k in METRICS}
            else:
                values = {k: 0.0 for k in METRICS}
                values["recall@10"] = None
            rows.append(
                {"query_id": query, "mode": modes[query], "method": method, **values}
            )

    def aggregate(selected: list[dict]) -> dict:
        """Return aggregate for this workflow.

        Args:
            selected: Selected records to aggregate.

        Returns:
            Result mapping produced by the workflow.
        """
        output: dict = {}
        for method in sorted(methods):
            current = [r for r in selected if r["method"] == method]
            output[method] = {}
            for key in METRICS:
                values = [r[key] for r in current if r[key] is not None]
                output[method][key] = float(np.mean(values)) if values else None
        return output

    summary = aggregate(rows)
    by_mode = {
        mode: aggregate([r for r in rows if r["mode"] == mode])
        for mode in sorted(set(modes.values()))
    }
    gates = {}
    for method in sorted(methods):
        failures = {}
        for scope, table in [("all", summary), *by_mode.items()]:
            observed = table[method]
            failed = []
            if (
                observed["precision@10"] is None
                or observed["precision@10"] < 0.6 - 1e-12
            ):
                failed.append("precision@10_below_60_percent")
            if observed["recall@10"] is None or observed["recall@10"] < 0.7 - 1e-12:
                failed.append("recall@10_below_70_percent_or_undefined")
            for baseline in ["global", "region_title"]:
                for key in METRICS[1:]:
                    reference = table[baseline][key]
                    if reference is not None and (
                        observed[key] is None or observed[key] < reference - 1e-12
                    ):
                        failed.append(f"{key}_regressed_vs_{baseline}")
            failures[scope] = failed
        gates[method] = {
            "passes_development_targets_and_nonregression": not any(failures.values()),
            "failures": failures,
            "formal_acceptance": False,
        }
    return {
        "quality_metrics": summary,
        "metrics_by_mode": by_mode,
        "query_metrics": rows,
        "development_gates": gates,
        "capacity": precision_capacity([len(relevant[q]) for q in allowed]),
        "zero_positive_query_ids": [q for q in allowed if not relevant[q]],
        "recall_effective_queries": sum(bool(v) for v in relevant.values()),
        "precision_effective_queries": len(allowed),
        "denominator_policy": "Precision/AP/NDCG/MRR/Hit1 include all queries; zero-positive Recall is undefined and its positive-query denominator is reported",
        "formal_acceptance": False,
        "formal_metrics": None,
        "goal_met": False,
        "goal_status": "Development thresholds are observations only; independent heldout acceptance remains required",
    }
