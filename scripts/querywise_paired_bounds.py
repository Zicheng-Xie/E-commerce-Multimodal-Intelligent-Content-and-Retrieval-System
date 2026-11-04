"""Query-local shared-label bounds, preserving exact pairs above other queries'
budgets.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.ai_review_metrics import optional_mean_bounds
from multimodal_ecommerce.business_evaluation import METRICS
from multimodal_ecommerce.interval_retrieval_metrics import query_bounds

MAX_EXACT_UNKNOWN = 16
CHUNK_SIZE = 2048
EXACT = "exact_shared_label_completions"
IDENTICAL = "exact_identical_order"
CONSERVATIVE = "conservative_per_query_interval_subtraction"
MIXED = "mixed_exact_and_per_query_conservative"
UNDEFINED = "undefined_zero_positive"
NON_RECALL = np.array([i for i, name in enumerate(METRICS) if name != "recall@10"])
RECALL_INDEX = METRICS.index("recall@10")


def _validate(rankings, labels, modes, comparisons):
    if not rankings or set(modes) != set(rankings):
        raise ValueError("Nonempty rankings and exact query/mode coverage required")
    if not set(modes.values()) <= {"text", "image"}:
        raise ValueError("Modes must be text or image")
    if not comparisons or len(set(comparisons)) != len(comparisons):
        raise ValueError("Distinct requested method pairs required")
    method_names = set(next(iter(rankings.values())))
    if not method_names:
        raise ValueError("At least one method required")
    if any(
        not isinstance(pair, tuple)
        or len(pair) != 2
        or any(name not in method_names for name in pair)
        for pair in comparisons
    ):
        raise ValueError("Each comparison must name an existing candidate and baseline")
    expected: set = set()
    for qid, methods in rankings.items():
        if set(methods) != method_names:
            raise ValueError("Methods differ between queries")
        first = next(iter(methods.values()))
        if not first or any(not isinstance(cid, str) for cid in first):
            raise ValueError("Nonempty full gallery with string IDs required")
        gallery = set(first)
        for order in methods.values():
            if len(order) != len(gallery) or set(order) != gallery:
                raise ValueError(
                    "Each method must rank every gallery item exactly once"
                )
        expected.update((qid, cid) for cid in gallery)
    lookup = {}
    for row in labels:
        key = row.get("query_id"), row.get("candidate_id")
        if key in lookup or key not in expected or "grade" not in row:
            raise ValueError("Unique, complete and in-gallery judgments required")
        if row.get("review_origin") != "model" or row.get("human_label") is not False:
            raise ValueError("AI review provenance required")
        value = row["grade"]
        if value is not None and (type(value) is not int or value not in (0, 1)):
            raise ValueError("Grade must be integer zero/one or unresolved null")
        lookup[key] = value
    if set(lookup) != expected:
        raise ValueError("Full query-gallery judgments required")
    return lookup


def _batch_metrics(hits: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """The original full-list AP/MRR and fixed-K binary retrieval definitions."""
    scores = np.zeros((len(hits), len(METRICS)), dtype=np.float64)
    top = hits[:, :10]
    top_counts = top.sum(axis=1)
    defined = totals > 0
    scores[:, 0] = top_counts / 10
    scores[:, RECALL_INDEX] = np.divide(
        top_counts,
        totals,
        out=np.full(len(hits), np.nan),
        where=defined,
    )
    positions = np.arange(1, hits.shape[1] + 1, dtype=np.float64)
    cumulative = hits.cumsum(axis=1, dtype=np.float64)
    scores[:, 2] = np.divide(
        (cumulative / positions * hits).sum(axis=1),
        totals,
        out=np.zeros(len(hits)),
        where=defined,
    )
    discounts = 1 / np.log2(np.arange(2, 12))
    ideal = np.concatenate(([0.0], discounts.cumsum()))[np.minimum(totals, 10)]
    scores[:, 3] = np.divide(
        (top * discounts[: top.shape[1]]).sum(axis=1),
        ideal,
        out=np.zeros(len(hits)),
        where=defined,
    )
    scores[:, 4] = np.where(defined, 1 / (hits.argmax(axis=1) + 1), 0)
    scores[:, 5] = hits[:, 0]
    return scores


def _exact_pairs(methods, grades, comparisons):
    ids = list(grades)
    positions = {cid: i for i, cid in enumerate(ids)}
    known = np.array([grades[cid] == 1 for cid in ids], dtype=bool)
    unknown = np.array([i for i, cid in enumerate(ids) if grades[cid] is None])
    order_keys = {name: tuple(methods[name]) for pair in comparisons for name in pair}
    indexes = {
        order: np.array([positions[cid] for cid in order])
        for order in set(order_keys.values())
    }
    lower = {pair: np.full(len(METRICS), np.inf) for pair in comparisons}
    upper = {pair: np.full(len(METRICS), -np.inf) for pair in comparisons}
    count = 1 << len(unknown)
    blocks = 0
    for start in range(0, count, CHUNK_SIZE):
        end = min(count, start + CHUNK_SIZE)
        assignments = (
            np.arange(start, end, dtype=np.uint32)[:, None]
            >> np.arange(len(unknown), dtype=np.uint32)
        ) & 1
        hits = np.broadcast_to(known, (end - start, len(ids))).copy()
        if len(unknown):
            hits[:, unknown] = assignments
        totals = hits.sum(axis=1)
        defined = totals > 0
        cache = {
            order: _batch_metrics(hits[:, index], totals)
            for order, index in indexes.items()
        }
        for pair in comparisons:
            candidate, baseline = pair
            delta = cache[order_keys[candidate]] - cache[order_keys[baseline]]
            lower[pair][NON_RECALL] = np.minimum(
                lower[pair][NON_RECALL], delta[:, NON_RECALL].min(axis=0)
            )
            upper[pair][NON_RECALL] = np.maximum(
                upper[pair][NON_RECALL], delta[:, NON_RECALL].max(axis=0)
            )
            if defined.any():
                values = delta[defined, RECALL_INDEX]
                lower[pair][RECALL_INDEX] = min(
                    lower[pair][RECALL_INDEX], float(values.min())
                )
                upper[pair][RECALL_INDEX] = max(
                    upper[pair][RECALL_INDEX], float(values.max())
                )
        blocks += 1
    result = {
        pair: {
            metric: (
                [float(lower[pair][i]), float(upper[pair][i])]
                if np.isfinite(lower[pair][i])
                else [None, None]
            )
            for i, metric in enumerate(METRICS)
        }
        for pair in comparisons
    }
    return result, dict(completions=count, blocks=blocks, unique_orders=len(indexes))


def _field(bounds, bound_type, conservative_queries=()):
    return dict(
        delta_range=list(bounds),
        nonregression_proven_for_shared_unknown_labels=bounds[0] is not None
        and bounds[0] >= -1e-12,
        bound_type=bound_type,
        conservative_query_ids=list(conservative_queries),
    )


def paired_querywise(
    rankings: dict,
    labels: list[dict],
    modes: dict,
    comparisons: list[tuple[str, str]],
) -> dict:
    """Compare only requested pairs, using a separate enumeration budget per query.

    Args:
        rankings: Candidate rankings keyed by query and retrieval method.
        labels: Judgments used for the stated development or evaluation protocol.
        modes: Modes used by this operation.
        comparisons: Comparisons used by this operation.

    Returns:
        Mapping with fields: scopes, query_pair_bounds, enumeration_work, fixed_k,
        exact_unknown_budget_per_query, enumeration_chunk_size,
        same_unknown_completion_for_candidate_and_baseline, conservative_fallback,
        recall_aggregation, ranges_are, review_origin, human_labels, formal_metrics,
        formal_acceptance.
    """
    lookup = _validate(rankings, labels, modes, comparisons)
    by_query: defaultdict = defaultdict(dict)
    for (qid, cid), grade in lookup.items():
        by_query[qid][cid] = grade
    rows, work = [], []
    for qid, methods in rankings.items():
        grades = by_query[qid]
        positive = sum(value == 1 for value in grades.values())
        unknown = sum(value is None for value in grades.values())
        different = [
            pair for pair in comparisons if methods[pair[0]] != methods[pair[1]]
        ]
        exact = {}
        if different and unknown <= MAX_EXACT_UNKNOWN:
            exact, evidence = _exact_pairs(methods, grades, different)
            work.append(dict(query_id=qid, **evidence))
        individual = {}
        for candidate, baseline in comparisons:
            pair = candidate, baseline
            if methods[candidate] == methods[baseline]:
                bounds: dict[str, list[float | None]] = dict.fromkeys(
                    METRICS, [0.0, 0.0]
                )
                bound_type = IDENTICAL
                if not positive and not unknown:
                    bounds["recall@10"] = [None, None]
            elif unknown <= MAX_EXACT_UNKNOWN:
                bounds, bound_type = exact[pair], EXACT
            else:
                for name in pair:
                    key = tuple(methods[name])
                    if key not in individual:
                        individual[key] = query_bounds(methods[name], grades)["ranges"]
                a = individual[tuple(methods[candidate])]
                b = individual[tuple(methods[baseline])]
                bounds = {
                    metric: [a[metric][0] - b[metric][1], a[metric][1] - b[metric][0]]
                    for metric in METRICS
                }
                bound_type = CONSERVATIVE
            rows.append(
                dict(
                    query_id=qid,
                    mode=modes[qid],
                    candidate=candidate,
                    baseline=baseline,
                    known_positive=positive,
                    unknown=unknown,
                    recall_presence=(
                        "mandatory"
                        if positive
                        else "optional" if unknown else "undefined"
                    ),
                    metrics={
                        metric: _field(
                            bounds[metric],
                            bound_type if bounds[metric][0] is not None else UNDEFINED,
                            [qid] if bound_type == CONSERVATIVE else [],
                        )
                        for metric in METRICS
                    },
                )
            )
    scopes: dict = {}
    for scope in ["all", *sorted(set(modes.values()))]:
        scopes[scope] = {}
        for candidate, baseline in comparisons:
            selected = [
                row
                for row in rows
                if row["candidate"] == candidate
                and row["baseline"] == baseline
                and (scope == "all" or row["mode"] == scope)
            ]
            metrics = {}
            for metric in METRICS:
                involved = selected
                if metric == "recall@10":
                    involved = [
                        r for r in selected if r["recall_presence"] != "undefined"
                    ]
                    mandatory = [
                        tuple(r["metrics"][metric]["delta_range"])
                        for r in involved
                        if r["recall_presence"] == "mandatory"
                    ]
                    optional = [
                        tuple(r["metrics"][metric]["delta_range"])
                        for r in involved
                        if r["recall_presence"] == "optional"
                    ]
                    scope_bounds = optional_mean_bounds(mandatory, optional)
                else:
                    scope_bounds = tuple(
                        sum(r["metrics"][metric]["delta_range"][i] for r in selected)
                        / len(selected)
                        for i in [0, 1]
                    )
                kinds = {r["metrics"][metric]["bound_type"] for r in involved}
                conservative = [
                    r["query_id"]
                    for r in involved
                    if r["metrics"][metric]["bound_type"] == CONSERVATIVE
                ]
                bound_type = (
                    UNDEFINED
                    if scope_bounds[0] is None
                    else (
                        MIXED
                        if CONSERVATIVE in kinds and len(kinds) > 1
                        else (
                            CONSERVATIVE
                            if conservative
                            else IDENTICAL if kinds == {IDENTICAL} else EXACT
                        )
                    )
                )
                metrics[metric] = _field(scope_bounds, bound_type, conservative)
            scopes[scope].setdefault(candidate, {})[baseline] = metrics
    return dict(
        scopes=scopes,
        query_pair_bounds=rows,
        enumeration_work=work,
        fixed_k=10,
        exact_unknown_budget_per_query=MAX_EXACT_UNKNOWN,
        enumeration_chunk_size=CHUNK_SIZE,
        same_unknown_completion_for_candidate_and_baseline=True,
        conservative_fallback="Only nonidentical pairs in queries with >16 unresolved labels; no unresolved label is filled in.",
        recall_aggregation="optional_mean_bounds over mandatory and possibly-positive query differences; zero-positive Recall stays undefined",
        ranges_are="shared-label-completion bounds, not confidence intervals or AI-error bounds",
        review_origin="model",
        human_labels=0,
        formal_metrics=None,
        formal_acceptance=False,
    )
