"""Human-only Top10 precision, full-gallery recall bounds, and capacity evidence."""

from __future__ import annotations

from multimodal_ecommerce.ai_review_metrics import optional_mean_bounds

K = 10
DEPTHS = (20, 50, 100)


def _human_grade(row: dict) -> int | None:
    grade = row.get("grade")
    if grade is not None and (type(grade) is not int or grade not in (0, 1)):
        raise ValueError("Binary integer or unresolved relevance required")
    if type(row.get("human", False)) is not bool:
        raise ValueError("Explicit Boolean human provenance required")
    return grade if row.get("human") else None


def _recall_bounds(positive: int, unknown: int, hits: int, top_unknown: int) -> list:
    outside = unknown - top_unknown
    if not positive:
        if not unknown:
            return [None, None]
        return [0.0 if outside else 1.0, 1.0 if top_unknown else 0.0]
    return [
        hits / (positive + outside),
        (hits + top_unknown) / (positive + top_unknown),
    ]


def _query(order: list[str], query: dict) -> dict:
    if (
        len(order) < K
        or len(order) != query["expected_pairs"]
        or len(order) != len(set(order))
        or set(order).intersection(query.get("exclusions", []))
    ):
        raise ValueError("Unique complete eligible rankings of at least ten required")
    observed = query["judgments"]
    grades = [_human_grade(observed.get(cid, {})) for cid in order]
    positive, negative, unknown = grades.count(1), grades.count(0), grades.count(None)
    head = grades[:K]
    hits, missing = head.count(1), head.count(None)
    uncertain = sum(
        observed.get(cid, {}).get("human") is True
        and observed[cid].get("grade") is None
        for cid in order
    )
    mixed = [observed.get(cid, {}).get("grade") for cid in order]
    status = (
        "confirmed_positive"
        if positive
        else "possibly_zero_positive" if unknown else "zero_positive_undefined"
    )
    return dict(
        mode=query["mode"],
        sources=query.get("sources", []),
        eligible_pairs=len(order),
        human_positive=positive,
        human_negative=negative,
        human_unresolved=unknown,
        human_uncertain=uncertain,
        not_human_reviewed=unknown - uncertain,
        mixed_binary=mixed.count(1) + mixed.count(0),
        mixed_unknown=mixed.count(None),
        top10_human_hits=hits,
        top10_human_negative=head.count(0),
        top10_human_judged=K - missing,
        top10_unresolved=missing,
        precision_at_10=dict(
            value=hits / K if not missing else None,
            bounds=[hits / K, (hits + missing) / K],
        ),
        recall_at_10=dict(
            value=hits / positive if not unknown and positive else None,
            bounds=_recall_bounds(positive, unknown, hits, missing),
            status=status,
        ),
        ideal_recall_at_10_upper=(
            min(K, positive) / positive if positive else 1.0 if unknown else None
        ),
        documented_positive_coverage={
            str(depth): (
                sum(g == 1 for g in grades[:depth]) / positive if positive else None
            )
            for depth in DEPTHS
        },
    )


def evaluate_human_retrieval(rankings: dict, queries: dict, target: dict) -> dict:
    """Separate human metric evidence from machine labels and impossible targets.

    Args:
        rankings: Complete eligible candidate order for every development query.
        queries: Query modes, exclusions, relation observations and human provenance.
        target: Existing fixed-Top10 precision and recall target contract.

    Returns:
        Per-query and per-scope precision, recall, capacity and coverage evidence.
    """
    if set(rankings) != set(queries) or target["k"] != K:
        raise ValueError("All original queries and the fixed Top10 contract required")
    if any(q["mode"] not in ("image", "text") for q in queries.values()):
        raise ValueError("Only the original text/image scopes are supported")
    rows = {key: _query(order, queries[key]) for key, order in rankings.items()}
    scopes = {}
    for scope in ("all", "image", "text"):
        selected = [row for row in rows.values() if scope in ("all", row["mode"])]
        mandatory = [
            tuple(row["recall_at_10"]["bounds"])
            for row in selected
            if row["human_positive"]
        ]
        optional = [
            tuple(row["recall_at_10"]["bounds"])
            for row in selected
            if not row["human_positive"] and row["human_unresolved"]
        ]
        recall = list(optional_mean_bounds(mandatory, optional))
        capacity = optional_mean_bounds(
            [
                (row["ideal_recall_at_10_upper"],) * 2
                for row in selected
                if row["human_positive"]
            ],
            [
                (1.0, 1.0)
                for row in selected
                if not row["human_positive"] and row["human_unresolved"]
            ],
        )[1]
        size = K * len(selected)
        hits = sum(row["top10_human_hits"] for row in selected)
        missing = sum(row["top10_unresolved"] for row in selected)
        precision = [hits / size, (hits + missing) / size] if size else [None, None]
        exact_p = precision[0] if size and not missing else None
        complete = bool(selected) and all(
            not row["human_unresolved"] for row in selected
        )
        exact_r = recall[0] if complete else None
        incompatible = (
            capacity is not None and capacity < target["recall_at_10_preferred"] - 1e-12
        )
        scopes[scope] = dict(
            queries=len(selected),
            precision_at_10=dict(
                value=exact_p,
                bounds=precision,
                judged=size - missing,
                expected=size,
                missing=missing,
                status=(
                    "no_queries"
                    if not size
                    else (
                        "pending_human_labels"
                        if exact_p is None
                        else (
                            "met"
                            if exact_p + 1e-12 >= target["precision_at_10_required"]
                            else "not_met"
                        )
                    )
                ),
            ),
            recall_at_10=dict(
                value=exact_r,
                bounds=recall,
                complete_relevance_sets=complete,
                queries_with_human_positives=len(mandatory),
                possibly_zero_positive_queries=len(optional),
                zero_positive_undefined_queries=sum(
                    row["recall_at_10"]["status"] == "zero_positive_undefined"
                    for row in selected
                ),
                status=(
                    "incompatible_with_confirmed_human_labels_and_top10"
                    if incompatible
                    else (
                        "pending_complete_relevance_sets"
                        if not complete
                        else (
                            "undefined_no_relevant_queries"
                            if exact_r is None
                            else (
                                "met"
                                if exact_r + 1e-12 >= target["recall_at_10_preferred"]
                                else "not_met"
                            )
                        )
                    )
                ),
            ),
            recall_capacity=dict(
                optimistic_ideal_macro_upper=capacity,
                required=target["recall_at_10_preferred"],
                incompatible=incompatible,
                positive_query_count=len(mandatory),
                basis="confirmed human positives; unresolved completions remain possible",
            ),
            coverage={
                key: sum(row[key] for row in selected)
                for key in (
                    "eligible_pairs",
                    "human_positive",
                    "human_negative",
                    "human_unresolved",
                    "human_uncertain",
                    "not_human_reviewed",
                    "mixed_binary",
                    "mixed_unknown",
                )
            },
            documented_positive_coverage={
                str(depth): (
                    sum(
                        row["documented_positive_coverage"][str(depth)]
                        for row in selected
                        if row["human_positive"]
                    )
                    / len(mandatory)
                    if mandatory
                    else None
                )
                for depth in DEPTHS
            },
        )
    return dict(
        label_basis="binary human judgments only; machine-only and uncertain relations are unresolved",
        ranges_are="label-completion bounds, not confidence intervals",
        recall_aggregation="macro average; zero-relevant completions undefined, cohort bounds explicit",
        diagnostic_depths_are="coverage of human-confirmed positives only, not complete-gallery recall or acceptance",
        scopes=scopes,
        query_metrics=rows,
        formal_acceptance=False,
    )


def retrieval_error_queue(
    rankings: dict, manifest: dict, annotations: dict
) -> list[dict]:
    """List every query's confirmed Top10 errors and known pool coverage gaps.

    Args:
        rankings: Full fixed prediction order per query.
        manifest: Shared catalog and effective human relevance observations.
        annotations: Verified human reason/source records keyed by query/product pair.

    Returns:
        All queries sorted by human-confirmed Top10 errors, with unresolved counts.
    """
    catalog = {row["id"]: row for row in manifest["catalog"]}
    output = []
    for key, query in manifest["query_manifest"].items():
        order = rankings[key]
        labels = query["judgments"]
        errors: list[dict] = []
        inside: list[dict] = []
        outside: list[dict] = []
        for rank, cid in enumerate(order, 1):
            grade = _human_grade(labels.get(cid, {}))
            row = dict(id=cid, title=catalog[cid]["title"], rank=rank)
            if grade == 0 and rank <= K:
                notes = annotations.get((key, cid), [])
                reasons = "\n".join(n["reason"] for n in notes if n["grade"] == 0)
                clean = reasons.replace("用途与明确属性匹配", "")
                kinds = []
                if any(
                    term in clean
                    for term in ("主体", "品类", "配件", "保护套", "保护壳")
                ):
                    kinds.append("subject_or_accessory")
                if any(
                    term in clean
                    for term in (
                        "属性",
                        "型号",
                        "颜色",
                        "尺寸",
                        "不兼容",
                        "长袖",
                        "短袖",
                        "层数",
                    )
                ):
                    kinds.append("attribute_conflict")
                errors.append(
                    dict(
                        **row,
                        error_types=kinds or ["unclassified"],
                        attribution_basis="human reason keywords only; not machine-inferred truth",
                        annotations=[n for n in notes if n["grade"] == 0],
                    )
                )
            elif grade == 1 and rank > K:
                (inside if rank <= 20 else outside).append(row)
        output.append(
            dict(
                query_id=key,
                mode=query["mode"],
                sources=query.get("sources", []),
                confirmed_top10_error_count=len(errors),
                errors=errors,
                unresolved_top10=sum(
                    _human_grade(labels.get(cid, {})) is None for cid in order[:K]
                ),
                known_relevant_in_top20_not_top10=inside,
                known_relevant_outside_top20=outside,
                diagnostic_actions=(
                    ["known_relevant_available_for_reranking"] if inside else []
                )
                + (["known_relevant_outside_current_pool"] if outside else [])
                or ["insufficient_confirmed_relevant_examples"],
            )
        )
    return sorted(
        output, key=lambda row: (-row["confirmed_top10_error_count"], row["query_id"])
    )
