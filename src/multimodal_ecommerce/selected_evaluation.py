"""Independent human scoring of sealed rankings, including unresolved-label ranges."""

from __future__ import annotations

from multimodal_ecommerce.ai_review_metrics import optional_mean_bounds
from multimodal_ecommerce.business_evaluation import METRICS
from multimodal_ecommerce.human_retrieval_evaluation import evaluate_human_retrieval
from multimodal_ecommerce.interval_retrieval_metrics import query_bounds


def evaluate_sealed_predictions(
    predictions: dict, packet: dict, gold: dict, target: dict
) -> dict:
    """Score the full independent relation matrix without guessing uncertain answers.

    Args:
        predictions: Complete sealed candidate and baseline rankings.
        packet: Independent public queries and unchanged complete gallery.
        gold: Frozen human binary or uncertain judgments.
        target: Original Top10 goals and baseline nonregression requirements.

    Returns:
        Six metric ranges, honest point values, capacity and target verdicts.
    """
    ids = [r["id"] for r in packet["catalog"]]
    modes = {q["id"]: q["mode"] for q in packet["queries"]}
    if (
        len(ids) != len(set(ids))
        or len(modes) != len(packet["queries"])
        or set(predictions["queries"]) != set(modes)
    ):
        raise ValueError("Preserve all unique frozen queries and products")
    expected = {(q, c) for q in modes for c in ids}
    labels = {}
    for row in gold["answers"]:
        pair = row["query_id"], row["candidate_id"]
        if pair in labels or pair not in expected or row.get("human") is not True:
            raise ValueError("Unique independent human gold required")
        labels[pair] = row["grade"]
    if set(labels) != expected:
        raise ValueError("Every frozen query/product relation must have an answer")
    orders, observations, rows = {}, {}, []
    for qid, mode in modes.items():
        current = predictions["queries"][qid]
        methods = dict(selected=current["ranking"], **current["baselines"])
        if set(methods) != {"selected", "global", "region_title"}:
            raise ValueError(
                "The selected policy and both fixed baselines are required"
            )
        grades = {cid: labels[qid, cid] for cid in ids}
        for method, order in methods.items():
            rows.append(
                dict(
                    query_id=qid,
                    mode=mode,
                    method=method,
                    **query_bounds(order, grades),
                )
            )
        orders[qid] = methods["selected"]
        observations[qid] = dict(
            mode=mode,
            exclusions=[],
            expected_pairs=len(ids),
            judgments={c: dict(grade=g, human=True) for c, g in grades.items()},
        )
    human = evaluate_human_retrieval(orders, observations, target)
    intervals: dict = {}
    tables: dict = {}
    failures: dict = {}
    unresolved = sum(g is None for g in labels.values())
    for scope in ("all", "image", "text"):
        intervals[scope], tables[scope], failures[scope] = {}, {}, []
        for method in ("selected", "global", "region_title"):
            current_rows = [
                r
                for r in rows
                if r["method"] == method and (scope == "all" or r["mode"] == scope)
            ]
            if not current_rows:
                failures[scope] = ["missing_required_scope"]
                continue
            ranges = {}
            for metric in METRICS:
                if metric == "recall@10":
                    ranges[metric] = list(
                        optional_mean_bounds(
                            [
                                tuple(r["ranges"][metric])
                                for r in current_rows
                                if r["known_positive"]
                            ],
                            [
                                tuple(r["ranges"][metric])
                                for r in current_rows
                                if not r["known_positive"] and r["unknown"]
                            ],
                        )
                    )
                else:
                    ranges[metric] = [
                        sum(r["ranges"][metric][side] for r in current_rows)
                        / len(current_rows)
                        for side in (0, 1)
                    ]
            intervals[scope][method] = ranges
            tables[scope][method] = {
                m: lo if lo == hi else None for m, (lo, hi) in ranges.items()
            }
        if failures[scope]:
            continue
        table = intervals[scope]
        for metric, required in (
            ("precision@10", target["precision_at_10_required"]),
            ("recall@10", target["recall_at_10_preferred"]),
        ):
            lower = table["selected"][metric][0]
            if lower is None or lower + 1e-12 < required:
                failures[scope].append(metric + "_below_target_or_undefined")
        for baseline in target["nonregression_baselines"]:
            for metric in target["nonregression_metrics"]:
                lower, upper = table["selected"][metric][0], table[baseline][metric][1]
                if upper is not None and (lower is None or lower + 1e-12 < upper):
                    failures[scope].append(
                        metric + "_nonregression_not_proven_vs_" + baseline
                    )
        if unresolved:
            failures[scope].append("unresolved_human_gold_intervals_only")
    met = not any(failures.values())
    return dict(
        evaluation_completed=True,
        formal_acceptance=met,
        goal_met=met,
        target_failures=failures,
        formal_metrics={
            scope: table.get("selected") for scope, table in tables.items()
        },
        metric_intervals=intervals,
        ranges_are="exact label-completion bounds, not confidence intervals",
        baseline_metrics=tables,
        query_metrics=rows,
        human_evaluation=human,
        unresolved_answers=unresolved,
        denominator_policy="All frozen queries included; zero-positive Recall undefined, its cohort bounds reported separately",
        no_query_dropped=True,
        further_tuning_on_this_dataset_allowed=False,
    )
