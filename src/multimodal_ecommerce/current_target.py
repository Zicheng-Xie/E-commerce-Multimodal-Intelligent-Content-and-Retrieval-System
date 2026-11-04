"""Apply the user's current hard Precision / preferred Recall development policy."""

from __future__ import annotations

import math


def assess_current_target(
    summary: dict | None, by_mode: dict | None, contract: dict
) -> dict:
    """Keep preferred Recall distinct from required gates and formal acceptance.

    Args:
        summary: Summary used by this operation.
        by_mode: By mode used by this operation.
        contract: Contract used by this operation.

    Returns:
        Mapping with fields: state, hard_gate_passed, recall_preferred_reached,
        development_targets_all_reached, candidates.

    Raises:
        ValueError: Both complete overall and mode metrics are required; Both unchanged
            baseline methods are required; Need overall, text and image reports; Method
            coverage differs between scopes; Metrics must be finite fractions or
            undefined.
    """
    output = dict(
        contract_version=contract["version"],
        precision_at_10_required=contract["precision_at_10_required"],
        recall_at_10_preferred=contract["recall_at_10_preferred"],
        recall_preferred_is_mandatory=False,
        formal_acceptance=False,
        formal_metrics=None,
        goal_met=False,
    )
    if summary is None and by_mode is None:
        return dict(
            output,
            state="awaiting_complete_human_development_labels",
            hard_gate_passed=None,
            recall_preferred_reached=None,
            development_targets_all_reached=None,
        )
    if not summary or not by_mode:
        raise ValueError("Both complete overall and mode metrics are required")
    methods = set(summary)
    if not set(contract["nonregression_baselines"]) <= methods:
        raise ValueError("Both unchanged baseline methods are required")
    scopes = {"all": summary, **by_mode}
    if set(scopes) != set(contract["report_scopes"]):
        raise ValueError("Need overall, text and image reports")
    metrics = ["precision@10", *contract["nonregression_metrics"]]
    for table in scopes.values():
        if set(table) != methods:
            raise ValueError("Method coverage differs between scopes")
        for values in table.values():
            for key in metrics:
                value = values[key]
                if value is not None and (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                ):
                    raise ValueError("Metrics must be finite fractions or undefined")
    candidates = {}
    for method in sorted(methods):
        required_failures = {}
        preferred = {}
        for scope, table in scopes.items():
            values = table[method]
            precision = values["precision@10"]
            recall = values["recall@10"]
            failed = []
            if (
                precision is None
                or precision < contract["precision_at_10_required"] - 1e-12
            ):
                failed.append("precision@10_below_required_or_undefined")
            for baseline in contract["nonregression_baselines"]:
                for metric in contract["nonregression_metrics"]:
                    reference = table[baseline][metric]
                    observed = values[metric]
                    if reference is not None and (
                        observed is None or observed < reference - 1e-12
                    ):
                        failed.append(f"{metric}_regressed_vs_{baseline}")
            required_failures[scope] = failed
            preferred[scope] = dict(
                recall_at_10=recall,
                reached=(
                    recall is not None
                    and recall >= contract["recall_at_10_preferred"] - 1e-12
                ),
                shortfall=(
                    None
                    if recall is None
                    else max(0.0, contract["recall_at_10_preferred"] - recall)
                ),
            )
        hard_passed = not any(required_failures.values())
        preferred_reached = all(r["reached"] for r in preferred.values())
        candidates[method] = dict(
            hard_gate_passed=hard_passed,
            required_failures=required_failures,
            recall_preferred_reached=preferred_reached,
            recall_preferred_by_scope=preferred,
            development_targets_all_reached=hard_passed and preferred_reached,
        )
    return dict(
        output,
        state="human_development_observations_only",
        candidates=candidates,
    )
