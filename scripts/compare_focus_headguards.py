"""A separately frozen adaptive development follow-up; original results stay intact."""

from __future__ import annotations

import argparse
import json
from typing import TypedDict

import run as project_bootstrap  # noqa: F401
from advance_focus_coverage import ROOT, load
from multimodal_ecommerce.ai_review_metrics import metric_ranges
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.head_guard import protected_prefix
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from report_focus_coverage import paired_differences

DESTINATION = ROOT / "head_guard"


class _PolicySpec(TypedDict):
    """Exact fields of the POLICY configuration and provenance record."""

    role: str
    anchor: str
    candidate_methods: list[str]
    keep_prefix: list[int]
    uses_query_ID_rules: bool
    uses_labels_in_runtime: bool
    parameter_search: bool
    changes_candidate_set: bool
    new_model_inferences: int
    formal_acceptance_allowed: bool


POLICY: _PolicySpec = dict(
    role="adaptive_AI_development_followup_after_observed_head_regression",
    anchor="cross_evidence_pool60",
    candidate_methods=["described_cross_pool20", "described_cross_pool60"],
    keep_prefix=[1, 2],
    uses_query_ID_rules=False,
    uses_labels_in_runtime=False,
    parameter_search=False,
    changes_candidate_set=False,
    new_model_inferences=0,
    formal_acceptance_allowed=False,
)


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        FileExistsError: Preserve follow-up protocol.

    Notes:
        Writes derived artifacts through DESTINATION.mkdir, write_json.
    """
    if DESTINATION.exists():
        raise FileExistsError("Preserve follow-up protocol")
    DESTINATION.mkdir()
    files = [
        "scripts/compare_focus_headguards.py",
        "src/multimodal_ecommerce/head_guard.py",
        "artifacts/module2_focus_coverage/rankings.json",
        "artifacts/module2_focus_coverage/judgments.ai.jsonl",
        "artifacts/module2_focus_coverage/internal_metrics.ai.development.json",
        "artifacts/module2_focus_coverage/internal_paired_differences.json",
    ]
    write_json(
        DESTINATION / "protocol.json",
        dict(
            policy=POLICY,
            files={p: sha256_file(PROJECT_ROOT / p) for p in files},
            reason="Completed fixed comparison shows described-image MRR and Hit@1 regressions. Freeze prefix1/2 follow-up before evaluating it; do not change judgments or original methods.",
            independent_validation=False,
            original_comparison_preserved=True,
        ),
    )
    print(
        json.dumps(
            dict(
                frozen=True,
                new_postprocessing_methods=4,
                original_methods_unchanged=True,
            )
        )
    )


def run() -> None:
    """Apply the run workflow step.

    Raises:
        FileExistsError: Preserve follow-up results.

    Notes:
        Writes derived artifacts through write_json.
    """
    protocol = load(DESTINATION / "protocol.json")
    verify_files(PROJECT_ROOT, protocol["files"])
    assert protocol["policy"] == POLICY
    if (DESTINATION / "receipt.json").exists():
        raise FileExistsError("Preserve follow-up results")
    original = load(ROOT / "rankings.json")
    rankings = {
        qid: {name: list(order) for name, order in methods.items()}
        for qid, methods in original.items()
    }
    for qid, methods in rankings.items():
        for candidate in POLICY["candidate_methods"]:
            for keep in POLICY["keep_prefix"]:
                methods[f"protected_top{keep}_{candidate}"] = protected_prefix(
                    original[qid][POLICY["anchor"]], original[qid][candidate], keep
                )
    labels = read_rows(ROOT / "judgments.ai.jsonl")
    modes = {
        q["id"]: "text" if q["text"] else "image"
        for q in load(ROOT / "inputs.public.json")["queries"]
    }
    metrics = metric_ranges(rankings, labels, modes)
    paired = paired_differences(rankings, labels, modes)
    write_json(DESTINATION / "rankings.json", rankings)
    write_json(DESTINATION / "internal_metrics.ai.development.json", metrics)
    write_json(DESTINATION / "internal_paired_differences.json", paired)
    assessments = {}
    for name in metrics["scopes"]["all"]:
        if not name.startswith("protected_"):
            continue
        assessments[name] = dict(
            precision_60_AI_lower_bound_all_modes=all(
                metrics["scopes"][scope][name]["precision_60_lower_bound_met"]
                for scope in ["all", "text", "image"]
            ),
            paired_nonregression_all_metrics_vs_both_original_baselines=all(
                item["nonregression_proven_for_shared_unknown_labels"]
                for scope in ["all", "text", "image"]
                for base in ["global", "region_title"]
                for item in paired["scopes"][scope][name][base].values()
            ),
            paired_nonregression_vs_cross_anchor=all(
                item["nonregression_proven_for_shared_unknown_labels"]
                for scope in ["all", "text", "image"]
                for item in paired["scopes"][scope][name][POLICY["anchor"]].values()
            ),
            formal_metrics=None,
            default_promotion_allowed=False,
        )
    write_json(DESTINATION / "internal_decisions.json", assessments)
    write_json(
        DESTINATION / "receipt.json",
        dict(
            original_methods=16,
            new_methods=4,
            new_full_rankings=72,
            total_methods=20,
            total_full_rankings=360,
            AI_labels_only=True,
            adaptive_followup_not_independent_validation=True,
            model_inferences=0,
            formal_metrics=None,
            goal_met=False,
            acceptance_predictions=0,
            files={
                p.name: sha256_file(p) for p in DESTINATION.iterdir() if p.is_file()
            },
        ),
    )
    print(
        json.dumps(
            dict(
                new_methods=4, new_full_rankings=72, formal_metrics=None, goal_met=False
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    globals()[parser.parse_args().action]()
