"""Correct the report schema in a separate root; preserve frozen inputs/results."""

from __future__ import annotations

import argparse
from pathlib import Path

import balanced_gallery_experiment as experiment
import balanced_gallery_runner as runner
import balanced_pair_review as review
import balanced_supplement_features as features
import run  # noqa: F401
from learned_fusion_experiment import load, once, rel
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_evaluation import METRICS
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file

ROOT = PROJECT_ROOT / "artifacts/module2_balanced_gallery_finalization"


def report_cells(values: dict) -> list:
    """Use the evaluator's real `ranges` schema and explicit metric order.

    Args:
        values: Values used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    ranges = values["ranges"]
    assert set(ranges) == set(METRICS)
    return [
        " / ".join("undefined" if v is None else f"{v * 100:.2f}" for v in ranges[m])
        for m in METRICS
    ]


def decision_values(metrics_by_cohort: dict, paired_by_cohort: dict) -> dict:
    """Return decision values for this workflow.

    Args:
        metrics_by_cohort: Metrics by cohort used by this operation.
        paired_by_cohort: Paired by cohort used by this operation.

    Returns:
        Mapping with fields: primary, checks, development_precision_target_met,
        development_recall_preference_met, development_all_metric_nonregression,
        formal_metrics, goal_met, default_strategy_changed,
        independent_AI_blind_predictions.
    """
    checks: dict = {}
    for cohort in experiment.COHORTS:
        checks[cohort] = {}
        for scope, methods in metrics_by_cohort[cohort]["scopes"].items():
            checks[cohort][scope] = {}
            for name in experiment.NEW:
                checks[cohort][scope][name] = dict(
                    precision_60=methods[name]["precision_60_lower_bound_met"],
                    recall_85=methods[name]["recall_85_lower_bound_met"],
                    nonregression=all(
                        value["nonregression_proven_for_shared_unknown_labels"]
                        for baseline in experiment.BASELINES
                        for value in paired_by_cohort[cohort]["scopes"][scope][name][
                            baseline
                        ].values()
                    ),
                )
    primary = [
        scope[experiment.PRIMARY]
        for group in checks.values()
        for scope in group.values()
    ]
    return dict(
        primary=experiment.PRIMARY,
        checks=checks,
        development_precision_target_met=all(g["precision_60"] for g in primary),
        development_recall_preference_met=all(g["recall_85"] for g in primary),
        development_all_metric_nonregression=all(g["nonregression"] for g in primary),
        formal_metrics=None,
        goal_met=False,
        default_strategy_changed=False,
        independent_AI_blind_predictions=0,
    )


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Notes:
        Writes derived artifacts through once.
    """
    runner.guard()
    experiment.checked("evaluation_receipt.json")
    if (ROOT / "protocol.json").exists():
        guard()
        return
    assert load(ROOT / "engineering_checks.pre.json")["passed"]
    failure = load(runner.ROOT / "failure.json")
    assert failure["type"] == "KeyError" and failure["message"] == "'macro_ranges'"
    paths = [
        Path(__file__),
        PROJECT_ROOT / "tests/test_finalize_balanced_gallery.py",
        ROOT / "engineering_checks.pre.json",
        runner.ROOT / "failure.json",
        runner.ROOT / "failure_exclusion_snapshot.json",
        runner.ROOT / "execution_source_manifest.json",
        features.ROOT / "post_features_exclusion_snapshot.json",
        review.ROOT / "label_receipt.json",
    ]
    paths += [p for p in sorted(experiment.ROOT.rglob("*")) if p.is_file()]
    once(
        ROOT / "protocol.json",
        dict(
            files={rel(p): sha256_file(p) for p in paths},
            change="Report rendering reads ranges rather than nonexistent macro_ranges; explicit METRICS order; replay and seal in a new root",
            original_failed_source_or_status_overwritten=False,
            original_rankings_labels_queries_K_weights_or_models_changed=False,
            new_neural_forward_calls=0,
            independent_acceptance=False,
            formal_metrics=None,
        ),
    )


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    verify_files(PROJECT_ROOT, load(ROOT / "protocol.json")["files"])
    runner.guard()


def run_finalization() -> None:
    """Return run finalization for this workflow.

    Notes:
        Writes derived artifacts through once, report.write_text.
    """
    guard()
    if (ROOT / "completion_receipt.json").exists():
        verify()
        return
    outputs, details = experiment.orders()
    metrics_by_cohort, paired_by_cohort = {}, {}
    rankings, old_labels, new_labels = 0, 0, 0
    for cohort in experiment.COHORTS:
        base = experiment.ROOT / cohort
        assert outputs[cohort] == load(base / "predictions.json")
        assert details[cohort] == load(base / "runtime_scores.json")
        qrels = read_rows(base / "judgments.ai.jsonl")
        original = read_rows(experiment.DATA / cohort / "judgments.ai.jsonl")
        assert qrels[: len(original)] == original and qrels == experiment.labels(cohort)
        old_labels += len(original)
        new_labels += len(qrels) - len(original)
        metrics, paired = experiment.evaluated(cohort)
        assert metrics == load(base / "internal_metrics.ai.development.json")
        assert paired == load(base / "internal_paired_querywise.json")
        metrics_by_cohort[cohort], paired_by_cohort[cohort] = metrics, paired
        for query in load(base / "queries.public.json"):
            methods = outputs[cohort][query["id"]]
            assert set(methods) == set(experiment.METHODS)
            assert all(len(v) == len(set(v)) == 315 for v in methods.values())
            if not query["image_path"]:
                assert all(
                    methods[m] == methods["cross20"]
                    for m in (*experiment.BASELINES[3:], *experiment.NEW)
                )
            rankings += len(methods)
    assert (rankings, old_labels, new_labels) == (320, 11840, 760)
    decision = decision_values(metrics_by_cohort, paired_by_cohort)
    once(ROOT / "decision.json", decision)
    lines = [
        "# 315件同图库内部AI开发对照（独立修复报告导出）",
        "",
        "原19件原生评审、完整模型推理、760新增关系、11840原关系与320份排序全部保留。",
        "本次只修复报告字段名并独立封存；新增神经前向0，没有调整查询、标签、K、模型、权重或方法选择。",
        "全部为已曝光开发；AI标签不是人工真值。共享未知完成范围不是统计置信区间，也不覆盖AI误标。",
        "315件图库结果只在本轮同条件比较，不能和296件历史数字拼接。默认原版、正式指标null、新独立验收0。",
        "以上先前工程失败证据保留于module2_balanced_runner/failure.json。",
        "",
    ]
    for cohort in experiment.COHORTS:
        lines += [
            f"## {cohort}",
            "",
            "|范围|方法|P@10|R@10|mAP|NDCG@10|MRR|Hit@1|",
            "|---|---|---|---|---|---|---|---|",
        ]
        for scope, methods in metrics_by_cohort[cohort]["scopes"].items():
            for name in experiment.METHODS:
                lines.append(
                    "|" + "|".join([scope, name, *report_cells(methods[name])]) + "|"
                )
    report = ROOT / "internal_comparison.ai.development.md"
    text = "\n".join(lines) + "\n"
    if report.exists():
        assert report.read_text(encoding="utf-8") == text
    else:
        report.write_text(text, encoding="utf-8")
    once(
        ROOT / "verification.json",
        dict(
            passed=True,
            replayed_rankings=rankings,
            original_relations_unchanged=old_labels,
            new_relations=new_labels,
            scopes_all_image_text_recomputed=True,
            full_rankings_and_runtime_details_identical=True,
            new_neural_forward_calls=0,
            human_labels=0,
            formal_metrics=None,
        ),
    )
    previous = load(features.ROOT / "post_features_exclusion_snapshot.json")
    partial = load(runner.ROOT / "failure_exclusion_snapshot.json")
    signatures = {
        r["sha256"]: r
        for r in previous["image_fingerprints"] + partial["image_fingerprints"]
    }
    once(
        ROOT / "post_experiment_exclusion_snapshot.json",
        dict(
            previous,
            files=dict(previous["files"], **partial["files"]),
            image_fingerprints=list(signatures.values()),
            scope="All prior sources, all 48 supplement metadata, 19 raw images, all 159 derived views and both full/partial derived registries remain development-exposed; report fix creates no new images",
        ),
    )
    once(
        ROOT / "public_status.json",
        dict(
            completed=True,
            phase="development_comparison_replayed_and_finalized",
            methods=8,
            rankings=320,
            gallery_per_cohort=315,
            AI_relations=12600,
            human_labels=0,
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
        ),
    )
    paths = [
        p
        for p in sorted(ROOT.rglob("*"))
        if p.is_file() and p.name != "completion_receipt.json"
    ]
    once(
        ROOT / "completion_receipt.json",
        dict(
            files={rel(p): sha256_file(p) for p in paths},
            input_files=load(ROOT / "protocol.json")["files"],
            completed=True,
            formal_metrics=None,
        ),
    )
    verify()
    print(
        dict(
            completed=True,
            replayed_rankings=320,
            formal_metrics=None,
            development_precision_target_met=decision[
                "development_precision_target_met"
            ],
            development_recall_preference_met=decision[
                "development_recall_preference_met"
            ],
            development_all_metric_nonregression=decision[
                "development_all_metric_nonregression"
            ],
        )
    )


def verify() -> None:
    """Verify the recorded artifacts and their consistency."""
    guard()
    receipt = load(ROOT / "completion_receipt.json")
    verify_files(PROJECT_ROOT, receipt["files"])
    verify_files(PROJECT_ROOT, receipt["input_files"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "verify"])
    action = parser.parse_args().action
    with runner.locked() as acquired:
        if not acquired:
            raise RuntimeError("Another balanced runner holds its lock")
        if action == "prepare":
            prepare()
        elif action == "verify":
            verify()
            print(dict(seal_verified=True))
        else:
            run_finalization()
