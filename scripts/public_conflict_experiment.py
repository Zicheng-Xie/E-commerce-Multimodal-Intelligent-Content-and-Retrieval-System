"""Frozen public-conflict priority comparison over real complete neural rankings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pointwise_recall_experiment as parent
import run  # noqa: F401
import subject_attribute_experiment as attributes
from finalize_balanced_gallery import report_cells
from learned_fusion_experiment import load, once, rel
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from public_conflict_priority import reorder
from querywise_paired_bounds import paired_querywise

ROOT = PROJECT_ROOT / "artifacts/module2_public_conflict_priority"
DATA = attributes.parent.ROOT
COHORTS = attributes.COHORTS
NEW = ("public_priority_full_order_ablation", "public_priority_fixed_CE_union")
PRIMARY = NEW[1]
METHODS = (*parent.METHODS, *NEW)
BASELINES = (
    "global",
    "region_title",
    "cross20",
    "whole_crop_context_union",
    attributes.PRIMARY,
    parent.PRIMARY,
)
COMPARISONS = [(m, b) for m in NEW for b in BASELINES] + [(PRIMARY, NEW[0])]
SOURCES = [
    "scripts/public_conflict_priority.py",
    "scripts/public_conflict_experiment.py",
    "tests/test_public_conflict_priority.py",
    "scripts/primary_subject_ranker.py",
    "scripts/subject_attribute_ranker.py",
    "scripts/subject_attribute_experiment.py",
    "scripts/pointwise_recall_experiment.py",
    "scripts/querywise_paired_bounds.py",
    "scripts/finalize_balanced_gallery.py",
]


def receipt(name: str, paths: list[Path], **extra: object) -> None:
    """Record file hashes and provenance for the completed stage.

    Args:
        name: Name used by this operation.
        paths: Paths used by this operation.
        **extra: Named options forwarded to the underlying operation.

    Notes:
        Writes derived artifacts through once.
    """
    once(ROOT / name, dict(files={rel(p): sha256_file(p) for p in paths}, **extra))


def checked(name: str) -> Any:
    """Return checked for this workflow.

    Args:
        name: Name used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    r = load(ROOT / name)
    verify_files(PROJECT_ROOT, r["files"])
    return r


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Notes:
        Writes derived artifacts through once.
    """
    parent.verify()
    if (ROOT / "protocol.json").exists():
        guard()
        return
    assert load(ROOT / "engineering_checks.pre.json")["passed"]
    paths = [PROJECT_ROOT / s for s in SOURCES] + sorted(
        (PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")
    )
    paths += [
        ROOT / "engineering_checks.pre.json",
        parent.ROOT / "completion_receipt.json",
        parent.ROOT / "proposals.public.json",
        parent.ROOT / "post_experiment_exclusion_snapshot.json",
        attributes.parent.DATA / "query_caption_receipt.json",
        attributes.parent.supplement.ROOT / "profile_receipt.json",
    ]
    for c in COHORTS:
        paths += [
            parent.ROOT / c / n
            for n in ("predictions.json", "internal_metrics.ai.development.json")
        ]
        paths += [
            DATA / c / n
            for n in (
                "queries.public.json",
                "catalog.public.json",
                "judgments.ai.jsonl",
            )
        ]
        paths.append(attributes.parent.DATA / c / "profiles.public.jsonl")
    paths.append(attributes.parent.supplement.ROOT / "profiles.candidate_only.jsonl")
    once(
        ROOT / "protocol.json",
        dict(
            files={rel(p): sha256_file(p) for p in paths},
            methods=METHODS,
            primary=PRIMARY,
            comparisons=COMPARISONS,
            role="Adaptive exposed AI business development only; not independent acceptance",
            hypothesis="Stable explicit public primary-item conflict priority prevents clear cross-type or incompatible part/form matches from dominating neural similarity",
            primary_execution="Same complete pointwise-parent CE union, raw full-model scores, fused order and tail; stable no-conflict tier before explicit-conflict tier inside that union only",
            full_order_ablation="Apply same public priority to existing original cross20 full ranking; promoted tail items have original base order, no invented CE scores",
            unknowns="No conflict inferred for unknown or contradictory public primary evidence; original relative order retained",
            relevance="All 12600 old AI business grades unchanged; phone model conflict not hard-gated, phone-case craft material retains broad phone_case type for this rubric",
            title_fallback="Longest public product compound only if primary description unresolved; ambiguous cabinet-character matches excluded; conflicting title/description abstain",
            neural_models="All existing full models and actual scores unchanged; additional priority component only, no reduced model or missing score imputation",
            K=10,
            gallery_per_cohort=315,
            queries=40,
            new_neural_forward_calls=0,
            new_images=0,
            human_labels=0,
            default_promotion_allowed=False,
            independent_acceptance=False,
            formal_metrics=None,
        ),
    )


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    verify_files(PROJECT_ROOT, load(ROOT / "protocol.json")["files"])
    parent.verify()


def orders() -> tuple[dict, dict]:
    """Return orders for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    proposals = {
        (p["cohort"], p["query_id"]): p
        for p in load(parent.ROOT / "proposals.public.json")
    }
    outputs, details = {}, {}
    for c in COHORTS:
        previous = load(parent.ROOT / c / "predictions.json")
        profiles = attributes.public_profiles(c)
        evidence: dict
        output, evidence = {}, {}
        for q in load(DATA / c / "queries.public.json"):
            qid = q["id"]
            if q["image_path"]:
                public = attributes.query_profile(c, qid)
                full, full_evidence = reorder(
                    previous[qid]["cross20"], public, profiles
                )
                union, union_evidence = reorder(
                    previous[qid][parent.PRIMARY],
                    public,
                    profiles,
                    pool=proposals[c, qid]["pool"],
                )
                new = dict(zip(NEW, [full, union]))
                evidence[qid] = dict(full=full_evidence, union=union_evidence)
            else:
                new = {name: list(previous[qid]["cross20"]) for name in NEW}
                evidence[qid] = dict(route="original_cross20_exact")
            output[qid] = dict(previous[qid], **new)
        outputs[c], details[c] = output, evidence
    return outputs, details


def prediction() -> None:
    """Apply the prediction workflow step.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    outputs, details = orders()
    paths = []
    for c in COHORTS:
        for name, value in [
            ("predictions.json", outputs[c]),
            ("runtime_public_evidence.json", details[c]),
        ]:
            p = ROOT / c / name
            once(p, value)
            paths.append(p)
    receipt(
        "prediction_receipt.json",
        paths,
        methods=19,
        rankings=760,
        parent_rankings_unchanged=680,
        new_neural_calls=0,
    )


def evaluated(c: str) -> tuple[dict, dict]:
    """Return evaluated for this workflow.

    Args:
        c: C used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    ranks = load(ROOT / c / "predictions.json")
    labels = read_rows(DATA / c / "judgments.ai.jsonl")
    modes = {
        q["id"]: "image" if q["image_path"] else "text"
        for q in load(DATA / c / "queries.public.json")
    }
    return evaluate_intervals(ranks, labels, modes), paired_querywise(
        ranks, labels, modes, COMPARISONS
    )


def evaluation() -> None:
    """Apply the evaluation workflow step.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    checked("prediction_receipt.json")
    paths = []
    for c in COHORTS:
        metrics, paired = evaluated(c)
        for name, value in [
            ("internal_metrics.ai.development.json", metrics),
            ("internal_paired_querywise.json", paired),
        ]:
            p = ROOT / c / name
            once(p, value)
            paths.append(p)
    receipt(
        "evaluation_receipt.json",
        paths,
        AI_relations_unchanged=12600,
        human_labels=0,
        formal_metrics=None,
    )


def finish() -> None:
    """Finalize the recorded outputs of the workflow.

    Notes:
        Writes derived artifacts through once, receipt, report.write_text.
    """
    guard()
    checked("evaluation_receipt.json")
    outputs, details = orders()
    gates: dict = {}
    losses = []
    lines = [
        "# 公开主售冲突优先级：内部AI开发对照",
        "",
        "315件同图库、40查询、12600冻结AI关系、K10，全部已曝光开发。",
        "主方法只在上一轮完整CE共享候选池内，将明确主类型/部件冲突稳定后移；同层原顺序和公共尾部不变。未知、矛盾字段不推断冲突；型号不硬过滤，不改相关性口径。",
        "全排序后处理为明确组件消融；原CE池外商品只有原base顺序，没有补造CE。原完整模型及全部680份父排序保持，新增神经前向0，19方法760份排序均报告。",
        "本轮没有修复已知错误ROI坐标，也未开启独立验收；未知范围不能覆盖AI误标。",
        "",
    ]
    for c in COHORTS:
        assert outputs[c] == load(ROOT / c / "predictions.json") and details[c] == load(
            ROOT / c / "runtime_public_evidence.json"
        )
        previous = load(parent.ROOT / c / "predictions.json")
        for qid, methods in outputs[c].items():
            assert set(methods) == set(METHODS) and all(
                len(v) == len(set(v)) == 315 for v in methods.values()
            )
            assert all(methods[m] == previous[qid][m] for m in parent.METHODS)
            if qid.startswith("text_"):
                assert all(methods[m] == methods["cross20"] for m in NEW)
        metrics, paired = evaluated(c)
        assert metrics == load(
            ROOT / c / "internal_metrics.ai.development.json"
        ) and paired == load(ROOT / c / "internal_paired_querywise.json")
        old = load(parent.ROOT / c / "internal_metrics.ai.development.json")
        gates[c] = {}
        lines += [
            f"## {c}",
            "",
            "|范围|方法|P@10|R@10|mAP|NDCG@10|MRR|Hit@1|",
            "|---|---|---|---|---|---|---|---|",
        ]
        for scope, methods in metrics["scopes"].items():
            assert all(methods[m] == old["scopes"][scope][m] for m in parent.METHODS)
            gates[c][scope] = {}
            for m in METHODS:
                lines.append(
                    "|" + "|".join([scope, m, *report_cells(methods[m])]) + "|"
                )
            for m in NEW:
                gates[c][scope][m] = dict(
                    precision_60=methods[m]["precision_60_lower_bound_met"],
                    recall_85=methods[m]["recall_85_lower_bound_met"],
                    nonregression=all(
                        v["nonregression_proven_for_shared_unknown_labels"]
                        for b in BASELINES
                        for v in paired["scopes"][scope][m][b].values()
                    ),
                )
        for row in paired["query_pair_bounds"]:
            if row["candidate"] == PRIMARY:
                bad = [
                    m
                    for m, v in row["metrics"].items()
                    if v["delta_range"][1] is not None and v["delta_range"][1] < -1e-12
                ]
                if bad:
                    losses.append(
                        dict(
                            cohort=c,
                            query_id=row["query_id"],
                            baseline=row["baseline"],
                            strict_regression_metrics=bad,
                        )
                    )
    primary = [scope[PRIMARY] for group in gates.values() for scope in group.values()]
    decision = dict(
        primary=PRIMARY,
        checks=gates,
        development_precision_target_met=all(v["precision_60"] for v in primary),
        development_recall_preference_met=all(v["recall_85"] for v in primary),
        development_all_metric_nonregression=all(v["nonregression"] for v in primary),
        goal_met=False,
        default_strategy_changed=False,
        independent_AI_blind_predictions=0,
        formal_metrics=None,
    )
    once(ROOT / "decision.json", decision)
    once(ROOT / "internal_strict_regressions.json", losses)
    text = "\n".join(lines) + "\n"
    report = ROOT / "internal_comparison.ai.development.md"
    if report.exists():
        assert report.read_text(encoding="utf-8") == text
    else:
        report.write_text(text, encoding="utf-8")
    once(
        ROOT / "verification.json",
        dict(
            passed=True,
            replayed_rankings=760,
            unchanged_parent_rankings=680,
            new_neural_calls=0,
            AI_relations_unchanged=12600,
            human_labels=0,
            formal_metrics=None,
        ),
    )
    once(
        ROOT / "post_experiment_exclusion_snapshot.json",
        load(parent.ROOT / "post_experiment_exclusion_snapshot.json"),
    )
    once(
        ROOT / "public_status.json",
        dict(
            completed=True,
            phase="public_conflict_priority_completed",
            methods=19,
            rankings=760,
            gallery_per_cohort=315,
            AI_relations=12600,
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
        ),
    )
    receipt(
        "completion_receipt.json",
        [
            p
            for p in sorted(ROOT.rglob("*"))
            if p.is_file() and p.name != "completion_receipt.json"
        ],
        source_files={s: sha256_file(PROJECT_ROOT / s) for s in SOURCES},
        completed=True,
        formal_metrics=None,
    )
    verify()
    print({k: v for k, v in decision.items() if k.startswith("development_")})


def verify() -> None:
    """Verify the recorded artifacts and their consistency."""
    guard()
    checked("completion_receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "verify"])
    action = parser.parse_args().action
    with attributes.locked() as acquired:
        if not acquired:
            raise RuntimeError("Another subject-learning process holds the lock")
        if action == "prepare":
            prepare()
        elif action == "verify":
            verify()
            print(dict(seal_verified=True))
        else:
            guard()
            for name, stage in [
                ("prediction_receipt.json", prediction),
                ("evaluation_receipt.json", evaluation),
                ("completion_receipt.json", finish),
            ]:
                if (ROOT / name).exists():
                    checked(name)
                else:
                    stage()
