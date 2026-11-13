"""Loss-only full-gallery proposal variant, with unchanged real full CE stages.

A fresh module namespace configures the sealed reusable stage implementation.
No original module, source file, checkpoint, score or label is mutated.
"""

from __future__ import annotations

import argparse
import traceback
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

import run  # noqa: F401
import subject_attribute_experiment as attributes
import subject_recall_experiment as parent
from finalize_balanced_gallery import report_cells
from learned_fusion_experiment import load, once, rel
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.module_loading import load_project_module
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file

ROOT = PROJECT_ROOT / "artifacts/module2_pointwise_recall_learning"
NEW = (
    "balanced_pointwise_proposal_ablation",
    "pointwise_recall_whole_union",
    "pointwise_recall_dual_union",
)
PRIMARY = NEW[2]
METHODS = (*parent.METHODS, *NEW)
BASELINES = (*parent.BASELINES, parent.PRIMARY)
COMPARISONS = [(m, b) for m in NEW for b in BASELINES] + [
    (PRIMARY, NEW[1]),
    (PRIMARY, NEW[0]),
]
SOURCES = [
    "scripts/pointwise_recall_experiment.py",
    "tests/test_pointwise_recall_experiment.py",
    "scripts/subject_recall_experiment.py",
    "scripts/subject_attribute_ranker.py",
    "scripts/subject_attribute_experiment.py",
    "scripts/querywise_paired_bounds.py",
    "scripts/finalize_balanced_gallery.py",
]
POLICY: dict = dict(
    parent.POLICY,
    loss="Per-query class-balanced pointwise logistic over every allowed known positive and negative",
    hard_negative_subsampling=False,
    bias_regularization=False,
    calibration="Allowed known training cells only; standardized logit, not a calibrated business probability",
)
POLICY["reference_parent_hard_negative_budget"] = POLICY.pop("hard_negatives")


def fit(
    features: np.ndarray,
    labels: np.ndarray,
    allowed: np.ndarray,
    base: np.ndarray,
    *,
    names: list[str],
) -> dict:
    """Return fit for this workflow.

    Args:
        features: Feature array aligned with the candidate or query records.
        labels: Judgments used for the stated development or evaluation protocol.
        allowed: Allowed used by this operation.
        base: Base used by this operation.
        names: Names to inspect or resolve.

    Returns:
        Mapping with fields: feature_names, mean, scale, weights, bias,
        score_location, score_scale, training_cells, positive, negative, policy,
        human_labels, formal_metrics.

    Raises:
        ValueError: Feature configuration mismatch; Both allowed training classes
            required; Missing complete training evidence.
    """
    del base
    known = allowed & np.isfinite(labels)
    if features.shape[-1] != len(names):
        raise ValueError("Feature configuration mismatch")
    if not known.any() or set(np.unique(labels[known])) != {0.0, 1.0}:
        raise ValueError("Both allowed training classes required")
    if not np.isfinite(features[known]).all():
        raise ValueError("Missing complete training evidence")
    mean, scale = features[known].mean(0), features[known].std(0)
    scale = np.where(scale < 1e-6, 1, scale)
    z = (features - mean) / scale
    xs, ys, weights = [], [], []
    for q in range(len(labels)):
        groups = [(np.flatnonzero(known[q] & (labels[q] == g)), g) for g in (0, 1)]
        groups = [(idx, g) for idx, g in groups if len(idx)]
        for idx, g in groups:
            xs.extend(z[q, idx])
            ys.extend([g] * len(idx))
            weights.extend([1 / (len(groups) * len(idx))] * len(idx))
    x = np.column_stack([np.asarray(xs), np.ones(len(xs))])
    y = np.asarray(ys)
    sample = np.asarray(weights)
    sample /= sample.sum()
    w = np.zeros(len(names) + 1)
    regularized = np.ones_like(w)
    regularized[-1] = 0
    for _ in range(POLICY["steps"]):
        prob = 1 / (1 + np.exp(-np.clip(x @ w, -40, 40)))
        w -= POLICY["learning_rate"] * (
            x.T @ ((prob - y) * sample) + POLICY["l2"] * w * regularized
        )
    raw = z[known] @ w[:-1] + w[-1]
    loc, spread = float(raw.mean()), float(raw.std())
    return dict(
        feature_names=list(names),
        mean=mean.tolist(),
        scale=scale.tolist(),
        weights=w[:-1].tolist(),
        bias=float(w[-1]),
        score_location=loc,
        score_scale=spread if spread > 1e-6 else 1.0,
        training_cells=int(known.sum()),
        positive=int((labels[known] == 1).sum()),
        negative=int((labels[known] == 0).sum()),
        policy=POLICY,
        human_labels=0,
        formal_metrics=None,
    )


def predict(features: np.ndarray, model: dict) -> np.ndarray:
    """Return predict for this workflow.

    Args:
        features: Feature array aligned with the candidate or query records.
        model: Model used by this operation.

    Returns:
        Computed array in the order of the supplied records.

    Raises:
        ValueError: Feature version mismatch.
    """
    if features.shape[-1] != len(model["feature_names"]):
        raise ValueError("Feature version mismatch")
    return (
        ((features - model["mean"]) / model["scale"]) @ model["weights"]
        + model["bias"]
        - model["score_location"]
    ) / model["score_scale"]


def diagnosis() -> dict:
    """Return diagnosis for this workflow.

    Returns:
        Mapping with fields: confirmed_negatives, base_top8_negatives,
        same_primary_type_all_negatives, same_primary_type_top8_negatives,
        primary_type_match_coefficient, interpretation, labels_or_queries_changed.
    """
    meta = load(parent.ROOT / "feature_metadata.json")
    i = meta["feature_names"].index("primary_type_match")
    with np.load(parent.ROOT / "feature_bank.npz", allow_pickle=False) as b:
        x, y, valid, base = [
            b[k].copy() for k in ("features", "labels", "valid", "base")
        ]
    all_neg, hard = [], []
    for q in range(len(y)):
        neg = np.flatnonzero(valid[q] & np.isfinite(y[q]) & (y[q] == 0))
        selected = neg[np.argsort(-base[q, neg], kind="stable")[:8]]
        all_neg.extend(x[q, neg, i].tolist())
        hard.extend(x[q, selected, i].tolist())
    model = load(parent.ROOT / "all_development_candidate.json")
    return dict(
        confirmed_negatives=len(all_neg),
        base_top8_negatives=len(hard),
        same_primary_type_all_negatives=int(sum(all_neg)),
        same_primary_type_top8_negatives=int(sum(hard)),
        primary_type_match_coefficient=model["weights"][i],
        interpretation="Internal development evidence of negative-sampling concentration; hypothesis for loss-only comparison, not proof of causal improvement",
        labels_or_queries_changed=False,
    )


def runtime() -> ModuleType:
    """Return runtime for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    path = PROJECT_ROOT / "scripts/subject_recall_experiment.py"
    module = load_project_module("pointwise_recall_runtime", path)
    facade = SimpleNamespace(**vars(attributes))
    facade.ROOT = parent.ROOT
    facade.METHODS = parent.METHODS
    facade.PRIMARY = parent.PRIMARY
    facade.verify = parent.verify
    module.previous = facade
    module.ROOT, module.NEW, module.PRIMARY, module.METHODS = (
        ROOT,
        NEW,
        PRIMARY,
        METHODS,
    )
    module.BASELINES, module.COMPARISONS, module.SOURCES = (
        BASELINES,
        COMPARISONS,
        SOURCES,
    )
    module.POLICY, module.fit, module.predict = POLICY, fit, predict
    module.guard = guard
    module.prepare = prepare
    original_plan = module.plan
    module.plan = lambda: plan(module, original_plan)
    module.prediction = lambda: prediction(module)
    module.finish = lambda: finish(module)
    return module


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
    once(ROOT / "negative_sampling_diagnosis.json", diagnosis())
    paths = [PROJECT_ROOT / s for s in SOURCES] + sorted(
        (PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")
    )
    paths += [
        ROOT / "engineering_checks.pre.json",
        ROOT / "negative_sampling_diagnosis.json",
        parent.ROOT / "completion_receipt.json",
        parent.ROOT / "feature_bank.npz",
        parent.ROOT / "feature_metadata.json",
        parent.ROOT / "exact_cache.json",
        parent.ROOT / "post_experiment_exclusion_snapshot.json",
    ]
    paths += sorted((parent.ROOT / "pairs").glob("*.json"))
    once(
        ROOT / "protocol.json",
        dict(
            files={rel(p): sha256_file(p) for p in paths},
            role="Loss-only adaptive exposed AI development, not independent acceptance",
            methods=METHODS,
            primary=PRIMARY,
            comparisons=COMPARISONS,
            features=parent.FEATURE_NAMES,
            training=POLICY,
            execution_bindings=dict(
                source="scripts/subject_recall_experiment.py",
                isolated_module_namespace=True,
                ROOT=rel(ROOT),
                previous_prediction_root=rel(parent.ROOT),
                DATA=rel(parent.DATA),
                fit="pointwise_recall_experiment.fit",
                predict="pointwise_recall_experiment.predict",
                plan="pointwise_recall_experiment.plan",
                report="pointwise_recall_experiment.finish",
            ),
            controlled_difference="Only proposal training objective/negative coverage changes; input feature arrays, labels, validity, folds, source-family masks, optimizer steps/LR/L2, stable original candidate union and full two-context CE formulas stay fixed",
            original_504_base_candidate_positions_retained=True,
            failed_ranknet_extra_pool_not_forced_into_new_method=True,
            actual_neural_models="Same full Qwen/FG evidence and full Qwen CE; exact compatible scores reused across all prior experiments, missing scores require real complete-model inference",
            proposal_ablation="Explicit no-CE component ablation, not substitute for main method",
            K=10,
            gallery_per_cohort=315,
            queries=40,
            AI_relations=12600,
            human_labels=0,
            formal_metrics=None,
            after_result_tuning=False,
            default_promotion_allowed=False,
            independent_acceptance=False,
        ),
    )
    runtime().status("protocol_frozen")


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    verify_files(PROJECT_ROOT, load(ROOT / "protocol.json")["files"])
    parent.verify()


def plan(module: ModuleType, original_plan: Callable[[], None]) -> None:
    # Reuse every exact real parent score, including its newly inferred pairs.
    # Its cache is an execution dependency; no evaluation label enters this plan.
    """Apply the plan workflow step.

    Args:
        module: Module used by this operation.
        original_plan: Original plan used by this operation.
    """
    original_loader = module.load

    def load_with_cache(path: Path) -> dict:
        """Return load with cache for this workflow.

        Args:
            path: Path of the input or output artifact used by this operation.

        Returns:
            Result mapping produced by the workflow.
        """
        value = original_loader(path)
        if path == parent.DATA / "exact_cache.json":
            value = dict(value)
            for source in sorted((parent.ROOT / "pairs").glob("*.json")):
                row = original_loader(source)
                identity = row["identity"]
                key = identity["sha256"]
                record = dict(
                    identity=identity,
                    logit=row["logit"],
                    source_file=rel(source),
                    source_sha256=sha256_file(source),
                )
                if key in value:
                    assert (
                        value[key]["identity"] == identity
                        and value[key]["logit"] == row["logit"]
                    )
                else:
                    value[key] = record
        return value

    module.__dict__["load"] = load_with_cache
    try:
        original_plan()
    finally:
        module.__dict__["load"] = original_loader


def prediction(module: ModuleType) -> None:
    """Apply the prediction workflow step.

    Args:
        module: Module used by this operation.

    Notes:
        Writes derived artifacts through once.
    """
    guard()
    module.checked("reranker_receipt.json")
    outputs, details = module.orders()
    paths = []
    for c in parent.COHORTS:
        for name, value in [
            ("predictions.json", outputs[c]),
            ("runtime_scores.json", details[c]),
        ]:
            path = ROOT / c / name
            once(path, value)
            paths.append(path)
    module.receipt(
        "prediction_receipt.json",
        paths,
        methods=17,
        rankings=680,
        unchanged_parent_rankings=560,
    )


def finish(module: ModuleType) -> None:
    """Finalize the recorded outputs of the workflow.

    Args:
        module: Module used by this operation.

    Notes:
        Writes derived artifacts through once, report.write_text.
    """
    guard()
    module.checked("evaluation_receipt.json")
    outputs, details = module.orders()
    scores, models, audits, candidate = module.train_values()
    assert (
        models == load(ROOT / "fold_models.json")
        and audits == load(ROOT / "fold_audits.json")
        and candidate == load(ROOT / "all_development_candidate.json")
    )
    with np.load(ROOT / "oof_scores.npz", allow_pickle=False) as b:
        assert np.array_equal(b["scores"], scores, equal_nan=True)
    with (
        np.load(ROOT / "feature_bank.npz", allow_pickle=False) as a,
        np.load(parent.ROOT / "feature_bank.npz", allow_pickle=False) as b,
    ):
        assert set(a.files) == set(b.files) and all(
            np.array_equal(a[k], b[k], equal_nan=True) for k in a.files
        )
    lines = [
        "# 类别平衡全负例候选学习：内部AI开发对照",
        "",
        "固定同一7560行33维真实特征、标签、有效掩码和家族折，改变提议层训练损失：从top8困难负例RankNet改为每查询类别平衡全正/负例逻辑损失。",
        "全模型证据及两种完整Qwen上下文精排保留；新共享池稳定保留原始504个候选位置，不强制继承失败RankNet新提议的干扰项。",
        "9个家族排除折模型和1个全开发候选；OOF拼接是开发证据，不是独立验收或单一部署模型。",
        "旧14方法560排序与全部12600条AI关系、40查询、K10保持。17方法680完整排序均报告，负面结果不隐藏。",
        "proposal_ablation是明确去除CE的组件消融，不替代主方法；打分不是业务相关概率。",
        "",
    ]
    gates: dict = {}
    losses = []
    for c in parent.COHORTS:
        assert outputs[c] == load(ROOT / c / "predictions.json") and details[c] == load(
            ROOT / c / "runtime_scores.json"
        )
        oldrank = load(parent.ROOT / c / "predictions.json")
        for qid, methods in outputs[c].items():
            assert set(methods) == set(METHODS) and all(
                len(v) == len(set(v)) == 315 for v in methods.values()
            )
            assert all(methods[m] == oldrank[qid][m] for m in parent.METHODS)
            if qid.startswith("text_"):
                assert all(methods[m] == methods["cross20"] for m in NEW)
        metrics, paired = module.evaluated(c)
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
        formal_metrics=None,
        independent_AI_blind_predictions=0,
    )
    once(ROOT / "decision.json", decision)
    once(ROOT / "internal_strict_regressions.json", losses)
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
            replayed_rankings=680,
            unchanged_parent_rankings=560,
            refitted_identical_models=10,
            identical_input_arrays_and_folds_to_parent=True,
            AI_relations_unchanged=12600,
            human_labels=0,
            formal_metrics=None,
        ),
    )
    once(
        ROOT / "post_experiment_exclusion_snapshot.json",
        load(parent.ROOT / "post_experiment_exclusion_snapshot.json"),
    )
    module.status(
        "completed",
        methods=17,
        rankings=680,
        gallery_per_cohort=315,
        AI_relations=12600,
    )
    module.receipt(
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
    print(
        {k: v for k, v in decision.items() if k.startswith("development_")}, flush=True
    )


def verify() -> None:
    """Verify the recorded artifacts and their consistency."""
    guard()
    runtime().checked("completion_receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "worker", "verify"])
    action = parser.parse_args().action
    module = runtime()
    with attributes.locked() as acquired:
        if not acquired:
            raise RuntimeError("Another subject-learning process holds the lock")
        if action == "prepare":
            prepare()
        elif action == "verify":
            verify()
            print(dict(seal_verified=True))
        else:
            try:
                module.run_all(wait=action == "worker")
            except Exception as exc:
                once(
                    ROOT / "failure.json",
                    dict(
                        type=type(exc).__name__,
                        message=str(exc),
                        traceback=traceback.format_exc(),
                        formal_metrics=None,
                    ),
                )
                module.status("engineering_failure")
                raise
