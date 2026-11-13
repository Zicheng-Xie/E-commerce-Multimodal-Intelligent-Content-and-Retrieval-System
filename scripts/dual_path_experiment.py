"""Frozen dual-context Qwen/ROI experiment on the same exposed 296-item catalogs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, unquote

import numpy as np

import dual_context_pair_cache as pair_cache
import run  # noqa: F401
from dual_path_ranker import METHODS as NEW_METHODS
from dual_path_ranker import (
    POLICY,
    DualPathRankerFuseContextsInputs,
    fuse_contexts,
    text_orders,
)
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_evaluation import METRICS
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.intent_retrieval import normalized_ranks
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)
from primary_subject_experiment import METHODS as OLD_METHODS
from querywise_paired_bounds import paired_querywise
from report_primary_subjects import baseline_scores
from subject_roi_experiment import load

ROOT = PROJECT_ROOT / "artifacts/module2_dual_path_development"
PARENT = PROJECT_ROOT / "artifacts/module2_primary_subjects"
COHORTS = ("tune", "regression", "fresh")
METHODS = (*OLD_METHODS, *NEW_METHODS)
PRIMARY = "dual_context_roi_union"
once = pair_cache.once
VIEW_METHODS = ("cross20", "dual_context_union", PRIMARY)
DISPLAY = {
    "global": "整图",
    "region_title": "区域与标题",
    "cross20": "原Qwen20",
    "fgclip2_local_fusion20": "原全模态FG融合20",
    "image_fusion20": "原图片融合20",
    "image_roi_fusion20": "原图片ROI融合20",
    "subject_caption20": "原新描述20",
    "subject_guard20": "原描述加类型扣分20",
    "original_context_union": "同并集原上下文",
    "new_context_union": "同并集新上下文",
    "dual_context_union": "双上下文并集",
    PRIMARY: "双上下文并集加原ROI",
}
SOURCES = [
    "scripts/dual_path_experiment.py",
    "scripts/dual_path_ranker.py",
    "scripts/dual_context_pair_cache.py",
    "scripts/querywise_paired_bounds.py",
    "tests/test_dual_path_ranker.py",
    "tests/test_querywise_paired_bounds.py",
]
COMPARISONS = [
    (candidate, baseline)
    for candidate in NEW_METHODS
    for baseline in ("cross20", "global", "region_title")
] + [
    ("new_context_union", "original_context_union"),
    ("dual_context_union", "original_context_union"),
    ("dual_context_union", "new_context_union"),
    (PRIMARY, "dual_context_union"),
    ("subject_caption20", "cross20"),
    ("subject_guard20", "cross20"),
    ("image_roi_fusion20", "cross20"),
    ("image_roi_fusion20", "image_fusion20"),
]


def relative(path: Path) -> str:
    """Represent a workspace path relative to the project root.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Text produced by this operation.
    """
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def parent_guard() -> None:
    """Apply the parent guard workflow step."""
    value = load(PARENT / "completion_receipt.json")
    verify_files(PARENT, value["files"])
    verify_files(PROJECT_ROOT, value["source_files"])
    verify_files(PROJECT_ROOT, value["project_files"])
    assert load(PARENT / "public_status.json")["sealed"]


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    verify_files(PROJECT_ROOT, load(ROOT / "protocol.json")["files"])
    if (ROOT / "reranker_receipt.json").exists():
        pair_cache.verify()
    for name in (
        "prediction_receipt.json",
        "evaluation_receipt.json",
        "report_receipt.json",
    ):
        if (ROOT / name).exists():
            verify_files(ROOT, load(ROOT / name)["files"])


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Notes:
        Writes derived artifacts through once, write_json.
    """
    parent_guard()
    assert load(ROOT / "engineering_checks.pre.json")["passed"]
    assert load(ROOT / "cache_model_identity_audit.json")["passed"]
    pair_cache.prepare()
    if (ROOT / "protocol.json").exists():
        guard()
        return
    assert not any((ROOT / c / "predictions.json").exists() for c in COHORTS)
    paths = [PROJECT_ROOT / p for p in SOURCES]
    paths += list((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py"))
    paths += [
        PARENT / "completion_receipt.json",
        PARENT / "post_experiment_exclusion_snapshot.json",
        ROOT / "pair_cache_protocol.json",
        ROOT / "pair_plan.json",
        ROOT / "diagnosis_before_experiment.json",
        ROOT / "cache_model_identity_audit.json",
        ROOT / "engineering_checks.pre.json",
        PARENT / "reranker_runtime.json",
        PROJECT_ROOT
        / "artifacts/module2_subject_coverage_expansion/reranker_runtime.json",
        PROJECT_ROOT / "configs/module2_current_target.json",
    ]
    for cohort in COHORTS:
        paths += [
            PARENT / cohort / name
            for name in [
                "catalog.public.json",
                "queries.public.json",
                "judgments.ai.jsonl",
                "profiles.public.jsonl",
                "predictions.json",
                "runtime_scores.json",
            ]
        ]
    once(
        ROOT / "protocol.json",
        dict(
            files={relative(p): sha256_file(p) for p in paths},
            policy=POLICY,
            primary_method=PRIMARY,
            methods=METHODS,
            comparisons=COMPARISONS,
            fixed_k=10,
            queries=40,
            image_queries=24,
            text_queries=16,
            gallery_per_cohort=296,
            unchanged_AI_relations=11840,
            new_AI_labels=0,
            human_labels=0,
            planned_full_rankings=480,
            full_original_models_reused=True,
            new_full_Qwen_pairs="exact missing pair identities only; no cross-context imputation",
            ROI="Frozen old-description whole/ROI FG evidence; no new caption FG or new ROI claim",
            paired_evaluation="Per-query shared-label exact enumeration when U<=16; identical order is exactly zero regardless U; changed U>16 query alone uses explicitly conservative bounds. Optional Recall denominator is retained.",
            metrics_changed_by_evaluation=False,
            selection="Fixed primary hypothesis and ablations. Report every method/group/scope. Require P60 lower bound and six-metric nonregression against cross20/global/region_title for a development proposal. Recall85 preferred, not mandatory. No after-result tuning, no query/label/K changes; no automatic serving promotion.",
            independent_acceptance_started=False,
            formal_metrics=None,
            default_strategy_changed=False,
        ),
    )
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=False,
            phase="frozen_before_new_inference",
            queries=40,
            gallery_per_cohort=296,
            new_labels=0,
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
        ),
    )
    print(dict(protocol_frozen=True, primary=PRIMARY, methods=len(METHODS)), flush=True)


def cohort_orders(cohort: str) -> tuple[dict, dict]:
    """Return cohort orders for this workflow.

    Args:
        cohort: Cohort used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    previous = load(PARENT / cohort / "predictions.json")
    evidence = load(PARENT / cohort / "runtime_scores.json")
    output, details = {}, {}
    for query in load(PARENT / cohort / "queries.public.json"):
        qid = query["id"]
        baseline = previous[qid]["cross20"]
        if query["image_path"]:
            old = load(PARENT / cohort / "runs" / qid / "features.json")
            new = load(PARENT / cohort / "runs" / qid / "caption_features.json")
            a = load(ROOT / cohort / "runs" / qid / "old_cross.json")
            b = load(ROOT / cohort / "runs" / qid / "new_cross.json")
            assert a["pool"] == b["pool"]
            entry = evidence[qid]
            ids = old["ids"]
            assert new["ids"] == entry["ids"] == ids
            whole = np.asarray(entry["whole_scores"]["fused_scores"])
            roi = np.asarray(entry["roi_scores"]["fused_scores"])
            auxiliary = 0.5 * normalized_ranks(ids, whole) + 0.5 * normalized_ranks(
                ids, roi
            )
            new_orders, detail = fuse_contexts(
                DualPathRankerFuseContextsInputs(
                    ids,
                    a["pool"],
                    a["logits"],
                    b["logits"],
                    old["base"],
                    new["base"],
                    auxiliary,
                    baseline,
                )
            )
        else:
            new_orders, detail = text_orders(baseline), dict(
                route="original_text_exact"
            )
        output[qid] = {**previous[qid], **new_orders}
        details[qid] = detail
    return output, details


def prediction() -> None:
    """Apply the prediction workflow step.

    Notes:
        Writes derived artifacts through once.
    """
    guard()
    files = []
    for cohort in COHORTS:
        rankings, details = cohort_orders(cohort)
        for name, value in [
            ("predictions.json", rankings),
            ("runtime_scores.json", details),
        ]:
            path = ROOT / cohort / name
            once(path, value)
            files.append(path)
    once(
        ROOT / "prediction_receipt.json",
        dict(
            files={p.relative_to(ROOT).as_posix(): sha256_file(p) for p in files},
            full_rankings=480,
            methods=len(METHODS),
            gallery_per_cohort=296,
            old_eight_methods_unchanged=True,
            text_queries_unchanged=16,
            labels_used_in_runtime=False,
        ),
    )


def evaluated(cohort: str) -> tuple[dict, dict]:
    """Return evaluated for this workflow.

    Args:
        cohort: Cohort used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    rankings = load(ROOT / cohort / "predictions.json")
    labels = read_rows(PARENT / cohort / "judgments.ai.jsonl")
    modes = {q: q.split("_")[0] for q in rankings}
    metrics = evaluate_intervals(rankings, labels, modes)
    paired = paired_querywise(rankings, labels, modes, COMPARISONS)
    return metrics, paired


def evaluation() -> None:
    """Apply the evaluation workflow step.

    Notes:
        Writes derived artifacts through once.
    """
    guard()
    verify_files(ROOT, load(ROOT / "prediction_receipt.json")["files"])
    files = []
    for cohort in COHORTS:
        metrics, paired = evaluated(cohort)
        for name, value in [
            ("internal_metrics.ai.development.json", metrics),
            ("internal_paired_querywise.json", paired),
        ]:
            path = ROOT / cohort / name
            once(path, value)
            files.append(path)
        print(dict(evaluated=cohort, role="development_only"), flush=True)
    once(
        ROOT / "evaluation_receipt.json",
        dict(
            files={p.relative_to(ROOT).as_posix(): sha256_file(p) for p in files},
            labels_unchanged=11840,
            zero_positive_queries_retained=True,
            formal_metrics=None,
        ),
    )


def verification() -> None:
    """Apply the verification workflow step.

    Notes:
        Writes derived artifacts through once.
    """
    guard()
    parent_guard()
    pair_checks = pair_cache.verify()
    verify_files(ROOT, load(ROOT / "evaluation_receipt.json")["files"])
    for cohort in COHORTS:
        ranks, details = cohort_orders(cohort)
        assert ranks == load(ROOT / cohort / "predictions.json")
        assert details == load(ROOT / cohort / "runtime_scores.json")
        previous = load(PARENT / cohort / "predictions.json")
        for qid, methods in ranks.items():
            assert set(methods) == set(METHODS)
            assert all(len(r) == len(set(r)) == 296 for r in methods.values())
            assert all(methods[m] == previous[qid][m] for m in OLD_METHODS)
            if qid.startswith("text_"):
                assert all(methods[m] == methods["cross20"] for m in NEW_METHODS)
        metrics, paired = evaluated(cohort)
        assert metrics == load(ROOT / cohort / "internal_metrics.ai.development.json")
        assert paired == load(ROOT / cohort / "internal_paired_querywise.json")
        old = load(PARENT / cohort / "internal_metrics.ai.development.json")
        for scope in old["scopes"]:
            for method in OLD_METHODS:
                assert metrics["scopes"][scope][method] == old["scopes"][scope][method]
    once(
        ROOT / "verification.json",
        dict(
            passed=True,
            replayed_full_rankings=480,
            preserved_prior_full_rankings=320,
            unchanged_relations=11840,
            text_queries_exact=16,
            raw_pair_checks=pair_checks,
            individual_metrics_unchanged_for_old_methods=True,
            tighter_pair_bounds_are_not_algorithm_gain=True,
            formal_metrics=None,
        ),
    )


def pct(bounds: dict) -> str:
    """Return pct for this workflow.

    Args:
        bounds: Bounds used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return "–".join("未定义" if v is None else f"{100 * v:.2f}" for v in bounds)


def viewer() -> None:
    """Apply the viewer workflow step.

    Notes:
        Writes derived artifacts through (ROOT / 'internal_comparison.html').write_text.
    """
    rows = []
    for cohort in COHORTS:
        rankings = load(ROOT / cohort / "predictions.json")
        details = load(ROOT / cohort / "runtime_scores.json")
        docs = {d["id"]: d for d in load(PARENT / cohort / "catalog.public.json")}
        profiles = {
            r["id"]: r["product_type"]
            for r in read_rows(PARENT / cohort / "profiles.public.jsonl")
        }
        grades = {
            (r["query_id"], r["candidate_id"]): r["grade"]
            for r in read_rows(PARENT / cohort / "judgments.ai.jsonl")
        }

        def url(path: Path) -> str:
            """Return url for this workflow.

            Args:
                path: Path of the input or output artifact used by this operation.

            Returns:
                Values produced by this operation, in the stated return order.
            """
            return quote(os.path.relpath(path, ROOT).replace("\\", "/"), safe="/:")

        for query in load(PARENT / cohort / "queries.public.json"):
            qid = query["id"]
            feature = load(PARENT / cohort / "runs" / qid / "features.json")
            raw = load(PARENT / cohort / "runs" / qid / "cross.json")
            scores = {"cross20": baseline_scores(feature, raw)}
            if query["image_path"]:
                scores.update(details[qid]["scores"])
            else:
                scores.update({m: scores["cross20"] for m in NEW_METHODS})
            results = {}
            for method in VIEW_METHODS:
                values = scores[method]
                assert (
                    sorted(values, key=lambda cid: (-values[cid], cid))[:20]
                    == rankings[qid][method][:20]
                )
                results[method] = [
                    dict(
                        id=cid,
                        title=docs[cid]["title"],
                        image=url(docs[cid]["image_path"]),
                        category=profiles[cid],
                        score=float(values[cid]),
                        AI_grade=grades[qid, cid],
                    )
                    for cid in rankings[qid][method][:20]
                ]
            description = query["text"]
            if query["image_path"]:
                new_description = load(
                    PARENT / cohort / "runs" / qid / "caption_features.json"
                )["description"]
                description = (
                    "原描述："
                    + feature["description"]
                    + "；新主售描述："
                    + new_description
                )
            rows.append(
                dict(
                    id=f"{cohort}/{qid}",
                    query=description,
                    image=url(query["image_path"]) if query["image_path"] else None,
                    crop=None,
                    results=results,
                )
            )
    html = (PARENT / "internal_comparison.html").read_text(encoding="utf-8")
    begin = '<script id="data" type="application/json">'
    start = html.index(begin) + len(begin)
    end = html.index("</script>", start)
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    html = html[:start] + payload + html[end:]
    names = (
        "const names="
        + json.dumps({m: DISPLAY[m] for m in VIEW_METHODS}, ensure_ascii=False)
        + ";"
    )
    html, n = re.subn(r"const names=\{.*?\};", lambda _: names, html, count=1)
    assert n == 1
    html = re.sub(
        r"<title>.*?</title>",
        "<title>模块二双上下文同图库开发对照</title>",
        html,
        count=1,
    )
    html = re.sub(
        r"<h1>.*?</h1>", "<h1>原Qwen、双上下文并集与ROI消融</h1>", html, count=1
    )
    (ROOT / "internal_comparison.html").write_text(html, encoding="utf-8")


def report() -> None:
    """Summarize the recorded results of this development workflow.

    Notes:
        Writes derived artifacts through (ROOT /
        'internal_comparison.ai.development.md').write_text, once.
    """
    guard()
    assert load(ROOT / "verification.json")["passed"]
    calls = load(ROOT / "reranker_receipt.json")
    lines = [
        "# 双上下文与ROI同图库开发对照",
        "",
        "内部AI开发证据；40条查询与296件图库均已曝光，人工标签0，正式指标为空。全部11840条关系不变，固定K=10，未知项仍为null。没有新标签或新独立验收。",
        "",
        "候选使用原20与新描述20的固定并集。四种新方法在同一并集上使用完整原/新上下文精排，公共尾部保持原cross20过滤并集后的相对顺序。无类型字典扣分，也不填补缺失模型分数。ROI复用原描述时期已冻结的whole/ROI FG证据，未生成新caption的FG文本或新主体ROI。",
        "",
        "原8种方法逐条保留，新增4种同并集消融，共480份完整排序。文本查询原样返回cross20。单方法质量区间完全保留原口径；逐query配对评估收紧了此前过宽的保守界，评估边界变化不能算作算法收益，也不覆盖AI误判。",
        "",
        "[商品对照页](internal_comparison.html) · [冻结诊断](diagnosis_before_experiment.json) · [精排实算及缓存](reranker_receipt.json)",
        "",
    ]
    gates: dict = {}
    for cohort in COHORTS:
        metrics = load(ROOT / cohort / "internal_metrics.ai.development.json")
        paired = load(ROOT / cohort / "internal_paired_querywise.json")
        lines += [
            f"## {cohort}（开发）",
            "",
            "| 范围 | 方法 | P@10 % | R@10 % | mAP % | NDCG@10 % | MRR % | Hit@1 % |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        gates[cohort] = {}
        for scope, methods in metrics["scopes"].items():
            gates[cohort][scope] = {}
            for method in METHODS:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            scope,
                            DISPLAY[method],
                            *[pct(methods[method]["ranges"][m]) for m in METRICS],
                        ]
                    )
                    + " |"
                )
            for method in NEW_METHODS:
                fields = paired["scopes"][scope][method]
                gates[cohort][scope][method] = dict(
                    precision_60=methods[method]["precision_60_lower_bound_met"],
                    recall_85=methods[method]["recall_85_lower_bound_met"],
                    nonregression_against_cross20=all(
                        v["nonregression_proven_for_shared_unknown_labels"]
                        for v in fields["cross20"].values()
                    ),
                    nonregression_against_feature_baselines=all(
                        v["nonregression_proven_for_shared_unknown_labels"]
                        for b in ("global", "region_title")
                        for v in fields[b].values()
                    ),
                )
        lines += [
            "",
            "### 同候选消融及原方法对照",
            "",
            "| 范围 | 候选方法 | 基线 | 指标 | 差值范围（百分点） | 配对界类型 | 不退化可证明 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for scope, candidates in paired["scopes"].items():
            for candidate, baselines in candidates.items():
                for baseline, fields in baselines.items():
                    for metric, value in fields.items():
                        lines.append(
                            "| "
                            + " | ".join(
                                [
                                    scope,
                                    DISPLAY[candidate],
                                    DISPLAY[baseline],
                                    metric,
                                    pct(value["delta_range"]),
                                    value["bound_type"],
                                    str(
                                        value[
                                            "nonregression_proven_for_shared_unknown_labels"
                                        ]
                                    ),
                                ]
                            )
                            + " |"
                        )
        lines += [
            "",
            "完整图库正例数决定Recall分母；正例数超过10时单查询Recall上限为10/R。冻结诊断已说明A/B泛文本与总体的85%目标有容量约束，不改分母、K或查询来宣称达成。零正例查询保留，Recall未定义而非满分。",
            "",
        ]
    primary = [
        group[PRIMARY] for cohorts in gates.values() for group in cohorts.values()
    ]
    decision = dict(
        primary_method=PRIMARY,
        scope_checks=gates,
        development_precision_target_met=all(v["precision_60"] for v in primary),
        development_recall_preference_met=all(v["recall_85"] for v in primary),
        development_six_metric_nonregression_against_cross20=all(
            v["nonregression_against_cross20"] for v in primary
        ),
        development_nonregression_against_feature_baselines=all(
            v["nonregression_against_feature_baselines"] for v in primary
        ),
        default_strategy_changed=False,
        formal_metrics=None,
        goal_met=False,
        independent_AI_blind_predictions=0,
        precision_60_is_required=True,
        recall_85_is_preferred_not_mandatory=True,
    )
    once(ROOT / "decision.json", decision)
    lines += [
        "## 结论与运行证据",
        "",
        f"主方法Precision全部组别/范围下界通过60%：{decision['development_precision_target_met']}；Recall偏好通过：{decision['development_recall_preference_met']}；对cross20六指标不退化：{decision['development_six_metric_nonregression_against_cross20']}。默认方法未替换，不能将开发结论当作正式验收。",
        "",
        "完整Qwen精排实际新增和身份一致复用数见运行收据；图像1008个上下文结果与文本320个冻结上下文结果分开记录，没有把缓存当成新模型调用。没有新增嵌入、DINO或FG模型调用。正式升级仍需要冻结方案后的新图库、新查询独立验收。",
        "",
    ]
    (ROOT / "internal_comparison.ai.development.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    once(
        ROOT / "runtime_audit.json",
        dict(
            raw_pair_receipt=calls,
            full_embedding_new_calls=0,
            DINO_new_calls=0,
            FG_new_calls=0,
            full_original_features_reused=True,
            full_rankings=480,
            AI_relations_unchanged=11840,
            new_AI_labels=0,
            formal_metrics=None,
        ),
    )
    viewer()
    once(
        ROOT / "report_receipt.json",
        dict(
            files={
                name: sha256_file(ROOT / name)
                for name in [
                    "decision.json",
                    "runtime_audit.json",
                    "internal_comparison.html",
                    "internal_comparison.ai.development.md",
                ]
            }
        ),
    )
    print(
        {k: v for k, v in decision.items() if k.startswith("development_")}, flush=True
    )


def checks() -> None:
    """Apply the checks workflow step.

    Notes:
        Writes derived artifacts through once, target.parent.mkdir, target.write_text.
    """
    import subprocess

    guard()
    html = (ROOT / "internal_comparison.html").read_text(encoding="utf-8")
    match = re.search(
        r'<script id="data" type="application/json">(.*?)</script>', html, re.S
    )
    if match is None:
        raise ValueError("Comparison page is missing its JSON payload")
    rows = json.loads(match[1])
    assert len(rows) == 40
    cards = 0
    for row in rows:
        assert set(row["results"]) == set(VIEW_METHODS)
        images = [row["image"]] if row.get("image") else []
        for results in row["results"].values():
            assert len(results) == len({r["id"] for r in results}) == 20
            assert all(
                r["title"] and r["category"] and isinstance(r["score"], float)
                for r in results
            )
            images.extend(r["image"] for r in results)
            cards += len(results)
        assert all((ROOT / unquote(path)).resolve().is_file() for path in images)
    scripts = [
        s
        for s in re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
        if s.lstrip().startswith("const rows=")
    ]
    assert len(scripts) == 1
    target = ROOT / "qa/viewer.js"
    target.parent.mkdir(exist_ok=True)
    target.write_text(scripts[0], encoding="utf-8")
    result = subprocess.run(
        ["E:/Node.js/node.exe", "--check", str(target)],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    once(
        ROOT / "checks.json",
        dict(
            passed=True,
            engineering=load(ROOT / "engineering_checks.pre.json"),
            full_rankings_replayed=480,
            prior_eight_methods_unchanged=True,
            original_AI_grid_unchanged=11840,
            cards=cards,
            local_images_exist=True,
            static_JS_syntax=True,
            browser_interaction_tested=False,
            online_latency_acceptance=False,
            formal_metrics=None,
        ),
    )


def seal() -> None:
    """Seal the completed outputs with their integrity metadata.

    Notes:
        Writes derived artifacts through once, release.mkdir, write_json.
    """
    guard()
    assert load(ROOT / "checks.json")["passed"]
    parent_guard()
    # No new images, metadata or crops: carry forward the complete exclusion registry.
    old_path = PARENT / "post_experiment_exclusion_snapshot.json"
    old = load(old_path)
    verify_files(PROJECT_ROOT, old["files"])
    snapshot = dict(old)
    snapshot.update(
        parent_snapshot_sha256=sha256_file(old_path),
        scope="All previous exposures, reused by this dual-context development experiment; zero new candidates or images; never independent acceptance",
    )
    once(ROOT / "post_experiment_exclusion_snapshot.json", snapshot)
    release = ROOT / "source_release"
    release.mkdir(exist_ok=True)
    source_files = {}
    for name in SOURCES:
        source = PROJECT_ROOT / name
        target = release / source.name
        if not target.exists():
            shutil.copyfile(source, target)
        assert sha256_file(target) == sha256_file(source)
        source_files[name] = sha256_file(source)
    once(release / "manifest.json", source_files)
    files = {
        p.relative_to(ROOT).as_posix(): sha256_file(p)
        for p in sorted(ROOT.rglob("*"))
        if p.is_file()
        and p.name not in {"completion_receipt.json", "public_status.json"}
    }
    decision = load(ROOT / "decision.json")
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=True,
            sealed=True,
            role="exposed_AI_dual_context_development",
            queries=40,
            image_queries=24,
            text_queries_unchanged=16,
            gallery_per_cohort=296,
            full_rankings=480,
            methods=12,
            unchanged_AI_relations=11840,
            human_labels=0,
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            development_precision_target_met=decision[
                "development_precision_target_met"
            ],
            development_recall_preference_met=decision[
                "development_recall_preference_met"
            ],
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
        ),
    )
    once(
        ROOT / "completion_receipt.json",
        dict(
            files=files,
            source_files=source_files,
            project_files=old["files"],
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
        ),
    )
    print(dict(sealed_files=len(files), source_snapshots=len(source_files)), flush=True)


def completed() -> bool:
    """Return completed for this workflow.

    Returns:
        Whether the condition checked by this operation holds.
    """
    if not (ROOT / "completion_receipt.json").exists():
        return False
    receipt = load(ROOT / "completion_receipt.json")
    verify_files(ROOT, receipt["files"])
    verify_files(PROJECT_ROOT, receipt["source_files"])
    verify_files(PROJECT_ROOT, receipt["project_files"])
    assert load(ROOT / "public_status.json")["completed"]
    return True


@contextmanager
def exclusive_runner() -> Iterator:
    """Return exclusive runner for this workflow.

    Yields:
        Values produced by this operation, in the stated return order.

    Notes:
        Writes derived artifacts through path.parent.mkdir.
    """
    from multimodal_ecommerce import platform_lock as msvcrt

    path = ensure_output_path(PROJECT_ROOT / "tmp/module2_dual_path.guard.lock")
    path.parent.mkdir(exist_ok=True)
    try:
        with path.open("xb") as first:
            first.write(b"0")
    except FileExistsError:
        logging.getLogger(__name__).debug("Reusing existing worker lock file")
    with path.open("r+b") as handle:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            logging.getLogger(__name__).debug(
                "Handled exception in exclusive_runner", exc_info=True
            )
            yield False
            return
        try:
            yield True
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def run_all() -> None:
    """Return run all for this workflow.

    Notes:
        Writes derived artifacts through write_json.
    """
    if completed():
        print(dict(seal_verified=True), flush=True)
        return
    with exclusive_runner() as acquired:
        if not acquired:
            return
        if completed():
            print(dict(seal_verified=True), flush=True)
            return
        prepare()
        if not (ROOT / "reranker_receipt.json").exists() and not pair_cache.infer():
            write_json(
                ROOT / "public_status.json",
                dict(
                    completed=False,
                    phase="waiting_for_natural_GPU_release",
                    quality_metrics=None,
                    formal_metrics=None,
                    default_strategy_changed=False,
                    independent_AI_blind_predictions=0,
                ),
            )
            print(dict(waiting_for_natural_GPU_release=True), flush=True)
            return
        stages = [
            (prediction, "prediction_receipt.json"),
            (evaluation, "evaluation_receipt.json"),
            (verification, "verification.json"),
            (report, "report_receipt.json"),
            (checks, "checks.json"),
        ]
        for action, output in stages:
            if not (ROOT / output).exists():
                print(dict(stage=action.__name__, phase="running"), flush=True)
                action()
        seal()
        assert completed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "run",
            "prediction",
            "evaluation",
            "verification",
            "report",
            "checks",
            "seal",
        ],
    )
    action = parser.parse_args().action
    (run_all if action == "run" else globals()[action])()
