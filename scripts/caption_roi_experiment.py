"""Frozen caption-guided crop/full-image Qwen development comparison."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, unquote

import numpy as np

import caption_roi_assets_safe as assets
import caption_roi_pair_cache as pair_cache
import run  # noqa: F401
from caption_roi_ranker import METHODS as NEW_METHODS
from caption_roi_ranker import (
    POLICY,
    PRIMARY,
    CaptionRoiRankerFuseCropsInputs,
    fuse_crops,
    text_orders,
)
from dual_path_experiment import COMPARISONS as PARENT_COMPARISONS
from dual_path_experiment import DISPLAY as PARENT_DISPLAY
from dual_path_experiment import METHODS as OLD_METHODS
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
from querywise_paired_bounds import paired_querywise
from report_primary_subjects import baseline_scores
from subject_roi_experiment import load

ROOT = PROJECT_ROOT / "artifacts/module2_caption_roi_development"
PARENT = PROJECT_ROOT / "artifacts/module2_dual_path_development"
DATA = PROJECT_ROOT / "artifacts/module2_primary_subjects"
COHORTS = ("tune", "regression", "fresh")
METHODS = (*OLD_METHODS, *NEW_METHODS)
VIEW_METHODS = ("cross20", "dual_context_roi_union", PRIMARY)
DISPLAY = {
    **PARENT_DISPLAY,
    "crop_context_union": "新描述与主体裁切完整Qwen",
    PRIMARY: "整图与主体裁切完整Qwen",
    "whole_crop_roi_union": "整图裁切Qwen加原FG区域辅助",
}
SOURCES = [
    "scripts/caption_roi_experiment.py",
    "scripts/caption_roi_ranker.py",
    "scripts/caption_roi_assets.py",
    "scripts/caption_roi_assets_safe.py",
    "scripts/caption_roi_pair_cache.py",
    "scripts/querywise_paired_bounds.py",
    "tests/test_caption_roi_ranker.py",
    "tests/test_querywise_paired_bounds.py",
]
REFERENCE_BASELINES = (
    "cross20",
    "global",
    "region_title",
    "dual_context_union",
    "dual_context_roi_union",
)
COMPARISONS = list(
    dict.fromkeys(
        [
            (candidate, baseline)
            for candidate in NEW_METHODS
            for baseline in REFERENCE_BASELINES
        ]
        + [
            (PRIMARY, "crop_context_union"),
            ("whole_crop_roi_union", PRIMARY),
            ("crop_context_union", "new_context_union"),
        ]
        + PARENT_COMPARISONS
    )
)


def relative(path: Path) -> str:
    """Represent a workspace path relative to the project root.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Text produced by this operation.
    """
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def once(path: Path, value: object) -> None:
    """Atomically preserve canonical JSON, including tuple/list round trips.

    Args:
        path: Path of the input or output artifact used by this operation.
        value: Input value to validate or transform.
    """
    path = ensure_output_path(path)
    value = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    if path.exists():
        assert load(path) == value, f"Preserve completed artifact: {path}"
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    if pending.exists():
        assert load(pending) == value, f"Preserve pending artifact: {pending}"
    else:
        with pending.open("x", encoding="utf-8", newline="") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    pending.replace(path)


def parent_guard() -> None:
    """Apply the parent guard workflow step."""
    for root in (DATA, PARENT):
        receipt = load(root / "completion_receipt.json")
        verify_files(root, receipt["files"])
        verify_files(PROJECT_ROOT, receipt["source_files"])
        verify_files(PROJECT_ROOT, receipt["project_files"])
        status = load(root / "public_status.json")
        assert status["sealed"] and status["completed"]


def guard() -> None:
    """Check every existing receipt before resuming any stage."""
    verify_protocol()
    assets.guard()
    for path in sorted(ROOT.glob("*receipt.json")):
        receipt = load(path)
        if "files" in receipt:
            base = PROJECT_ROOT if path.name == "exposure_receipt.json" else ROOT
            verify_files(base, receipt["files"])
        for key in ("source_files", "project_files"):
            if key in receipt:
                verify_files(PROJECT_ROOT, receipt[key])
    if (ROOT / "detection_receipt.json").exists():
        assets.verify()
    if (ROOT / "pair_cache_protocol.json").exists():
        pair_cache.guard()
    if (ROOT / "reranker_receipt.json").exists():
        pair_cache.verify()


def verify_protocol() -> None:
    """Verify protocol."""
    protocol = load(ROOT / "protocol.json")
    verify_files(PROJECT_ROOT, protocol["files"])
    expected = {
        name
        for name in protocol["files"]
        if name.startswith("src/multimodal_ecommerce/") and name.endswith(".py")
    }
    actual = {
        relative(path)
        for path in (PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")
    }
    assert len(expected) == 68 and actual == expected


def prepare() -> None:
    """Metadata only; freeze before any new detection or crop Qwen call."""
    parent_guard()
    assert load(ROOT / "engineering_checks.pre.json")["passed"]
    assets.prepare()
    if (ROOT / "protocol.json").exists():
        guard()
        return
    assert not (ROOT / "detection_receipt.json").exists()
    assert not (ROOT / "reranker_receipt.json").exists()
    assert not any((ROOT / "query_rois").rglob("result.json"))
    assert not any((ROOT / c / "predictions.json").exists() for c in COHORTS)
    paths = [PROJECT_ROOT / name for name in SOURCES]
    source_paths = sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py"))
    assert len(source_paths) == 68, "Keep the frozen 68-source project isolated"
    paths += source_paths
    paths += [
        ROOT / "engineering_checks.pre.json",
        ROOT / "crop_asset_protocol.json",
        ROOT / "crop_asset_execution_protocol.json",
        ROOT / "crop_plan.json",
        PARENT / "completion_receipt.json",
        PARENT / "protocol.json",
        PARENT / "post_experiment_exclusion_snapshot.json",
        PARENT / "pair_plan.json",
        PARENT / "reranker_runtime.json",
        DATA / "completion_receipt.json",
        DATA / "comparison_protocol.json",
        DATA / "feature_protocol.json",
        DATA / "query_caption_receipt.json",
        PROJECT_ROOT / "configs/reranker_model.json",
        PROJECT_ROOT / "configs/module2_current_target.json",
    ]
    for cohort in COHORTS:
        paths += [
            DATA / cohort / name
            for name in (
                "catalog.public.json",
                "queries.public.json",
                "judgments.ai.jsonl",
                "profiles.public.jsonl",
                "predictions.json",
                "runtime_scores.json",
            )
        ]
        paths += [
            PARENT / cohort / name
            for name in (
                "predictions.json",
                "runtime_scores.json",
                "internal_metrics.ai.development.json",
                "internal_paired_querywise.json",
            )
        ]
    files = {relative(path): sha256_file(path) for path in paths}
    files.update(load(ROOT / "crop_asset_protocol.json")["files"])
    files.update(load(ROOT / "crop_asset_execution_protocol.json")["files"])
    verify_files(PROJECT_ROOT, files)
    once(
        ROOT / "protocol.json",
        dict(
            files=files,
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
            planned_full_rankings=600,
            previous_methods_preserved=12,
            full_original_models_reused=True,
            new_query_ROI="Frozen new query-only caption coarse subject guides full GroundingDINO; unresolved/failed localization retains whole image. Actual calls and fallback counts come from receipts.",
            gallery_ROI="Reuse the original 296 catalog ROI request identities and actual zoom pixels, or original whole-image fallback; no gallery redetection or relabeling.",
            crop_Qwen="Complete pinned Qwen3-VL-Reranker interaction on every fixed-union query/candidate crop pair under the frozen new description; no lowered model configuration, proxy or score imputation.",
            pair_protocol_timing="Separate pair protocol is frozen after actual crop_inputs and detection_receipt exist; neither is required before new DINO by this main protocol.",
            common_pool="Parent original20/new20 stable union, unchanged 504 query-candidate pairs",
            common_tail="Original cross20 relative order after filtering the shared union",
            paired_evaluation="Unchanged per-query shared-label completion bounds; identical orders give exact zero, U<=16 uses exact shared enumeration, only changed U>16 queries use conservative bounds; optional zero-positive Recall remains undefined and retained.",
            selection="Fixed primary and three crop-context methods, all 15 methods/groups/scopes reported. P60 is required, R85 preferred; six-metric nonregression against cross20/global/region_title must be established for a development proposal. No post-result tuning, query/label/K changes or automatic promotion.",
            independent_acceptance_started=False,
            independent_AI_blind_predictions=0,
            formal_metrics=None,
            default_strategy_changed=False,
        ),
    )
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=False,
            phase="frozen_before_new_subject_detection",
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
    evidence = load(DATA / cohort / "runtime_scores.json")
    output, details = {}, {}
    for query in load(DATA / cohort / "queries.public.json"):
        qid = query["id"]
        baseline = previous[qid]["cross20"]
        if query["image_path"]:
            old = load(DATA / cohort / "runs" / qid / "features.json")
            new = load(DATA / cohort / "runs" / qid / "caption_features.json")
            a = load(PARENT / cohort / "runs" / qid / "old_cross.json")
            b = load(PARENT / cohort / "runs" / qid / "new_cross.json")
            crop = load(ROOT / cohort / "runs" / qid / "crop_cross.json")
            assert a["pool"] == b["pool"] == crop["pool"]
            entry = evidence[qid]
            ids = old["ids"]
            assert new["ids"] == entry["ids"] == ids
            assert len(ids) == len(set(ids)) == 296
            whole = np.asarray(entry["whole_scores"]["fused_scores"])
            roi = np.asarray(entry["roi_scores"]["fused_scores"])
            auxiliary = 0.5 * normalized_ranks(ids, whole) + 0.5 * normalized_ranks(
                ids, roi
            )
            new_orders, detail = fuse_crops(
                CaptionRoiRankerFuseCropsInputs(
                    ids,
                    a["pool"],
                    a["logits"],
                    b["logits"],
                    crop["logits"],
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
        assert not set(previous[qid]) & set(new_orders)
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
        for name, value in (
            ("predictions.json", rankings),
            ("runtime_scores.json", details),
        ):
            path = ROOT / cohort / name
            once(path, value)
            files.append(path)
    once(
        ROOT / "prediction_receipt.json",
        dict(
            files={
                path.relative_to(ROOT).as_posix(): sha256_file(path) for path in files
            },
            full_rankings=600,
            methods=len(METHODS),
            gallery_per_cohort=296,
            previous_twelve_methods_unchanged=True,
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
    labels = read_rows(DATA / cohort / "judgments.ai.jsonl")
    modes = {
        q["id"]: "image" if q["image_path"] else "text"
        for q in load(DATA / cohort / "queries.public.json")
    }
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
        for name, value in (
            ("internal_metrics.ai.development.json", metrics),
            ("internal_paired_querywise.json", paired),
        ):
            path = ROOT / cohort / name
            once(path, value)
            files.append(path)
        print(dict(evaluated=cohort, role="development_only"), flush=True)
    once(
        ROOT / "evaluation_receipt.json",
        dict(
            files={
                path.relative_to(ROOT).as_posix(): sha256_file(path) for path in files
            },
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
    asset_checks = assets.verify()
    pair_checks = pair_cache.verify()
    verify_files(ROOT, load(ROOT / "evaluation_receipt.json")["files"])
    for cohort in COHORTS:
        ranks, details = cohort_orders(cohort)
        assert ranks == load(ROOT / cohort / "predictions.json")
        assert details == load(ROOT / cohort / "runtime_scores.json")
        previous = load(PARENT / cohort / "predictions.json")
        for qid, methods in ranks.items():
            assert set(methods) == set(METHODS)
            assert all(
                len(order) == len(set(order)) == 296 for order in methods.values()
            )
            assert all(methods[name] == previous[qid][name] for name in OLD_METHODS)
            if qid.startswith("text_"):
                assert all(methods[name] == methods["cross20"] for name in NEW_METHODS)
            else:
                pool = details[qid]["shared_pool"]
                tail = [cid for cid in methods["cross20"] if cid not in set(pool)]
                assert all(methods[name][len(pool) :] == tail for name in NEW_METHODS)
        metrics, paired = evaluated(cohort)
        assert metrics == load(ROOT / cohort / "internal_metrics.ai.development.json")
        assert paired == load(ROOT / cohort / "internal_paired_querywise.json")
        old = load(PARENT / cohort / "internal_metrics.ai.development.json")
        for scope in old["scopes"]:
            for name in OLD_METHODS:
                assert metrics["scopes"][scope][name] == old["scopes"][scope][name]
    once(
        ROOT / "verification.json",
        dict(
            passed=True,
            replayed_full_rankings=600,
            preserved_prior_full_rankings=480,
            unchanged_relations=11840,
            text_queries_exact=16,
            identical_common_tail=True,
            crop_asset_checks=asset_checks,
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
    return "–".join(
        "未定义" if value is None else f"{100 * value:.2f}" for value in bounds
    )


def viewer() -> None:
    """Apply the viewer workflow step.

    Notes:
        Writes derived artifacts through (ROOT /
        'internal_comparison.html').write_text, once.
    """
    rows = []
    crop_inputs = load(ROOT / "crop_inputs.json")["cohorts"]

    def url(path: Path) -> str:
        """Return url for this workflow.

        Args:
            path: Path of the input or output artifact used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        return quote(os.path.relpath(path, ROOT).replace("\\", "/"), safe="/:")

    for cohort in COHORTS:
        rankings = load(ROOT / cohort / "predictions.json")
        details = load(ROOT / cohort / "runtime_scores.json")
        old_details = load(PARENT / cohort / "runtime_scores.json")
        docs = {doc["id"]: doc for doc in load(DATA / cohort / "catalog.public.json")}
        profiles = {
            row["id"]: row["product_type"]
            for row in read_rows(DATA / cohort / "profiles.public.jsonl")
        }
        grades = {
            (row["query_id"], row["candidate_id"]): row["grade"]
            for row in read_rows(DATA / cohort / "judgments.ai.jsonl")
        }
        gallery_crops = {row["id"]: row for row in crop_inputs[cohort]["gallery"]}
        query_crops = {row["id"]: row for row in crop_inputs[cohort]["queries"]}
        for query in load(DATA / cohort / "queries.public.json"):
            qid = query["id"]
            feature = load(DATA / cohort / "runs" / qid / "features.json")
            raw = load(DATA / cohort / "runs" / qid / "cross.json")
            scores = {"cross20": baseline_scores(feature, raw)}
            if query["image_path"]:
                scores.update(old_details[qid]["scores"])
                scores.update(details[qid]["scores"])
            else:
                scores.update({name: scores["cross20"] for name in VIEW_METHODS})
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
                        crop=(
                            url(PROJECT_ROOT / gallery_crops[cid]["selected_image"])
                            if query["image_path"]
                            else None
                        ),
                        category=profiles[cid],
                        score=float(values[cid]),
                        AI_grade=grades[qid, cid],
                    )
                    for cid in rankings[qid][method][:20]
                ]
            description = query["text"]
            if query["image_path"]:
                description = (
                    "原描述："
                    + feature["description"]
                    + "；新主售描述："
                    + query_crops[qid]["new_description"]
                )
            rows.append(
                dict(
                    id=f"{cohort}/{qid}",
                    query=description,
                    image=url(query["image_path"]) if query["image_path"] else None,
                    crop=(
                        url(PROJECT_ROOT / query_crops[qid]["selected_image"])
                        if query["image_path"]
                        else None
                    ),
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
        + json.dumps({name: DISPLAY[name] for name in VIEW_METHODS}, ensure_ascii=False)
        + ";"
    )
    html, count = re.subn(r"const names=\{.*?\};", lambda _: names, html, count=1)
    assert count == 1
    html = re.sub(
        r"<title>.*?</title>",
        "<title>模块二新主体裁切完整Qwen开发对照</title>",
        html,
        count=1,
    )
    html = re.sub(
        r"<h1>.*?</h1>", "<h1>原Qwen、父ROI方法与整图裁切Qwen</h1>", html, count=1
    )
    html = html.replace(
        "card.append(im);let c=",
        "card.append(im);if(d.crop){let ci=document.createElement('img');ci.src=d.crop;ci.alt='实际模型裁切或整图回退';card.append(ci)}let c=",
        1,
    )
    once(ROOT / "viewer_data.json", rows)
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
    detector = load(ROOT / "detection_receipt.json")
    lines = [
        "# 新主体裁切完整Qwen同图库开发对照",
        "",
        "内部AI开发证据；40条查询与296件图库均已曝光，人工标签0，正式指标为空。11840条原关系不变，固定K=10，未知项仍为null。没有新标签或新独立验收。",
        "",
        "冻结新query-only描述的粗主体类型引导完整GroundingDINO重新定位查询。定位失败或主体类型未决时使用原图。图库使用原请求身份对应的真实ROI放大图或整图回退，不重建、不按标签选框。实际调用、成功和回退数量见检测收据。",
        "",
        "在父轮固定504候选对上，用完整原配置Qwen3-VL-Reranker实际计算查询与商品裁切交互，文本仍使用同一冻结新描述。三种新方法分别只用裁切CE、等权整图/裁切CE、再加入原FG区域辅助；同候选、同base、同cross20过滤后的公共尾部。主方法预先固定，所有结果一起报告，不拼最高数字。",
        "",
        "保留父12方法的480份排序，新增3方法共600份完整排序。16条文本查询逐条保留原cross20。旧方法单方法指标与父轮逐项相同；共享未知配对界是证据范围，不是置信区间，也不覆盖AI误标。",
        "",
        "[商品与实际裁切对照页](internal_comparison.html) · [检测收据](detection_receipt.json) · [完整裁切精排收据](reranker_receipt.json)",
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
                            *[
                                pct(methods[method]["ranges"][metric])
                                for metric in METRICS
                            ],
                        ]
                    )
                    + " |"
                )
            for method in NEW_METHODS:
                fields = paired["scopes"][scope][method]

                def nonregression(baselines: Sequence[str]) -> bool:
                    """Return nonregression for this workflow.

                    Args:
                        baselines: Baselines used by this operation.

                    Returns:
                        Whether the condition checked by this operation holds.
                    """
                    return all(
                        value["nonregression_proven_for_shared_unknown_labels"]
                        for baseline in baselines
                        for value in fields[baseline].values()
                    )

                gates[cohort][scope][method] = dict(
                    precision_60=methods[method]["precision_60_lower_bound_met"],
                    recall_85=methods[method]["recall_85_lower_bound_met"],
                    nonregression_against_cross20=nonregression(("cross20",)),
                    nonregression_against_feature_baselines=nonregression(
                        ("global", "region_title")
                    ),
                    nonregression_against_parent_dual_methods=nonregression(
                        ("dual_context_union", "dual_context_roi_union")
                    ),
                )
        lines += [
            "",
            "### 固定候选、上下文与辅助ROI消融",
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
            "完整图库正例数决定Recall分母。正例超过10时单查询Recall上限为10/R；父轮冻结诊断已说明总体/泛文本的85%受容量约束。不改分母、K或删除查询来达标；零正例查询保留，Recall未定义而非满分。裁切可能丢失用途、尺度与整体结构，不能预先保证质量提升。",
            "",
        ]
    primary = [
        group[PRIMARY] for cohorts in gates.values() for group in cohorts.values()
    ]
    decision = dict(
        primary_method=PRIMARY,
        scope_checks=gates,
        development_precision_target_met=all(row["precision_60"] for row in primary),
        development_recall_preference_met=all(row["recall_85"] for row in primary),
        development_six_metric_nonregression_against_cross20=all(
            row["nonregression_against_cross20"] for row in primary
        ),
        development_nonregression_against_feature_baselines=all(
            row["nonregression_against_feature_baselines"] for row in primary
        ),
        development_nonregression_against_parent_dual_methods=all(
            row["nonregression_against_parent_dual_methods"] for row in primary
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
        f"主方法Precision全部组别/范围下界通过60%：{decision['development_precision_target_met']}；Recall偏好通过：{decision['development_recall_preference_met']}；对cross20六指标不退化：{decision['development_six_metric_nonregression_against_cross20']}。默认方法未替换，开发结论不能作为正式验收。",
        "",
        "完整模型实算与身份一致缓存分别登记，不把缓存称为新调用；图库没有新DINO调用，嵌入和FG未重新编码。最终上线升级仍须另建全新查询与图库、冻结方案后再独立验证。页面仅做数据、图片引用和静态JS检查，未声称浏览器交互或在线时延验收完成。",
        "",
    ]
    (ROOT / "internal_comparison.ai.development.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    once(
        ROOT / "runtime_audit.json",
        dict(
            crop_reranker_receipt=calls,
            query_detector_receipt=detector,
            full_embedding_new_calls=0,
            FG_new_calls=0,
            gallery_DINO_new_calls=0,
            original_whole_contexts_reused=1008,
            original_text_contexts_untouched=320,
            full_rankings=600,
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
                for name in (
                    "decision.json",
                    "runtime_audit.json",
                    "viewer_data.json",
                    "internal_comparison.html",
                    "internal_comparison.ai.development.md",
                )
            }
        ),
    )
    print(
        {
            key: value
            for key, value in decision.items()
            if key.startswith("development_")
        },
        flush=True,
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
    assert rows == load(ROOT / "viewer_data.json") and len(rows) == 40
    cards = 0
    for row in rows:
        assert set(row["results"]) == set(VIEW_METHODS)
        images = [row[key] for key in ("image", "crop") if row.get(key)]
        for results in row["results"].values():
            assert len(results) == len({result["id"] for result in results}) == 20
            assert all(
                result["title"]
                and result["category"]
                and isinstance(result["score"], float)
                for result in results
            )
            images.extend(
                result[key]
                for result in results
                for key in ("image", "crop")
                if result.get(key)
            )
            cards += len(results)
        assert all((ROOT / unquote(path)).resolve().is_file() for path in images)
    assert cards == 2400
    scripts = [
        script
        for script in re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
        if script.lstrip().startswith("const rows=")
    ]
    assert len(scripts) == 1
    target = ROOT / "qa/viewer.js"
    target.parent.mkdir(exist_ok=True)
    target.write_text(scripts[0], encoding="utf-8")
    node = shutil.which("node") or "E:/Node.js/node.exe"
    result = subprocess.run(
        [node, "--check", str(target)], capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    once(
        ROOT / "checks.json",
        dict(
            passed=True,
            engineering=load(ROOT / "engineering_checks.pre.json"),
            full_rankings_replayed=600,
            prior_twelve_methods_unchanged=True,
            original_AI_grid_unchanged=11840,
            cards=cards,
            local_original_and_crop_images_exist=True,
            static_JS_syntax=True,
            browser_interaction_tested=False,
            online_latency_acceptance=False,
            formal_metrics=None,
        ),
    )


def seal() -> None:
    """Seal the completed outputs with their integrity metadata.

    Notes:
        Writes derived artifacts through once, target.parent.mkdir, write_json.
    """
    guard()
    assert load(ROOT / "checks.json")["passed"]
    parent_guard()
    exposure = load(ROOT / "exposure_receipt.json")
    verify_files(PROJECT_ROOT, exposure["files"])
    snapshot = load(ROOT / "post_experiment_exclusion_snapshot.json")
    verify_files(PROJECT_ROOT, snapshot["files"])
    release = ROOT / "source_release"
    paths = [PROJECT_ROOT / name for name in SOURCES]
    paths += sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py"))
    source_files = {}
    for source in paths:
        name = relative(source)
        target = release / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        assert sha256_file(target) == sha256_file(source)
        source_files[name] = sha256_file(source)
    once(release / "manifest.json", source_files)
    decision = load(ROOT / "decision.json")
    # The final public status is immutable evidence before completion is authoritative.
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=True,
            sealed=True,
            role="exposed_AI_caption_subject_crop_development",
            queries=40,
            image_queries=24,
            text_queries_unchanged=16,
            gallery_per_cohort=296,
            full_rankings=600,
            methods=15,
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
    files = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in sorted(ROOT.rglob("*"))
        if path.is_file() and path.name != "completion_receipt.json"
    }
    assert not any(ROOT.rglob("*.pending")), "Do not seal incomplete artifacts"
    once(
        ROOT / "completion_receipt.json",
        dict(
            files=files,
            source_files=source_files,
            project_files=snapshot["files"],
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
        ),
    )
    print(dict(sealed_files=len(files), source_snapshots=len(source_files)), flush=True)


def completed() -> bool:
    """Read-only sealed replay; never create or refresh artifacts after completion.

    Returns:
        Whether the condition checked by this operation holds.
    """
    if not (ROOT / "completion_receipt.json").exists():
        return False
    receipt = load(ROOT / "completion_receipt.json")
    verify_files(ROOT, receipt["files"])
    verify_files(PROJECT_ROOT, receipt["source_files"])
    verify_files(PROJECT_ROOT, receipt["project_files"])
    verify_protocol()
    status = load(ROOT / "public_status.json")
    assert status["completed"] and status["sealed"]
    parent_guard()
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

    path = ensure_output_path(PROJECT_ROOT / "tmp/module2_caption_roi.guard.lock")
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


def wait_status(phase: str) -> None:
    """Apply the wait status workflow step.

    Args:
        phase: Phase used by this operation.

    Notes:
        Writes derived artifacts through write_json.
    """
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=False,
            phase=phase,
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
        ),
    )
    print(dict(waiting_for_natural_GPU_release=True, phase=phase), flush=True)


def run_all(freeze_only: bool = False) -> None:
    """Return run all for this workflow.

    Args:
        freeze_only: Freeze only used by this operation.
    """
    if completed():
        print(dict(seal_verified=True), flush=True)
        return
    with exclusive_runner() as acquired:
        if not acquired:
            print(dict(another_runner_holds_lock=True), flush=True)
            return
        if completed():
            print(dict(seal_verified=True), flush=True)
            return
        prepare()
        if freeze_only:
            return
        # Main protocol exists before the first full DINO or Qwen crop call.
        guard()
        if not assets.detect():
            wait_status("subject_detection_waiting_for_natural_GPU_release")
            return
        assets.verify()
        pair_cache.prepare()
        guard()
        if not pair_cache.infer():
            wait_status("crop_Qwen_waiting_for_natural_GPU_release")
            return
        stages = (
            (prediction, "prediction_receipt.json"),
            (evaluation, "evaluation_receipt.json"),
            (verification, "verification.json"),
            (report, "report_receipt.json"),
            (checks, "checks.json"),
            (assets.exposure, "exposure_receipt.json"),
        )
        for action, output in stages:
            guard()
            if not (ROOT / output).exists():
                print(dict(stage=action.__name__, phase="running"), flush=True)
                action()
        seal()
        assert completed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "verify"])
    action = parser.parse_args().action
    if action == "verify":
        assert completed(), "A sealed completion is required for read-only verification"
        print(dict(seal_verified=True), flush=True)
    else:
        run_all(freeze_only=action == "prepare")
