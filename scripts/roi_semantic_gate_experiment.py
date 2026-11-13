"""Query-only semantic ROI gating and fixed original-head development comparison."""

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

import roi_semantic_gate_review as review
import run  # noqa: F401
from caption_roi_experiment import COMPARISONS as PARENT_COMPARISONS
from caption_roi_experiment import DISPLAY as PARENT_DISPLAY
from caption_roi_experiment import METHODS as OLD_METHODS
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_evaluation import METRICS
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)
from querywise_paired_bounds import paired_querywise
from report_primary_subjects import baseline_scores
from roi_semantic_gate_ranker import METHODS as NEW_METHODS
from roi_semantic_gate_ranker import (
    POLICY,
    PRIMARY,
    RoiSemanticGateRankerGatedOrdersInputs,
    gated_orders,
    text_orders,
)
from subject_roi_experiment import load

ROOT = PROJECT_ROOT / "artifacts/module2_roi_semantic_gate"
PARENT = PROJECT_ROOT / "artifacts/module2_caption_roi_development"
WHOLE_PARENT = PROJECT_ROOT / "artifacts/module2_dual_path_development"
DATA = PROJECT_ROOT / "artifacts/module2_primary_subjects"
COHORTS = ("tune", "regression", "fresh")
METHODS = (*OLD_METHODS, *NEW_METHODS)
VIEW_METHODS = ("cross20", "whole_crop_context_union", PRIMARY)
DISPLAY = {
    **PARENT_DISPLAY,
    "semantic_crop_gate": "查询主体语义门控",
    "semantic_crop_gate_head1": "语义门控加原首位保护",
    "unverified_crop_head1": "未核验裁切加原首位保护",
    "whole_context_head1": "整图双路加原首位保护",
}
SOURCES = [
    "scripts/roi_semantic_gate_experiment.py",
    "scripts/roi_semantic_gate_ranker.py",
    "scripts/roi_semantic_gate_review.py",
    "scripts/querywise_paired_bounds.py",
    "tests/test_roi_semantic_gate_ranker.py",
    "tests/test_roi_semantic_gate_review.py",
    "tests/test_querywise_paired_bounds.py",
]
REFERENCE_BASELINES = (
    "cross20",
    "global",
    "region_title",
    "dual_context_union",
    "whole_crop_context_union",
)
COMPARISONS = list(
    dict.fromkeys(
        [
            (candidate, baseline)
            for candidate in NEW_METHODS
            for baseline in REFERENCE_BASELINES
        ]
        + [
            ("semantic_crop_gate_head1", "semantic_crop_gate"),
            ("semantic_crop_gate_head1", "unverified_crop_head1"),
            ("semantic_crop_gate_head1", "whole_context_head1"),
            ("semantic_crop_gate", "unverified_crop_head1"),
            ("semantic_crop_gate", "whole_context_head1"),
            ("unverified_crop_head1", "whole_context_head1"),
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
    for root in (DATA, WHOLE_PARENT, PARENT):
        receipt = load(root / "completion_receipt.json")
        verify_files(root, receipt["files"])
        verify_files(PROJECT_ROOT, receipt["source_files"])
        verify_files(PROJECT_ROOT, receipt["project_files"])
        status = load(root / "public_status.json")
        assert status["sealed"] and status["completed"]
    snapshot = load(PARENT / "post_experiment_exclusion_snapshot.json")
    verify_files(PROJECT_ROOT, snapshot["files"])


def guard() -> None:
    """Resume only unchanged inputs and every existing receipt."""
    verify_protocol()
    review.guard()
    for path in sorted(ROOT.glob("*receipt.json")):
        receipt = load(path)
        if "files" in receipt:
            verify_files(ROOT, receipt["files"])
        for key in ("source_files", "project_files"):
            if key in receipt:
                verify_files(PROJECT_ROOT, receipt[key])
    if (ROOT / "gate_receipt.json").exists():
        review.verify()


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
    """Freeze metadata and query-only review inputs before model inference."""
    parent_guard()
    assert load(ROOT / "engineering_checks.pre.json")["passed"]
    review.prepare()
    if (ROOT / "protocol.json").exists():
        guard()
        return
    assert not (ROOT / "gate_receipt.json").exists()
    assert not any((ROOT / "native_reviews").rglob("*.json"))
    assert not any((ROOT / "native_attempts").rglob("*.json"))
    assert not any((ROOT / c / "predictions.json").exists() for c in COHORTS)
    paths = [PROJECT_ROOT / name for name in SOURCES]
    source_paths = sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py"))
    assert len(source_paths) == 68, "Keep the frozen 68-source project isolated"
    paths += source_paths
    paths += [
        ROOT / "engineering_checks.pre.json",
        ROOT / "gate_review_protocol.json",
        ROOT / "gate_review_plan.json",
        ROOT / "gate_review_pre_receipt.json",
        ROOT / "gate_model_manifest.json",
        PARENT / "completion_receipt.json",
        PARENT / "protocol.json",
        PARENT / "post_experiment_exclusion_snapshot.json",
        PARENT / "crop_inputs.json",
        PARENT / "verification.json",
        WHOLE_PARENT / "completion_receipt.json",
        WHOLE_PARENT / "protocol.json",
        WHOLE_PARENT / "pair_plan.json",
        DATA / "completion_receipt.json",
        DATA / "comparison_protocol.json",
        DATA / "feature_protocol.json",
        DATA / "query_caption_receipt.json",
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
        paths += [
            WHOLE_PARENT / cohort / name
            for name in (
                "predictions.json",
                "runtime_scores.json",
            )
        ]
    files = {relative(path): sha256_file(path) for path in paths}
    for name, checksum in load(ROOT / "gate_review_protocol.json")["files"].items():
        assert name not in files or files[name] == checksum
        files[name] = checksum
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
            new_AI_relevance_labels=0,
            planned_query_ROI_AI_judgments=21,
            planned_inherited_query_fallbacks=3,
            semantic_and_caption_origin="model; human labels remain zero",
            human_labels=0,
            planned_full_rankings=760,
            previous_methods_preserved=15,
            query_only_review="21 previously accepted query ROIs use original image, actual zoom and frozen new query caption only; no candidate documents, ranking, qrels or per-query rule exceptions. Three inherited whole-image fallbacks conservatively bypass crop without a model judgment.",
            engineering_failure="All 21 schema-valid complete non-truncated model judgments plus 3 inherited fallbacks are required before gate_results and prediction. API/JSON/truncation failures stop incomplete, never become a semantic rejection; retries preserve the exact frozen request.",
            semantic_uncertainty="A valid uncertain or reject decision selects stage24 plain dual whole context; accept requires every fixed semantic check true. This checks query ROI only and cannot establish candidate ROI correctness or business relevance.",
            common_model_bias="The query caption and semantic judge reuse the same pinned model weights/digest with separately frozen prompts/options; self-consistency is not independent semantic ground truth and can reproduce caption errors.",
            gate_route="Accept -> parent's whole_crop_context_union; otherwise -> stage24 dual_context_union, with no FG auxiliary added to the whole fallback.",
            factorial="No gate/no head is the unchanged parent whole_crop_context_union. Gate alone, gate+head1, unverified crop+head1 and plain whole+head1 isolate route and head effects without duplicating the original corner as a new method.",
            fixed_head="Protect only original cross20 first item within the unchanged union. The displayed priority is a ranking composite, not a Qwen logit or probability. No MRR or all-metric guarantee.",
            common_pool="Unchanged stage24 union: 504 query-candidate positions across 24 queries, not 504 unique products",
            common_tail="Original cross20 relative order after filtering the common union",
            new_retrieval_model_calls=0,
            reused_evidence="All original full reranker, embedding, DINO and FG evidence remains frozen; this stage only selects pre-existing complete rankings/scores after query-only semantic review.",
            paired_evaluation="Unchanged querywise shared-label completion bounds; identical order is exactly zero, U<=16 exact enumeration, changed U>16 conservative per-query bounds. Retain undefined zero-positive Recall and optional denominators.",
            selection="Fixed primary and all four new methods, all 19 methods/groups/scopes reported. P60 required, R85 preferred; six-metric nonregression against cross20/global/region_title required for a development proposal. No post-result tuning, query/label/K changes or automatic promotion.",
            exposure="Inherit every prior image, metadata, derived crop and overlay; zero new raw or derived pixels. Semantic review uses separate native images directly, without constructing contact sheets.",
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
            phase="frozen_before_query_only_semantic_review",
            queries=40,
            gallery_per_cohort=296,
            new_AI_relevance_labels=0,
            planned_query_ROI_AI_judgments=21,
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
    whole = load(WHOLE_PARENT / cohort / "runtime_scores.json")
    crop = load(PARENT / cohort / "runtime_scores.json")
    gates = load(ROOT / "gate_results.json")["cohorts"][cohort]
    ids = [doc["id"] for doc in load(DATA / cohort / "catalog.public.json")]
    assert len(ids) == len(set(ids)) == 296
    queries = load(DATA / cohort / "queries.public.json")
    assert set(gates) == {q["id"] for q in queries if q["image_path"]}
    output, details = {}, {}
    for query in queries:
        qid = query["id"]
        baseline = previous[qid]["cross20"]
        if query["image_path"]:
            gate = gates[qid]
            assert type(gate["accepted"]) is bool
            assert gate["decision"] in {
                "accept",
                "reject",
                "uncertain",
                "inherited_global_fallback",
            }
            assert gate["accepted"] == (gate["decision"] == "accept")
            pool = load(WHOLE_PARENT / cohort / "runs" / qid / "old_cross.json")["pool"]
            new_orders, detail = gated_orders(
                RoiSemanticGateRankerGatedOrdersInputs(
                    ids,
                    pool,
                    whole[qid]["scores"]["dual_context_union"],
                    crop[qid]["scores"]["whole_crop_context_union"],
                    baseline,
                    gate["accepted"],
                )
            )
            detail["query_review_evidence"] = {
                key: gate[key]
                for key in (
                    "accepted",
                    "decision",
                    "reason",
                    "review_file",
                    "review_sha256",
                )
            }
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
    review.verify()
    assert (ROOT / "gate_receipt.json").exists()
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
            full_rankings=760,
            methods=len(METHODS),
            gallery_per_cohort=296,
            previous_fifteen_methods_unchanged=True,
            text_queries_unchanged=16,
            labels_used_in_runtime=False,
            query_ROI_AI_judgments=21,
            new_AI_relevance_labels=0,
            new_reranker_or_embedding_calls=0,
            semantic_input_is_query_only=True,
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
    review_checks = review.verify()
    verify_files(ROOT, load(ROOT / "evaluation_receipt.json")["files"])
    gate_results = load(ROOT / "gate_results.json")
    assert gate_results["query_count"] == 24
    assert gate_results["model_review_requests"] == 21
    assert gate_results["automatic_fallbacks"] == 3
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
                chosen = (
                    "whole_crop_context_union"
                    if gate_results["cohorts"][cohort][qid]["accepted"]
                    else "dual_context_union"
                )
                assert methods["semantic_crop_gate"] == previous[qid][chosen]
                for name in (PRIMARY, "unverified_crop_head1", "whole_context_head1"):
                    assert methods[name][0] == methods["cross20"][0]
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
            replayed_full_rankings=760,
            preserved_prior_full_rankings=600,
            unchanged_relations=11840,
            query_ROI_AI_judgments=21,
            new_AI_relevance_labels=0,
            text_queries_exact=16,
            identical_common_tail=True,
            original_first_item_exact=True,
            first_item_protection_is_not_MRR_or_all_metric_guarantee=True,
            query_only_semantic_review_checks=review_checks,
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
    crop_inputs = load(PARENT / "crop_inputs.json")["cohorts"]
    gate_results = load(ROOT / "gate_results.json")["cohorts"]

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
                gate = gate_results[cohort][qid]
                description = (
                    "冻结查询描述："
                    + query_crops[qid]["new_description"]
                    + "；查询ROI决定："
                    + gate["decision"]
                    + "；理由："
                    + gate["reason"]
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
        "<title>模块二查询ROI语义门控开发对照</title>",
        html,
        count=1,
    )
    html = re.sub(
        r"<h1>.*?</h1>", "<h1>原Qwen、未核验裁切与语义门控首位保护</h1>", html, count=1
    )
    html = html.replace("排序分数：", "排序优先值（含固定首位保护）：")
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
    review_receipt = load(ROOT / "gate_receipt.json")
    gate_results = load(ROOT / "gate_results.json")
    lines = [
        "# 查询ROI语义门控与固定原首位开发对照",
        "",
        "内部AI开发证据；40条查询、两组296件图库已曝光。人工标签0，新增AI相关性标签0，原11840条关系及K=10不变，未知仍为null。另新增21条query ROI的AI语义判断，不能与相关性标签混为一类；描述和核验均为模型来源。正式指标为空，本轮未启动新的独立验收，历史独立AI盲测未通过结论保留。",
        "",
        "在模型调用前冻结query-only评审协议。21条原ROI仅输入查询原图、实际zoom和冻结新描述，不输入图库、候选排名或相关性标签；3条原全图回退不调用评审模型。所有21条原生模型输出必须完整、schema有效且非截断才进入评分；API/JSON/截断等工程失败保留尝试并中止，不能转为语义拒绝。有效uncertain或reject按固定规则回退整图路线。",
        "",
        "语义检查只核验查询ROI，不核验候选ROI或业务相关性。查询描述与judge使用同一模型，可能复现描述错误和自一致偏误；模型accept不等同于人工正确。所有判断原样保留，不按本轮成绩重评或挑选框。",
        "",
        "固定四个新方法：gate在accept时用父整图裁切Qwen，否则用stage24不含FG的plain双路整图；主方法在gate路线保护原cross20首位；另外用未核验裁切+原首位及plain整图+原首位作控制。无gate无head角就是父whole_crop_context_union，不重复称新算法。保护仅覆盖首位，不保证MRR或全部指标。展示数值是排序优先组合值，不是原始Qwen logit或相关概率。",
        "",
        "原15方法600份排序及指标逐项保留，四新方法共760份完整排序；16条文本逐条使用原cross20。所有新方法同504个查询—商品候选关系位置及相同公共尾部，没有新增精排、嵌入、DINO或FG推理，没有新原图或衍生像素。",
        "",
        "[原方法/裁切/主方法商品对照](internal_comparison.html) · [query-only评审收据](gate_receipt.json) · [原生决定](gate_results.json)",
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
                    nonregression_against_parent_routes=nonregression(
                        ("dual_context_union", "whole_crop_context_union")
                    ),
                )
        lines += [
            "",
            "### 门控、首位保护及整图路线对照",
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
            "共享未知标签差值是证据界，不是统计置信区间，也不覆盖AI误判。Recall分母保留全图库相关集合，零正例查询保留且Recall未定义，泛文本/总体85%容量约束不通过改K或删正例解决。旧排序、失败和局部退化全部保留。",
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
        development_nonregression_against_parent_routes=all(
            row["nonregression_against_parent_routes"] for row in primary
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
        "## 结论与证据",
        "",
        f"主方法Precision全部组别/范围下界通过60%：{decision['development_precision_target_met']}；Recall偏好通过：{decision['development_recall_preference_met']}；对cross20六指标不退化：{decision['development_six_metric_nonregression_against_cross20']}。原默认策略保留，开发结论不能作为独立验收成绩。",
        "",
        "实际评审调用、完整性、accept/reject/uncertain及回退数以原生收据为准。原15方法没有重新推理或改写，先前63张裁切/框选和全部历史曝光继续排除于未来未见验收；本轮不新增图片。页面仅做数据、图片引用与静态JS检查，不声称浏览器交互或在线时延验收完成。",
        "",
    ]
    (ROOT / "internal_comparison.ai.development.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    once(
        ROOT / "runtime_audit.json",
        dict(
            query_only_semantic_receipt=review_receipt,
            semantic_query_count=gate_results["query_count"],
            model_review_requests=gate_results["model_review_requests"],
            query_ROI_AI_judgments=gate_results["model_review_requests"],
            automatic_fallbacks=gate_results["automatic_fallbacks"],
            new_reranker_calls=0,
            new_embedding_calls=0,
            new_DINO_calls=0,
            new_FG_calls=0,
            new_raw_images=0,
            new_derived_images=0,
            full_rankings=760,
            AI_relations_unchanged=11840,
            new_AI_relevance_labels=0,
            human_labels=0,
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
            full_rankings_replayed=760,
            prior_fifteen_methods_unchanged=True,
            original_AI_grid_unchanged=11840,
            cards=cards,
            local_original_and_crop_images_exist=True,
            static_JS_syntax=True,
            browser_interaction_tested=False,
            online_latency_acceptance=False,
            formal_metrics=None,
        ),
    )


def exposure() -> None:
    """Apply the exposure workflow step.

    Notes:
        Writes derived artifacts through once.
    """
    guard()
    prior_path = PARENT / "post_experiment_exclusion_snapshot.json"
    prior = load(prior_path)
    verify_files(PROJECT_ROOT, prior["files"])
    snapshot = dict(prior)
    snapshot.update(
        parent_snapshot_sha256=sha256_file(prior_path),
        scope="Every prior exposure and all 63 parent caption-query derived images remain excluded; semantic review directly reuses native images and creates zero new raw/derived pixels.",
        new_semantic_gate_raw_images=0,
        new_semantic_gate_derived_images=0,
        new_semantic_gate_source_metadata=0,
        inherited_parent_caption_query_derived_images=63,
    )
    once(ROOT / "post_experiment_exclusion_snapshot.json", snapshot)
    once(
        ROOT / "exposure_receipt.json",
        dict(
            files={
                "post_experiment_exclusion_snapshot.json": sha256_file(
                    ROOT / "post_experiment_exclusion_snapshot.json"
                )
            },
            project_files=prior["files"],
            parent_snapshot_sha256=sha256_file(prior_path),
            inherited_parent_caption_query_derived_images=63,
            new_raw_images=0,
            new_derived_images=0,
            new_source_metadata_records=0,
            original_ids_and_family_keys_preserved=True,
            parent_pixels_and_dhash_registry_preserved=True,
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
    exposure_receipt = load(ROOT / "exposure_receipt.json")
    verify_files(ROOT, exposure_receipt["files"])
    verify_files(PROJECT_ROOT, exposure_receipt["project_files"])
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
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=True,
            sealed=True,
            role="exposed_AI_query_ROI_semantic_gate_development",
            queries=40,
            image_queries=24,
            text_queries_unchanged=16,
            gallery_per_cohort=296,
            full_rankings=760,
            methods=19,
            unchanged_AI_relations=11840,
            new_AI_relevance_labels=0,
            query_ROI_AI_judgments=21,
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
    assert not any(ROOT.rglob("*.pending")), "Do not seal incomplete artifacts"
    files = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in sorted(ROOT.rglob("*"))
        if path.is_file() and path.name != "completion_receipt.json"
    }
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

    path = ensure_output_path(PROJECT_ROOT / "tmp/module2_roi_semantic_gate.guard.lock")
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


def wait_status() -> None:
    """Apply the wait status workflow step.

    Notes:
        Writes derived artifacts through write_json.
    """
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=False,
            phase="query_semantic_review_incomplete",
            reason="Busy resource or incomplete engineering evidence; inspect native attempts and retry only the same frozen request. No prediction or quality evaluation yet.",
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
        ),
    )
    print(
        dict(query_semantic_review_incomplete=True, quality_evaluation_started=False),
        flush=True,
    )


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
        guard()
        if not review.infer():
            wait_status()
            return
        review.verify()
        assert (ROOT / "gate_receipt.json").exists()
        stages = (
            (prediction, "prediction_receipt.json"),
            (evaluation, "evaluation_receipt.json"),
            (verification, "verification.json"),
            (report, "report_receipt.json"),
            (checks, "checks.json"),
            (exposure, "exposure_receipt.json"),
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
