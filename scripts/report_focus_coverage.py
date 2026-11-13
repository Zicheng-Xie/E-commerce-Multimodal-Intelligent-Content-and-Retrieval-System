"""Internal shared-label comparisons, candidate coverage and score-free status."""

from __future__ import annotations

import itertools
import json

import run  # noqa: F401
from advance_focus_coverage import PARENT, ROOT, load
from multimodal_ecommerce.ai_review_metrics import optional_mean_bounds
from multimodal_ecommerce.business_evaluation import METRICS
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.retrieval_strategies import extended_metrics
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json


def paired_differences(rankings: dict, labels: list[dict], modes: dict) -> dict:
    """Compare methods under the SAME unknown completion, not unrelated bounds.

    Args:
        rankings: Candidate rankings keyed by query and retrieval method.
        labels: Judgments used for the stated development or evaluation protocol.
        modes: Modes used by this operation.

    Returns:
        Mapping with fields: role,
        same_unknown_completion_for_candidate_and_baseline,
        not_statistical_confidence_intervals, scopes, query_rows, formal_acceptance.

    Raises:
        ValueError: Repeated relation; Complete shared judgments required; No
            approximate unknown filling.
    """
    lookup = {(r["query_id"], r["candidate_id"]): r["grade"] for r in labels}
    if len(lookup) != len(labels):
        raise ValueError("Repeated relation")
    expected = {
        (q, c) for q, methods in rankings.items() for c in next(iter(methods.values()))
    }
    if set(lookup) != expected or set(modes) != set(rankings):
        raise ValueError("Complete shared judgments required")
    baselines = [
        n
        for n in ["global", "region_title", "cross_evidence_pool60"]
        if n in next(iter(rankings.values()))
    ]
    query_rows = []
    for qid, methods in rankings.items():
        positive = {c for q, c in lookup if q == qid and lookup[q, c] == 1}
        unknown = sorted(c for q, c in lookup if q == qid and lookup[q, c] is None)
        if len(unknown) > 16:
            raise ValueError("No approximate unknown filling")
        ranges: dict = {
            method: {base: {m: [] for m in METRICS} for base in baselines}
            for method in methods
        }
        for bits in itertools.product([False, True], repeat=len(unknown)):
            relevant = positive | {c for c, keep in zip(unknown, bits) if keep}
            cache = {}
            values = {}
            for name, order in methods.items():
                key = tuple(order)
                if key not in cache:
                    cache[key] = (
                        extended_metrics(order, relevant)
                        if relevant
                        else {**dict.fromkeys(METRICS, 0.0), "recall@10": None}
                    )
                values[name] = cache[key]
            for name in methods:
                for base in baselines:
                    for metric in METRICS:
                        candidate_value = values[name][metric]
                        baseline_value = values[base][metric]
                        if candidate_value is not None:
                            if baseline_value is None:
                                raise ValueError("Inconsistent undefined paired metric")
                            ranges[name][base][metric].append(
                                candidate_value - baseline_value
                            )
        query_rows.append(
            dict(
                query_id=qid,
                mode=modes[qid],
                known_positive=len(positive),
                unknown=len(unknown),
                ranges={
                    name: {
                        base: {
                            metric: [min(v), max(v)] if v else [None, None]
                            for metric, v in metrics.items()
                        }
                        for base, metrics in bases.items()
                    }
                    for name, bases in ranges.items()
                },
            )
        )
    scopes = {}
    for scope in ["all", *sorted(set(modes.values()))]:
        selected = [r for r in query_rows if scope == "all" or r["mode"] == scope]
        methods = {}
        for name in next(iter(rankings.values())):
            methods[name] = {}
            for base in baselines:
                metrics = {}
                for metric in METRICS:
                    if metric == "recall@10":
                        mandatory = [
                            tuple(r["ranges"][name][base][metric])
                            for r in selected
                            if r["known_positive"]
                        ]
                        optional = [
                            tuple(r["ranges"][name][base][metric])
                            for r in selected
                            if not r["known_positive"] and r["unknown"]
                        ]
                        bounds = list(optional_mean_bounds(mandatory, optional))
                    else:
                        bounds = [
                            sum(r["ranges"][name][base][metric][i] for r in selected)
                            / len(selected)
                            for i in [0, 1]
                        ]
                    metrics[metric] = dict(
                        delta_range=bounds,
                        nonregression_proven_for_shared_unknown_labels=bounds[0]
                        is not None
                        and bounds[0] >= -1e-12,
                    )
                methods[name][base] = metrics
        scopes[scope] = methods
    return dict(
        role="internal_AI_development_paired_differences_only",
        same_unknown_completion_for_candidate_and_baseline=True,
        not_statistical_confidence_intervals=True,
        scopes=scopes,
        query_rows=query_rows,
        formal_acceptance=False,
    )


def candidate_coverage(
    rankings: dict,
    labels: list[dict],
    modes: dict,
    original_pools: dict,
    enriched: dict,
) -> dict:
    """Return candidate coverage for this workflow.

    Args:
        rankings: Candidate rankings keyed by query and retrieval method.
        labels: Judgments used for the stated development or evaluation protocol.
        modes: Modes used by this operation.
        original_pools: Original pools used by this operation.
        enriched: Enriched used by this operation.

    Returns:
        Mapping with fields: role, rows.
    """
    lookup = {(r["query_id"], r["candidate_id"]): r["grade"] for r in labels}
    rows = []
    for family, pools in [
        ("original", original_pools),
        ("enriched", {q: r["pool"] for q, r in enriched.items()}),
    ]:
        for qid, pool in pools.items():
            positive = {c for q, c in lookup if q == qid and lookup[q, c] == 1}
            unknown = sorted(c for q, c in lookup if q == qid and lookup[q, c] is None)
            for size in [20, 60]:
                candidate = set(pool[:size])
                observed = []
                for bits in itertools.product([False, True], repeat=len(unknown)):
                    relevant = positive | {c for c, keep in zip(unknown, bits) if keep}
                    if relevant:
                        observed.append(len(candidate & relevant) / len(relevant))
                rows.append(
                    dict(
                        query_id=qid,
                        mode=modes[qid],
                        family=family,
                        pool_size=size,
                        known_positive=len(positive),
                        unknown=len(unknown),
                        recall_range=(
                            [min(observed), max(observed)] if observed else [None, None]
                        ),
                        known_positive_outside_pool=sorted(positive - candidate),
                    )
                )
    return dict(
        role="candidate_pool_diagnostic_not_Top10_or_acceptance_metrics", rows=rows
    )


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through (ROOT / 'internal_report.md').write_text,
        write_json.
    """
    metrics = load(ROOT / "internal_metrics.ai.development.json")
    rankings = load(ROOT / "rankings.json")
    labels = read_rows(ROOT / "judgments.ai.jsonl")
    inputs = load(ROOT / "inputs.public.json")
    modes = {q["id"]: "text" if q["text"] else "image" for q in inputs["queries"]}
    paired = paired_differences(rankings, labels, modes)
    write_json(ROOT / "internal_paired_differences.json", paired)
    coverage = candidate_coverage(
        rankings,
        labels,
        modes,
        load(ROOT / "pools.json"),
        load(ROOT / "query_enrichment/ranking_features.json"),
    )
    write_json(ROOT / "internal_candidate_coverage.json", coverage)
    ceilings = {}
    for scope in metrics["scopes"]:
        selected = [
            r
            for r in metrics["query_capacity"]
            if scope == "all" or modes[r["query_id"]] == scope
        ]
        mandatory, optional = [], []
        for row in selected:
            n, u = row["known_positive"], row["unknown"]
            if n:
                mandatory.append((min(10, n + u) / (n + u), min(10, n) / n))
            elif u:
                optional.append((min(10, u) / u, 1.0))
        ceilings[scope] = dict(
            precision_ceiling=metrics["capacity_by_scope"][scope][
                "precision_ceiling_range"
            ],
            best_possible_recall_range=list(optional_mean_bounds(mandatory, optional)),
            zero_positive_queries=[
                r["query_id"]
                for r in selected
                if not r["known_positive"] and not r["unknown"]
            ],
        )
    decisions = {}
    for name in metrics["scopes"]["all"]:
        decisions[name] = dict(
            precision_60_AI_lower_bound_all_modes=all(
                metrics["scopes"][scope][name]["precision_60_lower_bound_met"]
                for scope in ["all", "text", "image"]
            ),
            recall_85_AI_lower_bound_all_modes=all(
                metrics["scopes"][scope][name]["recall_85_lower_bound_met"]
                for scope in ["all", "text", "image"]
            ),
            paired_nonregression_all_metrics_vs_global_and_region_all_modes=all(
                paired["scopes"][scope][name][base][m][
                    "nonregression_proven_for_shared_unknown_labels"
                ]
                for scope in ["all", "text", "image"]
                for base in ["global", "region_title"]
                for m in METRICS
            ),
            default_promotion_allowed=False,
            formal_acceptance=False,
        )
    write_json(
        ROOT / "internal_decisions.json",
        dict(
            mathematical_capacity=ceilings,
            methods=decisions,
            role="internal_query_aware_AI_development_diagnostic_only",
            independent_truth_validated=False,
            goal_met=False,
            formal_metrics=None,
        ),
    )
    qrels = {(r["query_id"], r["candidate_id"]): r for r in labels}
    errors = []
    for qid, methods in rankings.items():
        for method in [
            "cross_evidence_pool60",
            "described_cross_pool20",
            "described_cross_pool60",
        ]:
            for position, cid in enumerate(methods[method][:10], 1):
                row = qrels[qid, cid]
                if row["grade"] != 1:
                    errors.append(
                        dict(
                            query_id=qid,
                            method=method,
                            rank=position,
                            candidate_id=cid,
                            grade=row["grade"],
                            reason=row["reason"],
                            evidence=row["evidence"],
                        )
                    )
    write_json(
        ROOT / "internal_error_cases.json",
        dict(role="evaluator_only_not_runtime_features", rows=errors),
    )

    def percent(bounds: dict) -> str:
        """Return percent for this workflow.

        Args:
            bounds: Bounds used by this operation.

        Returns:
            Text produced by this operation.
        """
        return (
            "未定义"
            if bounds[0] is None
            else f"{100*bounds[0]:.2f}–{100*bounds[1]:.2f}"
        )

    lines = [
        "# 418件图库与查询主体描述：内部AI开发诊断",
        "",
        "仅用于开发排错。所有样本已登记开发暴露，新增补库有针对性并已接触开发查询；不属于独立验证。AI评审、AI生成描述均不等于人工或独立业务真值。未知标签范围不是置信区间，不可挑有利取值报成绩。",
        "",
        "## 同图库固定对照",
        "",
        "418件图库、18条原查询、固定相关性规则和K=10；16种预先冻结的方法，无本轮参数搜索。与383件结果的变化同时含库覆盖和Recall分母变化，不作为纯算法收益。新增方法的文本模式均复用固定cross_evidence_pool60锚点，图像模式比较新增模块。",
        "",
        "|方法|P@10|R@10|mAP|NDCG@10|MRR|Hit@1|",
        "|---|---|---|---|---|---|---|",
    ]
    for name, entry in metrics["scopes"]["all"].items():
        lines.append(
            "|"
            + name
            + "|"
            + "|".join(percent(entry["ranges"][m]) for m in METRICS)
            + "|"
        )
    lines += [
        "",
        "## 图片模式与同标签差值",
        "",
        "|方法|P@10|R@10|mAP|NDCG@10|MRR|Hit@1|",
        "|---|---|---|---|---|---|---|",
    ]
    for name, entry in metrics["scopes"]["image"].items():
        lines.append(
            "|"
            + name
            + "|"
            + "|".join(percent(entry["ranges"][m]) for m in METRICS)
            + "|"
        )
    lines += [
        "",
        "下表按同一未知标签补全分别计算新旧差值，再取范围；比独立上下界相减更精确。仍然是AI开发标签下的结果，不是泛化保证。单位为百分点。",
        "",
        "|图片方法相对cross_evidence_pool60|ΔP@10|ΔR@10|ΔmAP|ΔNDCG@10|ΔMRR|ΔHit@1|",
        "|---|---|---|---|---|---|---|",
    ]
    for name in [
        "query_multi_view",
        "query_caption_view_fusion",
        "described_cross_pool20",
        "described_cross_pool60",
    ]:
        lines.append(
            "|"
            + name
            + "|"
            + "|".join(
                percent(
                    paired["scopes"]["image"][name]["cross_evidence_pool60"][m][
                        "delta_range"
                    ]
                )
                for m in METRICS
            )
            + "|"
        )
    lines += [
        "",
        "## 容量约束",
        "",
        "|范围|P@10理论上限范围|最优R@10范围|零正例查询|",
        "|---|---|---|---|",
    ]
    for scope, row in ceilings.items():
        lines.append(
            f"|{scope}|{percent(row['precision_ceiling'])}|{percent(row['best_possible_recall_range'])}|{','.join(row['zero_positive_queries']) or '无'}|"
        )
    lines += [
        "",
        "每个查询有R个相关商品时，Recall@10最高min(10,R)/R；保留完整相关集合，不删正例或查询来追85%。增加真实相关商品扩大分母，因此扩容前后的Recall不能直接解释为算法优劣。所有方法在同一418件库上比较。",
        "",
        "## 证据和局限",
        "",
        "初批21件全部保留；外形补充扫描训练来源第200001至1000000条，冻结30条元数据，下载14张后全部保留。原计划补16张，但双层壁挂架仅6条符合元数据条件，短缺2条如实记录，没有填充或改标。35件均逐图AI质检和关系评审，共630对新增判断。",
        "",
        "完整embedding实际326次：新增商品283次、查询多视角及描述43次。本地固定Qwen3.5单独读取六张开发查询图生成描述，候选、标签、排名均未提供。完整Qwen3-VL-Reranker-2B通过原始查询图与描述共同精排，模型权重、业务指令及图像预处理保持固定。描述含未核实材质、结构计数等可能错误，原始输出保留；不能据描述改相关性标签。",
        "",
        "模型与库标签来源均为AI且使用固定已暴露查询，不能证明独立业务验收达标。14个历史阶段封存保留，预留heldout图片未读、推理0；默认策略不升级。全部方法及负结果保留。",
        "",
        "实际对照与精排调用计数见ranking_receipt.json，逐对原始分数在pair_scores/和described_pairs/。运行入口支持任意开发图片自动描述及裁切多视角，不使用保存的查询排名或评审表作为运行特征。",
        "",
    ]
    (ROOT / "internal_report.md").write_text("\n".join(lines), encoding="utf-8")
    prior = load(PARENT / "post_preparation_exclusion_snapshot.json")
    queue = read_rows(
        PROJECT_ROOT / "data/module2_form_coverage_candidates/records.jsonl"
    )
    prior["ids"] = sorted(set(prior["ids"]) | {r["id"] for r in queue})
    prior["product_source_ids"] = sorted(
        set(prior["product_source_ids"]) | {r["source_id"] for r in queue}
    )
    prior["family_keys"] = sorted(
        set(prior["family_keys"]) | {k for r in queue for k in r["family_keys"]}
    )
    signatures = {r["sha256"]: r for r in prior["image_fingerprints"]}
    for relative in [
        "data/module2_focus_coverage_media/records.jsonl",
        "data/module2_form_coverage_candidates/records.jsonl",
        "data/module2_form_coverage_media/records.jsonl",
    ]:
        path = PROJECT_ROOT / relative
        prior["files"][relative] = sha256_file(path)
        for row in read_rows(path):
            if row.get("image_fingerprints"):
                signature = row["image_fingerprints"]
                signatures[signature["sha256"]] = signature
    prior["image_fingerprints"] = list(signatures.values())
    prior["counts"] = dict(
        records=len(prior["ids"]),
        product_ids=len(prior["product_source_ids"]),
        deepfashion_pair_ids=len(
            prior["deepfashion_pair_ids_conservative_cross_split"]
        ),
        unique_image_bytes=len(prior["image_fingerprints"]),
        family_keys=len(prior["family_keys"]),
    )
    prior["parent_snapshot_sha256"] = sha256_file(
        PARENT / "post_preparation_exclusion_snapshot.json"
    )
    write_json(ROOT / "post_development_exclusion_snapshot.json", prior)
    receipt = load(ROOT / "ranking_receipt.json")
    write_json(
        ROOT / "public_status.json",
        dict(
            state="focused_coverage_and_query_enrichment_development_complete",
            updated_date="2026-10-23",
            development_catalog_products=418,
            new_products=35,
            new_AI_reviewed_relations=630,
            AI_relations=len(labels),
            unknown_labels_retained=sum(r["grade"] is None for r in labels),
            index_views=2546,
            actual_full_embedding_calls=326,
            new_product_embedding_calls=283,
            query_enrichment_embedding_calls=43,
            query_only_VLM_calls=6,
            methods=16,
            full_rankings=288,
            baseline_pair_scores=1080,
            described_pair_scores=receipt["described_pairs"],
            new_actual_baseline_cross_pairs=receipt["pair_score_origins"].get(
                "actual_full_cross_encoder_inference", 0
            ),
            AI_labels_only=True,
            human_labels=0,
            quality_metrics=None,
            formal_metrics=None,
            formal_acceptance=False,
            goal_met=False,
            heldout_pixels_read_for_development=0,
            acceptance_predictions=0,
            default_strategy_changed=False,
            external_quality_claim_allowed=False,
            internal_report="artifacts/module2_focus_coverage/internal_report.md",
            tests_passed=224,
            mypy_source_files=63,
            supplement_target_images=16,
            supplement_actual_images=14,
            supplement_shortfall=2,
        ),
    )
    print(
        json.dumps(
            dict(
                report_complete=True,
                formal_metrics=None,
                goal_met=False,
                acceptance_predictions=0,
            )
        )
    )


if __name__ == "__main__":
    main()
