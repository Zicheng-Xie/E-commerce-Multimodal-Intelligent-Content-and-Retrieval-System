"""Internal diagnostics, mathematical capacity and public score-free progress."""

from __future__ import annotations

import json

import run  # noqa: F401
from advance_visual_coverage import PARENT, ROOT, load
from multimodal_ecommerce.ai_review_metrics import optional_mean_bounds
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from prepare_visual_coverage import REGISTRY, SNAPSHOT


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through (ROOT / 'internal_report.md').write_text,
        write_json.
    """
    metrics = load(ROOT / "internal_metrics.ai.development.json")
    receipt = load(ROOT / "ranking_receipt.json")
    capacity = metrics["query_capacity"]
    modes = {
        q["id"]: "text" if q["text"] else "image"
        for q in load(ROOT / "inputs.public.json")["queries"]
    }
    decisions: dict = {}
    ceilings = {}
    for scope, methods in metrics["scopes"].items():
        selected = [
            r for r in capacity if scope == "all" or modes[r["query_id"]] == scope
        ]
        mandatory, optional = [], []
        for row in selected:
            n, u = row["known_positive"], row["unknown"]
            if n:
                mandatory.append((min(10, n + u) / (n + u), min(10, n) / n))
            elif u:
                optional.append((min(10, u) / u, 1.0))
        ceilings[scope] = dict(
            precision_ceiling_range=metrics["capacity_by_scope"][scope][
                "precision_ceiling_range"
            ],
            best_possible_recall_range=list(optional_mean_bounds(mandatory, optional)),
            zero_positive_queries=[
                r["query_id"]
                for r in selected
                if r["known_positive"] + r["unknown"] == 0
            ],
        )
        decisions[scope] = {}
        for method, entry in methods.items():
            ranges = entry["ranges"]
            regression = {}
            for baseline in ["global", "region_title"]:
                regression[baseline] = {
                    name: (
                        "undefined"
                        if bounds[0] is None
                        or methods[baseline]["ranges"][name][0] is None
                        else (
                            "nonregression_proven_for_all_unknown_completions"
                            if bounds[0] >= methods[baseline]["ranges"][name][1] - 1e-12
                            else (
                                "regression_proven"
                                if bounds[1]
                                < methods[baseline]["ranges"][name][0] - 1e-12
                                else "overlapping_unresolved_ranges"
                            )
                        )
                    )
                    for name, bounds in ranges.items()
                }
            decisions[scope][method] = dict(
                precision_60_lower_bound_met=ranges["precision@10"][0] >= 0.6 - 1e-12,
                recall_85_lower_bound_met=ranges["recall@10"][0] is not None
                and ranges["recall@10"][0] >= 0.85 - 1e-12,
                same_gallery_nonregression=regression,
                promote_default=False,
            )
    write_json(
        ROOT / "internal_decisions.json",
        dict(
            role="AI_development_diagnostic_only",
            mathematical_capacity=ceilings,
            fixed_methods=decisions,
            quality_truth_independently_validated=False,
            default_promotion=False,
            formal_metrics=None,
            goal_met=False,
        ),
    )
    ranks = load(ROOT / "rankings.json")
    labels = read_rows(ROOT / "judgments.ai.jsonl")
    qrels = {(r["query_id"], r["candidate_id"]): r for r in labels}
    errors = []
    for qid, methods in ranks.items():
        for position, cid in enumerate(methods["cross_evidence_pool60"][:10], 1):
            row = qrels[qid, cid]
            if row["grade"] != 1:
                errors.append(
                    dict(
                        query_id=qid,
                        candidate_id=cid,
                        rank=position,
                        AI_grade=row["grade"],
                        evidence=row["evidence"],
                        reason=row["reason"],
                    )
                )
    write_json(
        ROOT / "internal_error_cases.json",
        dict(
            method="cross_evidence_pool60",
            role="evaluator_only_not_runtime_features",
            rows=errors,
        ),
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
        "# 383件扩容与固定策略对照：内部AI开发诊断",
        "",
        "仅供开发排错，禁止作为正式验收成绩对外引用。不是独立验证；补库与描述制作都已接触开发查询，AI判断不等于人工或独立业务真值。未知项保留；范围是未知标签所有取值的精确范围，不是置信区间。",
        "",
        "目标沿用Precision@10≥60%、Recall@10尽量≥85%；固定K=10。没有删查询、改相关性规则、缩小相关集合或更换验收集。",
        "",
        "## 同图库策略比较",
        "",
        "以下每个方法都使用383件图库和相同18条开发查询；与上轮344件的差异另由容量变化解释，不能混为算法收益。",
        "",
        "|方法|P@10|R@10|mAP|NDCG@10|MRR|Hit@1|",
        "|---|---|---|---|---|---|---|",
    ]
    names = ["precision@10", "recall@10", "ap", "ndcg@10", "mrr", "hit@1"]
    for method, entry in metrics["scopes"]["all"].items():
        lines.append(
            "|"
            + method
            + "|"
            + "|".join(percent(entry["ranges"][name]) for name in names)
            + "|"
        )
    lines += ["", "## 分模式与容量", "", "|范围|方法|P@10|R@10|", "|---|---|---|---|"]
    for scope in ["text", "image"]:
        for method in [
            "global",
            "region_title",
            "public_evidence_soft",
            "AI_oof_hard_negative_ranknet",
            "cross_evidence_pool60",
        ]:
            e = metrics["scopes"][scope][method]["ranges"]
            lines.append(
                f"|{scope}|{method}|{percent(e['precision@10'])}|{percent(e['recall@10'])}|"
            )
    lines += [
        "",
        "|范围|P@10理论上限范围|最优R@10范围|零正例查询|",
        "|---|---|---|---|",
    ]
    for scope, e in ceilings.items():
        lines.append(
            f"|{scope}|{percent(e['precision_ceiling_range'])}|{percent(e['best_possible_recall_range'])}|{','.join(e['zero_positive_queries']) or '无'}|"
        )
    lines += [
        "",
        "零正例查询的Recall未定义，单独列出；Precision仍以固定10为分母。正例超过10的查询有min(10,R)/R召回上限。增加真实相关商品可能扩大Recall分母；不能通过删除商品或改标签追逐85%。",
        "",
        "|查询|原344件已知正例/未知|现383件已知正例/未知|",
        "|---|---|---|",
    ]
    before = {
        r["query_id"]: r
        for r in load(PARENT / "internal_metrics.ai.development.json")["query_capacity"]
    }
    for row in capacity:
        old = before[row["query_id"]]
        lines.append(
            f"|{row['query_id']}|{old['known_positive']}/{old['unknown']}|{row['known_positive']}/{row['unknown']}|"
        )
    lines += [
        "",
        "## 已执行与后续",
        "",
        "下载41次、技术去重后40件全部逐图AI评审，39件可用，1件图文变体冲突暂缓。没有因为不相关而用更容易的候选替换。实际316次完整embedding调用，12种策略216组完整排序，交叉分组训练18个模型与2个候选。完整模型精排原始分数与缓存来源均可校验。",
        "",
        "厨房补库仍混入刀架、纸架等非参考主体，需要用明确的调料瓶架、多层、壁挂等公开商品属性重新准备开发候选，继续保留真实负例。古典低长收纳柜也仍缺覆盖。优先补实物覆盖，再看同图库的排序误差；不在开发库上追加参数搜索。",
        "",
        "保留原服饰同款结果，保持13个历史封存目录、330条验收登记和模型推理0。当前AI开发结果不能证明Precision/Recall正式达标；默认策略未升级。新CLI可实际进行新查询的60候选完整交叉编码精排，无历史查询排名查表。",
        "",
    ]
    (ROOT / "internal_report.md").write_text("\n".join(lines), encoding="utf-8")
    snapshot = load(SNAPSHOT)
    registry = read_rows(REGISTRY)
    signatures = {r["sha256"]: r for r in snapshot["image_fingerprints"]}
    for row in registry:
        if row.get("image_fingerprints"):
            signatures[row["image_fingerprints"]["sha256"]] = row["image_fingerprints"]
    snapshot["image_fingerprints"] = list(signatures.values())
    snapshot["files"][REGISTRY.relative_to(PROJECT_ROOT).as_posix()] = sha256_file(
        REGISTRY
    )
    snapshot["role"] = (
        "all_exposed_development_and_reserved_sources_for_future_exclusion"
    )
    snapshot["parent_snapshot_sha256"] = sha256_file(SNAPSHOT)
    write_json(ROOT / "post_development_exclusion_snapshot.json", snapshot)
    write_json(
        ROOT / "public_status.json",
        dict(
            state="visual_coverage_development_comparison_complete",
            updated_date="2026-10-23",
            development_catalog_products=383,
            new_products=39,
            AI_reviewed_new_products=40,
            QA_uncertain_not_indexed=1,
            development_queries=18,
            methods=12,
            full_rankings=216,
            index_views=2333,
            actual_new_embedding_calls=316,
            actual_new_cross_inferences=receipt["pair_score_origins"].get(
                "actual_full_cross_encoder_inference", 0
            ),
            cached_cross_pairs=receipt["pair_score_origins"].get(
                "identical_frozen_input_cache", 0
            ),
            AI_labels_only=True,
            human_labels=0,
            quality_metrics=None,
            formal_metrics=None,
            formal_acceptance=False,
            goal_met=False,
            acceptance_predictions=0,
            heldout_pixels_read_for_development=0,
            external_quality_claim_allowed=False,
            default_strategy_changed=False,
            original_index_retained=True,
            reserved_registry_unchanged=True,
            public_summary_contains_development_scores=False,
            internal_report="artifacts/module2_visual_coverage/internal_report.md",
        ),
    )
    print(
        json.dumps(
            dict(
                report_written=True,
                public_quality_metrics=None,
                formal_acceptance=False,
                acceptance_predictions=0,
            )
        )
    )


if __name__ == "__main__":
    main()
