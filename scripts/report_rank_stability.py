"""Keep numerical diagnostics internal and emit a score-free progress record."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import run  # noqa: F401
from advance_rank_stability import CANDIDATE, NAME, ROOT
from multimodal_ecommerce.safety import write_json


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def table(metrics: dict, methods: list[str], scope: str) -> list[str]:
    """Return table for this workflow.

    Args:
        metrics: Metrics used by this operation.
        methods: Methods used by this operation.
        scope: Scope used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    lines = [
        "|策略|P@10|R@10|mAP|NDCG@10|MRR|Hit@1|",
        "|---|---|---|---|---|---|---|",
    ]
    for name in methods:
        values = metrics["scopes"][scope][name]["ranges"]
        cells = []
        for metric in ["precision@10", "recall@10", "ap", "ndcg@10", "mrr", "hit@1"]:
            low, high = values[metric]
            cells.append(
                "未定义" if low is None else f"{100 * low:.2f}–{100 * high:.2f}"
            )
        lines.append("|" + "|".join([name, *cells]) + "|")
    return lines


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through (ROOT / 'internal_report.md').write_text,
        write_json.
    """
    verification = load(ROOT / "verification.json")
    assert verification["verified"]
    fresh = ROOT / "fresh_query_check"
    original = load(ROOT / "internal_metrics.ai.development.json")
    current = load(fresh / "internal_metrics.ai.development.json")
    effect = load(ROOT / "internal_tail_effect.json")
    image = [r for r in effect["query_rows"] if r["mode"] == "image"]
    delta = [
        sum(r["delta_ranges"]["ap"][i] for r in image) / len(image) for i in [0, 1]
    ]
    lines = [
        "# 内部开发诊断：有限重排与新查询压力检查",
        "",
        "本文件含内部开发质量数字，不能对外作为验收成绩。全部评审来自AI，同一评审者参与过旧图库；固定图库418件已开发曝光。新查询属于不同商品族和新原图，但本轮不构成正式heldout验收或独立标注验证。",
        "",
        "## 旧18条查询：单一尾部方案",
        "",
        "先冻结规则：图片保留region_top2_described_cross_pool20的前10位，其他商品按原region_title相对顺序补全；文本沿用cross_evidence_pool60。K=10取自既有指标定义，未追加参数搜索、修改标签或新增模型推理。",
        "",
        *table(original, ["global", "region_title", CANDIDATE, NAME], "image"),
        "",
        f"前10位完全一致，P/R/NDCG/Hit范围一致。相同未知标签补全下，图片mAP配对差值为{delta[0] * 100:+.4f}至{delta[1] * 100:+.4f}个百分点，不能保证提高。旧18查询MRR完全相同；此事实不能推广成算法的无条件MRR保证。",
        "",
        "## 新6条图片查询：冻结后执行",
        "",
        "训练来源第1000001至1100000条按固定词面规则和种子抽18条元数据，技术可解码且不重复的每桶首张保留，共6张。未依据相关性、正例数量或检索结果换图；发现电钢琴、游戏充电复合机和枕套歧义均保留。",
        "",
        "先看6张查询图及全部418张图库图（12页网格），冻结2508对AI关系，再执行完整本地模型。8项未知保持未知，没有由预测结果反推标签。",
        "",
        *table(
            current,
            ["global", "region_title", "described_cross_pool20", CANDIDATE, NAME],
            "image",
        ),
        "",
        "### 图库容量与召回分母",
        "",
        "|查询|已知正例|未知|状态|",
        "|---|---|---|---|",
    ]
    for row in current["query_capacity"]:
        lines.append(
            f"|{row['query_id']}|{row['known_positive']}|{row['unknown']}|{row['recall_status']}|"
        )
    low, high = current["capacity_by_scope"]["image"]["precision_ceiling_range"]
    lines += [
        "",
        f"本组P@10的数学平均容量为{100 * low:.2f}–{100 * high:.2f}%，即使把所有未知都当正例也达不到60%。两条确认零正例查询保留在Precision、mAP、NDCG、MRR、Hit分母中，其Recall无定义；枕头查询仅有未知，Recall分母随标签补全变化，使用精确范围。不能把零正例伪造为Recall=0或删掉后另称总体。",
        "",
        "新查询上，区域前两位保护与尾部恢复仍有排序退化；尾部恢复甚至使MRR小幅下降。原描述精排、保护版本、恢复版本及全部负结果同时保留，不升级默认。现有原查询图经过针对性补库，不能将其成绩外推到任意新图片。",
        "",
        "## 实跑与复核",
        "",
        f"新增查询主体描述6次，完整嵌入编码{verification['full_embedding_calls']}次，完整图文精排120对。新查询5方法30组完整排序；旧查询新增18组排序，其他4方法仅作为原结果引用。新请求入口复用了本次实际请求的20对logit，未新增推理、未查历史评测查询或标签。",
        "",
        "15个历史封存根目录、来源元数据筛选、全部AI关系、实际向量与logit排序、预留隔离及新查询输出均已重放核验。冻结采集脚本在下载后生成网格时将catalog键误写成documents；原脚本保留，用单独finalizer验证已下载内容并补齐网格，未重抽样。修复证据见fresh_query_check/acquisition_repair.json。",
        "",
        "下一步优先补齐业务主体与外形覆盖，覆盖审计使用真实正例集合；不为满足指标限制正例数或放宽相关性。新6查询现已曝光，此后属于开发回归，不能再当盲测。正式heldout继续封存，只有最终版本与AI盲评协议冻结后才一次性验证。",
    ]
    (ROOT / "internal_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        ROOT / "public_status.json",
        dict(
            stage="bounded_reranking_and_new_query_development_check",
            updated_date="2026-10-23",
            scope="module2_only",
            catalog_products=418,
            original_queries=18,
            added_method_variants=1,
            added_original_query_full_rankings=18,
            fresh_development_queries=6,
            fresh_full_rankings=30,
            fresh_AI_relations=2508,
            fresh_unknown_relations=8,
            fresh_query_only_VLM_calls=6,
            fresh_full_embedding_calls=verification["full_embedding_calls"],
            fresh_full_cross_pairs=120,
            native_AI_review_complete=True,
            human_labels=0,
            all_difficult_queries_retained=True,
            prior_sealed_roots_verified=15,
            tests_passed=233,
            mypy_source_files=65,
            replay_verified=True,
            default_strategy_changed=False,
            default_upgrade_gate="not_passed_new_query_regressions_and_catalog_coverage_limits",
            external_quality_metrics_allowed=False,
            quality_metrics=None,
            formal_metrics=None,
            formal_acceptance=False,
            goal_met=False,
            heldout_sealed=True,
            acceptance_predictions=0,
            next_work="Prioritize business-subject and visible-form development catalog coverage; freeze future source sampling before retrieval and preserve all difficult cases. Final independent heldout remains unopened.",
        ),
    )
    print(
        json.dumps(
            dict(
                report_written=True,
                public_quality_metrics=None,
                formal_acceptance=False,
            )
        )
    )


if __name__ == "__main__":
    main()
