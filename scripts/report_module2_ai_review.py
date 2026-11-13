"""Render the existing AI development evidence; no quality numbers are invented."""

from __future__ import annotations

import json

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.safety import PROJECT_ROOT

ROOT = PROJECT_ROOT / "artifacts/module2_ai_review"


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through (ROOT / 'internal_report.md').write_text.
    """
    metrics = json.loads(
        (ROOT / "metrics.ai.development.json").read_text(encoding="utf-8")
    )
    capacity = json.loads((ROOT / "capacity.json").read_text(encoding="utf-8"))

    def interval(values: np.ndarray) -> str:
        """Return interval for this workflow.

        Args:
            values: Values used by this operation.

        Returns:
            Text produced by this operation.
        """
        if values[0] is None:
            return "未定义"
        return (
            f"{100 * values[0]:.2f}%"
            if abs(values[0] - values[1]) < 1e-12
            else f"{100 * values[0]:.2f}%–{100 * values[1]:.2f}%"
        )

    lines = [
        "# 模块二：全量AI评审与排序重训内部记录",
        "",
        "本轮由用户明确委托AI承担全部评审。以下均为AI评审下的开发结果，不是独立人工真值或正式验收成绩。评审不再等待真人，历史A/B真人模板仍为空。",
        "",
        "直接查看原图库286件和新增64件的图片及原标题，保留44张可核验网格图及逐件依据；生成完整6,295对关系：相关154、不相关6,122、未知19。原图库5,143对全部覆盖。59个历史冲突关系中，16对相关、42对不相关、1对未知。",
        "",
        "新增商品AI质检58件可用、3件剔除、3件未知。剔除相机专用电池、无袖背心、散装枕芯填料；保留连接方式冲突的游戏耳机、影视道具键盘及POLO/T恤边界三项未知。可用开发清单344件已生成，当前检索索引仍为原286件，尚无扩容后的实际检索分数。",
        "",
        "本地Qwen3.5完成79件自动图文评审后，发现多需求输出把背景道具、模特鞋及配件判作主体；停止该批，原始输出全部保留。最终全量裁决由Codex直接看图完成，不能将两种AI过程称为独立人工双评。Codex底层精确服务模型ID未向本任务公开，不编造版本标识。",
        "",
        "## 数据容量与目标",
        "",
        "上限是假设理想排序时的容量，并非模型成绩。Top10固定除以10；相关性标准及18条查询未改变。",
        "",
        "| 开发图库 | 总体P@10上限 | 文本上限 | 图片上限 |",
        "|---|---:|---:|---:|",
    ]
    for name in ["original286", "reviewed344"]:
        row = capacity[name]["by_scope"]
        lines.append(
            f"| {name} | {interval(row['all'])} | {interval(row['text'])} | {interval(row['image'])} |"
        )
    lines += [
        "",
        "原图库即使未知项全部按相关计算，P@10上限也仅45%，无法通过60%目标。344件可用清单的总体容量具备超过60%的可能，但图片模式上限仍不超过30%。image_03调料架、image_06气门帽在自身排除后均无相关候选；image_01只有3项未知、0项确定相关。需要继续补充开发覆盖，不能改分母或把未知改正例。",
        "",
        "## 相同原图库上的实际排序对照",
        "",
        "旧19种排序保留不变；固定既有13维公开特征、3×3查询/商品族隔离和训练参数，重新训练18个分组模型及2个候选，新增两种交叉预测排序，共21种方法、378组完整排序。未知项不进入训练，任何AI相关性答案均不作运行时特征。",
        "",
        "区间枚举全部未知标签的0/1补全。它描述AI判断未定项的影响，不是置信区间，也不量化AI已判定标签的错误。mAP为全排序AP的查询平均；P/AP/NDCG/MRR/Hit包含所有查询，确定零正例查询的Recall未定义，可能零正例查询允许退出Recall分母。",
    ]
    names = ["precision@10", "recall@10", "ap", "ndcg@10", "mrr", "hit@1"]
    for scope, table in metrics["scopes"].items():
        first = next(iter(table.values()))
        lines += [
            "",
            f"### {scope}",
            "",
            f"查询{first['queries']}条；Recall有效查询分母{first['recall_denominator_min']}–{first['recall_denominator_max']}。",
            "",
            "| 方法 | Precision@10 | Recall@10 | mAP | NDCG@10 | MRR | Hit@1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for method, values in table.items():
            lines.append(
                "| "
                + method
                + " | "
                + " | ".join(interval(values["ranges"][key]) for key in names)
                + " |"
            )
    lines += [
        "",
        "## 采用结论与后续",
        "",
        "两类新候选仍未达到P@10≥60%；图片模式的Recall存在相对原基线退化的可能，默认策略保留。当前目标状态为未达成，正式指标为null。以上分数仅用于开发定位。",
        "",
        "后续由AI继续执行：先把58件可用候选构建为独立开发索引，并按新完整候选池重跑；优先补充柜体、手机支架、厨房调料架、女式西装套装、气门帽等缺失图片需求覆盖，再结合主体/配件识别、明确属性校验、ROI多视图召回及精排改进。补充采样属于开发暴露，不作为验收收益。",
        "",
        "保留独立heldout预留状态及全部原始哈希；在最终检索版本、查询、评审规则和报告方式冻结后，才能执行独立数据的AI盲评。AI盲评仍需标明评审来源，不能宣称人工验证。当前预留数据模型预测0。",
        "",
        "## 重放",
        "",
        "```powershell",
        ".\\.venv312\\Scripts\\python.exe scripts\\finalize_module2_ai_review.py",
        ".\\.venv312\\Scripts\\python.exe scripts\\verify_module2_ai_review.py",
        ".\\.venv312\\Scripts\\python.exe scripts\\report_module2_ai_review.py",
        "```",
        "",
        "重放不调用GPU模型；原自动批次入口已关闭，避免覆盖最终直接看图裁决。关键文件为judgments.ai.jsonl、catalog_QA.ai.jsonl、catalog.reviewed.jsonl、capacity.json、metrics.ai.development.json、ranker/及native_audit/。",
    ]
    (ROOT / "internal_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("AI development report rendered")


if __name__ == "__main__":
    main()
