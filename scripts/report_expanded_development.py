"""Keep diagnostic scores internal and publish progress without acceptance claims."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.safety import PROJECT_ROOT, write_json

ROOT = PROJECT_ROOT / "artifacts/module2_expanded_development"


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through (ROOT / 'internal_report.md').write_text,
        write_json.
    """
    metrics = load(ROOT / "internal_metrics.ai.development.json")
    names = ["precision@10", "recall@10", "ap", "ndcg@10", "mrr", "hit@1"]
    decisions = {}
    for method in metrics["scopes"]["all"]:
        required, recall, nonregression = {}, {}, {}
        for scope, table in metrics["scopes"].items():
            current = table[method]["ranges"]
            required[scope] = current["precision@10"][0] >= 0.6 - 1e-12
            recall[scope] = (
                current["recall@10"][0] is not None
                and current["recall@10"][0] >= 0.85 - 1e-12
            )
            result = {}
            for baseline in ["global", "region_title"]:
                for metric in names[1:]:
                    a, b = current[metric], table[baseline]["ranges"][metric]
                    value = (
                        "undefined"
                        if a[0] is None or b[0] is None
                        else (
                            "nonregression_for_all_AI_unknown_completions"
                            if a[0] >= b[1] - 1e-12
                            else (
                                "regression_for_all_AI_unknown_completions"
                                if a[1] < b[0] - 1e-12
                                else "not_resolved_by_separate_ranges"
                            )
                        )
                    )
                    result[baseline + ":" + metric] = value
            nonregression[scope] = result
        decisions[method] = dict(
            internal_AI_precision_lower_bound_by_scope=required,
            recall_preferred_lower_bound_by_scope=recall,
            conservative_nonregression=nonregression,
            all_development_precision_scopes_passed=all(required.values()),
            promotion_allowed=False,
            formal_acceptance=False,
        )
    write_json(
        ROOT / "internal_decisions.json",
        dict(
            role="internal_AI_development_only",
            candidates=decisions,
            formal_metrics=None,
            default_strategy_changed=False,
            goal_met=False,
        ),
    )
    inputs = load(ROOT / "inputs.public.json")
    ranks = load(ROOT / "rankings.json")
    labels = {
        (r["query_id"], r["candidate_id"]): r
        for r in map(
            json.loads,
            (PROJECT_ROOT / "artifacts/module2_ai_review/judgments.ai.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
    }
    products = {p["id"]: p for p in inputs["catalog"]}
    errors = []
    for query in inputs["queries"]:
        for rank, cid in enumerate(ranks[query["id"]]["cross_evidence_pool60"][:10], 1):
            judgment = labels[query["id"], cid]
            if judgment["grade"] != 1:
                errors.append(
                    dict(
                        query_id=query["id"],
                        candidate_id=cid,
                        rank=rank,
                        title=products[cid]["title"],
                        image_path=products[cid]["image_path"],
                        AI_grade=judgment["grade"],
                        AI_reason=judgment["reason"],
                        review_origin="model",
                        role="internal_evaluator_error_analysis_never_runtime_features",
                    )
                )
    write_json(ROOT / "internal_error_cases.json", errors)

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
            f"{values[0] * 100:.2f}%"
            if abs(values[0] - values[1]) < 1e-12
            else f"{values[0] * 100:.2f}%–{values[1] * 100:.2f}%"
        )

    lines = [
        "# 模块二344件业务开发图库：内部实验记录",
        "",
        "**仅供开发调优，禁止将本文件成绩作为正式验收结果对外输出。** 本轮是AI评审的业务相关性开发集，与历史服饰同款开发实验分别保存；二者都不是未见数据的正式验收。",
        "",
        "## 协议和隔离",
        "",
        "推理前冻结344件图库、原18条查询、模型完整文件哈希、ROI参数、训练配方和12种固定策略。按ID、商品族、文件/定向像素哈希及dHash≤4检查与预留heldout不重叠；仅使用其元数据和预先保存的指纹，未读取验收图片。新增58件商品来自针对旧查询的开发补充采样，具有选择偏差。",
        "",
        "运行输入仅含原始标题、商品图和图像生成的公开主体/外观描述；不含AI相关性答案、评审理由、来源标签和采样bucket。查询和商品族按3×3分组隔离训练，仍是已暴露数据的内部交叉验证。",
        "",
        "## 实际执行",
        "",
        "原286件向量按商品ID复用；新增58件共470次实际完整嵌入调用，包括354个视图、58个原标题、58个主体/外观描述。合并后2,095个视图。首次合并时发现历史索引与评审清单顺序不同，断言阻止输出；保留失败和执行源码，改为按ID逐项对齐后合并，未重复推理。",
        "",
        "1,080对完整精排输入中193对新运行完整交叉编码器，887对复用模型、指令及公开输入一致的历史分数。固定配方训练18个分组模型及2个候选，生成216组完整排序。未知16对保留空值，不参与训练。",
        "",
        "## 图库扩容与策略收益分开看",
        "",
        "下表只表示同一策略面对不同开发图库的变化，不能当作策略提升或泛化证据。新图库增加了针对开发需求采样的候选和正例。",
        "",
        "| 相同区域/标题策略 | Precision@10 | Recall@10 | mAP |",
        "|---|---:|---:|---:|",
    ]
    prior = load(
        PROJECT_ROOT / "artifacts/module2_ai_review/metrics.ai.development.json"
    )
    for name, row in [
        ("原286件", prior["scopes"]["all"]["region_title"]),
        ("扩展344件", metrics["scopes"]["all"]["region_title"]),
    ]:
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(interval(row["ranges"][k]) for k in names[:3])
            + " |"
        )
    lines += [
        "",
        "## 同一344件候选池的策略对照",
        "",
        "以下区间枚举AI未知标签的全部0/1补全，不是统计置信区间，不包含AI确定判断出错的风险。P@10固定分母10；零正例查询保留在P/AP/NDCG/MRR/Hit分母，其Recall未定义并单列分母。mAP为全排序AP均值。",
    ]
    for scope, table in metrics["scopes"].items():
        first = next(iter(table.values()))
        lines += [
            "",
            "### " + scope,
            "",
            f"查询{first['queries']}条，Recall有效查询分母{first['recall_denominator_min']}–{first['recall_denominator_max']}。",
            "",
            "| 方法 | Precision@10 | Recall@10 | mAP | NDCG@10 | MRR | Hit@1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for method, row in table.items():
            lines.append(
                "| "
                + method
                + " | "
                + " | ".join(interval(row["ranges"][k]) for k in names)
                + " |"
            )
    lines += [
        "",
        "## 采用结论",
        "",
        "部分候选在本开发集总体Precision观察上达到目标，但图片模式仍未通过门槛，召回目标未完全满足。既有18条开发需求及AI标签已被用于多轮选择，不能据此宣布最终目标达成或替换默认策略。全部候选保持实验状态，正式指标为null。",
        "",
        "## 下一批开发覆盖",
        "",
        "已从固定训练来源前20万条中准备5类各16条、共80条元数据候选，覆盖中式木质边柜、车用手机支架、壁挂厨房调料架、女式西装长裤套装及气门帽；登记为开发暴露。目标后续下载及质检40张图片，当前未下载，也没有自动相关性标签。选择使用已知开发查询覆盖线索，不能用作独立验收。",
        "",
        "验收预留登记保持原哈希、模型推理0。最终策略、查询、相关性协议及报告口径冻结后才允许独立heldout AI盲测，开发分数不进入正式成绩。",
        "",
        "## 工程重放",
        "",
        "```powershell",
        ".\\.venv312\\Scripts\\python.exe scripts\\verify_expanded_development.py",
        ".\\.venv312\\Scripts\\python.exe scripts\\report_expanded_development.py",
        "```",
    ]
    (ROOT / "internal_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Internal development report written; no scores added to public status")


if __name__ == "__main__":
    main()
