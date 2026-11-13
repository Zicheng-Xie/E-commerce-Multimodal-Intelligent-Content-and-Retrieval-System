"""Deliver the completed independent-data AI blind evaluation without dev scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import run  # noqa: F401
from independent_business_blind import CONSUMED, METHOD, ROOT
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def interval(bounds: dict) -> str:
    """Return interval for this workflow.

    Args:
        bounds: Bounds used by this operation.

    Returns:
        Text produced by this operation.
    """
    if bounds[0] is None:
        return "未定义"
    low, high = 100 * bounds[0], 100 * bounds[1]
    return f"{low:.2f}–{high:.2f}%"


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve delivered blind report and documentation.

    Notes:
        Writes derived artifacts through (PROJECT_ROOT /
        'docs/module2_independent_blind.md').write_text, (ROOT /
        'AI_blind_report.md').write_text, environment_path.write_text,
        progress_path.write_text, readme_path.write_text, write_json.
    """
    if (ROOT / "AI_blind_report.md").exists():
        raise FileExistsError("Preserve delivered blind report and documentation")
    verification = load(ROOT / "verification.json")
    assert verification["verified"]
    metrics = load(ROOT / "AI_blind_metrics.json")
    decision = load(ROOT / "AI_blind_decision.json")
    capacity = load(ROOT / "AI_blind_capacity.json")
    overall_precision_met = metrics["scopes"]["all"][METHOD][
        "precision_60_lower_bound_met"
    ]
    result = (
        "总体Precision门槛已达，图片分支未达，分模态验收未通过"
        if not decision["precision_required_met"] and overall_precision_met
        else (
            "未通过Precision目标"
            if not decision["precision_required_met"]
            else "Precision目标下界通过，其他门槛见明细"
        )
    )
    primary = ["|模式|Precision@10|Recall@10|", "|---|---|---|"]
    for scope, label in [("all", "总体"), ("text", "文本"), ("image", "图片")]:
        ranges = metrics["scopes"][scope][METHOD]["ranges"]
        primary.append(
            f"|{label}|{interval(ranges['precision@10'])}|{interval(ranges['recall@10'])}|"
        )
    full = [
        "|模式|方法|P@10|R@10|mAP|NDCG@10|MRR|Hit@1|",
        "|---|---|---|---|---|---|---|---|",
    ]
    for scope in ["all", "text", "image"]:
        for name in ["global", "region_title", METHOD]:
            ranges = metrics["scopes"][scope][name]["ranges"]
            full.append(
                "|"
                + "|".join(
                    [
                        scope,
                        name,
                        *[
                            interval(ranges[m])
                            for m in [
                                "precision@10",
                                "recall@10",
                                "ap",
                                "ndcg@10",
                                "mrr",
                                "hit@1",
                            ]
                        ],
                    ]
                )
                + "|"
            )
    lines = [
        "# 模块二独立heldout的AI盲测结果",
        "",
        "日期：2026年10月23日。",
        "",
        f"结论：**{result}**。固定候选与两条原始基线完成60条查询、180组完整排序及1,200对完整模型精排；未据中途结果切换策略。默认策略保留。",
        "",
        "这是数据独立、排序分数对评审隐藏的AI盲测：30条文本与30张图片查询，对300件独立预留商品检索；18,000对AI判断在查询推理前冻结，59项未知保留。图库辅助描述也由AI在查看查询图前写成，评审者不独立于本项目。结果不能称作人工专家验收或线上流量表现。",
        "",
        "## 固定候选结果",
        "",
        *primary,
        "",
        "区间是对未知标签所有可能补全的精确上下界，不是置信区间，不涵盖AI误判风险，也没有将未知统一填正或填负。Precision分母固定10；无正例查询不删除，其Recall未定义；仅有未知的查询会使Recall参与平均的查询数随补全变化，计算已处理。",
        "",
        "目标：Precision@10≥60%为硬要求，Recall@10≥85%为争取目标。总体、文本、图片分别检查；不以总体平均掩盖图片分支的不足。",
        "本次总体Precision与Recall下界均达到目标；文本分支也达到门槛。图片Precision连上界仍低于60%，其Recall下界尚低于85%。因此没有将总体通过等同于两种检索模式都通过。",
        "",
        "## 两条基线与固定候选",
        "",
        *full,
        "",
        "## 图库容量",
        "",
        "|模式|P@10平均数学容量|确认零正例查询|仅未知查询|",
        "|---|---|---|---|",
    ]
    for scope in ["all", "text", "image"]:
        row = capacity[scope]
        lines.append(
            f"|{scope}|{interval(row['precision_ceiling_range'])}|{', '.join(row['zero_positive_queries']) or '无'}|{', '.join(row['uncertain_only_queries']) or '无'}|"
        )
    lines += [
        "",
        "数学容量按min(10,R)/10计算。低容量来自本次冻结查询与图库的实际相关商品数，不能靠重排突破。排序漏检与容量不足分别诊断，不删除正例以抬高Recall，不删除困难查询或重新抽样以提高Precision。",
        "",
        "## 冻结、隔离与执行",
        "",
        "先冻结最终版本、30条文本查询、预处理、图库辅助描述规约、业务相关性规约、指标和目标。用开发缓存精确重放两条分支的18组原排序后，才登记启用原预留heldout。来源ID、登记商品族、原图字节/像素指纹和dHash≤4均与已知开发曝光隔离。",
        "",
        "保留全部330条预留来源记录。图库存在鼠标宏软件、耳机替换线、牛仔裙、枕套等类目噪声，均按实际主体评审，没有用原细类直接生成相关性标签。148/150两条照片外观高度相近，仍按冻结采样保留；登记族与指纹隔离不等于排除了所有语义近重复。",
        "",
        "图库清单的image_path原本相对gallery/，首次网格生成误按dataset/解析，第一张图即报缺失。保留原冻结脚本和错误清单，单独finalizer从冻结lineage恢复正确路径，没有改变样本、模型或规则，也没有重置已使用登记。见gallery_path_failure_execution.json。",
        "",
        f"实际图库完整嵌入编码{verification['actual_gallery_embedding_calls']}次；查询完整嵌入编码{verification['actual_query_embedding_calls']}次；本地查询描述30次；完整图文精排1,200对。全部向量、原始logit、完整排序、AI关系与16个历史封存目录已重放核验。236项测试、66个源文件mypy通过。",
        "",
        "## 后续使用边界",
        "",
        "这批heldout已完成一次固定版本验证，已登记为consumed。不得继续在其上调参后再次称独立盲测。若根据本次错误改进模型或补库，这批数据只能作为已曝光回归；下一次验收必须另建并预先隔离新的未见集。",
        "",
        "没有把开发调优成绩混入本报告；未达到的指标如实保留，默认策略不升级。质量明细在AI_blind_metrics.json，配对不退化检查在AI_blind_paired_differences.json，逐查询容量在AI_blind_capacity.json。",
        "",
    ]
    (ROOT / "AI_blind_report.md").write_text("\n".join(lines), encoding="utf-8")
    doc = [
        "# 模块二：独立heldout盲测交付",
        "",
        "更新：2026年10月23日。目标保持Precision@10≥60%，Recall@10争取≥85%。",
        "",
        f"本轮已完成一次固定版本的独立数据AI盲测，结论为**{result}**；默认策略未替换。全部查询和困难样本保留，未根据盲测结果回调模型、规则或样本。",
        "",
        *primary,
        "",
        "以上为独立heldout的AI评审结果，区间反映59项未知标签的完整补全范围，不能称作人工真值或线上流量成绩。开发集数字未放入本表。",
        "",
        "## 版本与数据",
        "",
        "候选固定为文本20候选业务精排、图片20候选主体描述联合精排。沿用完整本地Qwen3-VL-Embedding-2B、Qwen3-VL-Reranker-2B与固定Qwen3.5查询描述；自动区域注意力、对比ROI、中心裁切与四角放大参数均在启用heldout前冻结，没有人工挑选验收ROI，没有叠加已出现退化的前缀和尾部变体。",
        "",
        "数据为事前准备的300件图库、30张图片查询及30条预先编写的文本查询，覆盖30个细类、三大业务品类。抽样为均衡封闭图库，不代表真实线上流量；来源ID、商品族与图像指纹均与已知开发曝光隔离。发现的类目噪声和图像相近记录全部保留并记录。",
        "",
        "图库辅助描述先于验收查询图查看冻结；18,000对AI相关性在查询推理前冻结，59项未知不强行判定。协议作者、图库描述者和评审者是同一AI，已知预先固定的文本查询与业务类别，也参与过开发；只对检索排名和分数隐藏，不声称独立评审员、上下文隔离或双盲人工验证。完整说明见review_context_disclosure.json。",
        "",
        "## 结果解释与后续",
        "",
        "分别检查总体、文本、图片的Precision硬目标及其他指标不退化；固定Top-10，保留零正例、歧义与低覆盖查询。Recall按完整相关集合计算，无正例查询的Recall未定义；容量不足与排序漏检分开解释，不能通过缩小正例集合追目标。",
        "",
        "这批数据已consumed，后续不得调参后重复作为盲测。若据此改进检索或补库，它只能用于开发回归；下一次正式验证需要新的未见集。完整基线、六项指标、容量审计和误差边界见[独立AI盲测报告](../artifacts/module2_independent_business_blind/AI_blind_report.md)。",
        "",
        "## 环境与复核",
        "",
        "使用既有`.venv312`、完整本地权重与既有依赖，无新增下载。模型、代码、预处理及AI协议均保存在release.json。原预留目录保持历史准备状态，当前使用情况由evaluation_registry.json覆盖；已使用记录在data/module2_business_blind_consumed/records.jsonl。",
        "",
        "```powershell",
        ".\\.venv312\\Scripts\\python.exe scripts/score_independent_business_blind.py",
        "```",
        "",
        "上述命令重放现有向量、logit和标签并核验原始数据及历史封存，不重跑模型或重新启用heldout。原freeze/open/predict入口拒绝覆盖已完成的一次性实验。图库路径错误及独立finalizer修复记录完整保留，未改变任何模型和评价规则。",
        "",
        f"实跑：图库编码{verification['actual_gallery_embedding_calls']}次，查询编码{verification['actual_query_embedding_calls']}次，查询描述30次，完整精排1,200对，180组完整排序。236项测试、66个源文件mypy及16个历史封存目录核验通过。",
        "",
    ]
    (PROJECT_ROOT / "docs/module2_independent_blind.md").write_text(
        "\n".join(doc), encoding="utf-8"
    )
    status = load(ROOT / "public_status.json")
    status.update(
        state=(
            "independent_AI_blind_completed_precision_not_met"
            if not decision["precision_required_met"]
            else "independent_AI_blind_completed_see_other_gates"
        ),
        independent_AI_blind_completed=True,
        heldout_unsealed=True,
        reusable_as_unseen=False,
        acceptance_predictions=60,
        actual_cross_pairs=1200,
        full_rankings=180,
        AI_relations=18000,
        unknown_relations=59,
        human_labels=0,
        precision_required_met=decision["precision_required_met"],
        overall_precision_required_met=overall_precision_met,
        overall_recall_preferred_met=metrics["scopes"]["all"][METHOD][
            "recall_85_lower_bound_met"
        ],
        recall_preferred_met=decision["recall_preferred_met"],
        default_strategy_changed=False,
        formal_acceptance=False,
        formal_metrics=None,
        goal_met=decision["AI_blind_primary_and_nonregression_goal_met"],
        quality_metrics=None,
        independent_AI_blind_metrics_file="AI_blind_metrics.json",
        report="AI_blind_report.md",
        tests_passed=236,
        mypy_source_files=66,
        replay_verified=True,
        old_sealed_roots_verified=16,
    )
    write_json(ROOT / "public_status.json", status)
    registry_path = PROJECT_ROOT / "configs/evaluation_registry.json"
    registry = load(registry_path)
    registry["formal_module2_acceptance"].update(
        state=status["state"],
        independent_AI_blind_completed=True,
        independent_AI_blind_metrics_file="artifacts/module2_independent_business_blind/AI_blind_metrics.json",
        formal_acceptance=False,
        metrics=None,
        independent_annotator_claim_allowed=False,
        human_labels=0,
        reusable_as_unseen=False,
        acceptance_predictions=60,
    )
    registry["current_module2_progress_file"] = (
        "artifacts/module2_independent_business_blind/public_status.json"
    )
    registry["current_preparation_exclusion_snapshot"] = (
        "artifacts/module2_independent_business_blind/post_blind_exclusion_snapshot.json"
    )
    registry["current_independent_AI_blind_metrics_file"] = (
        "artifacts/module2_independent_business_blind/AI_blind_metrics.json"
    )
    for rule in registry["rules"]:
        if rule["path_prefix"] == "artifacts/module2_independent_business_blind/":
            rule.update(
                role="consumed_single_frozen_version_independent_data_AI_reviewed_blind_result",
                external_quality_claim_allowed=True,
                formal_acceptance_claim_allowed=False,
                required_claim_qualifiers=[
                    "independent data AI-reviewed blind evaluation",
                    "balanced closed 300-product 60-query catalog",
                    "not human ground truth or traffic performance",
                ],
                reuse_as_unseen_test_allowed=False,
            )
    write_json(registry_path, registry)
    previous = PROJECT_ROOT / "artifacts/module2_rank_stability"
    parent = load(previous / "public_status.json")
    parent["next_stage"] = (
        "artifacts/module2_independent_business_blind/public_status.json"
    )
    write_json(previous / "public_status.json", parent)
    snapshot = load(previous / "post_development_exclusion_snapshot.json")
    snapshot["files"][CONSUMED.relative_to(PROJECT_ROOT).as_posix()] = sha256_file(
        CONSUMED
    )
    snapshot.update(
        scope="All prior development and consumed independent AI blind exposure; none reusable as unseen",
        consumed_business_blind_records=330,
        consumed_business_blind_release_sha256=sha256_file(ROOT / "release.json"),
    )
    write_json(ROOT / "post_blind_exclusion_snapshot.json", snapshot)
    summary = f"独立数据AI盲测已完成：300件图库、60条查询、18,000对预先冻结AI判断、1,200对完整模型精排。{result}，默认策略保留。236项测试、66个源文件mypy通过，16个历史封存目录已核验。该heldout已consumed，不能调参后复用为未见测试。结果明确区分AI标注、开发成绩及人工验收，详见[独立盲测交付](module2_independent_blind.md)。"
    progress_path = PROJECT_ROOT / "docs/current_progress.md"
    progress = progress_path.read_text(encoding="utf-8")
    progress = progress.replace(
        "## 最新：有限重排与6条新来源查询检查完成",
        "## 上一阶段：有限重排与6条新来源查询检查完成",
        1,
    )
    anchor = "## 上一阶段：有限重排与6条新来源查询检查完成"
    assert anchor in progress
    progress = progress.replace(
        anchor,
        "## 最新：一次独立heldout的AI盲测已完成\n\n" + summary + "\n\n" + anchor,
        1,
    )
    progress_path.write_text(progress, encoding="utf-8")
    readme_path = PROJECT_ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    oldline = next(
        line for line in readme.splitlines() if line.startswith("最新完成：")
    )
    readme = readme.replace(
        oldline,
        "最新完成："
        + summary.replace(
            "(module2_independent_blind.md)", "(docs/module2_independent_blind.md)"
        ),
        1,
    )
    readme_path.write_text(readme, encoding="utf-8")
    environment_path = PROJECT_ROOT / "docs/environment_setup.md"
    environment = environment_path.read_text(encoding="utf-8")
    header = "# 电商多模态项目环境配置与加载说明\n"
    assert environment.startswith(header)
    note = "\n2026年10月23日独立盲测补充：固定版本已使用预留独立数据完成300件图库、60条查询的AI盲测，复用既有完整本地模型和依赖，无新增下载。模型、参数、AI协议及输入绑定见release.json；重放复核入口为scripts/score_independent_business_blind.py。该heldout现已consumed，结果及运行说明见[独立盲测交付](module2_independent_blind.md)。\n"
    environment_path.write_text(
        header + note + environment[len(header) :], encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                report_written=True,
                independent_AI_blind_completed=True,
                precision_required_met=decision["precision_required_met"],
                formal_acceptance=False,
            )
        )
    )


if __name__ == "__main__":
    main()
