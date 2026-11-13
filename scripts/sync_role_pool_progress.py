"""Mutable progress overlay for the separately frozen role/pool experiment."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import run  # noqa: F401
from business_grounding_audit import load, relative
from multimodal_ecommerce.safety import PROJECT_ROOT, write_json
from role_pool_profiles import JOINT, ROOT


def replace_block(path: Path, marker: str, body: str) -> None:
    """Apply the replace block workflow step.

    Args:
        path: Path of the input or output artifact used by this operation.
        marker: Marker used by this operation.
        body: Body used by this operation.

    Notes:
        Writes derived artifacts through path.write_text.
    """
    original = path.read_text(encoding="utf-8")
    start, end = "<!-- " + marker + " -->", "<!-- /" + marker + " -->"
    block = start + "\n" + body + "\n" + end
    if start in original:
        before, rest = original.split(start, 1)
        _, after = rest.split(end, 1)
        updated = before + block + after
    else:
        updated = block + "\n\n" + original
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through (PROJECT_ROOT /
        'docs/module2_role_pool_development.md').write_text, write_json.
    """
    status = load(ROOT / "public_status.json")
    complete = (ROOT / "completion_receipt.json").exists()
    decision = load(ROOT / "decision.json") if complete else None
    now = datetime.now(timezone(timedelta(hours=8), name="Asia/Shanghai")).isoformat()
    profile_count = len(list((ROOT / "profiles").glob("S[0-9][0-9][0-9].json")))
    pair_count = len(list((ROOT / "pairs").glob("*.json")))
    if complete:
        assert decision is not None
        result_text = (
            f"四组归因对照及4组参照已完成，共320份完整排序。开发Precision硬目标{'通过' if decision['development_precision_target_met'] else '未整体通过'}；"
            f"Recall偏好{'通过' if decision['development_recall_preference_met'] else '未整体通过'}；"
            f"其他指标不退化{'已证明' if decision['development_all_metric_nonregression'] else '尚未整体证明'}。"
        )
        effects = load(ROOT / "internal_factorial_effects.json")
        role_effect = effects["fresh"]["image"]["joint_total"]
        if all(
            role_effect[m]["delta_range"][1] < 0
            for m in ("precision@10", "recall@10", "ap", "ndcg@10")
        ):
            result_text += (
                "共享未知标签成对对照确认：角色增强主方法在fresh开发组的Precision、Recall、mAP和NDCG均严格退步，"
                + "本轮不采用该增强分支。单独扩池也未取得稳定收益。"
            )
    else:
        result_text = (
            f"本轮进行中，阶段：{status['phase']}。原图主售角色画像{profile_count}/435条，已存储真实精排对{pair_count}条。"
            "本轮尚无完整指标结论。"
        )
    summary = (
        "模块二主售角色与候选扩池开发实验。"
        + result_text
        + "同图库、同40条查询、同12600条AI关系、K10；原池全部保留，扩池固定加入双图注意力前20条；主售不确定回退真实原分数。"
        + "新掩码全局停用。开发数据与独立验收严格隔离，历史盲测保持封存；本轮未新增独立盲测，默认原版、人工标签0、formal_metrics=null。"
    )
    for name, prefix in (("README.md", "docs/"), ("docs/current_progress.md", "")):
        replace_block(
            PROJECT_ROOT / name,
            "role_pool_latest",
            summary + f" 见[本轮进度]({prefix}module2_role_pool_development.md)。",
        )
    doc = f"""# 模块二：主售角色与候选扩池开发实验

最近同步：{now}。

{result_text}

目标保持Precision@10≥60%为硬门槛、Recall@10尽量≥85%，并检查mAP、NDCG@10、MRR、Hit@1和推理开销。以下均为已曝光开发实验；人工标签0，正式指标为null。

## 固定四组对照

| 候选池 | 原排序 | 主售角色排序 |
|---|---|---|
| 原池 | 精确重放原grounded_whole_zoom_union | 同原池，使用自身主售角色完整Qwen打分 |
| 扩池 | 保留全部原池，合并固定双图注意力前20；完整补分 | 同扩池，使用自身主售角色完整Qwen打分；预先指定主方法 |

另保留global、region_title、cross20和上一轮business_joint_attention_fusion作为参照，共8方法×40查询。315件/组的图库、全部查询、原12600条AI关系、K10及Recall完整分母保持。文本查询沿用原cross20，因此本轮主要隔离图片路径的两项变化。

## 主售语义及完整模型

435张原图包含411件图库商品和24条图片查询。完整本地Qwen3.5视觉模型逐件读取原生EXIF正向RGB图，生成主售类型、商品角色、配套对象、用途、展示道具及可见结构。图库商品只提供自身原始标题；查询图不提供图库、候选标题、相关性标签或排序。画像是模型观察，不是新增相关性标签。

原池/扩池均使用完整Qwen3-VL-Reranker-2B的whole及已确认zoom上下文。角色分支增加自身主售描述和预先固定的业务角色指令；任一自身画像不明确时复用真实原始上下文分数。不得为未运行的候选填充或猜测分数。融合固定为0.65精排秩+0.35原提议秩，whole/zoom各0.5；尾部保持同一原提议顺序。新SAM掩码继续全局停用。

双图注意力候选来自上一轮九个来源家族隔离折模型的OOF结果；本轮没有新骨干训练。这是开发归因实验，不能当作已冻结的单一部署模型或独立验收成绩。

## 执行与证据

- 画像协议：`artifacts/module2_role_pool_development/profile_protocol.json`
- 四组协议：`artifacts/module2_role_pool_development/experiment_protocol.json`
- 执行入口：`.venv312/Scripts/python.exe -X utf8 scripts/role_pool_experiment.py worker`
- 完整验证：`.venv312/Scripts/python.exe -X utf8 scripts/role_pool_experiment.py verify`（完整结束后）
- 当前状态：`artifacts/module2_role_pool_development/public_status.json`
- 本轮源码快照：`artifacts/module2_role_pool_development/source_release_manifest.json`

完整结束后输出逐方法六指标的未知标签范围、共享未知标签成对增量、四组归因增量、严格退化记录、实际推理耗时、3200张Top10商品卡片的内部结果页及封存收据。质量数值只进入内部报告。若硬门槛未过或退化，保留失败和原默认，不按结果改策略、标签、查询或K。

只在nvidia-smi占用低于1500MiB且Ollama驻留列表明确为空时接入GPU。自身新画像请求keep_alive为1分钟并自然到期；不得卸载、打断或缩短用户其他模型作业。可从收据续跑，并发由操作系统锁防止。

## 独立验收隔离

两轮已消耗盲测的全部元数据、下载图片及衍生物继续排除于训练及未来验收。当前435张原图已曝光，不能复用为独立验收。当前排除快照在本轮完成前继承上一轮，完成后只增补、不丢失历史。Precision门槛与其他要求通过后，仍须先产出单一可执行版本并冻结，再用全新查询和全新图库进行一次独立AI盲测。

部分旧开发组的Top10容量限制了Recall上限；该诊断不等于实际性能，不能通过删除困难查询或缩减完整正例分母达标。原PDF模块二的中文文本/图片检索、统一特征、品类属性排序及商品信息可视化任务保持。
"""
    if complete:
        assert decision is not None
        runtime = load(ROOT / "reranker_receipt.json")
        regressions = load(ROOT / "internal_strict_regressions.json")
        doc += (
            f"\n本轮完成435次原生角色画像调用；精排新增完整模型调用{runtime['actual_new_full_Qwen_calls']}次，"
            f"严格同输入复用旧完整分数{runtime['reused_exact_original_pairs']}对，"
            f"共{runtime['unique_pairs']}个唯一真实分数；未填补任何缺失分数。实际逐对耗时及模型诊断保留于`pairs/`和`reranker_sessions/`，不作为端到端服务时延。\n"
        )
        doc += (
            f"\n保留{len(regressions)}条严格退化对比记录，涉及"
            f"{len({(r['cohort'], r['query_id']) for r in regressions})}条不同查询；"
            "同一查询可对应多个基线和指标，记录数不是查询数。"
            + "固定扩池虽补入了部分已标注相关商品，但并未保证这些商品进入Top10。"
            + "画像质检同时发现主售及属性错误；这些观察解释了需要继续核验的环节，"
            + "当前四组对照没有单独证明每一项描述错误的因果贡献。\n"
            + "\n下一步优先研究主售名词与可见文字的可核对证据、剔除无依据属性，"
            + "以及在保留原候选前提下提升新增候选的精排可靠性。任何新机制必须另立冻结开发协议；"
            + "本轮方法、标签、查询、K10和已封存结果不改。\n"
        )
        doc += "\n内部完整结果：[六指标报告](../artifacts/module2_role_pool_development/internal_comparison.ai.development.md)，[结果页](../artifacts/module2_role_pool_development/internal_comparison.html)，[决策收据](../artifacts/module2_role_pool_development/decision.json)。\n"
    doc += (
        "\n原生原图抽查已保留一项主售语义失败：fresh/image_03仍将包内胆的展示手袋当成主售；"
        + "另两项复合词示例（金枕头榴莲、标题含手机壳的激光雕刻机）主售类型得到原图支持。"
        + "这三项为来源图质检，不是相关性重标注或总体质量结论；没有按查询编号修正本轮画像、策略或排序。"
        + "记录见`native_AI_role_QA.json`及`native_AI_role_QA_compounds.json`。\n"
    )
    if (ROOT / "native_query_role_QA.json").exists():
        doc += (
            "\n24条图片查询的原图均已由Codex逐张查看并核对自身新画像，"
            + "除主售误判外保留了鞋头形状、闭合结构、柜门/抽屉及未验证材质等属性问题。"
            + "完整来源质检见`native_query_role_QA.json`；该检查不是独立盲审或相关性标签，未修改冻结推理。\n"
        )
    if (ROOT / "profiles.json").exists():
        profiles = load(ROOT / "profiles.json")
        if not any(p["value"]["primary_unclear"] for p in profiles):
            doc += (
                "\n生成模型对435条画像均未标记主售不确定，但原图质检已发现错误；"
                + "因此本轮的不确定回退条件没有触发。这个标志不能当作经过校准的正确性保证。\n"
            )
    (PROJECT_ROOT / "docs/module2_role_pool_development.md").write_text(
        doc, encoding="utf-8"
    )
    runtime = (
        "主售角色/扩池实验复用完整本地Qwen3.5及Qwen3-VL-Reranker-2B，无新增依赖安装。"
        + "原生RGB画像采用16384上下文、1024输出上限、温度0；精排保持原完整模型配置与无静默截断检查。"
        + "九折双图注意力OOF仅提供开发候选，不能替代单一部署模型。显存准入及自然释放规则保持。"
    )
    replace_block(
        PROJECT_ROOT / "docs/environment_setup.md", "role_pool_runtime", runtime
    )
    registry_path = PROJECT_ROOT / "configs/evaluation_registry.json"
    registry = load(registry_path)
    prefix = relative(ROOT) + "/"
    if not any(r["path_prefix"] == prefix for r in registry["rules"]):
        registry["rules"].append(
            dict(
                path_prefix=prefix,
                role="exposed_AI_sale_role_pool_development_not_acceptance",
                external_quality_claim_allowed=False,
                reuse_as_unseen_test_allowed=False,
                human_labels=0,
                formal_metrics=None,
            )
        )
    registry.update(
        updated_at="2026-10-27",
        current_development_experiment=relative(ROOT),
        current_prepared_development_experiment=relative(ROOT),
        current_prepared_development_protocol=relative(
            ROOT / "experiment_protocol.json"
        ),
        current_prepared_development_state=status["phase"],
        current_module2_progress_file=relative(ROOT / "public_status.json"),
        current_role_pool_development_doc="docs/module2_role_pool_development.md",
        current_role_pool_development_status=relative(ROOT / "public_status.json"),
        role_pool_new_independent_AI_blind_started=False,
        formal_metrics=None,
        development_candidate_selected="original",
    )
    if complete:
        assert decision is not None
        registry.update(
            current_completed_development_experiment=relative(ROOT),
            current_development_decision_file=relative(ROOT / "decision.json"),
            current_module2_target_status_file=relative(ROOT / "decision.json"),
            current_role_pool_development_report=relative(
                ROOT / "internal_comparison.ai.development.md"
            ),
            current_preparation_exclusion_snapshot=relative(
                ROOT / "post_experiment_exclusion_snapshot.json"
            ),
        )
    else:
        registry["current_preparation_exclusion_snapshot"] = relative(
            JOINT / "post_experiment_exclusion_snapshot.json"
        )
    write_json(registry_path, registry)
    print(dict(progress_synchronized=True, completed=complete, formal_metrics=None))


if __name__ == "__main__":
    main()
