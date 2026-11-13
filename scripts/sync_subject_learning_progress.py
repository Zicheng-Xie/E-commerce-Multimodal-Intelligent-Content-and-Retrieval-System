"""Reconcile mutable project progress with immutable subject-learning receipts."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import run  # noqa: F401
from multimodal_ecommerce.safety import PROJECT_ROOT


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def replace_marker(path: Path, marker: str, body: str) -> None:
    """Apply the replace marker workflow step.

    Args:
        path: Path of the input or output artifact used by this operation.
        marker: Marker used by this operation.
        body: Body used by this operation.

    Notes:
        Writes derived artifacts through path.write_text.
    """
    text = path.read_text(encoding="utf-8")
    start = f"<!-- {marker} -->"
    end = f"<!-- /{marker} -->"
    block = start + "\n" + body + "\n" + end
    if start in text:
        a = text.index(start)
        b = text.index(end, a) + len(end)
        text = text[:a] + block + text[b:]
    else:
        text = block + "\n\n" + text
    path.write_text(text, encoding="utf-8")
    checked = path.read_text(encoding="utf-8")
    assert "\ufffd" not in checked and "开发" in checked


def _sync():
    attr = PROJECT_ROOT / "artifacts/module2_subject_attribute_learning"
    ranknet_recall = PROJECT_ROOT / "artifacts/module2_subject_recall_learning"
    pointwise = PROJECT_ROOT / "artifacts/module2_pointwise_recall_learning"
    priority = PROJECT_ROOT / "artifacts/module2_public_conflict_priority"
    pointwise_active = (pointwise / "public_status.json").exists()
    priority_active = (priority / "public_status.json").exists()
    recall = (
        priority
        if priority_active
        else pointwise if pointwise_active else ranknet_recall
    )
    state = load(recall / "public_status.json")
    done = state["completed"]
    attr_decision = load(attr / "decision.json")
    assert not attr_decision["development_precision_target_met"]
    prefix = "主售属性与困难负例重训已完成：11方法440份排序，原8方法320份排序及全部12600条AI关系保持不变；新主方法未整体通过Precision目标、Recall偏好和不退化检查。"
    if pointwise_active:
        prefix += "全图库RankNet提议与完整精排的14方法560份排序也已封存，仍未整体通过目标。类别平衡全负例损失的17方法680份排序随后完成；主类型系数转为正值，但图片Precision目标仍未整体通过。"
    if done:
        decision = load(recall / "decision.json")
        p = (
            "开发Precision硬目标通过"
            if decision["development_precision_target_met"]
            else "开发Precision硬目标未整体通过"
        )
        r = (
            "Recall偏好通过"
            if decision["development_recall_preference_met"]
            else "Recall偏好未整体通过"
        )
        n = (
            "六指标不退化通过"
            if decision["development_all_metric_nonregression"]
            else "六指标不退化未整体通过"
        )
        scope = (
            "公开主售冲突优先级对照已完成：19方法760份排序，原17方法680份排序不变"
            if priority_active
            else (
                "类别平衡全负例提议与完整精排已完成：17方法680份排序，原14方法560份排序不变"
                if pointwise_active
                else "全图库候选学习与完整双上下文精排已完成：14方法560份排序，原11方法440份排序不变"
            )
        )
        active = f"{scope}，完整重放与封存通过。{p}，{r}，{n}。"
    elif state["phase"] == "engineering_failure":
        active = "全图库候选学习实验遇到工程错误并已停止，原始证据保留在其failure.json；不能将未完成结果当作成绩。"
    else:
        plan = (
            load(recall / "plan_receipt.json")
            if (recall / "plan_receipt.json").exists()
            else None
        )
        active = "全图库候选学习实验正在执行，先以7560个真实查询商品组合训练33维提议层，保留所有旧候选，再完整Qwen精排。"
        if pointwise_active:
            active = "类别平衡全负例对照正在执行；使用与前轮完全相同的7560行33维真实输入、标签和家族折，覆盖所有允许的正/负例拟合，保持原始候选并集及完整双上下文精排公式。"
        if plan:
            active += f"已冻结追加{plan['additional_pool_positions']}个候选位置，需要{plan['planned_missing_unique_pairs']}个新增真实配对评分；当前阶段为{state['phase']}。尚无本轮完整检索结论。"
    tail = "全部为已曝光AI开发实验，人工标签0；原版默认保留，正式指标null，新的独立验收未启动。"
    body = f"最近同步：{datetime.now().astimezone().isoformat()}。\n\n{prefix}\n\n{active}{tail}"
    for filename, link in [
        ("README.md", "docs/module2_learned_fusion.md"),
        ("docs/current_progress.md", "module2_learned_fusion.md"),
    ]:
        replace_marker(
            PROJECT_ROOT / filename,
            "module2_subject_learning",
            body + f" 见[进度与证据]({link})。",
        )
    details = body + """

本轮归因覆盖24条图片查询：11条当前池遗漏部分已知正例，14条池内仍有已知排序空间。内胆包旧ROI裁到左下方手袋道具；华为Mate10查询的业务标签包含其他型号手机壳及部分DIY材料。标签保持冻结，型号继续作软诊断，不通过重写标签更换评价口径。属性训练使用软特征；后续公开主售冲突优先级作为单独实验披露。

属性训练对照使用20维原模型证据与38维加入公开属性、整图/裁切及FG一致性的证据；18个家族隔离折模型、2个全开发候选按固定400步、L2 0.05、每查询最多8个已确认困难负例训练。395条已知池内关系参与允许范围内的训练，109条未知排除于拟合。历史公开描述属于既有AI开发材料；本轮梯度家族掩码不能使它成为独立验证。

后续全图库提议层独立使用33维完整Qwen/FG/公开属性输入，覆盖7560个实际图库组合，其中7419条已知、141条未知；提议特征不含CE列，不填补缺失CE。9个家族排除折模型输出top20提议，稳定追加到原候选并集中，最多60件。共享池的每件商品均需完整Qwen两种原整图上下文真实评分或严格身份缓存。proposal_only为明确去掉CE的组件消融，不能替代主方法的完整精排成绩。

284项执行前测试通过；旧查询、标签、K10、完整图库Recall分母、零正例和未知null保持原口径。已确认的错误裁切坐标在这两轮保持原状，实验检验一致性融合和候选生成，不宣称主体定位已经修复。新实验不生成图片，曝光快照继承此前原图、159张衍生图和全部来源家族；全部材料排除于独立验收。

已完成属性实验：[封存收据](../artifacts/module2_subject_attribute_learning/completion_receipt.json)、[内部报告](../artifacts/module2_subject_attribute_learning/internal_comparison.ai.development.md)、[严格局部损失](../artifacts/module2_subject_attribute_learning/internal_strict_regressions.json)。全图库实验：[协议](../artifacts/module2_subject_recall_learning/protocol.json)、[候选计划收据](../artifacts/module2_subject_recall_learning/plan_receipt.json)、[运行状态](../artifacts/module2_subject_recall_learning/public_status.json)。开发质量数字只保留内部报告。
"""
    if done and not pointwise_active:
        details += "\n全图库结果：[封存收据](../artifacts/module2_subject_recall_learning/completion_receipt.json)、[完整内部对照](../artifacts/module2_subject_recall_learning/internal_comparison.ai.development.md)。只读核验：`.venv312/Scripts/python.exe scripts/subject_recall_experiment.py verify`。\n"
    if pointwise_active:
        details += "\n新增类别平衡损失的执行前286项测试通过，运行时使用独立模块命名空间和新输出目录，原已封存源码、模块对象与评分均不修改。全图库负例而非仅top8困难负例用于每查询类别平衡逻辑损失，偏置不正则；优化步数、学习率、L2及家族排除不变。提议层分数是标准化logit，不是业务概率。\n\n本轮入口：`.venv312/Scripts/python.exe scripts/pointwise_recall_experiment.py worker`；[冻结协议](../artifacts/module2_pointwise_recall_learning/protocol.json)、[运行状态](../artifacts/module2_pointwise_recall_learning/public_status.json)、[负例集中诊断](../artifacts/module2_pointwise_recall_learning/negative_sampling_diagnosis.json)。当前阶段不声称已验证训练问题的因果效果。\n"
        if load(pointwise / "public_status.json")["completed"]:
            details += "\n类别平衡结果：[封存收据](../artifacts/module2_pointwise_recall_learning/completion_receipt.json)、[内部对照](../artifacts/module2_pointwise_recall_learning/internal_comparison.ai.development.md)；只读核验：`scripts/pointwise_recall_experiment.py verify`。\n"
    if priority_active:
        details += "\n公开主售冲突优先级实验的288项执行前测试通过。主方法只在既有完整Qwen精排共享池内将明确主类型、部件或结构冲突稳定后移；同层原神经排序、完整尾部、所有商品ID和标签保留。未知或矛盾证据弃权，手机型号不硬过滤，手机壳材料按既有宽业务口径保留。全排序后处理是单独消融，池外商品保持原base顺序，不补造CE分数。本轮新增神经调用0、图片0；没有修复错误ROI坐标。\n\n[冻结协议](../artifacts/module2_public_conflict_priority/protocol.json)、[完整内部对照](../artifacts/module2_public_conflict_priority/internal_comparison.ai.development.md)、[严格局部退化](../artifacts/module2_public_conflict_priority/internal_strict_regressions.json)、[封存收据](../artifacts/module2_public_conflict_priority/completion_receipt.json)。只读核验：`.venv312/Scripts/python.exe scripts/public_conflict_experiment.py verify`。主方法与消融独立完整报告，不根据事后最高数字更换主方法。\n"
        details += "\n下一轮优先解决已确认的主体定位问题：先冻结适用于所有图片查询的query-only定位和不确定回退协议，再生成ROI并做完整模型评分。禁止按已知正例或检索名次选框，原整图候选必须保留；无法确认主售主体时保留整图。仍在本开发集检验Precision目标及其他指标变化，单独登记新增衍生图曝光。该定位修复尚未实现或运行，不列为已完成成果；新的独立盲测需另行冻结单一可执行候选后才启动。\n"
    replace_marker(
        PROJECT_ROOT / "docs/module2_learned_fusion.md",
        "subject_learning_current",
        details,
    )
    path = PROJECT_ROOT / "docs/module2_learned_fusion.md"
    text = path.read_text(encoding="utf-8")
    old = "下一阶段需先完成并冻结新增AI关系，再对所有保留商品进行完整模型编码，在同一扩充图库上重跑原方法与新融合方法。当前880份排序仍是旧296图库对照，不能与尚未完成的扩库实验混合。最终候选通过开发比较后再单独冻结一个可执行版本，使用全新查询和全新图库盲测。"
    text = text.replace(
        old,
        "315件图库扩库对照已完成，旧880份排序属于296件历史实验。当前主售属性和全图库候选学习各自按独立协议继续开发；候选通过同条件目标检查后，才冻结单一可执行版本，并使用全新查询和图库盲测。",
    )
    path.write_text(text, encoding="utf-8")
    replace_marker(
        PROJECT_ROOT / "docs/module2_primary_subjects.md",
        "latest_subject_learning_followup",
        body
        + "\n\n本页保留最初296件图库实验的历史证据；315件图库及后续学习对照见[module2_learned_fusion](module2_learned_fusion.md)，不同图库成绩不混合。",
    )
    runtime = "新增主售学习入口：`.venv312/Scripts/python.exe scripts/subject_attribute_experiment.py verify`。全图库候选实验使用`.venv312/Scripts/python.exe scripts/subject_recall_experiment.py worker`串行接续，已有进程时不要重复启动。该入口先校验冻结清单与并发锁；完整GPU精排前执行nvidia-smi低于1500MiB及Ollama明确空驻留列表检查，等待用户模型自然释放，不卸载或缩短keep-alive。完成后可执行`verify`只读核验。状态见`artifacts/module2_subject_recall_learning/public_status.json`。"
    if pointwise_active:
        runtime += " 当前类别平衡损失对照入口为`.venv312/Scripts/python.exe scripts/pointwise_recall_experiment.py worker`，已有本轮实例时不重复启动；同一完整模型、参数与GPU准入规则继续适用，使用新命名空间和输出目录，保留旧源码及模块对象。"
    if priority_active:
        runtime += " 最新公开主售冲突优先级实验已完成，入口`.venv312/Scripts/python.exe scripts/public_conflict_experiment.py verify`只读核验；该阶段不调用新模型，无需加载GPU。已有模型、原始标签、曝光排除和历史收据均保持。"
    replace_marker(
        PROJECT_ROOT / "docs/environment_setup.md", "subject_learning_runtime", runtime
    )
    path = PROJECT_ROOT / "configs/evaluation_registry.json"
    registry = load(path)
    completed = (
        recall
        if done
        else (
            ranknet_recall
            if load(ranknet_recall / "public_status.json")["completed"]
            else attr
        )
    )
    registry.update(
        current_development_experiment=completed.relative_to(PROJECT_ROOT).as_posix(),
        current_completed_development_experiment=completed.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        current_module2_progress_file=recall.relative_to(PROJECT_ROOT).as_posix()
        + "/public_status.json",
        current_module2_target_status_file=completed.relative_to(
            PROJECT_ROOT
        ).as_posix()
        + "/decision.json",
        current_development_decision_file=completed.relative_to(PROJECT_ROOT).as_posix()
        + "/decision.json",
        current_prepared_development_experiment=recall.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        current_prepared_development_protocol=recall.relative_to(
            PROJECT_ROOT
        ).as_posix()
        + "/protocol.json",
        current_prepared_development_state=state["phase"],
        current_preparation_exclusion_snapshot=completed.relative_to(
            PROJECT_ROOT
        ).as_posix()
        + "/post_experiment_exclusion_snapshot.json",
    )
    registry["current_subject_attribute_report"] = (
        attr.relative_to(PROJECT_ROOT).as_posix()
        + "/internal_comparison.ai.development.md"
    )
    registry["current_subject_recall_status"] = (
        recall.relative_to(PROJECT_ROOT).as_posix() + "/public_status.json"
    )
    if done:
        registry["current_subject_recall_report"] = (
            recall.relative_to(PROJECT_ROOT).as_posix()
            + "/internal_comparison.ai.development.md"
        )
    for root, role in [
        (
            attr,
            "exposed_AI_gradient_family_masked_subject_attribute_learning_not_acceptance",
        ),
        (
            ranknet_recall,
            "exposed_AI_full_gallery_candidate_learning_and_complete_CE_not_acceptance",
        ),
        (
            pointwise,
            "exposed_AI_pointwise_recall_learning_and_complete_CE_not_acceptance",
        ),
        (priority, "exposed_AI_public_conflict_priority_not_acceptance"),
    ]:
        if not (root / "protocol.json").exists():
            continue
        prefix = root.relative_to(PROJECT_ROOT).as_posix() + "/"
        if not any(r["path_prefix"] == prefix for r in registry["rules"]):
            registry["rules"].append(
                dict(
                    path_prefix=prefix,
                    role=role,
                    formal_acceptance_claim_allowed=False,
                    reusable_as_unseen=False,
                    development_scores_allowed_in_public_status=False,
                )
            )
    if pointwise_active:
        registry["current_pointwise_recall_status"] = (
            pointwise.relative_to(PROJECT_ROOT).as_posix() + "/public_status.json"
        )
        if load(pointwise / "public_status.json")["completed"]:
            registry["current_pointwise_recall_report"] = (
                pointwise.relative_to(PROJECT_ROOT).as_posix()
                + "/internal_comparison.ai.development.md"
            )
    if priority_active:
        registry["current_public_conflict_status"] = (
            priority.relative_to(PROJECT_ROOT).as_posix() + "/public_status.json"
        )
        registry["current_public_conflict_report"] = (
            priority.relative_to(PROJECT_ROOT).as_posix()
            + "/internal_comparison.ai.development.md"
        )
    assert (
        registry["formal_metrics"] is None
        and not registry["new_independent_AI_blind_after_development_started"]
    )
    path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(dict(documents_synchronized=True, phase=state["phase"], formal_metrics=None))


def sync() -> None:
    """Return sync for this workflow.

    Notes:
        Writes derived artifacts through path.parent.mkdir.
    """
    from multimodal_ecommerce import platform_lock as msvcrt

    path = PROJECT_ROOT / "tmp/module2_subject_progress_sync.lock"
    path.parent.mkdir(exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            logging.getLogger(__name__).debug(
                "Handled exception in sync", exc_info=True
            )
            print(dict(another_progress_sync_running=True))
            return
        try:
            _sync()
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    sync()
