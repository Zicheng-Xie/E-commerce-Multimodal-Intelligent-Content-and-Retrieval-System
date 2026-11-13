"""Publish development status without exposing tuning scores as acceptance."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from structure_development import MEDIA, QUEUE, ROOT, load


def report() -> None:
    """Summarize the recorded results of this development workflow.

    Raises:
        FileExistsError: Preserve completed development stage.

    Notes:
        Writes derived artifacts through (PROJECT_ROOT /
        'docs/module2_structure_development.md').write_text, (ROOT /
        'internal_report.ai.development.md').write_text, (ROOT /
        'review_context_disclosure.json').write_text, environment_path.write_text,
        path.write_text, write_json.
    """
    if (ROOT / "completion_receipt.json").exists():
        raise FileExistsError("Preserve completed development stage")
    decision = load(ROOT / "decision.json")
    verification = load(ROOT / "verification.json")
    selection = load(ROOT / "selection.json")
    selected = selection["selected"]
    internal = [
        "# 内部开发实验结果，禁止作为正式验收结果\n",
        "两组均为固定类别配额的小型封闭图库。标签由同一 AI 对话看图评审；排序前冻结，非人工真值。",
        "区间是未决标签的精确补全边界，未涵盖 AI 误判和抽样不确定性。",
        "补库得到新开发素材，但没有同查询、同图库的补库前后对照，因此不能宣称量化补库增益。\n",
    ]
    for cohort in ["tune", "validation"]:
        metrics = load(ROOT / cohort / "internal_metrics.ai.development.json")
        capacity = load(ROOT / cohort / "internal_capacity.json")
        internal += [
            f"\n## {cohort}\n",
            "|范围|方法|P@10|R@10|mAP|NDCG@10|MRR|Hit@1|",
            "|---|---|---|---|---|---|---|---|",
        ]
        for scope, methods in metrics["scopes"].items():
            for name, row in methods.items():
                values = [
                    (
                        "未定义"
                        if row["ranges"][m][0] is None
                        else "–".join(f"{100 * n:.2f}%" for n in row["ranges"][m])
                    )
                    for m in [
                        "precision@10",
                        "recall@10",
                        "ap",
                        "ndcg@10",
                        "mrr",
                        "hit@1",
                    ]
                ]
                internal.append("|" + "|".join([scope, name, *values]) + "|")
        internal += [
            "\nP@10 图库容量边界：`"
            + json.dumps(capacity["precision_at_10_capacity"], ensure_ascii=False)
            + "`。",
            "零正例查询保留，Recall 未定义；其他指标仍纳入。",
            "原始逐查询记录见本目录 internal_metrics.ai.development.json；同标签补全下的配对差见 internal_paired.json 和 internal_vs_original.json。",
        ]
    internal += [
        "\n## 固定选择与后续\n",
        json.dumps(decision, ensure_ascii=False, indent=2),
    ]
    (ROOT / "internal_report.ai.development.md").write_text(
        "\n".join(internal) + "\n", encoding="utf-8"
    )
    details = f"""# 模块二：主体结构开发实验与隔离验证

本轮完成了新开发素材采集、主体结构/描述输入对照，以及另一组商品家族隔离的固定方法验证。全部结果属于开发证据，正式验收指标为空，默认策略未升级。

## 实际完成

- 固定抽样规则：8 类商品，每组每类 12 个图库商品和 1 张查询图；两组各 96 个图库商品、8 条图片查询、8 条文本查询。
- 新增 256 条候选元数据，实际保留 208 张查询/图库图片。所有成功下载的图片，包括重复判退图片，都记入暴露账本。
- 商品 ID、家族键、字节/像素指纹及 dHash 距离 ≤4 同时排除历史开发数据、已消耗的独立 AI 盲测数据，并检查组间和组内隔离。
- 两组共 192 份公开商品画像、3072 条 AI 相关性标签；图片查询打开前先冻结图库画像，排序前冻结全部标签。保留原始标题、零正例查询与未决标签。
- 使用原来的完整 Qwen3-VL-Embedding-2B、Qwen3-VL-Reranker-2B、Qwen3.5 查询图描述模型。未更换权重、降低模型或改小候选池。

## 对照与选择

`original`：上一轮冻结的 20 候选业务重排方案。

`structure`：同一输入、同一候选池，增加主要售卖主体、轮廓、开合、部件排列等可见结构提示，并要求图像证据优先于有冲突的文字描述。

`structure_pixels`：保留结构提示与相同候选池，重排时移除自动描述，只保留查询图片，用于隔离描述带来的影响。自动描述仍参与原来的召回，所以该对照不能证明整个描述链路都无影响。

只在第一组选择：各项配对指标相对原方案不退化后，优先图片 P@10 下界，再比较 Recall 下界；同分保留原方案。选定 `{selected}` 后才打开第二组图片评审；第二组不能重新选方法或调参数。文本分支沿用同一份排序。

方法选择门控：`{json.dumps(selection['nonregression_gates'], ensure_ascii=False)}`。

## 当前判断

- 固定方法的隔离开发验证已完成：`{decision['independent_development_validation_completed']}`。
- 是否已具备冻结后另建新独立 AI 盲测的条件：`{decision['ready_to_freeze_for_new_independent_AI_blind']}`。
- 默认策略改变：`False`。正式验收结果：`null`。本轮新增独立盲测预测：`0`。
- 内部逐项数据与失败结果保存在 `artifacts/module2_structure_development/internal_report.ai.development.md`，不作为对外验收指标。

## 下一轮约束

{decision['next_action']}

已完成的上一批盲测及本轮两组数据均已暴露，不能作为下一轮未见数据。后续采样以 `post_development_exclusion_snapshot.json` 为完整排除账本。下一批独立 AI 盲测必须先冻结最终方法、模型和 AI 评审规则，再采集未见数据，一次固定验证，不在该批上选策略。

已输出 `next_development_coverage_queue.json`，逐条保留两组图片查询的主要主体/外形证据、当前正例与未决数量及补充匹配商品的缺口。它是已暴露开发数据的采集清单，不是验收数据设计；后续应保留原图库和全部查询作补库前后对照。每查询六个匹配商品只是采集诊断，不是整体达标证明，也不能将未决标签改成正例填补缺口。

图库覆盖需要以图片中的真实主要商品主体与可见结构核对，关键词只能帮助找候选。此次保留了“枕头”关键词实际指榴莲果肉、耳机词指收纳包/保护套等噪声；这些样例没有因为难检索而被删除。小型类别配额图库不能代表真实业务流量。

评审由同一 AI 对话完成，知道预定义文本查询及开发背景，标签对模型排序盲。商品家族与像素隔离不等于完全不存在所有语义近重复，也不构成人工或独立标注员验证。

## 复核

`python scripts/verify_structure_development.py` 可从实际模型向量、候选池和原始 logit 重放完整排序、标签区间及隔离检查。运行记录见 `verification.json`、`checks.json` 和 `completion_receipt.json`。
"""
    (PROJECT_ROOT / "docs/module2_structure_development.md").write_text(
        details, encoding="utf-8"
    )
    environment_path = PROJECT_ROOT / "docs/environment_setup.md"
    environment_text = environment_path.read_text(encoding="utf-8")
    if "<!-- module2_structure_development_environment -->" not in environment_text:
        first, rest = environment_text.split("\n", 1)
        supplement = "\n<!-- module2_structure_development_environment -->\n主体结构开发与隔离验证补充：两组新数据均使用既有完整本地 Qwen 嵌入、重排和 Qwen3.5 查询描述模型；无新增依赖或权重下载。新入口为 `scripts/structure_development.py`，重放入口为 `scripts/verify_structure_development.py`。原方案与结构提示/移除描述对照仅用于开发，模型和全部源码摘要绑定在 `artifacts/module2_structure_development/protocol.json`。默认检索入口保持原版，详见[主体结构开发与验证](module2_structure_development.md)。\n"
        environment_path.write_text(first + "\n" + supplement + rest, encoding="utf-8")
    title = "模块二已完成主体结构开发实验和商品家族隔离验证；正式验收指标为空，默认策略保持原版。"
    for rel in ["docs/current_progress.md", "README.md"]:
        path = PROJECT_ROOT / rel
        text = path.read_text(encoding="utf-8")
        marker = "<!-- module2_structure_development -->"
        if marker not in text:
            block = (
                marker
                + "\n"
                + title
                + "\n\n新增两组各 96 个图库商品、16 条查询，全部 AI 评审。先在开发组固定选择，再打开家族隔离验证组；本轮未新增独立盲测。详见 [主体结构开发与验证]("
                + (
                    "module2_structure_development.md"
                    if rel.startswith("docs/")
                    else "docs/module2_structure_development.md"
                )
                + ")。历史独立 AI 盲测结果及其图片分支未达标结论保持不变。\n\n"
            )
            path.write_text(block + text, encoding="utf-8")
    registry_path = PROJECT_ROOT / "configs/evaluation_registry.json"
    registry = load(registry_path)
    prefix = ROOT.relative_to(PROJECT_ROOT).as_posix()
    registry["current_module2_progress_file"] = prefix + "/public_status.json"
    registry["current_development_experiment"] = prefix
    registry["current_development_decision_file"] = prefix + "/decision.json"
    registry["current_structure_candidates_registry"] = QUEUE.relative_to(
        PROJECT_ROOT
    ).as_posix()
    registry["current_structure_media_registry"] = MEDIA.relative_to(
        PROJECT_ROOT
    ).as_posix()
    registry["current_structure_cohorts"] = {
        cohort: dict(
            catalog=prefix + "/" + cohort + "/catalog.public.json",
            index=prefix + "/" + cohort + "/index/manifest.json",
            judgments=prefix + "/" + cohort + "/judgments.ai.jsonl",
            metrics=prefix + "/" + cohort + "/internal_metrics.ai.development.json",
        )
        for cohort in ["tune", "validation"]
    }
    registry["current_preparation_exclusion_snapshot"] = (
        prefix + "/post_development_exclusion_snapshot.json"
    )
    registry["development_candidate_selected"] = selected
    registry["new_independent_AI_blind_after_development_started"] = False
    registry["formal_metrics"] = None
    rules = registry.setdefault("rules", [])
    if not any(r.get("path_prefix") == prefix + "/" for r in rules):
        rules.append(
            dict(
                path_prefix=prefix + "/",
                role="AI_development_and_fixed_family_disjoint_validation_only",
                formal_acceptance_claim_allowed=False,
                development_scores_allowed_in_public_status=False,
                reusable_as_unseen=False,
            )
        )
    write_json(registry_path, registry)
    status = dict(
        stage="structure_development_and_fixed_validation_completed",
        sealed=False,
        selected_method=selected,
        cohorts=2,
        gallery_per_cohort=96,
        queries_per_cohort=16,
        native_AI_profiles=192,
        AI_relations=3072,
        human_labels=0,
        development_metrics=None,
        formal_acceptance=False,
        formal_metrics=None,
        goal_met=False,
        default_strategy_changed=False,
        new_independent_AI_blind_predictions=0,
        previous_independent_AI_blind_preserved=True,
        validation_used_for_parameter_selection=False,
        ready_for_next_independent_AI_blind=decision[
            "ready_to_freeze_for_new_independent_AI_blind"
        ],
        latest_exclusion_snapshot=prefix + "/post_development_exclusion_snapshot.json",
    )
    write_json(ROOT / "public_status.json", status)
    (ROOT / "review_context_disclosure.json").write_text(
        json.dumps(
            dict(
                review_origin="model",
                human_labels=0,
                same_AI_conversation_for_protocol_profiles_and_judgments=True,
                annotator_knew_text_queries=True,
                judgments_frozen_before_model_predictions=True,
                independent_annotator_validation=False,
                family_disjoint_validation=True,
                traffic_representative=False,
                unknown_intervals_exclude_AI_error_and_sampling_uncertainty=True,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            dict(
                status_updated=True,
                decision=decision,
                verification=verification["cohorts"],
            )
        )
    )


def seal() -> None:
    """Seal the completed outputs with their integrity metadata.

    Raises:
        FileExistsError: Preserve sealed stage.

    Notes:
        Writes derived artifacts through dest.parent.mkdir, write_json.
    """
    if (ROOT / "completion_receipt.json").exists():
        raise FileExistsError("Preserve sealed stage")
    verify = load(ROOT / "verification.json")
    assert load(ROOT / "checks.json")["passed"] is True
    release = ROOT / "source_release"
    paths = [
        Path(__file__),
        PROJECT_ROOT / "scripts/structure_development.py",
        PROJECT_ROOT / "scripts/verify_structure_development.py",
        PROJECT_ROOT / "docs/module2_structure_development.md",
        PROJECT_ROOT / "docs/current_progress.md",
        PROJECT_ROOT / "docs/environment_setup.md",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "configs/evaluation_registry.json",
    ]
    for src in paths:
        dest = release / src.relative_to(PROJECT_ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    write_json(
        release / "manifest.json",
        dict(
            files={
                p.relative_to(release).as_posix(): sha256_file(p)
                for p in sorted(release.rglob("*"))
                if p.is_file()
            }
        ),
    )
    status = load(ROOT / "public_status.json")
    status["sealed"] = True
    status["replay_passed"] = True
    write_json(ROOT / "public_status.json", status)
    write_json(
        ROOT / "completion_receipt.json",
        dict(
            files={
                p.relative_to(ROOT).as_posix(): sha256_file(p)
                for p in sorted(ROOT.rglob("*"))
                if p.is_file()
                and p.name not in {"public_status.json", "completion_receipt.json"}
            },
            project_files={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                for p in [QUEUE, MEDIA]
            },
            previous_sealed_stages=verify["prior_sealed_stages_verified"],
            formal_acceptance=False,
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            new_independent_AI_blind_predictions=0,
            reusable_as_unseen=False,
        ),
    )
    print(
        json.dumps(
            dict(
                sealed=True, files=len(load(ROOT / "completion_receipt.json")["files"])
            )
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["report", "seal"])
    globals()[parser.parse_args().action]()
