"""Replay adaptive guards and verify a protected fresh request from real caches."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

import run  # noqa: F401
from advance_focus_coverage import ROOT, load
from compare_focus_headguards import POLICY as HEAD_POLICY
from compare_region_prefix import NAME, POLICY, route_guard
from multimodal_ecommerce.ai_review_metrics import metric_ranges
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.head_guard import protected_prefix
from multimodal_ecommerce.intent_retrieval import rerank_complete
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from report_focus_coverage import paired_differences


def without_method(value: object, name: str) -> Any:
    """Return without method for this workflow.

    Args:
        value: Input value to validate or transform.
        name: Name used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    result = json.loads(json.dumps(value))
    for scope in result["scopes"].values():
        scope.pop(name)
    if "query_metrics" in result:
        result["query_metrics"] = [
            r for r in result["query_metrics"] if r["method"] != name
        ]
    if "query_rows" in result:
        for row in result["query_rows"]:
            row["ranges"].pop(name)
    return result


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through report.write_text, write_json.
    """
    head = ROOT / "head_guard"
    region = ROOT / "region_guard"
    verify_files(head, load(head / "receipt.json")["files"])
    hp = load(head / "protocol.json")
    verify_files(PROJECT_ROOT, hp["files"])
    assert hp["policy"] == HEAD_POLICY
    original = load(ROOT / "rankings.json")
    rankings = {
        qid: {name: list(order) for name, order in methods.items()}
        for qid, methods in original.items()
    }
    for qid, methods in rankings.items():
        for candidate in HEAD_POLICY["candidate_methods"]:
            for keep in HEAD_POLICY["keep_prefix"]:
                methods[f"protected_top{keep}_{candidate}"] = protected_prefix(
                    original[qid][HEAD_POLICY["anchor"]], original[qid][candidate], keep
                )
    assert rankings == load(head / "rankings.json")
    image_modes = {
        q["id"]: bool(q["image_path"])
        for q in load(ROOT / "inputs.public.json")["queries"]
    }
    final = route_guard(rankings, image_modes)
    assert final == load(region / "rankings.json")
    assert all(
        final[q][NAME] == original[q]["cross_evidence_pool60"]
        for q in image_modes
        if not image_modes[q]
    )
    assert all(
        final[q][NAME][:2] == original[q]["region_title"][:2]
        for q in image_modes
        if image_modes[q]
    )
    rp = load(region / "protocol.json")
    verify_files(PROJECT_ROOT, rp["files"])
    assert rp["policy"] == POLICY
    verify_files(region, load(region / "receipt.json")["files"])
    repair = load(region / "modality_repair.json")
    old = ROOT / repair["original_execution_preserved"]
    assert (
        sha256_file(old / "compare_region_prefix.py")
        == repair["original_source_sha256"]
    )
    assert sha256_file(old / "protocol.json") == repair["original_protocol_sha256"]
    verify_files(old, load(old / "receipt.json")["files"])
    bad = {qid: dict(methods) for qid, methods in rankings.items()}
    for methods in bad.values():
        methods[NAME] = protected_prefix(
            methods["region_title"], methods["described_cross_pool20"], 2
        )
    assert bad == load(old / "rankings.json")
    assert repair["discovered_before_inspecting_this_followups_metric_values"] is True
    assert (
        repair["changes_relevance_labels"] is False
        and repair["changes_model_inputs_or_scores"] is False
    )
    labels = read_rows(ROOT / "judgments.ai.jsonl")
    modes = {q: "image" if image_modes[q] else "text" for q in image_modes}
    metrics = metric_ranges(final, labels, modes)
    paired = paired_differences(final, labels, modes)
    assert metrics == load(region / "internal_metrics.ai.development.json")
    assert paired == load(region / "internal_paired_differences.json")
    assert without_method(metrics, NAME) == load(
        head / "internal_metrics.ai.development.json"
    )
    assert without_method(paired, NAME) == load(
        head / "internal_paired_differences.json"
    )
    demo = load(ROOT / "demo_region_protected/results.json")
    verify_files(PROJECT_ROOT, demo["source_files"])
    assert (
        demo["method"] == NAME
        and demo["catalog_size"] == 418
        and demo["candidate_count"] == 417
    )
    assert (
        len(demo["results"]) == 10
        and len(demo["full_order"]) == len(set(demo["full_order"])) == 417
    )
    assert (
        demo["new_model_inferences"] == 0
        and demo["actual_cached_full_cross_pairs_reused"] == 20
    )
    assert (
        demo["historical_evaluation_query_rankings_used"] is False
        and demo["relevance_labels_used_as_runtime_features"] is False
    )
    parent = PROJECT_ROOT / demo["source_query_run"]
    fusion = load(parent / "query_fusion.json")
    cross = load(parent / "cross_encoder.json")
    candidate_order = rerank_complete(
        fusion["ids"],
        np.array(fusion["fused_scores"]),
        cross["pool"][:20],
        np.array(cross["logits"][:20]),
    )
    assert candidate_order == demo["underlying_joint_order"]
    assert (
        protected_prefix(demo["anchor_order"], candidate_order, 2) == demo["full_order"]
    )
    assert demo["full_order"][:2] == demo["anchor_order"][:2]

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
        "# 前缀保护：内部自适应开发跟进",
        "",
        "这一跟进在观察到16方法比较中的MRR/Hit@1回退后提出，不能称为独立验证。原16方法及全部负结果保持不变。先冻结cross_evidence_pool60前1/2位的四个变体，再冻结一个区域基线前两位保护。当前阶段不再追加其他组合。",
        "",
        "区域策略的首次离线实现把图片保护误用于文本；原执行、源码和协议完整保留在modality_mismatch_execution/。在检查其数值前发现静态模态不一致，修正为图片保护、文本保持原cross_evidence_pool60锚点，并加入回归测试。相关性标签、模型输入及原始分数均未改。",
        "",
        "|图片方法|P@10|R@10|mAP|NDCG@10|MRR|Hit@1|",
        "|---|---|---|---|---|---|---|",
    ]
    for name in [
        "global",
        "region_title",
        "cross_evidence_pool60",
        "described_cross_pool20",
        "described_cross_pool60",
        *[n for n in metrics["scopes"]["image"] if n.startswith("protected_")],
        NAME,
    ]:
        entry = metrics["scopes"]["image"][name]["ranges"]
        lines.append(
            "|"
            + name
            + "|"
            + "|".join(
                percent(entry[m])
                for m in ["precision@10", "recall@10", "ap", "ndcg@10", "mrr", "hit@1"]
            )
            + "|"
        )
    lines += [
        "",
        "保护前缀保证对应锚点的Hit@1不变；MRR还要求原首个相关结果位于所保留前缀内，不能作无条件保证。当前AI未知标签范围下，仍不能证明所有指标对两个原始基线同时不退化；不升级默认策略，不将开发分数当作正式验收。",
        "",
        "区域护头已在一个自动描述、ROI、真实完整模型精排的新请求上执行。复用的是该次请求已实际计算的20对原始logit及查询向量，未读取历史评测查询排名或标签作为运行特征，不新增模型推理。",
        "",
    ]
    report = region / "internal_report.md"
    content = "\n".join(lines)
    if report.exists():
        assert report.read_text(encoding="utf-8") == content
    else:
        report.write_text(content, encoding="utf-8")
    result = dict(
        original_methods_preserved=16,
        cross_anchor_followup_methods=4,
        region_anchor_followup_methods=1,
        total_methods=21,
        total_full_rankings=378,
        frozen_AI_relations=7519,
        labels_changed=False,
        modality_bug_original_execution_preserved=True,
        text_branch_invariance_verified=True,
        protected_prefix_actual_fresh_query_verified=True,
        actual_fresh_cached_cross_pairs_reused=20,
        new_model_inferences_for_guards=0,
        formal_metrics=None,
        goal_met=False,
        acceptance_predictions=0,
    )
    path = ROOT / "guard_verification.json"
    if path.exists():
        assert load(path) == result
    else:
        write_json(path, result)
    if (ROOT / "completion_receipt.json").exists():
        verify_files(ROOT, load(ROOT / "completion_receipt.json")["files"])
    print(json.dumps(result))


if __name__ == "__main__":
    main()
