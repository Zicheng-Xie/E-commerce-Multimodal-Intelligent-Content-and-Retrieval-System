"""Render the completed FG experiment without selecting variants after results."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import run  # noqa: F401
from compare_fgclip2_retrieval import COHORTS, PARENT, ROOT, check, load, relative
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

PRIMARY = "fgclip2_local_fusion20"
BASELINE = "cross20"
BASE_ALIAS = "cross_evidence_pool60"
METRICS = ["precision@10", "recall@10", "ap", "ndcg@10", "mrr", "hit@1"]


DISPLAY: dict[str, str] = {
    "cross20": "原 Qwen 20候选",
    "fgclip2_standalone": "FG-CLIP 2 独立评分",
    "fgclip2_global_fusion20": "新增全局分支消融",
    "fgclip2_local_fusion20": "新增局部融合方法",
}


COHORT_DISPLAY: dict[str, str] = {
    "tune": "开发组 A",
    "regression": "开发组 B",
    "fresh": "已曝光图片回归组",
}


def percent(bounds: dict, *, delta: bool = False) -> str:
    """Return percent for this workflow.

    Args:
        bounds: Bounds used by this operation.
        delta: Delta used by this operation.

    Returns:
        Text produced by this operation.
    """
    if bounds[0] is None:
        return "未定义"

    def value(x: np.ndarray) -> str:
        """Return value for this workflow.

        Args:
            x: X used by this operation.

        Returns:
            Text produced by this operation.
        """
        return f"{100 * x:+.2f}" if delta else f"{100 * x:.2f}"

    if abs(bounds[0] - bounds[1]) < 1e-10:
        return value(bounds[0])
    return value(bounds[0]) + "～" + value(bounds[1])


def freeze() -> None:
    """Freeze the inputs and policy used by this workflow.

    Raises:
        FileExistsError: Preserve report protocol.

    Notes:
        Writes derived artifacts through write_json.
    """
    if (ROOT / "report_protocol.json").exists():
        raise FileExistsError("Preserve report protocol")
    write_json(
        ROOT / "report_protocol.json",
        dict(
            files={relative(Path(__file__)): sha256_file(Path(__file__))},
            main_method=PRIMARY,
            baseline=BASELINE,
            variant_selection=False,
            report_all_completed_methods=True,
            gate="Primary versus original: all six shared-label paired lower bounds >= 0 in every cohort and scope. Development only; default never automatically changed.",
            public_summary_has_no_development_quality_scores=True,
            formal_metrics=None,
        ),
    )


def report() -> None:
    """Summarize the recorded results of this development workflow.

    Notes:
        Writes derived artifacts through (ROOT /
        'internal_comparison.ai.development.md').write_text, write_json.
    """
    _, inputs = check()
    assert load(ROOT / "report_protocol.json")["files"][
        relative(Path(__file__))
    ] == sha256_file(Path(__file__))
    assert load(ROOT / "verification.json")["passed"]
    comparisons: dict
    comparisons, cases, failed, text = {}, [], [], []
    header = [
        "# FG-CLIP 2 与原 Qwen 方法开发对照",
        "",
        "仅供内部开发诊断。全部40条查询均已曝光；不是独立验收。图库、查询、6720条AI关系和K=10完全相同，未改变未知标签。区间表示未知关系所有可能取值的精确范围，不是抽样置信区间，也不覆盖AI误判。两组168件图库共享72件商品；最后8张查询使用开发组A图库。",
        "",
        "主方法在查看新预测前固定为 `fgclip2_local_fusion20`，没有事后选择最好变体或继续调权重。两种融合与原方法使用完全相同的20候选和原始Qwen分数；独立FG评分覆盖全图库，属于单模型对照，计算预算不同。",
        "",
    ]
    for cohort in COHORTS:
        metrics = load(ROOT / cohort / "internal_metrics.ai.development.json")
        paired = load(ROOT / cohort / "internal_paired_differences.json")
        comparisons[cohort] = {}
        text.extend(
            [
                f"## {COHORT_DISPLAY[cohort]}",
                "",
                "| 范围 | 方法 | P@10 % | Recall@10 % | mAP % | NDCG@10 % | MRR % | Hit@1 % |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for scope, methods in metrics["scopes"].items():
            if (
                scope == "all"
                and set(methods) == set(metrics["scopes"].get("image", {}))
                and cohort == "fresh"
            ):
                continue
            for method in DISPLAY:
                ranges = methods[method]["ranges"]
                text.append(
                    "| "
                    + " | ".join(
                        [scope, DISPLAY[method], *[percent(ranges[m]) for m in METRICS]]
                    )
                    + " |"
                )
            delta = paired["scopes"][scope][PRIMARY][BASE_ALIAS]
            nonregression = all(
                delta[m]["nonregression_proven_for_shared_unknown_labels"]
                for m in METRICS
            )
            if not nonregression:
                failed.append(
                    dict(
                        cohort=cohort,
                        scope=scope,
                        metrics=[
                            m
                            for m in METRICS
                            if not delta[m][
                                "nonregression_proven_for_shared_unknown_labels"
                            ]
                        ],
                    )
                )
            comparisons[cohort][scope] = dict(
                original=methods[BASELINE],
                candidate=methods[PRIMARY],
                shared_completion_paired=delta,
                six_metric_nonregression_proven=nonregression,
            )
        text.extend(
            [
                "",
                "主方法减原方法，单位为百分点；逐项按同一未知标签取值比较：",
                "",
                "| 范围 | 指标 | 差值范围 | 不退化可证明 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for scope, row in comparisons[cohort].items():
            for metric, evidence in row["shared_completion_paired"].items():
                text.append(
                    f"| {scope} | {metric} | {percent(evidence['delta_range'], delta=True)} | {evidence['nonregression_proven_for_shared_unknown_labels']} |"
                )
        text.append("")
        ranks = load(ROOT / cohort / "predictions.json")
        grades = {
            (row["query_id"], row["candidate_id"]): row["grade"]
            for row in read_rows(PROJECT_ROOT / inputs["cohorts"][cohort]["labels"])
        }
        for qid, orders in ranks.items():
            before, after = orders[BASELINE][:10], orders[PRIMARY][:10]

            def known(order: list[str]) -> float:
                """Return known for this workflow.

                Args:
                    order: Order used by this operation.

                Returns:
                    Numeric result computed from the supplied values.
                """
                return sum(grades[(qid, c)] == 1 for c in order)

            def unknown(order: list[str]) -> float:
                """Return unknown for this workflow.

                Args:
                    order: Order used by this operation.

                Returns:
                    Numeric result computed from the supplied values.
                """
                return sum(grades[(qid, c)] is None for c in order)

            cases.append(
                dict(
                    cohort=cohort,
                    query_id=qid,
                    mode=qid.split("_")[0],
                    known_hits_before=known(before),
                    known_hits_after=known(after),
                    unknown_before=unknown(before),
                    unknown_after=unknown(after),
                    entered_top10=[c for c in after if c not in before],
                    left_top10=[c for c in before if c not in after],
                    original_pool20=load(
                        PROJECT_ROOT
                        / next(
                            q["original_cross"]
                            for q in inputs["cohorts"][cohort]["queries"]
                            if q["id"] == qid
                        )
                    )["pool"][:20],
                    fg_top20_additional_known_positive=[
                        c
                        for c in orders["fgclip2_standalone"][:20]
                        if c not in orders[BASELINE][:20] and grades[(qid, c)] == 1
                    ],
                )
            )
    improved = [r for r in cases if r["known_hits_after"] > r["known_hits_before"]]
    worsened = [r for r in cases if r["known_hits_after"] < r["known_hits_before"]]
    runtime = load(ROOT / "encoding_runtime.json")
    image_calls = sum(r["kind"] == "image_global_and_dense" for r in runtime["calls"])
    text_calls = len(runtime["calls"]) - image_calls
    summary = dict(
        main_method=PRIMARY,
        baseline=BASELINE,
        exposed_development_queries=len(cases),
        known_hit_improved_queries=len(improved),
        known_hit_worsened_queries=len(worsened),
        known_hit_unchanged_queries=len(cases) - len(improved) - len(worsened),
        six_metric_development_nonregression_gate_passed=not failed,
        failed_or_unproven_scopes=failed,
        full_model_parameters=runtime["parameter_count"],
        actual_new_image_requests=image_calls,
        actual_new_vision_forward_calls=2 * image_calls,
        actual_new_text_forward_calls=text_calls,
        actual_new_FG_forward_calls=2 * image_calls + text_calls,
        unique_global_texts=len(inputs["texts"]),
        unique_regional_texts=len(inputs["box_texts"]),
        reused_original_full_Qwen_pairs=800,
        new_Qwen_pairs=0,
        technical_probe_not_in_quality_results=True,
        default_strategy_changed=False,
        formal_acceptance=False,
        formal_metrics=None,
        goal_met=False,
        new_independent_AI_blind_predictions=0,
    )
    write_json(
        ROOT / "internal_before_after.json",
        dict(comparisons=comparisons, summary=summary),
    )
    write_json(
        ROOT / "internal_query_changes.json", dict(rows=cases, judgments_unchanged=True)
    )
    write_json(ROOT / "decision.json", summary)
    counts = [
        f"主方法相对原方法，已确认Top10命中数增加的查询{len(improved)}条、减少{len(worsened)}条、相同{len(cases) - len(improved) - len(worsened)}条。这只是已知标签诊断，未知项仍以完整区间报告。",
        "",
        f"六指标开发不退化门槛：{not failed}。默认策略保持原版。当前开发成绩不能用于正式60%/85%验收。",
        "",
        f"完整模型{runtime['parameter_count']:,}参数，{runtime['dtype']}，设备{runtime['device']}。新增{image_calls}次图片请求（每次全局、密集特征各一次完整视觉前向，共{image_calls * 2}次），{text_calls}次文本前向。原Qwen分数原样复用800对；新增Qwen精排0对。",
        "",
        f"模型加载{runtime['load_seconds']:.2f}秒；PyTorch峰值分配{runtime['peak_allocated_mib']:.1f}MiB、峰值保留{runtime['peak_reserved_mib']:.1f}MiB。该资源统计不等于整卡占用；逐图预处理、特征读写和完整在线时延另计。文本按官方196/64 token上限处理，截断明细保存在特征旁的JSON中。",
        "",
        "原独立AI盲测及本轮全部开发查询均保持已曝光状态。下一轮验收必须在方法冻结后另取全新查询与全新图库。",
    ]
    (ROOT / "internal_comparison.ai.development.md").write_text(
        "\n".join(header + counts + [""] + text) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def seal() -> None:
    """Seal the completed outputs with their integrity metadata.

    Notes:
        Writes derived artifacts through release.mkdir, write_json.
    """
    assert load(ROOT / "verification.json")["passed"]
    assert load(ROOT / "checks.json")["passed"]
    check()
    release = ROOT / "source_release"
    paths = [
        PROJECT_ROOT / "src/multimodal_ecommerce/fgclip_retrieval.py",
        PROJECT_ROOT / "scripts/prepare_fgclip2_model.py",
        PROJECT_ROOT / "scripts/compare_fgclip2_retrieval.py",
        Path(__file__),
        PROJECT_ROOT / "tests/test_fgclip_retrieval.py",
    ]
    release.mkdir(exist_ok=True)
    manifest = {}
    for source in paths:
        target = release / source.name
        if target.exists():
            assert sha256_file(target) == sha256_file(source)
        else:
            shutil.copyfile(source, target)
        manifest[relative(source)] = sha256_file(source)
    write_json(release / "manifest.json", manifest)
    files = {
        p.relative_to(ROOT).as_posix(): sha256_file(p)
        for p in sorted(ROOT.rglob("*"))
        if p.is_file()
        and p.name not in {"completion_receipt.json", "public_status.json"}
    }
    receipt = dict(
        files=files,
        source_files=manifest,
        parent_completion_sha256=sha256_file(PARENT / "completion_receipt.json"),
        formal_acceptance=False,
        formal_metrics=None,
        default_strategy_changed=False,
    )
    if (ROOT / "completion_receipt.json").exists():
        assert load(ROOT / "completion_receipt.json") == receipt
    else:
        write_json(ROOT / "completion_receipt.json", receipt)
    write_json(
        ROOT / "public_status.json",
        dict(
            role="FG_CLIP2_exposed_AI_development_comparison",
            completed=True,
            sealed=True,
            quality_metrics=None,
            formal_metrics=None,
            default_strategy_changed=False,
            goal_met=False,
            independent_AI_blind_predictions=0,
            internal_report="internal_comparison.ai.development.md",
        ),
    )
    print(json.dumps(dict(sealed_files=len(files))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "report", "seal"])
    args = parser.parse_args()
    globals()[args.action]()
