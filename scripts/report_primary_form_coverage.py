"""Audit, document and seal adaptive development without acceptance claims."""

from __future__ import annotations

import argparse
import json
import shutil

import run  # noqa: F401
from evaluate_primary_form_coverage import paired, persist
from multimodal_ecommerce.ai_review_metrics import optional_mean_bounds
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from primary_form_coverage import (
    FRESH_MEDIA,
    FRESH_QUEUE,
    MEDIA,
    PARENT,
    QUEUE,
    ROOT,
    load,
)


def capacity_detail(metrics: dict) -> dict:
    """Return capacity detail for this workflow.

    Args:
        metrics: Metrics used by this operation.

    Returns:
        Result mapping produced by the workflow.
    """
    result = {}
    for scope in metrics["scopes"]:
        rows = [
            r
            for r in metrics["query_capacity"]
            if scope == "all" or r["query_id"].startswith(scope)
        ]
        mandatory, optional = [], []
        for r in rows:
            p, u = r["known_positive"], r["unknown"]
            if p:
                mandatory.append((min(1, 10 / (p + u)), min(1, 10 / p)))
            elif u:
                optional.append((min(1, 10 / u), 1.0))
        result[scope] = dict(
            precision_oracle_range=[
                sum(r["precision_ceiling_range"][i] for r in rows) / len(rows)
                for i in [0, 1]
            ],
            recall_oracle_range=list(optional_mean_bounds(mandatory, optional)),
            fixed_k=10,
            positive_counts_are_full_catalog_not_just_pool=True,
        )
    return result


def audit() -> None:
    """Apply the audit workflow step."""
    protocol = load(ROOT / "protocol.json")
    assert sha256_file(PROJECT_ROOT / protocol["source"]) == protocol["source_sha256"]
    verify_files(ROOT, load(ROOT / "profile_receipt.json")["files"])
    if (ROOT / "completion_receipt.json").exists():
        verify_files(
            ROOT / "source_release",
            load(ROOT / "source_release/manifest.json")["files"],
        )
    docs = load(ROOT / "additions/catalog.public.json")
    profiles = read_rows(ROOT / "additions/profiles.public.jsonl")
    native = load(ROOT / "native/profiles.json")
    assert profiles == [
        dict(id=d["id"], **native[str(i)]) for i, d in enumerate(docs, 1)
    ]
    observations = load(ROOT / "native/judgments.json")
    combined = {}
    comparisons = {}
    for cohort, old in [("tune", "tune"), ("regression", "validation")]:
        folder = ROOT / cohort
        rows = read_rows(folder / "judgments.ai.jsonl")
        assert rows[:1536] == read_rows(PARENT / old / "judgments.ai.jsonl")
        added = rows[1536:]
        assert len(added) == 16 * 72
        for row in added:
            pos = row["addition_position"]
            v = observations[cohort][row["query_id"]]
            grade = (
                1 if pos in v["positive"] else None if str(pos) in v["unknown"] else 0
            )
            assert row["candidate_id"] == docs[pos - 1]["id"] and row["grade"] == grade
            assert row["review_origin"] == "model" and row["human_label"] is False
        ranks = load(folder / "predictions.json")
        metrics = load(folder / "internal_metrics.ai.development.json")
        assert paired(
            ranks, rows, {q: q.split("_")[0] for q in ranks}, metrics
        ) == load(folder / "internal_paired_differences.json")
        old_metrics = load(PARENT / old / "internal_metrics.ai.development.json")
        comparisons[cohort] = dict(
            before_gallery=96,
            after_gallery=168,
            unchanged_queries=16,
            unchanged_existing_labels=1536,
            before={s: v["original"] for s, v in old_metrics["scopes"].items()},
            after_same_algorithm={
                s: v["cross20"] for s, v in metrics["scopes"].items()
            },
            before_oracles=capacity_detail(old_metrics),
            after_oracles=capacity_detail(metrics),
            interpretation="Adaptive coverage intervention with the same original algorithm. Catalog and full relevance denominators change; this is not a fixed-catalog ranking improvement or blind validation.",
        )
        combined[cohort] = dict(
            relations=len(rows), unknown=sum(r["grade"] is None for r in rows)
        )
    selected = load(ROOT / "selection.json")
    pm = load(ROOT / "tune/internal_paired_differences.json")["scopes"]
    mm = load(ROOT / "tune/internal_metrics.ai.development.json")["scopes"]
    gate = all(
        v["nonregression_proven_for_shared_unknown_labels"]
        for s in ["all", "text", "image"]
        for v in pm[s]["cross60"]["cross_evidence_pool60"].values()
    )
    eligible = ["cross20"] + (["cross60"] if gate else [])
    expected = max(
        eligible,
        key=lambda n: (
            mm["image"][n]["ranges"]["precision@10"][0],
            mm["image"][n]["ranges"]["recall@10"][0] or 0,
            n == "cross20",
        ),
    )
    assert selected["selected"] == expected and selected["eligible"] == eligible
    assert selected["cross60_nonregression_gate"] == gate
    assert (ROOT / "selection_receipt.json").stat().st_mtime_ns < (
        ROOT / "fresh/protocol.json"
    ).stat().st_mtime_ns
    fresh = ROOT / "fresh"
    fresh_rows = read_rows(fresh / "judgments.ai.jsonl")
    native_fresh = load(fresh / "judgments.native.json")
    gallery = load(ROOT / "tune/catalog.public.json")
    for row in fresh_rows:
        pos = row["native_position"]
        v = native_fresh[row["query_id"]]
        grade = 1 if pos in v["positive"] else None if str(pos) in v["unknown"] else 0
        assert row["grade"] == grade and row["candidate_id"] == gallery[pos - 1]["id"]
        assert row["review_origin"] == "model" and row["human_label"] is False
    assert len(fresh_rows) == 8 * 168
    assert (fresh / "label_receipt.json").stat().st_mtime_ns < (
        fresh / "predictions.json"
    ).stat().st_mtime_ns
    combined["fresh"] = dict(
        relations=len(fresh_rows), unknown=sum(r["grade"] is None for r in fresh_rows)
    )
    comparisons["fresh"] = dict(
        oracles=capacity_detail(load(fresh / "internal_metrics.ai.development.json")),
        query_families_disjoint=True,
        gallery_independent=False,
    )
    persist(ROOT / "internal_before_after_and_capacity.json", comparisons)
    worklist = []
    for cohort in ["tune", "regression", "fresh"]:
        metrics = load(ROOT / cohort / "internal_metrics.ai.development.json")
        attribution = load(ROOT / cohort / "internal_error_attribution.json")["rows"]
        by_query = {r["query_id"]: r for r in attribution if r["method"] == "cross20"}
        for row in metrics["query_capacity"]:
            p, u = row["known_positive"], row["unknown"]
            a = by_query[row["query_id"]]
            worklist.append(
                dict(
                    cohort=cohort,
                    exposed_query_id=row["query_id"],
                    known_positive=p,
                    unknown=u,
                    precision_capacity=row["precision_ceiling_range"],
                    precision_60_impossible_even_with_optimistic_unknowns=row[
                        "precision_ceiling_range"
                    ][1]
                    < 0.6,
                    additional_known_matches_to_six=max(0, 6 - p),
                    known_pool20_capacity_gap=a["known_oracle_at_10"]
                    - a["known_oracle_in_pool20"],
                    known_ranking_loss_inside_pool20=a["known_oracle_in_pool20"]
                    - a["known_hits_at_10"],
                    recall_85_impossible_from_known_positive_count=p > 10 / 0.85,
                    reusable_as_unseen=False,
                )
            )
    persist(
        ROOT / "next_development_worklist.json",
        dict(
            role="posthoc_exposed_development_diagnosis_not_runtime_features_or_acceptance",
            entries=worklist,
            original_queries_and_labels_must_be_retained=True,
            never_shrink_recall_denominators_or_relabel_to_pass=True,
            six_matches_is_a_per_query_collection_diagnostic_not_macro_goal_proof=True,
            later_selection_requires_another_fresh_query_cohort=True,
            next_acceptance_requires_both_new_queries_and_new_gallery=True,
        ),
    )
    persist(
        ROOT / "review_audit.json",
        dict(
            cohorts=combined,
            original_labels_unchanged=True,
            selection_replayed=True,
            original_regression_cohort_is_exposed_development=True,
            review_origin="model",
            human_labels=0,
            same_AI_conversation=True,
            rank_blinded_judgments=True,
            independent_annotator_validation=False,
        ),
    )
    print(json.dumps(dict(review_audit_passed=True, cohorts=combined)), flush=True)


def report() -> None:
    """Summarize the recorded results of this development workflow.

    Raises:
        FileExistsError: Preserve sealed development stage.

    Notes:
        Writes derived artifacts through (PROJECT_ROOT /
        'docs/module2_primary_form_coverage.md').write_text, (ROOT /
        'internal_report.ai.development.md').write_text, env.write_text,
        path.write_text, write_json.
    """
    if (ROOT / "completion_receipt.json").exists():
        raise FileExistsError("Preserve sealed development stage")
    verification = load(ROOT / "verification.json")
    review = load(ROOT / "review_audit.json")
    assert sum(v["relations"] for v in review["cohorts"].values()) == 6720
    selection = load(ROOT / "selection.json")
    decision = load(ROOT / "decision.json")
    comparisons = load(ROOT / "internal_before_after_and_capacity.json")
    internal = [
        "# 内部开发结果，不是正式验收\n",
        "同一 AI 对话评审，标签对扩充图库排序盲。区间只涵盖未决标签补全，不涵盖 AI 误判或抽样误差。",
        "补库使用了已暴露的两组查询，不能把补库增益当成未见查询或真实流量增益。\n",
    ]
    for cohort in ["tune", "regression", "fresh"]:
        metrics = load(ROOT / cohort / "internal_metrics.ai.development.json")
        internal += [
            f"\n## {cohort}\n",
            "|范围|方法|P@10|R@10|mAP|NDCG@10|MRR|Hit@1|",
            "|---|---|---|---|---|---|---|---|",
        ]
        for scope, methods in metrics["scopes"].items():
            for name, row in methods.items():
                fields = [
                    (
                        "未定义"
                        if row["ranges"][k][0] is None
                        else "–".join(f"{100 * v:.2f}%" for v in row["ranges"][k])
                    )
                    for k in [
                        "precision@10",
                        "recall@10",
                        "ap",
                        "ndcg@10",
                        "mrr",
                        "hit@1",
                    ]
                ]
                internal.append("|" + "|".join([scope, name, *fields]) + "|")
        cap = comparisons[cohort].get(
            "after_oracles", comparisons[cohort].get("oracles")
        )
        internal += [
            "\n图库可达上限：`" + json.dumps(cap, ensure_ascii=False) + "`。",
            "完整逐查询指标、候选池容量和排序损失见 internal_metrics.ai.development.json 与 internal_error_attribution.json。",
        ]
    internal += [
        "\n## 同算法补库前后\n",
        "```json",
        json.dumps(comparisons, ensure_ascii=False, indent=2),
        "```",
        "\n## 选择与决策\n",
        "```json",
        json.dumps(
            dict(selection=selection, decision=decision), ensure_ascii=False, indent=2
        ),
        "```",
    ]
    (ROOT / "internal_report.ai.development.md").write_text(
        "\n".join(internal) + "\n", encoding="utf-8"
    )
    total_new = sum(
        v["actual_new_full_pairs"] for v in verification["cohorts"].values()
    )
    reused = sum(v["reused_exact_raw_pairs"] for v in verification["cohorts"].values())
    doc = f"""# 模块二：按主体外形补库与新查询开发验证

本轮新增 72 件看图评审的开发图库商品，保留原来的两组各 96 件图库、全部查询和已有标签，形成两组各 168 件开发图库。两组图库共享这 72 件新增商品。补库依据两组已经暴露的查询，因此两组现在均属于开发/回归数据。本轮没有新增独立图库验收结果，默认策略未升级。

## 已完成

- 固定 9 个外形采集方向：头戴耳机、尖头细跟短靴、拱形颈枕、无袖羽绒马甲、玻璃餐边柜、带抽屉餐边柜、双把手吸管杯、厚底运动鞋、榴莲果肉。
- 从固定来源区间按哈希顺序准备 108 条候选元数据，每方向取 8 张可读且去重通过的图片；没有按相关性或指标挑掉困难商品。耳罩套、智能眼镜、敞口杯和外形不匹配的柜体均保留。
- 72 份新增公开商品画像在新图库排序前冻结；新增 2304 条 AI 关系，原 3072 条标签逐条保持不变。两组共 5376 条关系。
- 使用相同查询图片、相同自动描述、完整本地 Qwen 权重和区域注意力配置，只比较 20 与 60 个重排候选；原方案 20 候选同时构成补库前后的同算法对照。
- 排序方法只按 tune 组预先冻结的规则选择，选定 `{selection['selected']}` 后，才采集并打开另外 8 张新来源图片查询。新查询商品家族与所有历史暴露数据及新增图库隔离；图库共享，所以这是新查询开发验证，不能称为独立图库盲测。
- 新查询共 1344 条 AI 关系；本轮完整记录共 6720 条关系（含原有标签）。所有不确定关系保留 null，零正例查询不删除。
- 实际新增完整重排计算 {total_new} 对，复用 {reused} 对输入完全相同的历史原始 logit；缓存来源、摘要、模型身份、每对来源和时延均保留。模型推理未使用评审标签。

## 如何解释结果

补库前后图库不同，Recall 的真实正例分母也不同；补库后有更多相关商品，不等于排序变差或变好。内部报告同时保留原始 P@10、Recall@10、mAP、NDCG@10、MRR、Hit@1、完整相关商品数量以及固定 Top-10 的理论容量。

当宽泛文本需求在图库中有超过 10 个相关商品时，即使完美排序，Recall@10 也受 `min(10,R)/R` 限制。本轮不缩小分母、不把未知标签当负例、不改 K、不替换原始 Recall。容量诊断只解释瓶颈，不代替用户要求的指标。

新图库内部的 20/60 候选对照使用完全相同的标签、图库、特征、模型和排序指令，用于判断候选预算的影响。内部详细数值保存在 `internal_report.ai.development.md`，不作为对外验收成绩。

## 当前状态

- 新查询开发验证完成：`True`。
- 新查询图片 Precision 硬目标是否通过：`{decision['fresh_image_precision_threshold_met']}`。
- 新查询图片 Recall 优先目标是否通过：`{decision['fresh_image_recall_preference_met']}`。
- 独立图库验收：`False`。正式验收指标：`null`。默认策略改变：`False`。本轮新增独立盲测预测：`0`。

此前独立 AI 盲测的原始结果保留，它仍是已经消耗的数据，不能重新当成未见集。后续如要冻结下一轮验收方案，必须使用全新查询和全新图库，并先冻结模型、方法、采样与 AI 评审协议；本轮新查询也不能继续当作下一轮未见数据。

已输出 `next_development_worklist.json`，把全部40条已暴露查询分成图库容量不足、20候选池漏召回、池内排序损失和固定Top-10的Recall容量约束，供后续开发定位使用。该清单不参与本轮推理，不能用于删掉困难查询或改动相关性标签。

评审由同一 AI 对话完成，知道开发背景；不是人工真值、独立标注员验证或真实业务流量评估。未决区间也不是统计置信区间。

## 复核入口

`python scripts/evaluate_primary_form_coverage.py verify` 重放隔离、实际向量、候选池、原始分数和全排序；`python scripts/report_primary_form_coverage.py audit` 核对新旧标签、方法选择与理论容量。完整检查见 `verification.json`、`review_audit.json`、`checks.json` 和 `completion_receipt.json`。
"""
    (PROJECT_ROOT / "docs/module2_primary_form_coverage.md").write_text(
        doc, encoding="utf-8"
    )
    marker = "<!-- module2_primary_form_coverage -->"
    for rel in ["README.md", "docs/current_progress.md"]:
        path = PROJECT_ROOT / rel
        text = path.read_text(encoding="utf-8")
        if marker not in text:
            link = (
                "docs/module2_primary_form_coverage.md"
                if rel == "README.md"
                else "module2_primary_form_coverage.md"
            )
            block = (
                marker
                + "\n模块二已完成按主体外形补库、20/60 候选对照和新查询家族隔离的开发验证。新增72件开发商品，原查询与已有标签保留；正式验收指标为空，默认策略未升级。本轮新查询使用共享图库，不能作为独立图库盲测结果。详见[按外形补库与验证]("
                + link
                + ")。\n\n"
            )
            path.write_text(block + text, encoding="utf-8")
    env = PROJECT_ROOT / "docs/environment_setup.md"
    text = env.read_text(encoding="utf-8")
    if marker not in text:
        first, rest = text.split("\n", 1)
        env.write_text(
            first
            + "\n\n"
            + marker
            + "\n按主体外形补库仍使用既有完整本地模型和依赖，无新模型或依赖下载。新入口为 `scripts/primary_form_coverage.py`；索引引用已核验的历史向量和裁切图，仅对新增商品实际编码。重放入口为 `scripts/evaluate_primary_form_coverage.py verify`。详见[补库与新查询开发验证](module2_primary_form_coverage.md)。\n"
            + rest,
            encoding="utf-8",
        )
    prefix = ROOT.relative_to(PROJECT_ROOT).as_posix()
    registry_path = PROJECT_ROOT / "configs/evaluation_registry.json"
    registry = load(registry_path)
    registry.update(
        current_module2_progress_file=prefix + "/public_status.json",
        current_development_experiment=prefix,
        current_preparation_exclusion_snapshot=prefix
        + "/post_development_exclusion_snapshot.json",
        current_development_decision_file=prefix + "/decision.json",
        new_independent_AI_blind_after_development_started=False,
        current_primary_form_candidates_registry=QUEUE.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        current_primary_form_media_registry=MEDIA.relative_to(PROJECT_ROOT).as_posix(),
        current_primary_form_fresh_candidates_registry=FRESH_QUEUE.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        current_primary_form_fresh_media_registry=FRESH_MEDIA.relative_to(
            PROJECT_ROOT
        ).as_posix(),
    )
    if not any(r.get("path_prefix") == prefix + "/" for r in registry["rules"]):
        registry["rules"].append(
            dict(
                path_prefix=prefix + "/",
                role="adaptive_coverage_development_and_new_query_validation_only",
                formal_acceptance_claim_allowed=False,
                development_scores_allowed_in_public_status=False,
                reusable_as_unseen=False,
            )
        )
    write_json(registry_path, registry)
    status = dict(
        stage="primary_form_coverage_and_new_query_development_validation_completed",
        sealed=False,
        new_gallery_products=72,
        expanded_gallery_per_exposed_cohort=168,
        original_queries_retained=32,
        new_query_family_validation_queries=8,
        AI_relations=6720,
        human_labels=0,
        selected_method=selection["selected"],
        quality_metrics=None,
        formal_metrics=None,
        formal_acceptance=False,
        goal_met=False,
        default_strategy_changed=False,
        new_independent_AI_blind_predictions=0,
        fresh_query_validation_uses_shared_development_gallery=True,
        previous_consumed_AI_blind_preserved=True,
    )
    write_json(ROOT / "public_status.json", status)
    print(
        json.dumps(
            dict(
                report_saved=True,
                actual_new_full_pairs=total_new,
                reused_exact_pairs=reused,
                selected=selection["selected"],
            )
        ),
        flush=True,
    )


def seal() -> None:
    """Seal the completed outputs with their integrity metadata.

    Raises:
        FileExistsError: Preserve completed stage.

    Notes:
        Writes derived artifacts through target.parent.mkdir, write_json.
    """
    if (ROOT / "completion_receipt.json").exists():
        raise FileExistsError("Preserve completed stage")
    assert load(ROOT / "checks.json")["passed"] is True
    verification = load(ROOT / "verification.json")
    paths = [
        PROJECT_ROOT / "scripts" / n
        for n in [
            "primary_form_coverage.py",
            "evaluate_primary_form_coverage.py",
            "report_primary_form_coverage.py",
        ]
    ]
    paths += [
        PROJECT_ROOT / p
        for p in [
            "docs/module2_primary_form_coverage.md",
            "docs/current_progress.md",
            "docs/environment_setup.md",
            "README.md",
            "configs/evaluation_registry.json",
        ]
    ]
    release = ROOT / "source_release"
    for src in paths:
        target = release / src.relative_to(PROJECT_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
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
    status.update(sealed=True, replay_passed=True)
    write_json(ROOT / "public_status.json", status)
    write_json(
        ROOT / "completion_receipt.json",
        dict(
            files={
                p.relative_to(ROOT).as_posix(): sha256_file(p)
                for p in sorted(ROOT.rglob("*"))
                if p.is_file()
                and p.name not in {"completion_receipt.json", "public_status.json"}
            },
            project_files={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                for p in [QUEUE, MEDIA, FRESH_QUEUE, FRESH_MEDIA]
            },
            previous_sealed_stages=verification["previous_sealed_stages_verified"],
            formal_metrics=None,
            formal_acceptance=False,
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
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["audit", "report", "seal"])
    globals()[parser.parse_args().action]()
