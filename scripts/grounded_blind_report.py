"""Evaluate the single frozen candidate once; seal consumed independent AI data."""

from __future__ import annotations

import argparse

import grounded_blind_dataset as dataset
import grounded_blind_runtime as runtime
import run  # noqa: F401
from finalize_balanced_gallery import report_cells
from learned_fusion_experiment import load, once, rel
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.holdout_data import fingerprints
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.safety import sha256_file, write_json
from querywise_paired_bounds import paired_querywise

ROOT = dataset.ROOT
COMPARISONS = [(runtime.PRIMARY, b) for b in ("global", "region_title", "cross20")]


def calculated() -> tuple[dict, dict]:
    """Return calculated for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    ranks = load(ROOT / "predictions.json")
    labels = read_rows(ROOT / "judgments.ai.jsonl")
    modes = {
        q["id"]: "image" if q["image_path"] else "text"
        for q in load(ROOT / "queries.public.json")
    }
    return evaluate_intervals(ranks, labels, modes), paired_querywise(
        ranks, labels, modes, COMPARISONS
    )


def exposure() -> None:
    """Apply the exposure workflow step.

    Notes:
        Writes derived artifacts through once.
    """
    previous = load(dataset.DEVELOPMENT / "post_experiment_exclusion_snapshot.json")
    candidates = read_rows(dataset.QUEUE)
    downloaded = [load(p) for p in sorted((ROOT / "downloads").glob("*.json"))]
    images = [d["fingerprints"] for d in downloaded if d["status"] == "downloaded"]
    derived = [
        dict(path=rel(p), fingerprints=fingerprints(p))
        for folder in ("review_gallery", "review_queries", "query_preparation", "index")
        for p in sorted((ROOT / folder).rglob("*.png"))
    ]
    once(ROOT / "derived_image_exposure.json", derived)
    snapshot = dict(previous)
    snapshot.update(
        files=dict(
            previous["files"],
            **{
                rel(p): sha256_file(p)
                for p in [
                    dataset.QUEUE,
                    dataset.CONSUMED,
                    ROOT / "derived_image_exposure.json",
                ]
            },
        ),
        ids=sorted(set(previous["ids"]) | {r["id"] for r in candidates}),
        product_source_ids=sorted(
            set(previous["product_source_ids"]) | {r["source_id"] for r in candidates}
        ),
        family_keys=sorted(
            set(previous["family_keys"])
            | {k for r in candidates for k in r["family_keys"]}
        ),
        image_fingerprints=previous["image_fingerprints"]
        + images
        + [d["fingerprints"] for d in derived],
        scope="All previous development, every new blind candidate and acquired image, selected independent AI blind and all derived views are consumed and never reusable as unseen",
    )
    once(ROOT / "post_blind_exclusion_snapshot.json", snapshot)
    dataset.receipt(
        "exposure_receipt.json",
        [
            ROOT / "derived_image_exposure.json",
            ROOT / "post_blind_exclusion_snapshot.json",
        ],
        all_metadata_candidates_excluded=True,
        all_downloaded_and_derived_images_excluded=True,
    )


def finish() -> None:
    """Finalize the recorded outputs of the workflow.

    Notes:
        Writes derived artifacts through once, p.write_text, write_json.
    """
    dataset.guard()
    dataset.checked("label_receipt.json")
    dataset.checked("prediction_receipt.json")
    ranks = load(ROOT / "predictions.json")
    assert ranks == runtime.orders()
    ids = {d["id"] for d in load(ROOT / "catalog.public.json")}
    assert len(ranks) == 24 and all(
        set(ms) == set(runtime.METHODS)
        and all(len(v) == len(set(v)) == 96 and set(v) == ids for v in ms.values())
        for ms in ranks.values()
    )
    metrics, paired = calculated()
    once(ROOT / "internal_metrics.ai.blind.json", metrics)
    once(ROOT / "internal_paired_querywise.ai.blind.json", paired)
    gates = {}
    lines = [
        "# 独立AI盲测：固定主售定位候选",
        "",
        "全新查询与全新图库；96件商品、12图片查询与12中文文本查询、2304条逐对AI关系、K10。",
        "方法与采样规则在新候选元数据扫描前冻结；标签在所有检索预测前冻结。全图库分母、未知null、零正例和所有选中记录保留。仅运行一个预声明候选与三个基线，不在盲测上选方法、调参、改标或换图。",
        "数据来源/商品族/原图与衍生图隔离按冻结记录和sha/pixels/dHash<=4检查，不等于所有语义近重复均可证明不存在。平衡12个预声明类型的封闭小图库，不代表真实流量分布。",
        "评审为Codex原生视觉AI，人工0，评审者知道项目背景但未看本轮预测；不声称独立人工或独立标注模型。区间是未知标签补全范围，不是AI错误界或置信区间。正式验收指标仍为null，开发结果不混入盲测。",
        "",
    ]
    for scope, ms in metrics["scopes"].items():
        lines += [
            f"## {scope}",
            "",
            "|方法|P@10|R@10|mAP|NDCG@10|MRR|Hit@1|",
            "|---|---|---|---|---|---|---|",
        ]
        for method in runtime.METHODS:
            lines.append("|" + "|".join([method, *report_cells(ms[method])]) + "|")
        p = ms[runtime.PRIMARY]
        gates[scope] = dict(
            precision_60=p["precision_60_lower_bound_met"],
            recall_85=p["recall_85_lower_bound_met"],
            nonregression=all(
                v["nonregression_proven_for_shared_unknown_labels"]
                for b in ("global", "region_title")
                for metric, v in paired["scopes"][scope][runtime.PRIMARY][b].items()
                if metric != "precision@10"
            ),
        )
    decision = dict(
        candidate=runtime.PRIMARY,
        independent_AI_blind_completed=True,
        query_predictions=24,
        checks=gates,
        independent_AI_precision_target_met=all(
            v["precision_60"] for v in gates.values()
        ),
        independent_AI_recall_preference_met=all(
            v["recall_85"] for v in gates.values()
        ),
        independent_AI_nonregression_proven=all(
            v["nonregression"] for v in gates.values()
        ),
        development_decision=load(dataset.DEVELOPMENT / "decision.json"),
        goal_met=False,
        formal_metrics=None,
        human_labels=0,
        default_strategy_changed=False,
        failed_or_passed_blind_data_reusable_as_unseen=False,
    )
    once(ROOT / "decision.json", decision)
    p = ROOT / "internal_comparison.ai.blind.md"
    text = "\n".join(lines) + "\n"
    if p.exists():
        assert p.read_text(encoding="utf-8") == text
    else:
        p.write_text(text, encoding="utf-8")
    losses = []
    for row in paired["query_pair_bounds"]:
        bad = [
            k
            for k, v in row["metrics"].items()
            if v["delta_range"][1] is not None and v["delta_range"][1] < -1e-12
        ]
        if bad:
            losses.append(
                dict(
                    query_id=row["query_id"],
                    baseline=row["baseline"],
                    strict_regression_metrics=bad,
                )
            )
    once(ROOT / "internal_strict_regressions.json", losses)
    exposure()
    once(
        ROOT / "verification.json",
        dict(
            passed=True,
            replayed_rankings=96,
            methods=4,
            queries=24,
            AI_relations=2304,
            labels_before_predictions=True,
            formal_metrics=None,
            human_labels=0,
        ),
    )
    dataset.receipt(
        "completion_receipt.json",
        [
            p
            for p in sorted(ROOT.rglob("*"))
            if p.is_file()
            and p.name
            not in {"completion_receipt.json", "public_status.json", "runtime.lock"}
        ],
        completed=True,
        independent_AI_blind_completed=True,
        formal_metrics=None,
    )
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=True,
            phase="independent_AI_blind_completed_sealed",
            independent_AI_blind_completed=True,
            query_predictions=24,
            methods=4,
            rankings=96,
            gallery=96,
            AI_relations=2304,
            quality_metrics=None,
            formal_metrics=None,
            human_labels=0,
            goal_met=False,
            default_strategy_changed=False,
        ),
    )
    verify()
    print(
        dict(
            independent_AI_blind_completed=True,
            formal_metrics=None,
            default_strategy_changed=False,
        )
    )


def verify() -> None:
    """Verify the recorded artifacts and their consistency."""
    dataset.guard()
    dataset.checked("completion_receipt.json")
    assert runtime.orders() == load(ROOT / "predictions.json")
    metrics, paired = calculated()
    assert metrics == load(ROOT / "internal_metrics.ai.blind.json") and paired == load(
        ROOT / "internal_paired_querywise.ai.blind.json"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["finish", "verify"])
    a = parser.parse_args().action
    if a == "finish":
        finish()
    else:
        verify()
        print(dict(seal_and_blind_metrics_replay_passed=True))
