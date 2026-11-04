"""Native query-view QA and strictly family-masked business training inputs.

The QA is diagnostic and never an ID-based inference override. Training and
STIR are gated pending a reusable subject-quality rule; this file trains none.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run  # noqa: F401
from business_grounding_audit import ROOT, load, once, relative
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from multimodal_ecommerce.semantic_ranker import (
    SemanticRankerTrainingMaskInputs,
    training_mask,
)

# Actual source/box/mask panels were inspected by Codex for every image query.
# accept means plausible development QA, not human truth or localization accuracy.


QUERY_QA: dict[str, tuple[str, str, str]] = {
    "S001": ("accept", "accept", "三款单耳设备主体保留，背景基本去除"),
    "S002": ("fallback", "fallback", "无有效候选框，保留华为Mate10保护壳原图"),
    "S003": ("reject", "reject", "内胆包与展示手袋同时入框入掩码；仍保留了非主售道具"),
    "S004": ("accept", "accept", "两只厚底靴的可见鞋体及鞋底保留"),
    "S005": (
        "accept",
        "accept",
        "外套主体保留，内搭及模特被移除；不保证遮挡后的完整性",
    ),
    "S006": ("accept", "accept", "柜体与可见柜脚基本保留，旁侧有少量残余"),
    "S007": (
        "accept",
        "uncertain",
        "框取真实杯子的插图视图；掩码带红色框线残余，不确认轮廓清洁",
    ),
    "S008": ("accept", "accept", "运动鞋鞋面和鞋底保留"),
    "S420": (
        "uncertain",
        "uncertain",
        "耳麦下方连线完整性未确认，掩码还保留遮挡头梁的文字",
    ),
    "S421": (
        "accept",
        "uncertain",
        "保护壳主体入框；手指遮挡处抠除后形成缺口，轮廓可靠性未确认",
    ),
    "S422": ("fallback", "fallback", "木制拱形颈枕未获得有效框，保留原图"),
    "S423": ("accept", "reject", "框保留靴子，但掩码抹掉可见鞋跟，破坏关键形态"),
    "S424": ("accept", "accept", "外套可见主体基本保留，原图边界和原始遮挡仍在"),
    "S425": ("accept", "accept", "柜门、抽屉及可见柜脚保留"),
    "S426": ("accept", "accept", "两只杯子及手柄保留"),
    "S427": ("accept", "accept", "鞋体可见轮廓基本保留"),
    "S428": (
        "accept",
        "reject",
        "掩码在黑色头梁上产生大面积孔洞，不能作为完整耳麦前景",
    ),
    "S429": ("accept", "accept", "保护壳主体、摄像头开孔和支架保留"),
    "S430": (
        "reject",
        "reject",
        "裁切只取包装上的果肉印图，掩码丢弃外包装，不是完整可见主售包装",
    ),
    "S431": ("accept", "reject", "裁切保留细高跟；掩码删除鞋跟，丢失关键结构"),
    "S432": ("accept", "accept", "无袖马甲主体及开口轮廓保留"),
    "S433": (
        "accept",
        "uncertain",
        "柜体框可靠，掩码在玻璃内可见内容产生白色孔洞，外观改变未确认",
    ),
    "S434": (
        "accept",
        "reject",
        "透明杯体与手部接触区域被大块抹除，关键可见杯体轮廓受损",
    ),
    "S435": (
        "accept",
        "reject",
        "掩码在右鞋鞋口和后跟附件产生明显缺损，不能确认完整前景",
    ),
}


def verified_pairs(
    labels: np.ndarray, allowed: np.ndarray, scores: np.ndarray, limit: int = 16
) -> tuple[list, np.ndarray]:
    """Known business grades only. SKU inequality never creates a negative.

    Args:
        labels: Judgments used for the stated development or evaluation protocol.
        allowed: Allowed used by this operation.
        scores: Scores used by this operation.
        limit: Limit used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Pair matrices misaligned; Known binary business grades and real
            scores required.
    """
    labels, allowed, scores = (
        np.asarray(labels),
        np.asarray(allowed),
        np.asarray(scores),
    )
    if labels.shape != allowed.shape or labels.shape != scores.shape:
        raise ValueError("Pair matrices misaligned")
    known = allowed & np.isfinite(labels)
    if (
        not np.isin(labels[known], [0.0, 1.0]).all()
        or not np.isfinite(scores[known]).all()
    ):
        raise ValueError("Known binary business grades and real scores required")
    result = []
    for q in range(len(labels)):
        positives = np.flatnonzero(known[q] & (labels[q] == 1))
        negatives = np.flatnonzero(known[q] & (labels[q] == 0))
        negatives = negatives[np.argsort(-scores[q, negatives], kind="stable")[:limit]]
        if len(positives) and len(negatives):
            result.append(
                {
                    "query_index": q,
                    "positive_indices": positives.tolist(),
                    "hard_negative_indices": negatives.tolist(),
                }
            )
    return result, known


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Notes:
        Writes derived artifacts through once.
    """
    views = load(ROOT / "subject_views_results.json")
    queries = {r["opaque"]: r for r in views if not r["key"].startswith("product/")}
    assert set(queries) == set(QUERY_QA)
    observations = []
    for opaque, (box, mask, note) in QUERY_QA.items():
        r = queries[opaque]
        observations.append(
            {
                "opaque": opaque,
                "query_key": r["key"],
                "source_sha256": r["source_sha256"],
                "box_decision": box,
                "mask_decision": mask,
                "native_visual_evidence": note,
                "review_origin": "model",
                "annotation_tool": "Codex native image view",
                "human_label": False,
                "source_evidence": r["paths"]
                .get("review", {})
                .get("path", r["source_image"]),
                "runtime_override": False,
                "relevance_grade_changed": False,
            }
        )
    once(
        ROOT / "QA/query_views.native_AI.json",
        {
            "observations": observations,
            "all_24_image_queries_inspected": True,
            "purpose": "subject localization/mask quality only; not retrieval relevance",
            "mask_failure_modes": [
                "prop retained with sale item",
                "shoe heel erased",
                "headband holes",
                "transparent item erased",
                "packaging replaced by printed illustration",
            ],
            "diagnostic_decisions_are_not_deployable_ID_rules": True,
            "human_labels": 0,
            "formal_metrics": None,
        },
    )
    # Strictly follow the user's sequence: known semantic failures must be dealt
    # with before training can present these masks as a validated attention input.
    once(
        ROOT / "subject_quality_gate.json",
        {
            "automatic_mask_training_ready": False,
            "automatic_mask_default_promotion_allowed": False,
            "reason": "Native development inspection found destroyed sale-item structures and retained props. No reusable semantic/structure preservation rule is validated yet.",
            "next_action": "Freeze a candidate/query-only mask verifier using original+crop+mask evidence, evaluate it on development, and keep full-image fallback. Do not hardcode these query IDs.",
            "training_started": False,
            "STIR_started": False,
            "fusion_quality_comparison_started": False,
            "all_original_queries_gallery_labels_retained": True,
            "formal_metrics": None,
        },
    )
    original = PROJECT_ROOT / "artifacts/module2_subject_recall_learning"
    meta = load(original / "feature_metadata.json")
    with np.load(original / "feature_bank.npz", allow_pickle=False) as bank:
        labels, valid, scores, qf, gf = [
            bank[k].copy()
            for k in ("labels", "valid", "base", "query_folds", "gallery_folds")
        ]
    folds = []
    for hq in range(3):
        for hg in range(3):
            allowed, evidence = training_mask(
                SemanticRankerTrainingMaskInputs(
                    meta["query_ids"],
                    qf,
                    meta["query_families"],
                    meta["gallery_families"],
                    gf,
                    hq,
                    hg,
                )
            )
            allowed &= valid
            pairs, known = verified_pairs(labels, allowed, scores)
            held = valid & (qf[:, None] == hq) & (gf[None, :] == hg)
            assert not np.any(known & held)
            assert all(
                known[p["query_index"], p["positive_indices"]].all()
                and known[p["query_index"], p["hard_negative_indices"]].all()
                for p in pairs
            )
            folds.append(
                {
                    "name": f"q{hq}_g{hg}",
                    "family_audit": evidence,
                    "verified_business_pairs": pairs,
                    "known_training_cells": int(known.sum()),
                    "unknown_training_cells_excluded": int(
                        (allowed & ~np.isfinite(labels)).sum()
                    ),
                    "pointwise_training_cells": np.argwhere(known).tolist(),
                    "no_positive_query_not_removed_from_evaluation": True,
                }
            )
    paths = [
        Path(__file__),
        PROJECT_ROOT / "tests/test_business_view_qa_and_pairs.py",
        ROOT / "audit_receipt.json",
        ROOT / "views_receipt.json",
        ROOT / "QA/source_review_receipt.json",
        ROOT / "QA/query_views.native_AI.json",
        ROOT / "subject_quality_gate.json",
        original / "feature_metadata.json",
        original / "feature_bank.npz",
        PROJECT_ROOT / "src/multimodal_ecommerce/semantic_ranker.py",
    ]
    once(
        ROOT / "business_training_inputs.json",
        {
            "query_ids": meta["query_ids"],
            "gallery_ids": meta["gallery_ids"],
            "folds": folds,
            "all_valid_query_gallery_cells": int(valid.sum()),
            "known_cells": int((valid & np.isfinite(labels)).sum()),
            "unknown_cells_excluded": int((valid & ~np.isfinite(labels)).sum()),
            "pair_scope": "frozen business relevance, not same-SKU identity; all exposed development",
            "hard_negatives_per_training_query": 16,
            "sampling_source": "previous full-gallery base scores, restricted to each allowed training family cell",
            "features_from_consumed_blind": False,
            "training_started": False,
            "formal_metrics": None,
        },
    )
    paths.append(ROOT / "business_training_inputs.json")
    once(
        ROOT / "QA_and_pairs_receipt.json",
        {
            "files": {relative(p): sha256_file(p) for p in paths},
            "native_query_QA_completed": True,
            "family_masked_training_inputs_prepared": True,
            "neural_training_completed": False,
            "formal_metrics": None,
        },
    )
    print(
        json.dumps(
            {
                "query_view_QA": 24,
                "family_masked_training_folds": 9,
                "automatic_mask_quality_gate_passed": False,
                "training_started": False,
                "formal_metrics": None,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    prepare()
