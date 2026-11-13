"""Replay the single frozen blind run and publish explicitly AI-reviewed results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import run  # noqa: F401
from independent_business_blind import (
    CONSUMED,
    DATASET,
    METHOD,
    RESERVED,
    ROOT,
    TEXTS,
    audit_isolation,
    check,
    public_features,
)
from multimodal_ecommerce.ai_review_metrics import metric_ranges
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_evaluation import METRICS
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.intent_retrieval import (
    balanced_pool,
    normalized_ranks,
    public_evidence,
    recall_channels,
    rerank_complete,
)
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.query_fusion import (
    QueryFusionVisualChannelsInputs,
    visual_channels,
)
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.semantic_ranker import (
    SemanticRankerFeatureMatrixInputs,
    feature_matrix,
    ranked_ids,
)
from report_focus_coverage import paired_differences
from verify_rank_stability import verify_history


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def persist(path: Path, value: object) -> None:
    """Apply the persist workflow step.

    Args:
        path: Path of the input or output artifact used by this operation.
        value: Input value to validate or transform.

    Notes:
        Writes derived artifacts through write_json.
    """
    if path.exists():
        assert load(path) == value, f"Existing blind evidence differs: {path}"
    else:
        write_json(path, value)


def verify_native_review() -> tuple[list, Any]:
    """Verify native review.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    public = load(ROOT / "catalog.public.json")
    source = read_rows(DATASET / "gallery/products.jsonl")
    lineage = {
        r["opaque_id"]: r for r in read_rows(DATASET / "evaluator/lineage.jsonl")
    }
    assert len(public) == 300
    assert all(set(r) == {"id", "title", "image_path"} for r in public)
    for a, b in zip(public, source):
        assert a["id"] == b["id"] and a["title"] == b["title"]
        assert (
            Path(a["image_path"])
            == (DATASET / lineage[a["id"]]["image_path"]).resolve()
        )
        assert sha256_file(Path(a["image_path"])) == b["image_sha256"]
    authored = {}
    for line in (ROOT / "native/profiles.tsv").read_text(encoding="utf-8").splitlines():
        pos, kind, appearance = line.split("\t")
        assert int(pos) not in authored
        authored[int(pos)] = dict(product_type=kind, appearance=appearance)
    profiles = [dict(id=r["id"], **authored[i]) for i, r in enumerate(public, 1)]
    assert profiles == read_rows(ROOT / "profiles.public.jsonl")
    native = load(ROOT / "native/judgments.json")
    assert (
        not native["query_model_predictions_seen"]
        and not native["ranking_or_score_inputs_used"]
    )
    assert native["gallery_sheets_viewed"] == list(range(1, 31))
    assert native["query_sheets_viewed"] == [1, 2, 3]
    labels = []
    for position, doc in enumerate(public, 1):
        entry = native["products"][str(position)]
        for mode in ["text", "image"]:
            for number in range(1, 31):
                positive = entry[mode + "_positive"]
                unknown = entry[mode + "_unknown"]
                grade = (
                    1 if number in positive else None if str(number) in unknown else 0
                )
                reason = (
                    "主要主体、用途及该查询明确要求的外形/属性匹配；按冻结规则直接看图与原标题判断。"
                    if grade == 1
                    else (
                        unknown[str(number)]
                        if grade is None
                        else "按冻结业务规则复核，非该需求匹配的主要商品主体/用途/可见外形或明确属性。"
                    )
                )
                labels.append(
                    dict(
                        query_id=f"{mode}_{number:02d}",
                        candidate_id=doc["id"],
                        grade=grade,
                        reason=reason,
                        native_position=position,
                        review_origin="model",
                        human_label=False,
                        evidence="native/judgments.json",
                    )
                )
    assert labels == read_rows(ROOT / "judgments.ai.jsonl")
    assert len(labels) == 18000 and sum(r["grade"] is None for r in labels) == 59
    drafts = read_rows(DATASET / "evaluator/queries.draft.jsonl")
    queries = load(ROOT / "queries.public.json")
    assert len(queries) == 60 and all(
        set(q) == {"id", "text", "image_path"} for q in queries
    )
    for q, draft in zip(queries, drafts):
        assert q["id"] == draft["id"]
        if draft["kind"] == "text":
            assert (
                q["text"] == TEXTS[int(q["id"].split("_")[1]) - 1]
                and q["image_path"] is None
            )
        else:
            assert (
                q["text"] is None
                and Path(q["image_path"]) == (DATASET / draft["image_path"]).resolve()
            )
    failure = load(ROOT / "gallery_path_failure_execution.json")
    assert failure["continuation_sha256"] == sha256_file(
        PROJECT_ROOT / "scripts/finish_blind_gallery.py"
    )
    assert [r["id"] for r in failure["catalog_before_repair"]] == [
        r["id"] for r in public
    ]
    consumed = read_rows(CONSUMED)
    assert len(consumed) == 330 and all(
        r["project_role"] == "consumed_business_blind_not_reusable_as_unseen"
        for r in consumed
    )
    return labels, queries


def reconstruct_queries(queries: list[dict]) -> dict:
    """Return reconstruct queries for this workflow.

    Args:
        queries: Queries used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    index, titles, profiles, profile_texts, colors = public_features()
    ids = [d["id"] for d in index.records]
    result, pair_count = {}, 0
    for q in queries:
        folder = ROOT / "query_preparation" / q["id"]
        with np.load(folder / "vectors.npz", allow_pickle=False) as bank:
            views, caption = bank["views"].copy(), bank["caption"].copy()
        image = Path(q["image_path"]) if q["image_path"] else None
        features = feature_matrix(
            SemanticRankerFeatureMatrixInputs(
                index,
                titles,
                profiles,
                profile_texts,
                views[0],
                text=q["text"],
                image=image,
                color_bank=colors,
            )
        )
        target = ROOT / "raw_predictions" / q["id"]
        saved = load(target / "features.json")
        assert saved["ids"] == ids
        assert np.allclose(saved["features"], features, atol=1e-14, rtol=0)
        if image:
            description = load(folder / "description.json")
            assert description["gallery_or_judgments_supplied"] is False
            fusion = visual_channels(
                QueryFusionVisualChannelsInputs(
                    index,
                    titles,
                    profiles,
                    features,
                    views,
                    caption,
                    description["text"],
                    np.arange(300, dtype=int),
                )
            )
            pool, base, offsets = (
                fusion["pool"][:20],
                np.array(fusion["fused_scores"]),
                None,
            )
        else:
            base, channels = recall_channels(ids, features)
            offsets = np.array(
                [
                    public_evidence(q["text"], d["title"], p)["offset"]
                    for d, p in zip(index.records, profile_texts)
                ]
            )
            channels.append(ranked_ids(ids, normalized_ranks(ids, base) + offsets))
            pool = balanced_pool(channels, 60)[:20]
        assert saved["pool"] == pool and np.allclose(
            saved["base"], base, atol=1e-14, rtol=0
        )
        if offsets is None:
            assert saved["offsets"] is None
        else:
            assert np.allclose(saved["offsets"], offsets, atol=1e-14, rtol=0)
        raw = load(target / "cross.json")
        assert raw["query_sha256"] == sha256_file(folder / "vectors.npz")
        assert raw["pool"] == pool and len(raw["logits"]) == len(raw["timings"]) == 20
        assert np.isfinite(raw["logits"]).all()
        assert [r["document_id"] for r in raw["timings"]] == pool
        orders = dict(
            global_=ranked_ids(ids, features[:, 0]),
            region_title=ranked_ids(ids, 0.8 * features[:, 1] + 0.2 * features[:, 3]),
        )
        orders["global"] = orders.pop("global_")
        orders[METHOD] = rerank_complete(
            ids, base, pool, np.array(raw["logits"]), offsets
        )
        assert orders == load(target / "rankings.json")
        result[q["id"]] = orders
        pair_count += 20
    assert pair_count == 1200 and result == load(ROOT / "predictions.json")
    return result


def main() -> None:
    """Run the command-line workflow defined by this module."""
    release = check()
    verify_files(RESERVED, load(RESERVED / "preparation_receipt.json")["files"])
    assert audit_isolation() == load(ROOT / "isolation_before_opening.json")
    for name in [
        "profile_receipt.json",
        "label_receipt.json",
        "query_preparation_receipt.json",
        "prediction_receipt.json",
    ]:
        verify_files(ROOT, load(ROOT / name)["files"])
    assert load(ROOT / "predictions_started.json")[
        "label_receipt_sha256"
    ] == sha256_file(ROOT / "label_receipt.json")
    labels, queries = verify_native_review()
    rankings = reconstruct_queries(queries)
    index, _, _, _, _ = public_features()
    index_runtime = load(ROOT / "index/embedding_runtime.json")
    assert (
        index_runtime["device"] == "cuda:0"
        and index_runtime["dtype"] == "torch.bfloat16"
    )
    expected_gallery_calls = sum(len(r["views"]) for r in index.records) + 600
    assert len(index_runtime["calls"]) == expected_gallery_calls
    for cfg in [release["embedding"], release["reranker"]]:
        verify_files(Path(cfg["path"]), cfg["checkpoint_files"])
    modes = {q["id"]: "text" if q["text"] else "image" for q in queries}
    metrics = evaluate_intervals(rankings, labels, modes)
    exhaustive = metric_ranges(rankings, labels, modes)
    for scope in metrics["scopes"]:
        for name in metrics["scopes"][scope]:
            for metric in METRICS:
                assert np.allclose(
                    metrics["scopes"][scope][name]["ranges"][metric],
                    exhaustive["scopes"][scope][name]["ranges"][metric],
                    atol=1e-13,
                    rtol=0,
                )
    paired = paired_differences(rankings, labels, modes)
    paired["role"] = (
        "independent_data_AI_blind_paired_differences_with_shared_unknown_labels"
    )
    decisions = dict(
        precision_required_met=all(
            metrics["scopes"][s][METHOD]["precision_60_lower_bound_met"]
            for s in ["all", "text", "image"]
        ),
        recall_preferred_met=all(
            metrics["scopes"][s][METHOD]["recall_85_lower_bound_met"]
            for s in ["all", "text", "image"]
        ),
        other_metrics_nonregression_proven=all(
            paired["scopes"][s][METHOD][base][m][
                "nonregression_proven_for_shared_unknown_labels"
            ]
            for s in ["all", "text", "image"]
            for base in ["global", "region_title"]
            for m in METRICS
        ),
        recall_preferred_not_required=True,
        default_promotion_allowed=False,
        formal_acceptance=False,
        formal_metrics=None,
        review_origin="model",
        human_labels=0,
        independent_annotator_validation=False,
    )
    decisions["AI_blind_primary_and_nonregression_goal_met"] = (
        decisions["precision_required_met"]
        and decisions["other_metrics_nonregression_proven"]
    )
    capacities = {}
    for scope in ["all", "text", "image"]:
        current = [
            r
            for r in metrics["query_capacity"]
            if scope == "all" or modes[r["query_id"]] == scope
        ]
        capacities[scope] = dict(
            precision_ceiling_range=[
                sum(r["precision_ceiling_range"][i] for r in current) / len(current)
                for i in [0, 1]
            ],
            zero_positive_queries=[
                r["query_id"] for r in current if r["recall_status"] == "zero_positive"
            ],
            uncertain_only_queries=[
                r["query_id"]
                for r in current
                if r["recall_status"] == "possibly_zero_positive"
            ],
        )
    persist(
        ROOT / "AI_blind_metrics.json",
        dict(
            **metrics,
            role="single_frozen_version_independent_data_AI_reviewed_blind_result",
            scope=release["policy"]["scope"],
            formal_acceptance=False,
        ),
    )
    persist(ROOT / "AI_blind_paired_differences.json", paired)
    persist(ROOT / "AI_blind_capacity.json", capacities)
    persist(ROOT / "AI_blind_decision.json", decisions)
    drafts = read_rows(DATASET / "evaluator/queries.draft.jsonl")
    by_category = {}
    for category in sorted({q["category_for_reporting"] for q in drafts}):
        selected = {q["id"] for q in drafts if q["category_for_reporting"] == category}
        by_category[category] = evaluate_intervals(
            {q: v for q, v in rankings.items() if q in selected},
            [r for r in labels if r["query_id"] in selected],
            {q: v for q, v in modes.items() if q in selected},
        )
    persist(ROOT / "AI_blind_by_category.json", by_category)
    history = verify_history()
    previous = PROJECT_ROOT / "artifacts/module2_rank_stability"
    prior_receipt = load(previous / "completion_receipt.json")
    verify_files(previous, prior_receipt["files"])
    verify_files(PROJECT_ROOT, prior_receipt["project_files"])
    verification = dict(
        verified=True,
        all_source_models_and_inputs_bound=True,
        heldout_records=330,
        gallery=300,
        queries=60,
        full_rankings=180,
        AI_relations=18000,
        unknown=59,
        actual_gallery_embedding_calls=expected_gallery_calls,
        actual_query_embedding_calls=len(
            load(ROOT / "query_preparation/runtime.json")["calls"]
        ),
        actual_query_VLM_calls=30,
        actual_full_cross_pairs=1200,
        all_prediction_orders_replayed=True,
        all_AI_labels_replayed=True,
        exact_dynamic_bounds_match_exhaustive_completion=True,
        previous_sealed_roots_verified=len(history) + 1,
        human_labels=0,
        reusable_as_unseen=False,
        default_strategy_changed=False,
        formal_acceptance=False,
    )
    persist(ROOT / "verification.json", verification)
    if (ROOT / "completion_receipt.json").exists():
        receipt = load(ROOT / "completion_receipt.json")
        verify_files(ROOT, receipt["files"])
        verify_files(PROJECT_ROOT, receipt["project_files"])
        verify_files(
            ROOT / "source_release",
            load(ROOT / "source_release/manifest.json")["files"],
        )
    print(
        json.dumps(
            dict(
                verified=True,
                queries=60,
                full_cross_pairs=1200,
                precision_required_met=decisions["precision_required_met"],
                formal_acceptance=False,
            )
        )
    )


if __name__ == "__main__":
    main()
