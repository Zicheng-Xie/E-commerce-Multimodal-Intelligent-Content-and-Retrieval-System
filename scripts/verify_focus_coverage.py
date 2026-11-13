"""Replay all development comparisons and verify heldout isolation without inference."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

import run  # noqa: F401
from advance_focus_coverage import (
    PARENT,
    RESERVED,
    ROOT,
    check,
    key,
    load,
    raw_catalog,
)
from multimodal_ecommerce.ai_review_metrics import metric_ranges
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_retrieval import BUSINESS_INSTRUCTION, hybrid_scores
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.development_expansion import (
    acceptance_guard,
    fold_metadata,
    load_expanded_index,
)
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.intent_retrieval import (
    balanced_pool,
    normalized_ranks,
    public_evidence,
    recall_channels,
    rerank_complete,
)
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
from multimodal_ecommerce.supervised_pipeline import (
    SupervisedPipelineFitSupervisedBundleInputs,
    fit_supervised_bundle,
)
from multimodal_ecommerce.visual_query import caption_text
from prepare_focus_coverage import QUEUE, REGISTRY, SNAPSHOT


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through write_json.
    """
    protocol = check()
    verify_files(ROOT, load(ROOT / "review_receipt.json")["files"])
    verify_files(ROOT / "dataset", load(ROOT / "dataset/manifest.json")["files"])
    inputs = load(ROOT / "inputs.public.json")
    assert all(set(r) == {"id", "title", "image_path"} for r in inputs["catalog"])
    assert all(
        set(q) == {"id", "text", "image_path", "exclude_id"} for q in inputs["queries"]
    )
    assert inputs["queries"] == load(PARENT / "inputs.public.json")["queries"]
    guard = acceptance_guard(inputs["catalog"], raw_catalog(), read_rows(RESERVED))
    assert guard == load(ROOT / "guard.json")
    folds = fold_metadata(inputs["catalog"], inputs["queries"], guard)
    assert folds == load(ROOT / "folds.json")
    index, titles = load_expanded_index(ROOT / "index")
    assert [r["id"] for r in index.records] == [r["id"] for r in inputs["catalog"]]
    old_index, old_titles = load_expanded_index(PARENT / "index")
    old_positions = {r["id"]: i for i, r in enumerate(old_index.records)}
    with np.load(PARENT / "index/vectors.npz", allow_pickle=False) as archive:
        old_profiles = archive["profiles"].copy()
    with np.load(ROOT / "index/vectors.npz", allow_pickle=False) as archive:
        regions, profiles = archive["regions"].copy(), archive["profiles"].copy()
        raw_titles = archive["texts"].copy()
    actual_calls = 0
    for i, product in enumerate(inputs["catalog"]):
        cid = product["id"]
        assert (
            sha256_file(Path(product["image_path"])) == index.records[i]["image_sha256"]
        )
        if cid in old_positions:
            j = old_positions[cid]
            assert np.array_equal(regions[i], old_index.vectors[j])
            assert np.array_equal(raw_titles[i], old_titles[j])
            assert np.array_equal(profiles[i], old_profiles[j])
        else:
            cache_path = ROOT / "embedding_cache" / (key(cid) + ".json")
            cached = load(cache_path)
            assert (
                cached["id"] == cid
                and cached["title"] == product["title"]
                and cached["image_path"] == product["image_path"]
            )
            asset = cache_path.parent / cached["vector_asset"]
            assert sha256_file(asset) == cached["vector_sha256"]
            with np.load(asset, allow_pickle=False) as values:
                assert np.array_equal(
                    regions[i, : len(cached["views"])], values["regions"]
                )
                assert np.array_equal(raw_titles[i], values["title"])
                assert np.array_equal(profiles[i], values["profile"])
            actual_calls += len(cached["views"]) + 2
    gallery_calls = actual_calls
    enrichment = load(ROOT / "query_enrichment/embedding_receipt.json")
    actual_calls += enrichment["actual_embedding_calls"]
    for qid, entry in enrichment["queries"].items():
        caption = load(ROOT / "query_enrichment/captions" / (qid + ".json"))
        assert caption["gallery_or_judgments_supplied"] is False
        assert entry["description"] == caption["text"] == caption_text(caption["value"])
        asset = ROOT / "query_enrichment" / entry["vector_asset"]
        assert sha256_file(asset) == entry["vector_sha256"]
        for view in entry["views"]:
            assert sha256_file(Path(view["image_path"])) == view["sha256"]
    amendment = load(ROOT / "pre_inference_type_amendment.json")
    original_protocol = load(ROOT / amendment["original_protocol"])
    before = ROOT / amendment["original_frozen_source"]
    after = PROJECT_ROOT / amendment["changed_source"]
    assert (
        sha256_file(before) == amendment["before_sha256"]
        and sha256_file(after) == amendment["after_sha256"]
    )
    assert before.read_text(encoding="utf-8").replace(
        'min(weights["rerank_pool"], len(ids))',
        'min(int(weights["rerank_pool"]), len(ids))',
    ) == after.read_text(encoding="utf-8")
    expected_protocol = dict(original_protocol)
    expected_protocol["implementation"] = dict(original_protocol["implementation"])
    expected_protocol["implementation"][amendment["changed_source"]] = amendment[
        "after_sha256"
    ]
    expected_protocol["pre_inference_type_amendment_sha256"] = sha256_file(
        ROOT / "pre_inference_type_amendment.json"
    )
    assert protocol == expected_protocol
    runtime_paths = [
        ROOT / "embedding_runtime.json",
        *sorted(ROOT.glob("embedding_resume_*.json")),
    ]
    assert actual_calls == sum(len(load(p)["calls"]) for p in runtime_paths)
    for runtime_path in runtime_paths:
        runtime = load(runtime_path)
        assert runtime["calls"] and runtime["device"] == "cuda:0"
    profile_texts = load(ROOT / "index/profile_texts.json")
    with np.load(ROOT / "features.npz", allow_pickle=False) as archive:
        x, valid, y, bases, colors = [
            archive[k]
            for k in ["features", "valid", "labels", "base_scores", "color_bank"]
        ]
    with np.load(ROOT / "query_vectors.npz", allow_pickle=False) as archive:
        queries = archive["vectors"].copy()
    recomputed = np.stack(
        [
            feature_matrix(
                SemanticRankerFeatureMatrixInputs(
                    index,
                    titles,
                    profiles,
                    profile_texts,
                    queries[i],
                    text=q["text"],
                    image=Path(q["image_path"]) if q["image_path"] else None,
                    color_bank=colors,
                )
            )
            for i, q in enumerate(inputs["queries"])
        ]
    )
    # Public lexical IDF sums iterate sets; cross-process last-bit differences
    # are bounded here. All ranks are replayed from the actual frozen features.
    assert np.allclose(x, recomputed, rtol=0, atol=1e-15)
    ids = [r["id"] for r in index.records]
    original_labels = read_rows(ROOT / "judgments.ai.jsonl")
    lookup = {(r["query_id"], r["candidate_id"]): r["grade"] for r in original_labels}
    for i, q in enumerate(inputs["queries"]):
        for j, cid in enumerate(ids):
            if valid[i, j]:
                grade = lookup[q["id"], cid]
                assert (np.isnan(y[i, j]) and grade is None) or y[i, j] == grade
    assert valid.sum() == len(original_labels)
    assert int((valid & np.isnan(y)).sum()) == sum(
        r["grade"] is None for r in original_labels
    )
    rankings, pools, evidence = {}, {}, {}
    for i, q in enumerate(inputs["queries"]):
        positions = np.flatnonzero(valid[i])
        current_ids = [ids[j] for j in positions]
        current = x[i, positions]
        hybrid = hybrid_scores(
            current_ids,
            current[:, 1],
            current[:, 3],
            text=q["text"],
            titles=[index.records[j]["title"] for j in positions],
        )
        assert np.array_equal(bases[i, positions], hybrid)
        rrf, channels = recall_channels(current_ids, current)
        why = [
            public_evidence(q["text"], index.records[j]["title"], profile_texts[j])
            for j in positions
        ]
        evidence[q["id"]] = dict(zip(current_ids, why))
        adjusted = normalized_ranks(current_ids, rrf) + np.array(
            [v["offset"] for v in why]
        )
        channels.append(ranked_ids(current_ids, adjusted))
        pools[q["id"]] = balanced_pool(channels, 60)
        rankings[q["id"]] = dict(
            global_=ranked_ids(current_ids, current[:, 0]),
            region_title=ranked_ids(
                current_ids, 0.8 * current[:, 1] + 0.2 * current[:, 3]
            ),
            hybrid=ranked_ids(current_ids, hybrid),
            semantic_blend=ranked_ids(
                current_ids,
                0.5 * current[:, 1] + 0.2 * current[:, 3] + 0.3 * current[:, 4],
            ),
            multi_channel_rrf=ranked_ids(current_ids, rrf),
            public_evidence_soft=ranked_ids(current_ids, adjusted),
        )
        rankings[q["id"]]["global"] = rankings[q["id"]].pop("global_")
    assert rankings == load(ROOT / "baseline_rankings.json")
    assert pools == load(ROOT / "pools.json") and evidence == load(
        ROOT / "public_evidence.json"
    )
    metadata = load(ROOT / "feature_metadata.json")
    fitted = fit_supervised_bundle(
        SupervisedPipelineFitSupervisedBundleInputs(
            x,
            y,
            valid,
            np.array(folds["query_folds"]),
            np.array(folds["gallery_folds"]),
            bases,
            metadata,
            protocol["training_policy"],
            "machine_pixel_refined_development",
        )
    )
    for name in ["models", "candidates", "audits"]:
        assert fitted[name] == load(ROOT / ("trained_" + name + ".json"))
    for i, q in enumerate(inputs["queries"]):
        positions = np.flatnonzero(valid[i])
        current_ids = [ids[j] for j in positions]
        for name, scores in fitted["predictions"].items():
            rankings[q["id"]]["AI_oof_" + name] = ranked_ids(
                current_ids, scores[i, positions]
            )
    assert rankings == load(ROOT / "rankings.before_cross.json")
    by_id = {r["id"]: r for r in index.records}
    by_query = {q["id"]: q for q in inputs["queries"]}
    origins: Counter
    scored, origins = {}, Counter()
    for path in (ROOT / "pair_scores").glob("*.json"):
        row = load(path)
        qid, cid = row["query_id"], row["candidate_id"]
        public = dict(
            query={k: by_query[qid][k] for k in ["id", "text", "image_path"]},
            document=dict(
                id=cid, image_path=by_id[cid]["image_path"], text=by_id[cid]["title"]
            ),
        )
        digest = hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
        assert digest == row["model_input_sha256"]
        if row["origin"] == "identical_frozen_input_cache":
            assert (
                sha256_file(PROJECT_ROOT / row["cache_source"])
                == row["cache_source_sha256"]
            )
            previous = load(PROJECT_ROOT / row["cache_source"])
            assert (
                previous["model_input_sha256"] == digest
                and previous["logit"] == row["logit"]
                and row["seconds"] == 0
            )
        else:
            assert (
                row["origin"] == "actual_full_cross_encoder_inference"
                and row["seconds"] > 0
            )
        assert (qid, cid) not in scored
        scored[qid, cid] = row["logit"]
        origins[row["origin"]] += 1
    assert set(scored) == {(q, cid) for q, pool in pools.items() for cid in pool}
    assert sum(origins.values()) == 1080
    runtime = load(ROOT / "reranker_runtime.json")
    assert runtime["model"]["instruction"] == BUSINESS_INSTRUCTION
    assert (
        len(runtime["timings"]) == origins["actual_full_cross_encoder_inference"] + 360
    )
    for i, q in enumerate(inputs["queries"]):
        qid = q["id"]
        positions = np.flatnonzero(valid[i])
        current_ids = [ids[j] for j in positions]
        rrf, _ = recall_channels(current_ids, x[i, positions])
        offsets = np.array([evidence[qid][cid]["offset"] for cid in current_ids])
        for size in [20, 60]:
            pool = pools[qid][:size]
            logits = np.array([scored[qid, cid] for cid in pool])
            for name, offset in [("cross", None), ("cross_evidence", offsets)]:
                rankings[qid][f"{name}_pool{size}"] = rerank_complete(
                    current_ids, rrf, pool, logits, offset
                )
    enriched = {}
    by_query = {q["id"]: q for q in inputs["queries"]}
    for i, query in enumerate(inputs["queries"]):
        qid = query["id"]
        if not query["image_path"]:
            continue
        row = enrichment["queries"][qid]
        with np.load(
            ROOT / "query_enrichment" / row["vector_asset"], allow_pickle=False
        ) as archive:
            enriched[qid] = visual_channels(
                QueryFusionVisualChannelsInputs(
                    index,
                    titles,
                    profiles,
                    x[i, np.flatnonzero(valid[i])],
                    archive["views"],
                    archive["caption"],
                    row["description"],
                    np.flatnonzero(valid[i]),
                )
            )
    assert enriched == load(ROOT / "query_enrichment/ranking_features.json")
    described_scores = {}
    for path in (ROOT / "described_pairs").glob("*.json"):
        row = load(path)
        qid, cid = row["query_id"], row["candidate_id"]
        public = dict(
            query=dict(
                id=qid,
                text=enriched[qid]["description"] or None,
                image_path=by_query[qid]["image_path"],
            ),
            document=dict(
                id=cid, image_path=by_id[cid]["image_path"], text=by_id[cid]["title"]
            ),
        )
        assert (
            hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
            == row["model_input_sha256"]
        )
        assert (
            row["origin"] == "actual_full_described_image_cross_encoder_inference"
            and row["seconds"] > 0
        )
        assert row["query_modalities"] == (
            ["image", "text"] if enriched[qid]["description"] else ["image"]
        )
        assert (qid, cid) not in described_scores
        described_scores[qid, cid] = row["logit"]
    assert set(described_scores) == {
        (qid, cid) for qid, r in enriched.items() for cid in r["pool"]
    }
    for query in inputs["queries"]:
        qid = query["id"]
        if query["text"]:
            for name in [
                "query_multi_view",
                "query_caption_view_fusion",
                "described_cross_pool20",
                "described_cross_pool60",
            ]:
                rankings[qid][name] = list(rankings[qid]["cross_evidence_pool60"])
        else:
            data = enriched[qid]
            current_ids = data["ids"]
            rankings[qid]["query_multi_view"] = ranked_ids(
                current_ids, np.array(data["multi_scores"])
            )
            rankings[qid]["query_caption_view_fusion"] = ranked_ids(
                current_ids, np.array(data["fused_scores"])
            )
            for size in [20, 60]:
                pool = data["pool"][:size]
                logits = np.array([described_scores[qid, cid] for cid in pool])
                rankings[qid][f"described_cross_pool{size}"] = rerank_complete(
                    current_ids, np.array(data["fused_scores"]), pool, logits
                )
    assert rankings == load(ROOT / "rankings.json")
    labels = [r for r in original_labels if r["candidate_id"] in set(ids)]
    modes = {q["id"]: "text" if q["text"] else "image" for q in inputs["queries"]}
    assert metric_ranges(rankings, labels, modes) == load(
        ROOT / "internal_metrics.ai.development.json"
    )
    for folder, queue_path, media_registry, snapshot_path in [
        (ROOT, QUEUE, REGISTRY, SNAPSHOT),
        (
            ROOT / "supplement",
            PROJECT_ROOT / "data/module2_form_coverage_candidates/records.jsonl",
            PROJECT_ROOT / "data/module2_form_coverage_media/records.jsonl",
            ROOT / "supplement/prior_exclusion.json",
        ),
    ]:
        verify_files(
            folder / "dataset", load(folder / "dataset/manifest.json")["files"]
        )
        queue = read_rows(queue_path)
        original_candidates = {r["id"]: r for r in queue}
        snapshot = load(snapshot_path)
        selected = read_rows(folder / "dataset/products.jsonl")
        downloaded = load(folder / "download_cache.json")
        assert {r["id"] for r in read_rows(media_registry)} == set(downloaded)
        signatures: list = []
        for row in selected:
            assert (
                row["source_record_sha256"]
                == original_candidates[row["id"]]["source_record_sha256"]
            )
            signature = fingerprints(Path(row["image_path"]))
            assert signature == row["fingerprints"]
            assert not is_exposed(signature, snapshot["image_fingerprints"], 4)
            assert not is_exposed(signature, signatures, 4)
            signatures.append(signature)
    qa = read_rows(ROOT / "catalog_QA.ai.jsonl")
    new_ids = {r["id"] for r in qa if r["status"] == "usable"}
    assert set(ids) == set(old_positions) | new_ids
    new_judgments = read_rows(ROOT / "judgments.new.ai.jsonl")
    assert len(new_judgments) == 630
    assert all(
        r["review_origin"] == "model" and r["human_label"] is False
        for r in new_judgments
    )
    assert all(
        r["review_origin"] == "model" and r["human_label"] is False
        for r in original_labels
    )
    old_AI = read_rows(PARENT / "judgments.ai.jsonl")
    assert [r for r in original_labels if r["candidate_id"] in old_positions] == [
        r for r in old_AI if r["candidate_id"] in old_positions
    ]
    assert all(
        set(r) == {"id", "product_type", "appearance", "caption_origin", "image_sha256"}
        for r in read_rows(ROOT / "profiles.public.jsonl")
    )
    history = [
        "module2_blind",
        "module2_precision",
        "module2_training",
        "module2_precision60",
        "module2_business_review",
        "module2_business_acceptance",
        "module2_machine_assist",
        "module2_continuation",
        "module2_target_p60_r85",
        "module2_precision_recall",
        "module2_catalog_review",
        "module2_ai_review",
        "module2_expanded_development",
        "module2_visual_coverage",
    ]
    for name in history:
        folder = PROJECT_ROOT / "artifacts" / name
        receipt_name = (
            "preparation_receipt.json"
            if name
            in {
                "module2_business_review",
                "module2_business_acceptance",
                "module2_catalog_review",
            }
            else "completion_receipt.json"
        )
        receipt = load(folder / receipt_name)
        verify_files(folder, receipt["files"])
        if receipt.get("project_files"):
            verify_files(PROJECT_ROOT, receipt["project_files"])
    status = load(ROOT / "public_status.json")
    assert status["quality_metrics"] is None and status["formal_metrics"] is None
    assert (
        status["acceptance_predictions"] == 0
        and status["formal_acceptance"] is False
        and status["goal_met"] is False
    )
    result = dict(
        development_products=len(ids),
        original_vectors_reused_by_ID=len(old_positions),
        new_products_encoded=len(new_ids),
        actual_embedding_calls=actual_calls,
        new_gallery_embedding_calls=gallery_calls,
        query_embedding_calls=enrichment["actual_embedding_calls"],
        actual_query_description_VLM_calls=6,
        described_cross_pairs=len(described_scores),
        image_views=sum(len(r["views"]) for r in index.records),
        new_products_reviewed=35,
        all_new_relations=630,
        indexed_AI_relations=len(original_labels),
        unknown_labels_retained=int((valid & np.isnan(y)).sum()),
        methods_replayed=16,
        full_rankings_replayed=288,
        fold_models_retrained=18,
        final_candidates_retrained=2,
        pair_scores=len(scored),
        actual_cross_encoder_inferences=origins["actual_full_cross_encoder_inference"],
        identical_input_cached_pairs=origins["identical_frozen_input_cache"],
        old_sealed_roots_unchanged=len(history),
        heldout_pixels_opened=0,
        acceptance_predictions=0,
        formal_metrics=None,
        goal_met=False,
    )
    if (ROOT / "verification.json").exists():
        assert load(ROOT / "verification.json") == result
    else:
        write_json(ROOT / "verification.json", result)
    if (ROOT / "completion_receipt.json").exists():
        verify_files(ROOT, load(ROOT / "completion_receipt.json")["files"])
    print(json.dumps(result))


if __name__ == "__main__":
    main()
