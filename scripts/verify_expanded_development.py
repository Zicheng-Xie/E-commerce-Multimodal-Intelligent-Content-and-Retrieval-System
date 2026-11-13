"""Replay expanded development without GPU inference or heldout pixel access."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import ijson
import numpy as np

import run  # noqa: F401
from advance_development_catalog import AI, BANK, OLD, RESERVED, ROOT, check, load
from multimodal_ecommerce.ai_review_metrics import metric_ranges
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_retrieval import BUSINESS_INSTRUCTION, hybrid_scores
from multimodal_ecommerce.business_review_jobs import canonical_hash
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.development_expansion import (
    acceptance_guard,
    fold_metadata,
    load_expanded_index,
)
from multimodal_ecommerce.intent_retrieval import (
    balanced_pool,
    normalized_ranks,
    public_evidence,
    recall_channels,
    rerank_complete,
)
from multimodal_ecommerce.regional_retrieval import RegionalIndex
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
from prepare_visual_coverage_candidates import bucket


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through write_json.
    """
    protocol = check()
    inputs = load(ROOT / "inputs.public.json")
    assert all(set(r) == {"id", "title", "image_path"} for r in inputs["catalog"])
    assert all(
        set(q) == {"id", "text", "image_path", "exclude_id"} for q in inputs["queries"]
    )
    raw = {
        r["id"]: r
        for r in read_rows(PROJECT_ROOT / "data/stage2_taxonomy_v2/products.jsonl")
    }
    raw.update(
        {
            r["id"]: r
            for r in read_rows(
                PROJECT_ROOT / "artifacts/module2_catalog_review/dataset/products.jsonl"
            )
        }
    )
    guard = acceptance_guard(inputs["catalog"], raw, read_rows(RESERVED))
    assert guard == load(ROOT / "guard.json")
    folds = fold_metadata(inputs["catalog"], inputs["queries"], guard)
    assert folds == load(ROOT / "folds.json")
    index, titles = load_expanded_index(ROOT / "index")
    assert [r["id"] for r in index.records] == [r["id"] for r in inputs["catalog"]]
    old_index, old_titles, _ = RegionalIndex.load(OLD)
    assert old_titles is not None
    old_positions = {r["id"]: i for i, r in enumerate(old_index.records)}
    old_metadata = load(BANK / "feature_metadata.json")
    old_profile_positions = {
        cid: i for i, cid in enumerate(old_metadata["gallery_ids"])
    }
    with np.load(BANK / "profile_vectors.npz", allow_pickle=False) as archive:
        old_profiles = archive["vectors"].copy()
    with np.load(ROOT / "index/vectors.npz", allow_pickle=False) as archive:
        regions, profiles = archive["regions"].copy(), archive["profiles"].copy()
        raw_titles = archive["texts"].copy()
    for i, product in enumerate(inputs["catalog"][:286]):
        j = old_positions[product["id"]]
        assert np.array_equal(regions[i], old_index.vectors[j])
        assert np.array_equal(raw_titles[i], old_titles[j])
        assert np.array_equal(
            profiles[i], old_profiles[old_profile_positions[product["id"]]]
        )
    for record in index.records:
        assert sha256_file(Path(record["image_path"])) == record["image_sha256"]
    calls = 0
    for i, product in enumerate(inputs["catalog"][286:], 286):
        path = (
            ROOT
            / "embedding_cache"
            / (hashlib.sha256(product["id"].encode()).hexdigest()[:24] + ".json")
        )
        cache = load(path)
        asset = path.parent / cache["vector_asset"]
        assert cache["id"] == product["id"] and cache["title"] == product["title"]
        assert sha256_file(asset) == cache["vector_sha256"]
        with np.load(asset, allow_pickle=False) as values:
            assert np.array_equal(regions[i, : len(cache["views"])], values["regions"])
            assert np.array_equal(raw_titles[i], values["title"])
            assert np.array_equal(profiles[i], values["profile"])
        calls += len(cache["views"]) + 2
    assert calls == len(load(ROOT / "embedding_runtime.json")["calls"]) == 470
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
    original_labels = read_rows(AI / "judgments.ai.jsonl")
    lookup = {(r["query_id"], r["candidate_id"]): r["grade"] for r in original_labels}
    for i, q in enumerate(inputs["queries"]):
        for j, cid in enumerate(ids):
            if valid[i, j]:
                grade = lookup[q["id"], cid]
                assert (np.isnan(y[i, j]) and grade is None) or y[i, j] == grade
    assert valid.sum() == 6187 and (valid & np.isnan(y)).sum() == 16
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
    prior_root = PROJECT_ROOT / "artifacts/module2_precision_recall"
    previous = load(prior_root / "inputs.json")
    pq = {q["id"]: q for q in previous["queries"]}
    pd = {r["id"]: r for r in previous["catalog"]}
    previous_scores = {}
    for path in (prior_root / "pair_scores").glob("*.json"):
        value = load(path)
        qid, cid = value["query_id"], value["candidate_id"]
        public = dict(
            query=pq[qid],
            document=dict(
                id=cid, image_path=pd[cid]["image_path"], text=pd[cid]["title"]
            ),
        )
        previous_scores[
            hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
        ] = value["logit"]
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
            assert row["logit"] == previous_scores[digest] and row["seconds"] == 0
        else:
            assert (
                row["origin"] == "actual_full_cross_encoder_inference"
                and row["seconds"] > 0
            )
        assert (qid, cid) not in scored
        scored[qid, cid] = row["logit"]
        origins[row["origin"]] += 1
    assert set(scored) == {(q, cid) for q, pool in pools.items() for cid in pool}
    assert origins == dict(
        identical_frozen_input_cache=887, actual_full_cross_encoder_inference=193
    )
    assert (
        load(ROOT / "reranker_runtime.json")["model"]["instruction"]
        == BUSINESS_INSTRUCTION
    )
    assert len(load(ROOT / "reranker_runtime.json")["timings"]) == 193
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
    assert rankings == load(ROOT / "rankings.json")
    labels = [r for r in original_labels if r["candidate_id"] in set(ids)]
    modes = {q["id"]: "text" if q["text"] else "image" for q in inputs["queries"]}
    assert metric_ranges(rankings, labels, modes) == load(
        ROOT / "internal_metrics.ai.development.json"
    )
    coverage = ROOT / "coverage_next"
    cp = load(coverage / "protocol.json")
    receipt = load(coverage / "receipt.json")
    verify_files(coverage, receipt["files"])
    assert (
        sha256_file(PROJECT_ROOT / receipt["exposure_registry"])
        == receipt["exposure_registry_sha256"]
    )
    expected = {
        r["source_id"]: r for r in read_rows(coverage / "source_candidates.jsonl")
    }
    found = set()
    assert sha256_file(PROJECT_ROOT / cp["source"]) == cp["source_sha256"]
    with (PROJECT_ROOT / cp["source"]).open("rb") as handle:
        for number, (sid, raw) in enumerate(ijson.kvitems(handle, ""), 1):
            if str(sid) in expected:
                row = expected[str(sid)]
                assert number == row["source_position"]
                assert (
                    canonical_hash(raw) == row["source_record_sha256"]
                    and bucket(raw["title"]) == row["bucket"]
                )
                found.add(str(sid))
            if number == cp["policy"]["scan_limit"]:
                break
    assert len(found) == 80
    amendment = load(ROOT / "query_import_amendment.json")
    old_source = ROOT / amendment["executed_source"]
    current_source = PROJECT_ROOT / amendment["current_source"]
    assert (
        sha256_file(old_source) == amendment["executed_sha256"]
        and sha256_file(current_source) == amendment["current_sha256"]
    )
    assert old_source.read_text(encoding="utf-8").replace(
        "from multimodal_ecommerce.precision_reranking import color_descriptor\n", ""
    ) == current_source.read_text(encoding="utf-8")
    for name in ["demo_text", "demo_image_roi"]:
        demo = load(ROOT / name / "results.json")
        assert demo["catalog_size"] == 344 and len(demo["results"]) == 10
        assert (
            not demo["historical_query_rankings_used"]
            and not demo["relevance_table_used_as_runtime_feature"]
        )
        assert demo["implementation_sha256"] in {
            amendment["executed_sha256"],
            amendment["current_sha256"],
        }
    assert load(ROOT / "query_guard_probe.json")["heldout_pixels_opened"] is False
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
        verify_files(folder, load(folder / receipt_name)["files"])
    status = load(ROOT / "public_status.json")
    assert (
        status["formal_metrics"] is None
        and status["quality_metrics"] is None
        and status["acceptance_predictions"] == 0
    )
    snapshot = load(ROOT / "post_development_exclusion_snapshot.json")
    verify_files(PROJECT_ROOT, snapshot["files"])
    assert {r["id"] for r in expected.values()} <= set(snapshot["ids"])
    assert {key for r in expected.values() for key in r["family_keys"]} <= set(
        snapshot["family_keys"]
    )
    result = dict(
        expanded_products=344,
        original_vectors_aligned_by_ID=286,
        new_products_encoded=58,
        full_views=2095,
        actual_embedding_calls=470,
        allowed_AI_development_relations=6187,
        unknown_labels_retained=16,
        cross_fit_models_retrained=18,
        candidate_models_retrained=2,
        methods_replayed=12,
        full_rankings_replayed=216,
        pair_scores=1080,
        actual_new_cross_inferences=193,
        identical_input_cached_pairs=887,
        new_coverage_metadata_verified=80,
        future_target_images=40,
        actual_query_modes_verified=2,
        reserved_registry_unchanged=True,
        reserved_query_blocked_before_pixel_read=True,
        old_sealed_roots_unchanged=12,
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
