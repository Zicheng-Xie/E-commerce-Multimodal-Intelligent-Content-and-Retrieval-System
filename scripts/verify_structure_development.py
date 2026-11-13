"""Replay the development experiment and preserve its exposure and decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import ijson
import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_acceptance import product_family_keys
from multimodal_ecommerce.business_review_jobs import canonical_hash
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
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
from structure_development import (
    BUCKETS,
    MEDIA,
    METHODS,
    PARENT,
    POLICY,
    QUEUE,
    ROOT,
    check,
    inputs,
    load,
)


def persist(path: Path, value: object) -> None:
    """Apply the persist workflow step.

    Args:
        path: Path of the input or output artifact used by this operation.
        value: Input value to validate or transform.

    Notes:
        Writes derived artifacts through write_json.
    """
    if path.exists():
        assert load(path) == value, str(path)
    else:
        write_json(path, value)


def isolation() -> dict:
    """Return isolation for this workflow.

    Returns:
        Mapping with fields: metadata, selected_images, downloaded,
        overlap_family_keys, overlap_image_fingerprints_dhash4,
        previous_independent_AI_blind_reused.
    """
    protocol = check()
    verify_files(ROOT, load(ROOT / "acquisition_receipt.json")["files"])
    prior = load(PARENT / "post_blind_exclusion_snapshot.json")
    metadata = read_rows(QUEUE)
    assert len(metadata) == len({r["id"] for r in metadata}) == 256
    seen = set(prior["family_keys"])
    for row in metadata:
        assert row["family_keys"] == sorted(product_family_keys(row))
        assert not seen & set(row["family_keys"])
        seen.update(row["family_keys"])
    source = PROJECT_ROOT / protocol["source"]
    assert sha256_file(source) == protocol["source_sha256"]
    # Reconstruct hash sampling from the protected source.
    # The receipt alone does not establish the selected source rows.
    groups: dict = {b[0]: [] for b in BUCKETS}
    rows_by_id = {r["id"]: r for r in metadata}
    prior_keys = set(prior["family_keys"])
    with source.open("rb") as handle:
        for number, (sid, raw) in enumerate(ijson.kvitems(handle, ""), 1):
            if number > POLICY["scan_after"] and raw.get("url"):
                title = str(raw.get("title") or "")
                matches = [
                    b[0]
                    for b in BUCKETS
                    if any(t in title for t in b[2])
                    and not any(t in title for t in b[3])
                ]
                if matches:
                    cid = f"product/train/{sid}"
                    row = dict(
                        id=cid,
                        dataset="product",
                        split="train",
                        source_id=str(sid),
                        text=title,
                        image_url=raw["url"],
                        pv=raw.get("pv") or "",
                    )
                    keys = product_family_keys(row)
                    if not keys & prior_keys:
                        groups[matches[0]].append(
                            dict(
                                id=cid,
                                keys=keys,
                                priority=hashlib.sha256(
                                    f"{POLICY['seed']}:{cid}".encode()
                                ).hexdigest(),
                            )
                        )
                    if cid in rows_by_id:
                        assert rows_by_id[cid][
                            "source_record_sha256"
                        ] == canonical_hash(raw)
                        assert rows_by_id[cid]["source_position"] == number
                        assert (
                            rows_by_id[cid]["text"] == title
                            and rows_by_id[cid]["image_url"] == raw["url"]
                        )
            if number == POLICY["scan_limit"]:
                break
    selected, seen_again = [], set(prior["family_keys"])
    for bucket, group in groups.items():
        accepted = 0
        for row in sorted(group, key=lambda r: r["priority"]):
            if row["keys"] & seen_again:
                continue
            selected.append(
                (
                    row["id"],
                    bucket,
                    "tune" if accepted % 2 == 0 else "validation",
                    accepted // 2,
                )
            )
            seen_again.update(row["keys"])
            accepted += 1
            if accepted == 32:
                break
    assert selected == [
        (r["id"], r["bucket"], r["cohort"], r["cohort_priority"]) for r in metadata
    ]
    chosen = read_rows(ROOT / "selected.evaluator.jsonl")
    signatures = list(prior["image_fingerprints"])
    assert len(chosen) == 208
    for row in chosen:
        actual = fingerprints(Path(row["image_path"]))
        assert actual == row["fingerprints"]
        assert not is_exposed(actual, signatures, 4)
        signatures.append(actual)
    assert {
        c: sum(r["cohort"] == c for r in chosen) for c in POLICY["cohorts"]
    } == dict(tune=104, validation=104)
    # Every successfully fetched image, including rejected duplicates, stays exposed.
    downloaded = read_rows(MEDIA)
    for row in downloaded:
        if row["status"] == "downloaded":
            assert fingerprints(Path(row["image_path"])) == row["fingerprints"]
    snapshot = dict(prior)
    snapshot["scope"] = (
        "All prior exposures, consumed AI blind and structure-development cohorts; none reusable as unseen"
    )
    snapshot["parent_snapshot_sha256"] = sha256_file(
        PARENT / "post_blind_exclusion_snapshot.json"
    )
    snapshot["ids"] = sorted(set(prior["ids"]) | {r["id"] for r in metadata})
    snapshot["family_keys"] = sorted(seen)
    snapshot["product_source_ids"] = sorted(
        set(prior["product_source_ids"]) | {r["source_id"] for r in metadata}
    )
    snapshot["image_fingerprints"] = list(prior["image_fingerprints"]) + [
        r["fingerprints"] for r in downloaded if r["status"] == "downloaded"
    ]
    snapshot["files"] = dict(
        prior["files"],
        **{
            p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
            for p in [QUEUE, MEDIA]
        },
    )
    snapshot["structure_development_metadata"] = 256
    snapshot["structure_development_selected_images"] = 208
    snapshot["structure_development_queries"] = 32
    persist(ROOT / "post_development_exclusion_snapshot.json", snapshot)
    return dict(
        metadata=256,
        selected_images=208,
        downloaded=len(downloaded),
        overlap_family_keys=0,
        overlap_image_fingerprints_dhash4=0,
        previous_independent_AI_blind_reused=False,
    )


def cohort_replay(cohort: str) -> dict:
    """Return cohort replay for this workflow.

    Args:
        cohort: Cohort used by this operation.

    Returns:
        Mapping with fields: queries, gallery, full_rankings,
        actual_full_cross_pairs, actual_embedding_calls, unknown, zero_positive_queries,
        all_orders_and_metrics_replayed.
    """
    folder = ROOT / cohort
    for name in [
        "profile_receipt.json",
        "label_receipt.json",
        "encoding_receipt.json",
        "prediction_receipt.json",
        "evaluation_receipt.json",
    ]:
        verify_files(folder, load(folder / name)["files"])
    catalog = load(folder / "catalog.public.json")
    assert len(catalog) == 96 and all(
        set(d) == {"id", "title", "image_path"} for d in catalog
    )
    public_profiles = read_rows(folder / "profiles.public.jsonl")
    assert len(public_profiles) == 96 and all(
        set(p) == {"id", "product_type", "appearance"} for p in public_profiles
    )
    native = load(folder / "native/judgments.json")["queries"]
    labels = read_rows(folder / "judgments.ai.jsonl")
    assert (
        len(labels) == len({(r["query_id"], r["candidate_id"]) for r in labels}) == 1536
    )
    for row in labels:
        v = native[row["query_id"]]
        pos = row["native_position"]
        expected = (
            1 if pos in v["positive"] else None if str(pos) in v["unknown"] else 0
        )
        assert (
            row["grade"] == expected and row["candidate_id"] == catalog[pos - 1]["id"]
        )
        assert row["review_origin"] == "model" and row["human_label"] is False
    selected = load(ROOT / "selection.json")["selected"]
    methods = (
        list(METHODS)
        if cohort == "tune"
        else list(dict.fromkeys(["original", selected]))
    )
    index, titles, profiles, texts, colors = inputs(cohort)
    ids = [d["id"] for d in index.records]
    full_pairs = 0
    ranks = {}
    queries = load(folder / "queries.public.json")
    assert len(queries) == 16
    for q in queries:
        assert set(q) == {"id", "text", "image_path"}
        target = folder / "runs" / q["id"]
        with np.load(
            folder / "queries" / q["id"] / "vectors.npz", allow_pickle=False
        ) as bank:
            views, caption = bank["views"].copy(), bank["caption"].copy()
        image = Path(q["image_path"]) if q["image_path"] else None
        feats = feature_matrix(
            SemanticRankerFeatureMatrixInputs(
                index,
                titles,
                profiles,
                texts,
                views[0],
                text=q["text"],
                image=image,
                color_bank=colors,
            )
        )
        saved = load(target / "features.json")
        assert ids == saved["ids"] and np.allclose(
            saved["features"], feats, atol=1e-14, rtol=0
        )
        offsets = None
        if image:
            desc = load(folder / "queries" / q["id"] / "description.json")
            assert (
                desc["gallery_or_judgments_supplied"] is False
                and desc["human_label"] is False
            )
            channels = visual_channels(
                QueryFusionVisualChannelsInputs(
                    index,
                    titles,
                    profiles,
                    feats,
                    views,
                    caption,
                    desc["text"],
                    np.arange(len(ids), dtype=int),
                )
            )
            base, pool = np.array(channels["fused_scores"]), channels["pool"][:20]
        else:
            base, recall_orders = recall_channels(ids, feats)
            offsets = np.array(
                [
                    public_evidence(q["text"], d["title"], p)["offset"]
                    for d, p in zip(index.records, texts)
                ]
            )
            recall_orders.append(ranked_ids(ids, normalized_ranks(ids, base) + offsets))
            pool = balanced_pool(recall_orders, 60)[:20]
        assert saved["pool"] == pool and np.allclose(
            saved["base"], base, atol=1e-14, rtol=0
        )
        assert (saved["offsets"] is None) == (offsets is None)
        if offsets is not None:
            assert np.array_equal(saved["offsets"], offsets)
        current = {
            "global": ranked_ids(ids, feats[:, 0]),
            "region_title": ranked_ids(ids, 0.8 * feats[:, 1] + 0.2 * feats[:, 3]),
        }
        for method in methods:
            if not image and method != "original":
                current[method] = current["original"]
                continue
            raw = load(target / (method + ".json"))
            assert raw["policy"] == METHODS[method] and raw["pool"] == pool
            assert (
                len(raw["logits"]) == len(raw["timings"]) == 20
                and np.isfinite(raw["logits"]).all()
            )
            assert [v["document_id"] for v in raw["timings"]] == pool
            current[method] = rerank_complete(
                ids, base, pool, np.array(raw["logits"]), offsets
            )
            full_pairs += 20
        ranks[q["id"]] = current
    assert ranks == load(folder / "predictions.json")
    modes = {q: q.split("_")[0] for q in ranks}
    metrics = evaluate_intervals(ranks, labels, modes)
    assert metrics == load(folder / "internal_metrics.ai.development.json")
    assert paired_differences(ranks, labels, modes) == load(
        folder / "internal_paired.json"
    )
    renamed = {
        q: {
            ("cross_evidence_pool60" if m == "original" else m): v
            for m, v in methods.items()
        }
        for q, methods in ranks.items()
    }
    assert paired_differences(renamed, labels, modes) == load(
        folder / "internal_vs_original.json"
    )
    runtime = load(folder / "index/embedding_runtime.json")
    assert runtime["device"].startswith("cuda") and runtime["dtype"] == "torch.bfloat16"
    assert runtime["config"] == load(PROJECT_ROOT / "configs/embedding_model.json")
    cross = load(folder / "cross_runtime.json")
    for key in [
        "repository",
        "revision",
        "code_revision",
        "path",
        "files",
        "vendor_sha256",
        "dtype",
        "max_pixels",
        "max_length",
        "batch_size",
    ]:
        assert (
            cross["model"][key]
            == load(PROJECT_ROOT / "configs/reranker_model.json")[key]
        )
    assert full_pairs == len(cross["timings"])
    capacity = {
        scope: [
            sum(
                r["precision_ceiling_range"][i]
                for r in metrics["query_capacity"]
                if scope == "all" or r["query_id"].startswith(scope)
            )
            / (16 if scope == "all" else 8)
            for i in [0, 1]
        ]
        for scope in ["all", "image", "text"]
    }
    persist(
        folder / "internal_capacity.json",
        dict(
            precision_at_10_capacity=capacity,
            full_catalog_evaluated=True,
            fixed_k=10,
            formal_acceptance=False,
        ),
    )
    lookup = {(r["query_id"], r["candidate_id"]): r["grade"] for r in labels}
    error_rows = []
    for qid, orders in ranks.items():
        positives = {cid for cid in ids if lookup[qid, cid] == 1}
        unknown = {cid for cid in ids if lookup[qid, cid] is None}
        pool = load(folder / "runs" / qid / "features.json")["pool"]
        oracle = min(10, len(positives))
        for method, order in orders.items():
            top = set(order[:10])
            original_top = set(orders["original"][:10])
            error_rows.append(
                dict(
                    query_id=qid,
                    mode=modes[qid],
                    method=method,
                    known_positive_total=len(positives),
                    unknown_total=len(unknown),
                    known_hits_10=len(top & positives),
                    unknown_at_10=len(top & unknown),
                    known_oracle_10=oracle,
                    known_recoverable_positions_missed=oracle - len(top & positives),
                    known_candidate_pool_capacity=min(10, len(positives & set(pool))),
                    known_candidate_pool_gap=oracle
                    - min(10, len(positives & set(pool))),
                    known_positive_ids_lost_vs_original=sorted(
                        (original_top - top) & positives
                    ),
                    known_positive_ids_gained_vs_original=sorted(
                        (top - original_top) & positives
                    ),
                )
            )
    persist(
        folder / "internal_error_attribution.json",
        dict(
            rows=error_rows,
            labels_used_after_prediction_only=True,
            unknown_labels_not_imputed=True,
            no_quality_claim_outside_development=True,
        ),
    )
    return dict(
        queries=16,
        gallery=96,
        full_rankings=sum(len(m) for m in ranks.values()),
        actual_full_cross_pairs=full_pairs,
        actual_embedding_calls=len(runtime["calls"]),
        unknown=sum(r["grade"] is None for r in labels),
        zero_positive_queries=[
            r["query_id"]
            for r in metrics["query_capacity"]
            if r["known_positive"] == 0 and r["unknown"] == 0
        ],
        all_orders_and_metrics_replayed=True,
    )


def verify() -> None:
    """Verify the recorded artifacts and their consistency."""
    from verify_rank_stability import verify_history

    isolation_result = isolation()
    verify_files(ROOT, load(ROOT / "selection_receipt.json")["files"])
    selection = load(ROOT / "selection.json")
    tune_metrics = load(ROOT / "tune/internal_metrics.ai.development.json")["scopes"]
    tune_paired = load(ROOT / "tune/internal_vs_original.json")["scopes"]
    gates = {
        m: all(
            v["nonregression_proven_for_shared_unknown_labels"]
            for s in ["all", "text", "image"]
            for v in tune_paired[s][m]["cross_evidence_pool60"].values()
        )
        for m in ["structure", "structure_pixels"]
    }
    eligible = ["original", *[m for m in ["structure", "structure_pixels"] if gates[m]]]
    expected_method = max(
        eligible,
        key=lambda m: (
            tune_metrics["image"][m]["ranges"]["precision@10"][0],
            tune_metrics["image"][m]["ranges"]["recall@10"][0] or 0,
            m == "original",
        ),
    )
    assert (
        selection["selected"] == expected_method
        and selection["nonregression_gates"] == gates
        and selection["eligible"] == eligible
    )
    assert (ROOT / "selection_receipt.json").stat().st_mtime_ns < (
        ROOT / "validation/native/gallery_01.jpg"
    ).stat().st_mtime_ns
    assert (ROOT / "validation/label_receipt.json").stat().st_mtime_ns < (
        ROOT / "validation/predictions.json"
    ).stat().st_mtime_ns
    results = {c: cohort_replay(c) for c in POLICY["cohorts"]}
    previous = verify_history()
    for stage in [PROJECT_ROOT / "artifacts/module2_rank_stability", PARENT]:
        r = load(stage / "completion_receipt.json")
        verify_files(stage, r["files"])
        verify_files(PROJECT_ROOT, r["project_files"])
    chosen = load(ROOT / "selection.json")["selected"]
    metrics = load(ROOT / "validation/internal_metrics.ai.development.json")["scopes"]
    pairs = load(ROOT / "validation/internal_paired.json")["scopes"]
    threshold = all(
        metrics[s][chosen]["precision_60_lower_bound_met"]
        for s in ["all", "image", "text"]
    )
    recall = all(
        metrics[s][chosen]["recall_85_lower_bound_met"]
        for s in ["all", "image", "text"]
    )
    nonreg = all(
        v["nonregression_proven_for_shared_unknown_labels"]
        for s in ["all", "image", "text"]
        for b in ["global", "region_title"]
        for v in pairs[s][chosen][b].values()
    )
    decision = dict(
        selected=chosen,
        independent_development_validation_completed=True,
        all_modes_precision_threshold_met=threshold,
        all_modes_recall_preference_met=recall,
        nonregression_against_global_and_region_proven=nonreg,
        ready_to_freeze_for_new_independent_AI_blind=threshold and recall and nonreg,
        old_blind_reuse_as_unseen=False,
        default_strategy_changed=False,
        formal_acceptance=False,
        formal_metrics=None,
        goal_met=False,
        human_labels=0,
        next_action=(
            "Freeze selected full method then acquire a fresh unseen independent cohort"
            if threshold and recall and nonreg
            else "Continue development coverage and ranking; retain this validation as exposed regression evidence and use another fresh validation for any later selection"
        ),
    )
    persist(ROOT / "decision.json", decision)
    queue = []
    for cohort in POLICY["cohorts"]:
        metrics_detail = load(ROOT / cohort / "internal_metrics.ai.development.json")
        queries = {q["id"]: q for q in load(ROOT / cohort / "queries.public.json")}
        observations = load(ROOT / cohort / "native/judgments.json")["queries"]
        for row in metrics_detail["query_capacity"]:
            qid = row["query_id"]
            if qid.startswith("image"):
                queue.append(
                    dict(
                        cohort=cohort,
                        exposed_query_id=qid,
                        query_image=queries[qid]["image_path"],
                        current_known_positive=row["known_positive"],
                        current_unknown=row["unknown"],
                        additional_known_matches_to_reach_six_per_query=max(
                            0, 6 - row["known_positive"]
                        ),
                        required_primary_form_evidence=observations[qid]["reason"],
                        query_is_exposed_development_only=True,
                        query_removal_allowed=False,
                    )
                )
    persist(
        ROOT / "next_development_coverage_queue.json",
        dict(
            role="exposed_development_collection_priorities_not_an_acceptance_design",
            entries=queue,
            **{
                "six_matches_per_query_is_a_collection_diagnostic_not_a_macro_goal_proof": True
            },
            uncertain_labels_cannot_be_changed_to_close_deficits=True,
            keep_every_candidate_and_original_query_for_before_after_comparison=True,
            preserve_original_same_item_and_consumed_blind_evidence=True,
            any_later_method_selection_requires_another_family_disjoint_validation=True,
            **{
                "future_independent_AI_blind_requires_new_frozen_protocol_and_never_uses_these_queries": True
            },
        ),
    )
    value = dict(
        isolation=isolation_result,
        cohorts=results,
        prior_sealed_stages_verified=len(previous) + 2,
        provenance="Same AI conversation, rank-blinded labels, not independent annotators or human validation",
        exact_intervals_are_not_statistical_confidence_intervals=True,
        formal_metrics=None,
    )
    persist(ROOT / "verification.json", value)
    if (ROOT / "completion_receipt.json").exists():
        r = load(ROOT / "completion_receipt.json")
        verify_files(ROOT, r["files"])
        verify_files(PROJECT_ROOT, r["project_files"])
        verify_files(
            ROOT / "source_release",
            load(ROOT / "source_release/manifest.json")["files"],
        )
    print(
        json.dumps(dict(replay_passed=True, decision=decision, cohorts=results)),
        flush=True,
    )


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    verify()
