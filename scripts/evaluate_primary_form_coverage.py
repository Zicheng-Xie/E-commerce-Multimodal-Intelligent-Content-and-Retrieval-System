"""Exact-label-bound development scoring, fixed selection and data isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.ai_review_metrics import metric_ranges
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_acceptance import product_family_keys
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.development_expansion import load_expanded_index
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
from primary_form_coverage import (
    FACETS,
    FRESH_MEDIA,
    FRESH_QUEUE,
    MEDIA,
    PARENT,
    POLICY,
    QUEUE,
    ROOT,
    PrimaryFormCoverageScanInputs,
    check,
    facet,
    load,
    query_bucket,
    receipt,
    scan,
)
from report_focus_coverage import paired_differences
from structure_development import BUCKETS


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


def snapshot(final: bool = False) -> dict:
    """Return snapshot for this workflow.

    Args:
        final: Final used by this operation.

    Returns:
        Result mapping produced by the workflow.
    """
    prior = load(PARENT / "post_development_exclusion_snapshot.json")
    queues = [QUEUE] + ([FRESH_QUEUE] if final else [])
    registries = [MEDIA] + ([FRESH_MEDIA] if final else [])
    rows = [r for p in queues for r in read_rows(p)]
    images = [
        r["fingerprints"]
        for p in registries
        for r in read_rows(p)
        if r["status"] == "downloaded"
    ]
    out = dict(prior)
    out.update(
        scope="All exposed development, consumed independent AI blind, primary-form coverage and any completed fresh-query validation; not reusable as unseen",
        parent_snapshot_sha256=sha256_file(
            PARENT / "post_development_exclusion_snapshot.json"
        ),
        ids=sorted(set(prior["ids"]) | {r["id"] for r in rows}),
        family_keys=sorted(
            set(prior["family_keys"]) | {k for r in rows for k in r["family_keys"]}
        ),
        product_source_ids=sorted(
            set(prior["product_source_ids"]) | {r["source_id"] for r in rows}
        ),
        image_fingerprints=prior["image_fingerprints"] + images,
        files=dict(
            prior["files"],
            **{
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                for p in queues + registries
            },
        ),
        primary_form_metadata=len(rows),
        primary_form_downloaded=len(images),
        primary_form_fresh_queries=8 if final else 0,
    )
    persist(
        ROOT
        / (
            "post_development_exclusion_snapshot.json"
            if final
            else "post_coverage_exclusion_snapshot.json"
        ),
        out,
    )
    return out


def freeze_evaluation() -> None:
    """Freeze evaluation.

    Raises:
        RuntimeError: Freeze scoring and selection before new predictions.
        FileExistsError: Preserve scoring freeze.

    Notes:
        Writes derived artifacts through write_json.
    """
    check()
    if any(
        (ROOT / c / "predictions.json").exists()
        for c in ["tune", "regression", "fresh"]
    ):
        raise RuntimeError("Freeze scoring and selection before new predictions")
    if (ROOT / "evaluation_protocol.json").exists():
        raise FileExistsError("Preserve scoring freeze")
    snapshot()
    write_json(
        ROOT / "evaluation_protocol.json",
        dict(
            implementation_sha256=sha256_file(Path(__file__)),
            runner_sha256=sha256_file(
                PROJECT_ROOT / "scripts/primary_form_coverage.py"
            ),
            selection=POLICY["selection"],
            fixed_k=10,
            original_labels_unchanged=True,
            paired_bounds="Exact shared-completion enumeration up to 16 unknown labels per query; otherwise conservative subtraction of exact macro intervals, with identical orders fixed to zero. Never impute unresolved labels.",
            sampling_uncertainty_or_AI_error_not_covered_by_intervals=True,
            formal_acceptance=False,
            formal_metrics=None,
        ),
    )


def paired(rankings: dict, labels: list[dict], modes: dict, metrics: dict) -> dict:
    """Return paired for this workflow.

    Args:
        rankings: Candidate rankings keyed by query and retrieval method.
        labels: Judgments used for the stated development or evaluation protocol.
        modes: Modes used by this operation.
        metrics: Metrics used by this operation.

    Returns:
        Mapping with fields: scopes, bound_type, baseline_alias,
        same_unknown_completion_for_candidate_and_baseline, unknown_imputation,
        formal_acceptance.
    """
    counts = [
        sum(r["grade"] is None for r in labels if r["query_id"] == q) for q in rankings
    ]
    renamed = {
        q: {
            ("cross_evidence_pool60" if m == "cross20" else m): v
            for m, v in methods.items()
        }
        for q, methods in rankings.items()
    }
    if max(counts) <= 16:
        value = paired_differences(renamed, labels, modes)
        # The legacy alias is internal only.
        # Raw rankings keep their actual method names.
        value["baseline_alias"] = {"cross_evidence_pool60": "cross20"}
        value["bound_type"] = "exact shared-label-completion differences"
        return value
    scopes: dict = {}
    for scope, methods in metrics["scopes"].items():
        scoped_queries = [q for q in rankings if scope == "all" or modes[q] == scope]
        scopes[scope] = {}
        for method in methods:
            name = "cross_evidence_pool60" if method == "cross20" else method
            scopes[scope][name] = {}
            for base in ["global", "region_title", "cross20"]:
                target = "cross_evidence_pool60" if base == "cross20" else base
                fields = {}
                identical = all(
                    rankings[q][method] == rankings[q][base] for q in scoped_queries
                )
                for metric, bounds in methods[method]["ranges"].items():
                    baseline = methods[base]["ranges"][metric]
                    delta = (
                        [0.0, 0.0]
                        if identical
                        else (
                            [None, None]
                            if bounds[0] is None or baseline[0] is None
                            else [bounds[0] - baseline[1], bounds[1] - baseline[0]]
                        )
                    )
                    fields[metric] = dict(
                        delta_range=delta,
                        nonregression_proven_for_shared_unknown_labels=delta[0]
                        is not None
                        and delta[0] >= -1e-12,
                    )
                scopes[scope][name][target] = fields
    return dict(
        scopes=scopes,
        bound_type="conservative subtraction of exact individual macro intervals, not exact paired optima",
        baseline_alias={"cross_evidence_pool60": "cross20"},
        same_unknown_completion_for_candidate_and_baseline=True,
        unknown_imputation=False,
        formal_acceptance=False,
    )


def evaluate(cohort: str) -> None:
    """Apply the evaluate workflow step.

    Args:
        cohort: Cohort used by this operation.

    Notes:
        Writes derived artifacts through receipt.
    """
    check()
    assert load(ROOT / "evaluation_protocol.json")[
        "implementation_sha256"
    ] == sha256_file(Path(__file__))
    folder = ROOT / cohort
    verify_files(folder, load(folder / "prediction_receipt.json")["files"])
    rankings = load(folder / "predictions.json")
    labels = read_rows(folder / "judgments.ai.jsonl")
    modes = {q: q.split("_")[0] for q in rankings}
    metrics = evaluate_intervals(rankings, labels, modes)
    if (
        max(
            sum(r["grade"] is None for r in labels if r["query_id"] == q)
            for q in rankings
        )
        <= 16
    ):
        exhaustive = metric_ranges(rankings, labels, modes)
        for scope, methods in metrics["scopes"].items():
            for method, row in methods.items():
                for key, bounds in row["ranges"].items():
                    expected = exhaustive["scopes"][scope][method]["ranges"][key]
                    assert all(
                        a is b if a is None or b is None else abs(a - b) < 1e-12
                        for a, b in zip(bounds, expected)
                    )
    deltas = paired(rankings, labels, modes, metrics)
    lookup = {(r["query_id"], r["candidate_id"]): r["grade"] for r in labels}
    attribution = []
    for q, methods in rankings.items():
        pos = {cid for cid in next(iter(methods.values())) if lookup[q, cid] == 1}
        pool = load(folder / "runs" / q / "features.json")["pool"]
        for method, order in methods.items():
            attribution.append(
                dict(
                    query_id=q,
                    method=method,
                    mode=modes[q],
                    known_positive_total=len(pos),
                    known_oracle_at_10=min(10, len(pos)),
                    known_hits_at_10=len(pos & set(order[:10])),
                    known_oracle_in_pool20=min(10, len(pos & set(pool[:20]))),
                    known_oracle_in_pool60=min(10, len(pos & set(pool[:60]))),
                )
            )
    capacity = {
        scope: [
            sum(
                r["precision_ceiling_range"][i]
                for r in metrics["query_capacity"]
                if scope == "all" or modes[r["query_id"]] == scope
            )
            / sum(scope == "all" or v == scope for v in modes.values())
            for i in [0, 1]
        ]
        for scope in metrics["scopes"]
    }
    persist(folder / "internal_metrics.ai.development.json", metrics)
    persist(folder / "internal_paired_differences.json", deltas)
    persist(
        folder / "internal_error_attribution.json",
        dict(rows=attribution, uses_labels_after_predictions_only=True),
    )
    persist(
        folder / "internal_capacity.json",
        dict(precision_at_10_capacity=capacity, fixed_k=10, unknown_not_imputed=True),
    )
    receipt(
        folder,
        "evaluation_receipt.json",
        [
            folder / n
            for n in [
                "internal_metrics.ai.development.json",
                "internal_paired_differences.json",
                "internal_error_attribution.json",
                "internal_capacity.json",
            ]
        ],
        formal_metrics=None,
        review_origin="model",
        human_labels=0,
    )
    print(json.dumps(dict(evaluated=cohort, formal_acceptance=False)), flush=True)


def choose() -> None:
    """Apply the choose workflow step.

    Raises:
        FileExistsError: Never choose after fresh validation.

    Notes:
        Writes derived artifacts through receipt, write_json.
    """
    check()
    assert load(ROOT / "evaluation_protocol.json")[
        "implementation_sha256"
    ] == sha256_file(Path(__file__))
    if (ROOT / "selection.json").exists() or (ROOT / "fresh").exists():
        raise FileExistsError("Never choose after fresh validation")
    verify_files(ROOT / "tune", load(ROOT / "tune/evaluation_receipt.json")["files"])
    p = load(ROOT / "tune/internal_paired_differences.json")["scopes"]
    m = load(ROOT / "tune/internal_metrics.ai.development.json")["scopes"]
    gate = all(
        v["nonregression_proven_for_shared_unknown_labels"]
        for s in ["all", "text", "image"]
        for v in p[s]["cross60"]["cross_evidence_pool60"].values()
    )
    eligible = ["cross20"] + (["cross60"] if gate else [])
    selected = max(
        eligible,
        key=lambda n: (
            m["image"][n]["ranges"]["precision@10"][0],
            m["image"][n]["ranges"]["recall@10"][0] or 0,
            n == "cross20",
        ),
    )
    write_json(
        ROOT / "selection.json",
        dict(
            selected=selected,
            cross60_nonregression_gate=gate,
            eligible=eligible,
            policy=POLICY["selection"],
            uses_tune_only_for_ranking_selection=True,
            coverage_design_used_both_exposed_cohorts=True,
            fresh_validation_queries_inspected=0,
            fresh_validation_cannot_change_selection=True,
            default_promotion_allowed=False,
            formal_acceptance=False,
        ),
    )
    receipt(
        ROOT,
        "selection_receipt.json",
        [
            ROOT / "selection.json",
            ROOT / "evaluation_protocol.json",
            ROOT / "tune/evaluation_receipt.json",
        ],
        fresh_predictions=0,
    )
    print(json.dumps(dict(selected=selected, cross60_gate=gate)), flush=True)


def verify_isolation(final: bool) -> dict:
    """Verify isolation.

    Args:
        final: Final used by this operation.

    Returns:
        Mapping with fields: new_metadata, new_gallery, family_overlap_with_history,
        image_overlap_dhash4, fresh_query_families_disjoint, fresh_gallery_independent,
        old_consumed_blind_reused_as_unseen.
    """
    protocol = check()
    verify_files(ROOT, load(ROOT / "acquisition_receipt.json")["files"])
    prior = load(PARENT / "post_development_exclusion_snapshot.json")
    rows, meta = scan(
        PrimaryFormCoverageScanInputs(
            PROJECT_ROOT / protocol["source"],
            prior,
            POLICY["scan_after"],
            POLICY["scan_limit"],
            POLICY["seed"],
            FACETS,
            facet,
            POLICY["metadata_per_facet"],
        )
    )
    assert rows == read_rows(QUEUE)
    assert meta["eligible"] == load(ROOT / "metadata_receipt.json")["eligible"]
    seen = set(prior["family_keys"])
    for r in rows:
        assert sorted(product_family_keys(r)) == r["family_keys"] and not seen & set(
            r["family_keys"]
        )
        seen.update(r["family_keys"])
    signatures = list(prior["image_fingerprints"])
    selected = read_rows(ROOT / "selected.evaluator.jsonl")
    for r in selected:
        actual = fingerprints(Path(r["image_path"]))
        assert actual == r["fingerprints"] and not is_exposed(actual, signatures, 4)
        signatures.append(actual)
    assert len(selected) == len(load(ROOT / "additions/catalog.public.json")) == 72
    for r in read_rows(MEDIA):
        if r["status"] == "downloaded":
            assert fingerprints(Path(r["image_path"])) == r["fingerprints"]
    snap = snapshot()
    if final:
        r, _ = scan(
            PrimaryFormCoverageScanInputs(
                PROJECT_ROOT / protocol["source"],
                snap,
                POLICY["fresh_scan_after"],
                POLICY["fresh_scan_limit"],
                POLICY["fresh_seed"],
                [b[0] for b in BUCKETS],
                query_bucket,
                POLICY["fresh_metadata_per_bucket"],
            )
        )
        assert r == read_rows(FRESH_QUEUE)
        selected_fresh = read_rows(ROOT / "fresh/selected.evaluator.jsonl")
        assert len(selected_fresh) == 8
        signatures = list(snap["image_fingerprints"])
        for row in selected_fresh:
            assert not set(row["family_keys"]) & set(snap["family_keys"])
            actual = fingerprints(Path(row["image_path"]))
            assert actual == row["fingerprints"] and not is_exposed(
                actual, signatures, 4
            )
            signatures.append(actual)
        assert (ROOT / "selection_receipt.json").stat().st_mtime_ns < (
            ROOT / "fresh/protocol.json"
        ).stat().st_mtime_ns
        snapshot(True)
    return dict(
        new_metadata=len(rows),
        new_gallery=72,
        family_overlap_with_history=0,
        image_overlap_dhash4=0,
        fresh_query_families_disjoint=final,
        fresh_gallery_independent=False,
        old_consumed_blind_reused_as_unseen=False,
    )


def replay(cohort: str) -> dict:
    """Return replay for this workflow.

    Args:
        cohort: Cohort used by this operation.

    Returns:
        Mapping with fields: queries, gallery, full_rankings, actual_new_full_pairs,
        reused_exact_raw_pairs, all_orders_replayed.
    """
    folder = ROOT / cohort
    is_fresh = cohort == "fresh"
    if is_fresh:
        verify_files(folder, load(folder / "label_receipt.json")["files"])
        verify_files(folder, load(folder / "query_encoding_receipt.json")["files"])
        index_root = ROOT / "tune/index"
        bank_root = folder / "queries"
    else:
        old = PARENT / ("tune" if cohort == "tune" else "validation")
        index_root = folder / "index"
        bank_root = old / "queries"
        assert load(folder / "queries.public.json") == load(old / "queries.public.json")
        assert load(folder / "catalog.public.json")[:96] == load(
            old / "catalog.public.json"
        )
        assert read_rows(folder / "judgments.ai.jsonl")[:1536] == read_rows(
            old / "judgments.ai.jsonl"
        )
    for name in ["prediction_receipt.json", "evaluation_receipt.json"]:
        verify_files(folder, load(folder / name)["files"])
    index, titles = load_expanded_index(index_root)
    with np.load(index_root / "vectors.npz", allow_pickle=False) as bank:
        profiles = bank["profiles"].copy()
    texts = load(index_root / "profile_texts.json")
    colors = np.load(index_root / "colors.npy", allow_pickle=False)
    ids = [r["id"] for r in index.records]
    assert len(ids) == 168
    rankings = load(folder / "predictions.json")
    new_pairs, reused_pairs = 0, 0
    for q in load(folder / "queries.public.json"):
        target = folder / "runs" / q["id"]
        saved = load(target / "features.json")
        with np.load(bank_root / q["id"] / "vectors.npz", allow_pickle=False) as bank:
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
        assert saved["ids"] == ids and np.allclose(
            saved["features"], feats, atol=1e-14, rtol=0
        )
        offsets = None
        if image:
            description = load(bank_root / q["id"] / "description.json")
            assert description["gallery_or_judgments_supplied"] is False
            ch = visual_channels(
                QueryFusionVisualChannelsInputs(
                    index,
                    titles,
                    profiles,
                    feats,
                    views,
                    caption,
                    description["text"],
                    np.arange(len(ids), dtype=int),
                )
            )
            base, pool = np.array(ch["fused_scores"]), ch["pool"]
            assert saved["description"] == description["text"]
        else:
            base, recall_orders = recall_channels(ids, feats)
            offsets = np.array(
                [
                    public_evidence(q["text"], d["title"], p)["offset"]
                    for d, p in zip(index.records, texts)
                ]
            )
            recall_orders.append(ranked_ids(ids, normalized_ranks(ids, base) + offsets))
            pool = balanced_pool(recall_orders, 60)
        methods = list(rankings[q["id"]])
        pool = pool[: 60 if "cross60" in methods else 20]
        assert saved["pool"] == pool and np.allclose(
            saved["base"], base, atol=1e-14, rtol=0
        )
        raw = load(target / "cross.json")
        assert (
            raw["pool"] == pool
            and len(raw["logits"]) == len(pool)
            and np.isfinite(raw["logits"]).all()
        )
        assert set(raw["new_pair_ids"]) | set(raw["reused_pair_ids"]) == set(pool)
        assert not set(raw["new_pair_ids"]) & set(raw["reused_pair_ids"])
        assert [t["document_id"] for t in raw["timings"]] == raw["new_pair_ids"]
        if raw["cache_file"]:
            path = PROJECT_ROOT / raw["cache_file"]
            assert sha256_file(path) == raw["cache_sha256"]
            cache = load(path)
            lookup = dict(zip(cache["pool"], cache["logits"]))
            assert all(
                raw["logits"][pool.index(cid)] == lookup[cid]
                for cid in raw["reused_pair_ids"]
            )
        else:
            assert not raw["reused_pair_ids"]
        current = {
            "global": ranked_ids(ids, feats[:, 0]),
            "region_title": ranked_ids(ids, 0.8 * feats[:, 1] + 0.2 * feats[:, 3]),
        }
        for method in [m for m in methods if m.startswith("cross")]:
            n = 20 if method == "cross20" else 60
            current[method] = rerank_complete(
                ids, base, pool[:n], np.array(raw["logits"][:n]), offsets
            )
        assert rankings[q["id"]] == current
        new_pairs += len(raw["new_pair_ids"])
        reused_pairs += len(raw["reused_pair_ids"])
    runtime = load(folder / "cross_runtime.json")
    assert len(runtime["timings"]) == new_pairs
    config = load(PROJECT_ROOT / "configs/reranker_model.json")
    for name, value in config.items():
        if name != "instruction":
            assert runtime["model"][name] == value
    metrics = evaluate_intervals(
        rankings,
        read_rows(folder / "judgments.ai.jsonl"),
        {q: q.split("_")[0] for q in rankings},
    )
    assert metrics == load(folder / "internal_metrics.ai.development.json")
    return dict(
        queries=len(rankings),
        gallery=168,
        full_rankings=sum(len(v) for v in rankings.values()),
        actual_new_full_pairs=new_pairs,
        reused_exact_raw_pairs=reused_pairs,
        all_orders_replayed=True,
    )


def verify() -> None:
    """Verify the recorded artifacts and their consistency."""
    from verify_rank_stability import verify_history

    check()
    assert load(ROOT / "evaluation_protocol.json")[
        "implementation_sha256"
    ] == sha256_file(Path(__file__))
    isolation = verify_isolation(True)
    verify_files(ROOT, load(ROOT / "encoding_receipt.json")["files"])
    verify_files(ROOT, load(ROOT / "label_receipt.json")["files"])
    verify_files(ROOT, load(ROOT / "selection_receipt.json")["files"])
    summaries = {c: replay(c) for c in ["tune", "regression", "fresh"]}
    history = verify_history()
    for name in [
        "module2_rank_stability",
        "module2_independent_business_blind",
        "module2_structure_development",
    ]:
        stage = PROJECT_ROOT / "artifacts" / name
        r = load(stage / "completion_receipt.json")
        verify_files(stage, r["files"])
        verify_files(PROJECT_ROOT, r["project_files"])
    m = load(ROOT / "fresh/internal_metrics.ai.development.json")["scopes"]
    selection = load(ROOT / "selection.json")["selected"]
    fresh = m["image"][selection]
    decision = dict(
        selected=selection,
        fresh_query_development_validation_completed=True,
        fresh_image_precision_threshold_met=fresh["precision_60_lower_bound_met"],
        fresh_image_recall_preference_met=fresh["recall_85_lower_bound_met"],
        independent_gallery_validation=False,
        formal_acceptance=False,
        formal_metrics=None,
        goal_met=False,
        default_strategy_changed=False,
        new_independent_AI_blind_predictions=0,
        next_independent_AI_blind_started=False,
        reason="This is adaptive coverage development plus new query-family validation, not an independent-gallery acceptance result. All six-metric gates must be reconciled before proposing a new frozen unseen acceptance cohort.",
    )
    persist(ROOT / "decision.json", decision)
    runtime = load(ROOT / "additions/index/runtime.json")
    assert runtime["device"].startswith("cuda") and runtime["dtype"] == "torch.bfloat16"
    value = dict(
        isolation=isolation,
        cohorts=summaries,
        actual_gallery_embedding_calls=len(runtime["calls"]),
        actual_fresh_query_embedding_calls=len(
            load(ROOT / "fresh/query_runtime.json")["calls"]
        ),
        previous_sealed_stages_verified=len(history) + 3,
        labels_are_AI_not_human=True,
        unknown_intervals_are_not_confidence_intervals=True,
        formal_metrics=None,
    )
    persist(ROOT / "verification.json", value)
    if (ROOT / "completion_receipt.json").exists():
        r = load(ROOT / "completion_receipt.json")
        verify_files(ROOT, r["files"])
        verify_files(PROJECT_ROOT, r["project_files"])
    print(
        json.dumps(dict(replay_passed=True, decision=decision, cohorts=summaries)),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["freeze_evaluation", "evaluate", "choose", "verify"]
    )
    parser.add_argument(
        "--cohort", choices=["tune", "regression", "fresh"], default="tune"
    )
    args = parser.parse_args()
    evaluate(args.cohort) if args.action == "evaluate" else globals()[args.action]()
