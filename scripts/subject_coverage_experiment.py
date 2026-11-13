"""Identical expanded-gallery comparison; runtime never reads relevance labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import run  # noqa: F401
from compare_fgclip2_retrieval import read_array, relative
from evaluate_primary_form_coverage import paired
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.contracts import FeatureBanks
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.development_expansion import load_expanded_index
from multimodal_ecommerce.fgclip_retrieval import (
    FgclipRetrievalAuxiliaryScoresInputs,
    FgclipRetrievalFusePoolInputs,
    auxiliary_scores,
    fuse_pool,
)
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
from multimodal_ecommerce.regional_retrieval import RegionalIndex
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from multimodal_ecommerce.semantic_ranker import (
    SemanticRankerFeatureMatrixInputs,
    feature_matrix,
    ranked_ids,
)
from multimodal_ecommerce.subject_roi import (
    SubjectRoiImageOnlyOrderInputs,
    image_only_order,
)
from multimodal_ecommerce.visual_query import DescribedImageReranker
from primary_form_coverage import receipt
from subject_coverage_acquisition import PARENT, ROOT, check
from subject_roi_experiment import COVERAGE, load, once

COHORTS = ("tune", "regression", "fresh")
METHODS = (
    "global",
    "region_title",
    "cross20",
    "fgclip2_local_fusion20",
    "image_fusion20",
    "image_roi_fusion20",
)
STRUCTURE = PROJECT_ROOT / "artifacts/module2_structure_development"


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        FileExistsError: Preserve comparison protocol.

    Notes:
        Writes derived artifacts through once.
    """
    check()
    if (ROOT / "comparison_protocol.json").exists():
        raise FileExistsError("Preserve comparison protocol")
    assert not any((ROOT / c / "predictions.json").exists() for c in COHORTS)
    paths = [
        Path(__file__),
        ROOT / "encoding_protocol.json",
        ROOT / "label_receipt.json",
        ROOT / "auxiliary_protocol.json",
        ROOT / "inputs.json",
        ROOT / "encoding_receipt.json",
        PARENT / "completion_receipt.json",
        COVERAGE / "completion_receipt.json",
        PROJECT_ROOT / "configs/reranker_model.json",
        PROJECT_ROOT / "scripts/evaluate_primary_form_coverage.py",
        PROJECT_ROOT / "scripts/report_focus_coverage.py",
        *sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")),
    ]
    once(
        ROOT / "comparison_protocol.json",
        dict(
            files={relative(p): sha256_file(p) for p in paths},
            role="exposed_AI_development_only",
            methods=METHODS,
            primary_method="image_roi_fusion20",
            fixed_k=10,
            precision_required=0.6,
            recall_preferred=0.85,
            full_official_models=True,
            same_gallery_queries_labels=True,
            queries=40,
            gallery_per_cohort=193,
            AI_relations=7720,
            original_labels_unchanged=True,
            human_labels=0,
            formal_metrics=None,
            comparison="Recompute original Qwen recall pool over all 193 items; compute missing real reranker pairs. All fusion methods use the identical original twenty-candidate set and retain its full tail.",
            selection="No tuning after expanded results; compare all six paired metrics for all/text/image. No default promotion without later independent validation.",
            unknowns="Exact label-completion metric bounds; shared-completion differences up to sixteen unknowns, otherwise documented conservative bounds; AI judgment error and sampling uncertainty are not covered.",
            new_independent_blind_queries=0,
            default_promotion_allowed=False,
        ),
    )
    print(dict(protocol_frozen=True, methods=len(METHODS)), flush=True)


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    check()
    verify_files(PROJECT_ROOT, load(ROOT / "comparison_protocol.json")["files"])
    verify_files(ROOT, load(ROOT / "encoding_receipt.json")["files"])


def bank_path(cohort: str, qid: str) -> Path:
    """Return bank path for this workflow.

    Args:
        cohort: Cohort used by this operation.
        qid: Qid used by this operation.

    Returns:
        Numeric result computed from the supplied values.
    """
    if cohort == "fresh":
        return COVERAGE / "fresh/queries" / qid
    return STRUCTURE / ("tune" if cohort == "tune" else "validation") / "queries" / qid


def feature_inputs(
    cohort: str,
) -> tuple[RegionalIndex, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Return feature inputs for this workflow.

    Args:
        cohort: Cohort used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    folder = ROOT / ("tune" if cohort == "fresh" else cohort) / "index"
    index, titles = load_expanded_index(folder)
    with np.load(folder / "vectors.npz", allow_pickle=False) as bank:
        profiles = bank["profiles"].copy()
    return (
        index,
        titles,
        profiles,
        load(folder / "profile_texts.json"),
        np.load(folder / "colors.npy", allow_pickle=False),
    )


def calculate_features(cohort: str, query: dict, banks: FeatureBanks) -> dict:
    """Return calculate features for this workflow.

    Args:
        cohort: Cohort used by this operation.
        query: Query record or representation used by this operation.
        banks: Banks used by this operation.

    Returns:
        Mapping with fields: ids, base, pool, features, description, offsets.
    """
    index, titles, profiles, texts, colors = banks
    ids = [d["id"] for d in index.records]
    with np.load(
        bank_path(cohort, query["id"]) / "vectors.npz", allow_pickle=False
    ) as bank:
        views, caption = bank["views"].copy(), bank["caption"].copy()
    image = Path(query["image_path"]) if query["image_path"] else None
    features = feature_matrix(
        SemanticRankerFeatureMatrixInputs(
            index,
            titles,
            profiles,
            texts,
            views[0],
            text=query["text"],
            image=image,
            color_bank=colors,
        )
    )
    description, offsets = "", None
    if image:
        description = load(bank_path(cohort, query["id"]) / "description.json")["text"]
        fusion = visual_channels(
            QueryFusionVisualChannelsInputs(
                index,
                titles,
                profiles,
                features,
                views,
                caption,
                description,
                np.arange(len(ids), dtype=int),
            )
        )
        base, pool = np.array(fusion["fused_scores"]), fusion["pool"]
    else:
        base, channels = recall_channels(ids, features)
        offsets = np.array(
            [
                public_evidence(query["text"], d["title"], p)["offset"]
                for d, p in zip(index.records, texts)
            ]
        )
        channels.append(ranked_ids(ids, normalized_ranks(ids, base) + offsets))
        pool = balanced_pool(channels, 60)
    return dict(
        ids=ids,
        base=base.tolist(),
        pool=pool[:20],
        features=features.tolist(),
        description=description,
        offsets=None if offsets is None else offsets.tolist(),
    )


def pair_identity(query: dict, description: str, doc: dict) -> dict:
    """Return pair identity for this workflow.

    Args:
        query: Query record or representation used by this operation.
        description: Description used by this operation.
        doc: Doc used by this operation.

    Returns:
        Mapping with fields: sha256, inputs.
    """
    identity = dict(
        query_text=query["text"],
        query_image_sha256=(
            sha256_file(Path(query["image_path"])) if query["image_path"] else None
        ),
        query_description=description,
        candidate_id=doc["id"],
        candidate_title=doc["title"],
        candidate_image_sha256=sha256_file(Path(doc["image_path"])),
        reranker_config_sha256=sha256_file(
            PROJECT_ROOT / "configs/reranker_model.json"
        ),
        instruction_source_sha256=sha256_file(
            PROJECT_ROOT / "src/multimodal_ecommerce/visual_query.py"
        ),
    )
    key = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return dict(sha256=key, inputs=identity)


def infer() -> None:
    """Run model inference using the configured inputs.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    reranker = None
    total_new, total_reuse = 0, 0
    try:
        for cohort in COHORTS:
            banks = feature_inputs(cohort)
            by_id = {d["id"]: d for d in banks[0].records}
            old_docs = {
                d["id"]: d
                for d in load(
                    COVERAGE
                    / ("tune" if cohort == "fresh" else cohort)
                    / "index/records.json"
                )
            }
            old_queries = load(COVERAGE / cohort / "queries.public.json")
            for query in load(ROOT / cohort / "queries.public.json"):
                assert query in old_queries
                target = ROOT / cohort / "runs" / query["id"]
                features = calculate_features(cohort, query, banks)
                once(target / "features.json", features)
                old_features = load(
                    COVERAGE / cohort / "runs" / query["id"] / "features.json"
                )
                assert features["description"] == old_features["description"]
                pool = features["pool"]
                old_path = COVERAGE / cohort / "runs" / query["id"] / "cross.json"
                old = load(old_path)
                cache = dict(zip(old["pool"], old["logits"]))
                reused = [cid for cid in pool if cid in cache]
                identities = {}
                for cid in pool:
                    identities[cid] = pair_identity(
                        query, features["description"], by_id[cid]
                    )
                    if cid in cache:
                        assert identities[cid] == pair_identity(
                            query, old_features["description"], old_docs[cid]
                        )
                missing = [cid for cid in pool if cid not in cache]
                raw_path = target / "cross.json"
                if raw_path.exists():
                    raw = load(raw_path)
                    assert raw["pool"] == pool and raw["pair_identities"] == identities
                else:
                    if missing and reranker is None:
                        reranker = DescribedImageReranker()
                    pairs = [
                        dict(
                            id=cid,
                            image_path=by_id[cid]["image_path"],
                            text=by_id[cid]["title"],
                        )
                        for cid in missing
                    ]
                    before = len(reranker.adapter.timings) if reranker else 0
                    image = Path(query["image_path"]) if query["image_path"] else None
                    if missing:
                        assert reranker is not None
                        values = (
                            reranker.score_described(
                                pairs,
                                query_image=image,
                                description=features["description"],
                            )
                            if image and features["description"]
                            else reranker.score(
                                pairs, query_image=image, query_text=query["text"]
                            )
                        )
                    else:
                        values = np.array([])
                    computed = dict(zip(missing, values.tolist()))
                    once(
                        raw_path,
                        dict(
                            pool=pool,
                            logits=[
                                cache[cid] if cid in cache else computed[cid]
                                for cid in pool
                            ],
                            new_pair_ids=missing,
                            reused_pair_ids=reused,
                            pair_identities=identities,
                            cache_file=relative(old_path),
                            cache_sha256=sha256_file(old_path),
                            timings=(
                                reranker.adapter.timings[before:] if reranker else []
                            ),
                        ),
                    )
                total_new += len(missing)
                total_reuse += len(reused)
                print(
                    dict(
                        cohort=cohort,
                        query=query["id"],
                        new_pairs=len(missing),
                        reused=len(reused),
                    ),
                    flush=True,
                )
        runtime = reranker.diagnostics() if reranker else dict(timings=[])
    finally:
        if reranker:
            reranker.close()
    once(ROOT / "reranker_runtime.json", runtime)
    receipt(
        ROOT,
        "reranker_receipt.json",
        [
            ROOT / "reranker_runtime.json",
            *[p for c in COHORTS for p in sorted((ROOT / c / "runs").rglob("*.json"))],
        ],
        new_full_reranker_pairs=total_new,
        exact_reused_pairs=total_reuse,
        total_pairs=total_new + total_reuse,
        qrels_used_in_runtime=False,
    )


def calculate_orders(cohort: str) -> tuple[dict, dict]:
    """Return calculate orders for this workflow.

    Args:
        cohort: Cohort used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    inputs, paths, fmap = (
        load(ROOT / "inputs.json"),
        load(ROOT / "feature_paths.json"),
        load(ROOT / "roi_feature_index.json"),
    )
    data = inputs["cohorts"][cohort]
    ids = [d["id"] for d in data["gallery"]]
    titles = np.stack(
        [
            read_array(PROJECT_ROOT / paths["texts"][d["title_key"]], "vector")
            for d in data["gallery"]
        ]
    )
    full_images = np.stack(
        [
            read_array(PROJECT_ROOT / paths["images"][d["image_key"]], "global_vector")
            for d in data["gallery"]
        ]
    )
    full_dense = [
        read_array(PROJECT_ROOT / paths["images"][d["image_key"]], "dense")
        for d in data["gallery"]
    ]
    roi_images = np.stack(
        [
            read_array(PROJECT_ROOT / fmap[d["roi_request"]]["file"], "global_vector")
            for d in data["gallery"]
        ]
    )
    roi_dense = [
        read_array(PROJECT_ROOT / fmap[d["roi_request"]]["file"], "dense")
        for d in data["gallery"]
    ]
    predictions, details = {}, {}
    for query in data["queries"]:
        qid = query["id"]
        features, raw = load(ROOT / cohort / "runs" / qid / "features.json"), load(
            ROOT / cohort / "runs" / qid / "cross.json"
        )
        assert features["ids"] == ids and len(raw["pool"]) == 20
        offsets = (
            np.array(features["offsets"]) if features["offsets"] is not None else None
        )
        base, logits = np.array(features["base"]), np.array(raw["logits"])
        matrix = np.array(features["features"])
        common = dict(
            ids=ids, base=base, pool=raw["pool"], cross_logits=logits, offsets=offsets
        )
        original = rerank_complete(ids, base, raw["pool"], logits, offsets)
        txt = (
            read_array(PROJECT_ROOT / paths["texts"][query["text_key"]], "vector")
            if query["text_key"]
            else None
        )
        box = (
            read_array(PROJECT_ROOT / paths["box_texts"][query["text_key"]], "vector")
            if query["text_key"]
            else None
        )
        image = (
            read_array(
                PROJECT_ROOT / paths["images"][query["image_key"]], "global_vector"
            )
            if query["image_key"]
            else None
        )
        whole = auxiliary_scores(
            FgclipRetrievalAuxiliaryScoresInputs(
                ids,
                full_images,
                titles,
                full_dense,
                query_image=image,
                query_text=txt,
                query_box=box,
            )
        )
        fg_order = fuse_pool(
            FgclipRetrievalFusePoolInputs(
                auxiliary=np.array(whole["fused_scores"]), **common
            )
        )
        orders = dict(
            global_=ranked_ids(ids, matrix[:, 0]),
            region_title=ranked_ids(ids, 0.8 * matrix[:, 1] + 0.2 * matrix[:, 3]),
            cross20=original,
            fgclip2_local_fusion20=fg_order,
        )
        orders["global"] = orders.pop("global_")
        roi = None
        if query["image_key"]:
            roi = auxiliary_scores(
                FgclipRetrievalAuxiliaryScoresInputs(
                    ids,
                    roi_images,
                    titles,
                    roi_dense,
                    query_image=read_array(
                        PROJECT_ROOT / fmap[query["roi_request"]]["file"],
                        "global_vector",
                    ),
                    query_text=txt,
                    query_box=box,
                )
            )
            kwargs = dict(
                is_image=True,
                ids=ids,
                base=base,
                pool=raw["pool"],
                logits=logits,
                whole_scores=np.array(whole["fused_scores"]),
                offsets=offsets,
            )
            orders["image_fusion20"] = image_only_order(
                SubjectRoiImageOnlyOrderInputs(original, roi_scores=None, **kwargs)
            )
            orders["image_roi_fusion20"] = image_only_order(
                SubjectRoiImageOnlyOrderInputs(
                    original, roi_scores=np.array(roi["fused_scores"]), **kwargs
                )
            )
        else:
            orders.update(
                image_fusion20=list(original), image_roi_fusion20=list(original)
            )
        assert set(orders) == set(METHODS)
        assert all(len(o) == len(ids) and set(o) == set(ids) for o in orders.values())
        for method in METHODS[3:]:
            assert (
                set(orders[method][:20]) == set(original[:20])
                and orders[method][20:] == original[20:]
            )
        predictions[qid] = orders
        details[qid] = dict(
            ids=ids,
            whole_scores=whole,
            roi_scores=roi,
            route=(
                "image_only_auxiliary" if image is not None else "original_text_exact"
            ),
        )
    return predictions, details


def predict() -> None:
    """Apply the predict workflow step.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    verify_files(ROOT, load(ROOT / "fg_encoding_receipt.json")["files"])
    verify_files(ROOT, load(ROOT / "reranker_receipt.json")["files"])
    for cohort in COHORTS:
        ranks, scores = calculate_orders(cohort)
        once(ROOT / cohort / "predictions.json", ranks)
        once(ROOT / cohort / "runtime_scores.json", scores)
        print(
            dict(predicted=cohort, queries=len(ranks), methods=len(METHODS)), flush=True
        )
    receipt(
        ROOT,
        "prediction_receipt.json",
        [
            ROOT / c / n
            for c in COHORTS
            for n in ["predictions.json", "runtime_scores.json"]
        ],
        full_rankings=240,
        full_gallery=193,
        qrels_used_in_runtime=False,
    )


def evaluate() -> None:
    """Apply the evaluate workflow step.

    Notes:
        Writes derived artifacts through once.
    """
    guard()
    verify_files(ROOT, load(ROOT / "prediction_receipt.json")["files"])
    for cohort in COHORTS:
        ranks = load(ROOT / cohort / "predictions.json")
        labels = read_rows(ROOT / cohort / "judgments.ai.jsonl")
        modes = {qid: qid.split("_")[0] for qid in ranks}
        metrics = evaluate_intervals(ranks, labels, modes)
        once(ROOT / cohort / "internal_metrics.ai.development.json", metrics)
        once(
            ROOT / cohort / "internal_paired_differences.json",
            paired(ranks, labels, modes, metrics),
        )
        print(dict(evaluated=cohort, relations=len(labels)), flush=True)


def verify() -> None:
    """Verify the recorded artifacts and their consistency.

    Notes:
        Writes derived artifacts through once.
    """
    guard()
    for name in [
        "label_receipt.json",
        "detection_receipt.json",
        "fg_encoding_receipt.json",
        "reranker_receipt.json",
        "prediction_receipt.json",
    ]:
        verify_files(ROOT, load(ROOT / name)["files"])
    old_receipt = load(PARENT / "completion_receipt.json")
    verify_files(PARENT, old_receipt["files"])
    for cohort in COHORTS:
        ranks, details = calculate_orders(cohort)
        assert ranks == load(ROOT / cohort / "predictions.json") and details == load(
            ROOT / cohort / "runtime_scores.json"
        )
        banks = feature_inputs(cohort)
        for query in load(ROOT / cohort / "queries.public.json"):
            assert calculate_features(cohort, query, banks) == load(
                ROOT / cohort / "runs" / query["id"] / "features.json"
            )
        labels = read_rows(ROOT / cohort / "judgments.ai.jsonl")
        original = read_rows(COVERAGE / cohort / "judgments.ai.jsonl")
        assert labels[: len(original)] == original
        assert len(labels) == 193 * len(ranks)
        modes = {qid: qid.split("_")[0] for qid in ranks}
        metrics = evaluate_intervals(ranks, labels, modes)
        assert metrics == load(ROOT / cohort / "internal_metrics.ai.development.json")
        assert paired(ranks, labels, modes, metrics) == load(
            ROOT / cohort / "internal_paired_differences.json"
        )
        for qid, mode in modes.items():
            if mode == "text":
                assert (
                    ranks[qid]["image_fusion20"]
                    == ranks[qid]["image_roi_fusion20"]
                    == ranks[qid]["cross20"]
                )
    once(
        ROOT / "verification.json",
        dict(
            passed=True,
            same_gallery_and_queries=True,
            complete_full_rankings_replayed=240,
            Qwen_recall_features_replayed=True,
            original_6720_labels_unchanged=True,
            AI_labels=7720,
            human_labels=0,
            original_text_route_exact=True,
            metrics_and_shared_label_differences_replayed=True,
            formal_metrics=None,
            independent_acceptance_queries=0,
            default_changed=False,
        ),
    )
    print(dict(verified=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["prepare", "infer", "predict", "evaluate", "verify"]
    )
    globals()[parser.parse_args().action]()
