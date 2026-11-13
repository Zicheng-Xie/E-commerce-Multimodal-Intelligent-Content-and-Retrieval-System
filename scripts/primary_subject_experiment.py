"""Identical expanded-gallery comparison; runtime never reads relevance labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import run  # noqa: F401
from acquire_primary_subjects import PARENT, ROOT, check
from compare_fgclip2_retrieval import read_array, relative
from encode_primary_subjects import gpu_guard
from encode_primary_subjects import guard as feature_guard
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
from primary_subject_ranker import POLICY as GUARD_POLICY
from primary_subject_ranker import PrimarySubjectRankerGuardedOrderInputs, guarded_order
from subject_roi_experiment import COVERAGE, load, once

COHORTS = ("tune", "regression", "fresh")
METHODS = (
    "global",
    "region_title",
    "cross20",
    "fgclip2_local_fusion20",
    "image_fusion20",
    "image_roi_fusion20",
    "subject_caption20",
    "subject_guard20",
)
STRUCTURE = PROJECT_ROOT / "artifacts/module2_structure_development"


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        FileExistsError: Preserve frozen comparison.

    Notes:
        Writes derived artifacts through once.
    """
    check()
    feature_guard()
    if (ROOT / "comparison_protocol.json").exists():
        raise FileExistsError("Preserve frozen comparison")
    assert load(ROOT / "method_design.json")["public_conflict_policy"] == GUARD_POLICY
    paths = [
        Path(__file__),
        ROOT / "method_design.json",
        PROJECT_ROOT / "scripts/primary_subject_ranker.py",
        PROJECT_ROOT / "tests/test_primary_subject_ranker.py",
        ROOT / "feature_protocol.json",
        ROOT / "label_receipt.json",
        ROOT / "inputs.json",
        ROOT / "encoding_receipt.json",
        ROOT / "query_caption_protocol.json",
        ROOT / "query_caption_receipt.json",
        PARENT / "completion_receipt.json",
        PROJECT_ROOT / "configs/reranker_model.json",
        *sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")),
    ]
    size = len(load(ROOT / "tune/catalog.public.json"))
    once(
        ROOT / "comparison_protocol.json",
        dict(
            files={relative(p): sha256_file(p) for p in paths},
            role="query_aware_exposed_AI_development_only",
            methods=METHODS,
            primary_method="subject_guard20",
            ablation="subject_caption20: same new query caption, pool, raw CE logits and tail; remove public conflict penalties",
            fixed_k=10,
            precision_required=0.6,
            recall_preferred=0.85,
            guard_policy=GUARD_POLICY,
            full_official_models=True,
            queries=40,
            gallery_per_cohort=size,
            AI_relations=40 * size,
            original_AI_relations_unchanged=7720,
            human_labels=0,
            formal_metrics=None,
            query_caption_uses_candidates_or_labels=False,
            text_route="return this expanded-gallery original cross20 exactly",
            selection="No after-result tuning. All methods/scopes/six metrics retained. Require Precision lower bound >=.60; prefer Recall lower bound >=.85; require paired nonregression before candidate promotion. Independent acceptance still requires another new gallery and queries.",
            default_promotion_allowed=False,
            independent_blind_queries=0,
            unknown_bounds="Exact per-method label-completion intervals; shared-label exact differences up to sixteen unknowns, otherwise documented conservative subtraction. Not confidence intervals or AI-error bounds.",
        ),
    )
    print(dict(comparison_frozen=True, gallery=size, methods=len(METHODS)), flush=True)


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    check()
    verify_files(PROJECT_ROOT, load(ROOT / "comparison_protocol.json")["files"])
    verify_files(ROOT, load(ROOT / "encoding_receipt.json")["files"])
    feature_guard()


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


def calculate_features(
    cohort: str, query: dict, banks: FeatureBanks, *, new_caption: bool = False
) -> dict:
    """Return calculate features for this workflow.

    Args:
        cohort: Cohort used by this operation.
        query: Query record or representation used by this operation.
        banks: Banks used by this operation.
        new_caption: New caption used by this operation.

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
        if new_caption:
            description = load(
                ROOT / "query_captions" / cohort / (query["id"] + ".json")
            )["text"]
            with np.load(
                ROOT / "new_query_features" / cohort / (query["id"] + ".npz"),
                allow_pickle=False,
            ) as bank:
                caption = bank["caption"].copy()
        else:
            description = load(bank_path(cohort, query["id"]) / "description.json")[
                "text"
            ]
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
    engine = None
    new_count, reused_count = 0, 0
    try:
        for cohort in COHORTS:
            banks = feature_inputs(cohort)
            documents = {d["id"]: d for d in banks[0].records}
            for query in load(ROOT / cohort / "queries.public.json"):
                qid = query["id"]
                target = ROOT / cohort / "runs" / qid
                original = calculate_features(cohort, query, banks)
                once(target / "features.json", original)
                old_path = PARENT / cohort / "runs" / qid / "cross.json"
                old = load(old_path)
                parent_description = load(
                    PARENT / cohort / "runs" / qid / "features.json"
                )["description"]
                assert original["description"] == parent_description
                stages = [("cross.json", original, old, old_path)]
                if query["image_path"]:
                    updated = calculate_features(cohort, query, banks, new_caption=True)
                    once(target / "caption_features.json", updated)
                    stages.append(
                        ("caption_cross.json", updated, None, target / "cross.json")
                    )
                for filename, feature, prior, prior_path in stages:
                    if prior is None:
                        prior = load(prior_path)
                    pool = feature["pool"]
                    identities = {
                        cid: pair_identity(
                            query, feature["description"], documents[cid]
                        )
                        for cid in pool
                    }
                    cache = {
                        cid: score
                        for cid, score in zip(prior["pool"], prior["logits"])
                        if cid in identities
                        and identities[cid] == prior["pair_identities"].get(cid)
                    }
                    missing = [cid for cid in pool if cid not in cache]
                    path = target / filename
                    if path.exists():
                        raw = load(path)
                        assert (
                            raw["pool"] == pool and raw["pair_identities"] == identities
                        )
                    else:
                        if missing and engine is None:
                            gpu_guard()
                            engine = DescribedImageReranker()
                        pairs = [
                            dict(
                                id=cid,
                                image_path=documents[cid]["image_path"],
                                text=documents[cid]["title"],
                            )
                            for cid in missing
                        ]
                        before = len(engine.adapter.timings) if engine else 0
                        image = (
                            Path(query["image_path"]) if query["image_path"] else None
                        )
                        if pairs:
                            assert engine is not None
                            values = (
                                engine.score_described(
                                    pairs,
                                    query_image=image,
                                    description=feature["description"],
                                )
                                if image and feature["description"]
                                else engine.score(
                                    pairs, query_image=image, query_text=query["text"]
                                )
                            )
                        else:
                            values = np.array([])
                        computed = dict(zip(missing, values.tolist()))
                        once(
                            path,
                            dict(
                                pool=pool,
                                logits=[
                                    cache[cid] if cid in cache else computed[cid]
                                    for cid in pool
                                ],
                                pair_identities=identities,
                                new_pair_ids=missing,
                                reused_pair_ids=[cid for cid in pool if cid in cache],
                                cache_file=relative(prior_path),
                                cache_sha256=sha256_file(prior_path),
                                timings=(
                                    engine.adapter.timings[before:] if engine else []
                                ),
                            ),
                        )
                        raw = load(path)
                    new_count += len(raw["new_pair_ids"])
                    reused_count += len(raw["reused_pair_ids"])
                    print(
                        dict(
                            cohort=cohort,
                            query=qid,
                            role=filename,
                            new=len(raw["new_pair_ids"]),
                            reused=len(raw["reused_pair_ids"]),
                        ),
                        flush=True,
                    )
        runtime = engine.diagnostics() if engine else dict(timings=[])
    finally:
        if engine:
            engine.close()
    raw_paths = [p for c in COHORTS for p in (ROOT / c / "runs").glob("*/*cross.json")]
    assert len(raw_paths) == 64 and new_count + reused_count == 1280
    for path in raw_paths:
        row = load(path)
        assert len(row["pool"]) == len(set(row["pool"])) == len(row["logits"]) == 20
        assert not set(row["new_pair_ids"]) & set(row["reused_pair_ids"])
        assert set(row["new_pair_ids"]) | set(row["reused_pair_ids"]) == set(
            row["pool"]
        )
        assert len(row["timings"]) == len(row["new_pair_ids"])
    once(ROOT / "reranker_runtime.json", runtime)
    receipt(
        ROOT,
        "reranker_receipt.json",
        [
            ROOT / "reranker_runtime.json",
            *[p for c in COHORTS for p in sorted((ROOT / c / "runs").rglob("*.json"))],
        ],
        new_full_reranker_pairs=new_count,
        exact_reused_pairs=reused_count,
        total_pairs=new_count + reused_count,
        expected_original_plus_caption_pairs=1280,
        caption_and_guard_share_raw_pairs=True,
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
        subject = None
        if query["image_key"]:
            updated = load(ROOT / cohort / "runs" / qid / "caption_features.json")
            extra = load(ROOT / cohort / "runs" / qid / "caption_cross.json")
            updated_offsets = (
                np.array(updated["offsets"]) if updated["offsets"] is not None else None
            )
            orders["subject_caption20"] = rerank_complete(
                ids,
                np.array(updated["base"]),
                extra["pool"],
                np.array(extra["logits"]),
                updated_offsets,
            )
            public_profiles = {
                r["id"]: r for r in read_rows(ROOT / cohort / "profiles.public.jsonl")
            }
            qvalue = load(ROOT / "query_captions" / cohort / (qid + ".json"))["value"]
            orders["subject_guard20"], subject = guarded_order(
                PrimarySubjectRankerGuardedOrderInputs(
                    ids,
                    np.array(updated["base"]),
                    extra["pool"],
                    np.array(extra["logits"]),
                    updated_offsets,
                    qvalue,
                    public_profiles,
                )
            )
            assert set(orders["subject_guard20"][:20]) == set(
                orders["subject_caption20"][:20]
            )
            assert orders["subject_guard20"][20:] == orders["subject_caption20"][20:]
        else:
            orders.update(
                subject_caption20=list(original), subject_guard20=list(original)
            )
        assert set(orders) == set(METHODS)
        assert all(len(o) == len(ids) and set(o) == set(ids) for o in orders.values())
        for method in METHODS[3:6]:
            assert (
                set(orders[method][:20]) == set(original[:20])
                and orders[method][20:] == original[20:]
            )
        predictions[qid] = orders
        details[qid] = dict(
            ids=ids,
            whole_scores=whole,
            roi_scores=roi,
            public_subject_guard=subject,
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
        full_rankings=320,
        full_gallery=len(load(ROOT / "tune/catalog.public.json")),
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
        once(
            ROOT / cohort / "internal_guard_incremental.json",
            guard_incremental(ranks, labels, modes),
        )
        once(
            ROOT / cohort / "internal_roi_incremental.json",
            roi_incremental(ranks, labels, modes),
        )
        print(dict(evaluated=cohort, relations=len(labels)), flush=True)


def guard_incremental(ranks: dict, labels: list[dict], modes: dict) -> dict:
    """Return guard incremental for this workflow.

    Args:
        ranks: Ranks used by this operation.
        labels: Judgments used for the stated development or evaluation protocol.
        modes: Modes used by this operation.

    Returns:
        Result mapping produced by the workflow.
    """
    comparison = {
        qid: dict(
            global_=order["global"],
            region_title=order["region_title"],
            cross20=order["subject_caption20"],
            subject_guard20=order["subject_guard20"],
        )
        for qid, order in ranks.items()
    }
    for methods in comparison.values():
        methods["global"] = methods.pop("global_")
    metrics = evaluate_intervals(comparison, labels, modes)
    result = paired(comparison, labels, modes, metrics)
    result["baseline_alias"] = {"cross_evidence_pool60": "subject_caption20"}
    result["meaning"] = (
        "Public subject penalties only; same new caption, twenty candidates, raw full CE logits and full tail."
    )
    return result


def roi_incremental(ranks: dict, labels: list[dict], modes: dict) -> dict:
    """Return roi incremental for this workflow.

    Args:
        ranks: Ranks used by this operation.
        labels: Judgments used for the stated development or evaluation protocol.
        modes: Modes used by this operation.

    Returns:
        Result mapping produced by the workflow.
    """
    proxy = {
        qid: dict(
            order,
            subject_caption20=order["image_fusion20"],
            subject_guard20=order["image_roi_fusion20"],
        )
        for qid, order in ranks.items()
    }
    value = guard_incremental(proxy, labels, modes)
    value["baseline_alias"] = {"cross_evidence_pool60": "image_fusion20"}
    value["candidate_alias"] = {"subject_guard20": "image_roi_fusion20"}
    value["meaning"] = (
        "ROI contribution only; same original query description, twenty candidates, raw CE scores and full tail."
    )
    return value


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
    verify_files(PARENT, load(PARENT / "completion_receipt.json")["files"])
    errors = []
    for cohort in COHORTS:
        ranks, details = calculate_orders(cohort)
        assert ranks == load(ROOT / cohort / "predictions.json")
        assert details == load(ROOT / cohort / "runtime_scores.json")
        banks = feature_inputs(cohort)
        documents = {d["id"]: d for d in banks[0].records}
        for query in load(ROOT / cohort / "queries.public.json"):
            stages = [(False, "features.json", "cross.json")]
            if query["image_path"]:
                stages.append((True, "caption_features.json", "caption_cross.json"))
            for updated, feature_file, raw_file in stages:
                actual = calculate_features(cohort, query, banks, new_caption=updated)
                target = ROOT / cohort / "runs" / query["id"]
                saved = load(target / feature_file)
                for field in actual:
                    if field in {"features", "base"}:
                        a, b = np.array(actual[field]), np.array(saved[field])
                        assert np.allclose(a, b, rtol=0, atol=1e-12)
                        delta = float(np.max(np.abs(a - b)))
                        if delta:
                            errors.append(
                                dict(
                                    cohort=cohort,
                                    query=query["id"],
                                    file=feature_file,
                                    field=field,
                                    max_absolute_difference=delta,
                                )
                            )
                    else:
                        assert actual[field] == saved[field]
                raw = load(target / raw_file)
                prior_path = PROJECT_ROOT / raw["cache_file"]
                assert sha256_file(prior_path) == raw["cache_sha256"]
                prior = load(prior_path)
                cache = dict(zip(prior["pool"], prior["logits"]))
                for cid, value in zip(raw["pool"], raw["logits"]):
                    identity = pair_identity(
                        query, saved["description"], documents[cid]
                    )
                    assert identity == raw["pair_identities"][cid]
                    if cid in raw["reused_pair_ids"]:
                        assert (
                            identity == prior["pair_identities"][cid]
                            and value == cache[cid]
                        )
                order = rerank_complete(
                    actual["ids"],
                    np.array(actual["base"]),
                    raw["pool"],
                    np.array(raw["logits"]),
                    (
                        np.array(actual["offsets"])
                        if actual["offsets"] is not None
                        else None
                    ),
                )
                assert (
                    order
                    == ranks[query["id"]]["subject_caption20" if updated else "cross20"]
                )
            if query["text"]:
                assert all(
                    ranks[query["id"]][m] == ranks[query["id"]]["cross20"]
                    for m in [
                        "image_fusion20",
                        "image_roi_fusion20",
                        "subject_caption20",
                        "subject_guard20",
                    ]
                )
        labels = read_rows(ROOT / cohort / "judgments.ai.jsonl")
        old = read_rows(PARENT / cohort / "judgments.ai.jsonl")
        assert labels[: len(old)] == old
        assert len(labels) == len(banks[0].records) * len(ranks)
        modes = {qid: qid.split("_")[0] for qid in ranks}
        metrics = evaluate_intervals(ranks, labels, modes)
        assert metrics == load(ROOT / cohort / "internal_metrics.ai.development.json")
        assert paired(ranks, labels, modes, metrics) == load(
            ROOT / cohort / "internal_paired_differences.json"
        )
        assert guard_incremental(ranks, labels, modes) == load(
            ROOT / cohort / "internal_guard_incremental.json"
        )
        assert roi_incremental(ranks, labels, modes) == load(
            ROOT / cohort / "internal_roi_incremental.json"
        )
    once(
        ROOT / "verification.json",
        dict(
            passed=True,
            same_gallery_queries_and_labels=True,
            full_rankings_replayed=320,
            original_relations_unchanged=7720,
            runtime_uses_qrels=False,
            feature_absolute_tolerance=1e-12,
            feature_roundoff=errors,
            pools_orders_pair_identities_cached_logits_and_metrics_exact=True,
            changed_captions_never_reuse_old_query_context_scores=True,
            four_image_only_methods_text_route_exact=True,
            human_labels=0,
            formal_metrics=None,
            independent_acceptance_queries=0,
            default_changed=False,
        ),
    )
    print(dict(verified=True, roundoff=errors), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["prepare", "infer", "predict", "evaluate", "verify"]
    )
    globals()[parser.parse_args().action]()
