"""Extend any frozen development index by explicit ID, then compare fixed methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.ai_review_metrics import metric_ranges
from multimodal_ecommerce.blind_evaluation import file_map, verify_files
from multimodal_ecommerce.business_retrieval import BusinessReranker, hybrid_scores
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.development_expansion import (
    acceptance_guard,
    fold_metadata,
    load_expanded_index,
)
from multimodal_ecommerce.embedding import QwenEmbedder, model_configuration
from multimodal_ecommerce.holdout_data import fingerprints
from multimodal_ecommerce.intent_retrieval import (
    balanced_pool,
    normalized_ranks,
    public_evidence,
    recall_channels,
    rerank_complete,
)
from multimodal_ecommerce.pair_reranker import reranker_configuration
from multimodal_ecommerce.precision_reranking import color_descriptor
from multimodal_ecommerce.regional_retrieval import (
    RegionalRetrievalEncodeViewsInputs,
    encode_views,
    region_configuration,
)
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.semantic_ranker import (
    FEATURE_NAMES,
    SemanticRankerFeatureMatrixInputs,
    feature_matrix,
    ranked_ids,
)
from multimodal_ecommerce.supervised_pipeline import (
    SupervisedPipelineFitSupervisedBundleInputs,
    fit_supervised_bundle,
)
from prepare_visual_coverage import PARENT, REGISTRY, ROOT
from prepare_visual_coverage import check as check_acquisition

RESERVED = (
    PROJECT_ROOT
    / "artifacts/module2_business_acceptance/dataset/evaluator/lineage.jsonl"
)


class _PolicySpec(TypedDict):
    """Exact fields of the POLICY configuration and provenance record."""

    role: str
    fixed_k: int
    unchanged_queries: int
    precision_required: float
    recall_preferred: float
    family_folds: int
    fold_seed: int
    duplicate_dhash_distance: int
    hyperparameter_search: bool
    relevance_from_acquisition_bucket: bool
    acceptance_inference_allowed: bool
    public_quality_metrics_allowed: bool
    default_promotion_allowed: bool


POLICY: _PolicySpec = dict(
    role="AI_assessed_query_aware_exposed_business_development_only",
    fixed_k=10,
    unchanged_queries=18,
    precision_required=0.6,
    recall_preferred=0.85,
    family_folds=3,
    fold_seed=2026102205,
    duplicate_dhash_distance=4,
    hyperparameter_search=False,
    relevance_from_acquisition_bucket=False,
    acceptance_inference_allowed=False,
    public_quality_metrics_allowed=False,
    default_promotion_allowed=False,
)


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def key(value: str) -> str:
    """Return key for this workflow.

    Args:
        value: Input value to validate or transform.

    Returns:
        Text produced by this operation.
    """
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def raw_catalog() -> dict:
    """Return raw catalog for this workflow.

    Returns:
        Result mapping produced by the workflow.
    """
    paths = [
        PROJECT_ROOT / "data/stage2_taxonomy_v2/products.jsonl",
        PROJECT_ROOT / "artifacts/module2_catalog_review/dataset/products.jsonl",
        ROOT / "dataset/products.jsonl",
    ]
    return {r["id"]: r for path in paths for r in read_rows(path)}


def check() -> Any:
    """Verify the prerequisites and integrity of the current workflow.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    protocol = load(ROOT / "protocol.json")
    assert protocol["policy"] == POLICY
    check_acquisition()
    verify_files(PROJECT_ROOT, protocol["implementation"])
    verify_files(PROJECT_ROOT, protocol["parents"])
    verify_files(PROJECT_ROOT, protocol["development_image_files"])
    verify_files(ROOT, protocol["prepared_files"])
    for cfg in [protocol["embedding"], protocol["reranker"]]:
        verify_files(Path(cfg["path"]), cfg["checkpoint_files"])
    return protocol


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        FileExistsError: Preserve frozen inference protocol.

    Notes:
        Writes derived artifacts through np.savez_compressed, write_json.
    """
    check_acquisition()
    if (ROOT / "protocol.json").exists():
        raise FileExistsError("Preserve frozen inference protocol")
    verify_files(ROOT, load(ROOT / "review_receipt.json")["files"])
    inputs = load(ROOT / "inputs.public.json")
    previous = load(PARENT / "inputs.public.json")
    assert inputs["queries"] == previous["queries"]
    public = inputs["catalog"]
    assert all(set(r) == {"id", "title", "image_path"} for r in public)
    old_ids = {r["id"] for r in previous["catalog"]}
    assert old_ids < {r["id"] for r in public}
    new = [r for r in public if r["id"] not in old_ids]
    captions = read_rows(ROOT / "profiles.public.jsonl")
    assert {r["id"] for r in captions} == {r["id"] for r in new}
    assert all(
        set(r) == {"id", "product_type", "appearance", "caption_origin", "image_sha256"}
        for r in captions
    )
    guard = acceptance_guard(public, raw_catalog(), read_rows(RESERVED))
    write_json(ROOT / "guard.json", guard)
    write_json(ROOT / "folds.json", fold_metadata(public, inputs["queries"], guard))
    with np.load(PARENT / "query_vectors.npz", allow_pickle=False) as archive:
        np.savez_compressed(ROOT / "query_vectors.npz", vectors=archive["vectors"])
    code = [
        "scripts/advance_visual_coverage.py",
        "scripts/apply_visual_coverage_review.py",
        "scripts/prepare_visual_coverage.py",
        "src/multimodal_ecommerce/development_expansion.py",
        "src/multimodal_ecommerce/embedding.py",
        "src/multimodal_ecommerce/regional_retrieval.py",
        "src/multimodal_ecommerce/semantic_ranker.py",
        "src/multimodal_ecommerce/intent_retrieval.py",
        "src/multimodal_ecommerce/business_retrieval.py",
        "src/multimodal_ecommerce/pair_reranker.py",
        "src/multimodal_ecommerce/supervised_pipeline.py",
        "src/multimodal_ecommerce/ai_review_metrics.py",
    ]
    parents = [
        PARENT / "completion_receipt.json",
        PARENT / "protocol.json",
        PARENT / "index/manifest.json",
        PARENT / "ranking_receipt.json",
        PARENT / "internal_metrics.ai.development.json",
        PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl",
        RESERVED,
        REGISTRY,
        PROJECT_ROOT / "configs/module2_current_target.json",
        PROJECT_ROOT / "configs/embedding_model.json",
        PROJECT_ROOT / "configs/reranker_model.json",
        PROJECT_ROOT / "configs/retrieval_regions_v1.json",
    ]
    checkpoints = [
        dict(cfg, checkpoint_files=file_map(Path(cfg["path"])))
        for cfg in [model_configuration(), reranker_configuration()]
    ]
    write_json(
        ROOT / "protocol.json",
        dict(
            frozen_at=datetime.now(timezone.utc).isoformat(),
            policy=POLICY,
            parent_products=len(old_ids),
            new_products=len(new),
            total_products=len(public),
            implementation={p: sha256_file(PROJECT_ROOT / p) for p in code},
            parents={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in parents
            },
            prepared_files={
                p: sha256_file(ROOT / p)
                for p in [
                    "inputs.public.json",
                    "profiles.public.jsonl",
                    "judgments.ai.jsonl",
                    "review_receipt.json",
                    "guard.json",
                    "folds.json",
                    "query_vectors.npz",
                ]
            },
            development_image_files={
                Path(r["image_path"])
                .relative_to(PROJECT_ROOT)
                .as_posix(): sha256_file(Path(r["image_path"]))
                for r in [*public, *[q for q in inputs["queries"] if q["image_path"]]]
            },
            embedding=checkpoints[0],
            reranker=checkpoints[1],
            region_policy=region_configuration(),
            training_policy=load(PARENT / "protocol.json")["training_policy"],
            comparison="Same unchanged exposed queries, same expanded gallery, same AI qrels across all fixed methods; do not attribute gallery changes to ranking strategies",
            method_names=[
                "global",
                "region_title",
                "hybrid",
                "semantic_blend",
                "multi_channel_rrf",
                "public_evidence_soft",
                "AI_oof_balanced_logistic",
                "AI_oof_hard_negative_ranknet",
                "cross_pool20",
                "cross_evidence_pool20",
                "cross_pool60",
                "cross_evidence_pool60",
            ],
            label_provenance="AI native direct pixel judgments only, unknown retained; not independent acceptance evidence",
            heldout_pixels_opened=0,
            acceptance_predictions=0,
        ),
    )
    print(
        json.dumps(
            dict(
                frozen=True,
                catalog=len(public),
                new=len(new),
                reserved_overlap=0,
                acceptance_predictions=0,
            )
        )
    )


def encode() -> None:
    """Encode the configured inputs into model feature representations.

    Raises:
        FileExistsError: Preserve complete index.

    Notes:
        Writes derived artifacts through cache_dir.mkdir, np.savez_compressed,
        output.mkdir, write_json.
    """
    protocol = check()
    if (ROOT / "index/manifest.json").exists():
        raise FileExistsError("Preserve complete index")
    inputs = load(ROOT / "inputs.public.json")
    base, base_titles = load_expanded_index(PARENT / "index")
    positions = {r["id"]: i for i, r in enumerate(base.records)}
    with np.load(PARENT / "index/vectors.npz", allow_pickle=False) as archive:
        base_profiles = archive["profiles"].copy()
    base_texts = load(PARENT / "index/profile_texts.json")
    captions = {r["id"]: r for r in read_rows(ROOT / "profiles.public.jsonl")}
    new = [r for r in inputs["catalog"] if r["id"] not in positions]
    output = ROOT / "index"
    output.mkdir(exist_ok=True)
    cache_dir = ROOT / "embedding_cache"
    cache_dir.mkdir(exist_ok=True)
    pending = [r for r in new if not (cache_dir / (key(r["id"]) + ".json")).exists()]
    encoder = QwenEmbedder() if pending else None
    try:
        for number, product in enumerate(new, 1):
            cache_path = cache_dir / (key(product["id"]) + ".json")
            if not cache_path.exists():
                assert encoder is not None
                destination = output / "views" / key(product["id"])
                attempt = 0
                while (destination / f"attempt_{attempt}").exists():
                    attempt += 1
                vectors, views = encode_views(
                    RegionalRetrievalEncodeViewsInputs(
                        encoder,
                        Path(product["image_path"]),
                        destination / f"attempt_{attempt}",
                        protocol["region_policy"],
                    )
                )
                title = encoder.encode(text=product["title"])
                cap = captions[product["id"]]
                profile_text = (
                    "商品主体："
                    + cap["product_type"]
                    + "。可见外观："
                    + cap["appearance"]
                    + "。"
                )
                profile = encoder.encode(text=profile_text)
                asset = cache_path.with_suffix(".npz")
                np.savez_compressed(
                    asset, regions=vectors, title=title, profile=profile
                )
                write_json(
                    cache_path,
                    dict(
                        id=product["id"],
                        title=product["title"],
                        image_path=product["image_path"],
                        image_sha256=sha256_file(Path(product["image_path"])),
                        views=views,
                        profile_text=profile_text,
                        vector_asset=asset.name,
                        vector_sha256=sha256_file(asset),
                    ),
                )
            print(f"Coverage embeddings {number}/{len(new)}", flush=True)
        if encoder is not None:
            runtime = ROOT / "embedding_runtime.json"
            if runtime.exists():
                runtime = ROOT / (
                    "embedding_resume_"
                    + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                    + ".json"
                )
            write_json(runtime, encoder.diagnostics())
    finally:
        if encoder is not None:
            encoder.close()
    regions = np.zeros(
        (len(inputs["catalog"]), protocol["region_policy"]["max_views"], 2048),
        dtype=np.float32,
    )
    records, titles, profiles, texts = [], [], [], []
    for i, product in enumerate(inputs["catalog"]):
        cid = product["id"]
        if cid in positions:
            j = positions[cid]
            record = base.records[j]
            regions[i] = base.vectors[j]
            titles.append(base_titles[j])
            profiles.append(base_profiles[j])
            texts.append(base_texts[j])
        else:
            cached = load(cache_dir / (key(cid) + ".json"))
            asset = cache_dir / cached["vector_asset"]
            assert sha256_file(asset) == cached["vector_sha256"]
            assert (
                cached["title"] == product["title"]
                and cached["image_path"] == product["image_path"]
            )
            with np.load(asset, allow_pickle=False) as archive:
                regions[i, : len(cached["views"])] = archive["regions"]
                titles.append(archive["title"].copy())
                profiles.append(archive["profile"].copy())
            texts.append(cached["profile_text"])
            record = {
                k: cached[k]
                for k in ["id", "title", "image_path", "image_sha256", "views"]
            }
        records.append(record)
    np.savez_compressed(
        output / "vectors.npz",
        regions=regions,
        texts=np.stack(titles),
        profiles=np.stack(profiles),
    )
    write_json(output / "records.json", records)
    write_json(output / "profile_texts.json", texts)
    upstream = dict(load(PARENT / "index/manifest.json")["upstream_files"])
    upstream.update(
        {
            (PARENT / "index" / name).relative_to(PROJECT_ROOT).as_posix(): checksum
            for name, checksum in load(PARENT / "index/manifest.json")["files"].items()
        }
    )
    upstream[(PARENT / "index/manifest.json").relative_to(PROJECT_ROOT).as_posix()] = (
        sha256_file(PARENT / "index/manifest.json")
    )
    write_json(
        output / "manifest.json",
        dict(
            role=POLICY["role"],
            count=len(records),
            new_records=len(new),
            reused_records=len(positions),
            model=protocol["embedding"],
            regions=protocol["region_policy"],
            files=file_map(output),
            upstream_files=upstream,
            formal_acceptance=False,
            acceptance_predictions=0,
        ),
    )
    load_expanded_index(output)
    print(
        json.dumps(
            dict(
                index_complete=True,
                count=len(records),
                new=len(new),
                full_views=sum(len(r["views"]) for r in records),
            )
        )
    )


def rank() -> None:
    """Apply the rank workflow step.

    Raises:
        ValueError: Preserve completed development rankings.

    Notes:
        Writes derived artifacts through directory.mkdir, np.savez_compressed,
        write_json.
    """
    protocol = check()
    if (ROOT / "ranking_receipt.json").exists():
        raise ValueError("Preserve completed development rankings")
    inputs = load(ROOT / "inputs.public.json")
    index, titles = load_expanded_index(ROOT / "index")
    records = index.records
    with np.load(ROOT / "index/vectors.npz", allow_pickle=False) as bank:
        profiles = bank["profiles"].copy()
    profile_texts = load(ROOT / "index/profile_texts.json")
    with np.load(ROOT / "query_vectors.npz", allow_pickle=False) as bank:
        qvectors = bank["vectors"].copy()
    colors = np.stack([color_descriptor(Path(r["image_path"])) for r in records])
    x = np.stack(
        [
            feature_matrix(
                SemanticRankerFeatureMatrixInputs(
                    index,
                    titles,
                    profiles,
                    profile_texts,
                    qvectors[i],
                    text=q["text"],
                    image=Path(q["image_path"]) if q["image_path"] else None,
                    color_bank=colors,
                )
            )
            for i, q in enumerate(inputs["queries"])
        ]
    )
    ids = [r["id"] for r in records]
    valid = np.ones((len(inputs["queries"]), len(records)), dtype=bool)
    rankings, pools, evidence, bases = (
        {},
        {},
        {},
        np.zeros((len(inputs["queries"]), len(records))),
    )
    signatures = load(ROOT / "guard.json")["signatures"]
    for i, query in enumerate(inputs["queries"]):
        excluded = {query["exclude_id"]} if query["exclude_id"] else set()
        if query["image_path"]:
            signature = fingerprints(Path(query["image_path"]))
            excluded.update(
                r["id"]
                for r in records
                if signatures[r["id"]]["sha256"] == signature["sha256"]
                or signatures[r["id"]]["pixels"] == signature["pixels"]
            )
        valid[i] = [cid not in excluded for cid in ids]
        positions = np.flatnonzero(valid[i])
        current_ids = [ids[j] for j in positions]
        current = x[i, positions]
        hybrid = hybrid_scores(
            current_ids,
            current[:, 1],
            current[:, 3],
            text=query["text"],
            titles=[records[j]["title"] for j in positions],
        )
        rrf, channels = recall_channels(current_ids, current)
        why = [
            public_evidence(query["text"], records[j]["title"], profile_texts[j])
            for j in positions
        ]
        evidence[query["id"]] = dict(zip(current_ids, why))
        adjusted = normalized_ranks(current_ids, rrf) + np.array(
            [r["offset"] for r in why]
        )
        channels.append(ranked_ids(current_ids, adjusted))
        pools[query["id"]] = balanced_pool(channels, 60)
        rankings[query["id"]] = dict(
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
        rankings[query["id"]]["global"] = rankings[query["id"]].pop("global_")
        bases[i, positions] = hybrid
    write_json(ROOT / "baseline_rankings.json", rankings)
    write_json(ROOT / "pools.json", pools)
    write_json(ROOT / "public_evidence.json", evidence)
    folds = load(ROOT / "folds.json")
    metadata = {**folds, "feature_names": FEATURE_NAMES, "profile_texts": profile_texts}
    qrels = {
        (r["query_id"], r["candidate_id"]): r["grade"]
        for r in read_rows(ROOT / "judgments.ai.jsonl")
    }
    y = np.array(
        [
            [
                np.nan if qrels.get((q["id"], cid)) is None else qrels[q["id"], cid]
                for cid in ids
            ]
            for q in inputs["queries"]
        ],
        dtype=float,
    )
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
    if not fitted["trained"]:
        raise ValueError(fitted)
    for name in ["models", "candidates", "audits"]:
        write_json(ROOT / ("trained_" + name + ".json"), fitted[name])
    for i, query in enumerate(inputs["queries"]):
        positions = np.flatnonzero(valid[i])
        current_ids = [ids[j] for j in positions]
        for name, scores in fitted["predictions"].items():
            rankings[query["id"]]["AI_oof_" + name] = ranked_ids(
                current_ids, scores[i, positions]
            )
    np.savez_compressed(
        ROOT / "features.npz",
        features=x,
        valid=valid,
        labels=y,
        base_scores=bases,
        color_bank=colors,
    )
    write_json(ROOT / "feature_metadata.json", metadata)
    write_json(ROOT / "rankings.before_cross.json", rankings)
    # Cached pair scores are eligible only for byte-identical public model inputs.
    cache: dict = {}
    for parent_root in [PROJECT_ROOT / "artifacts/module2_precision_recall", PARENT]:
        prior = load(
            parent_root
            / ("inputs.public.json" if parent_root == PARENT else "inputs.json")
        )
        prior_queries = {q["id"]: q for q in prior["queries"]}
        prior_docs = {r["id"]: r for r in prior["catalog"]}
        for path in sorted((parent_root / "pair_scores").glob("*.json")):
            row = load(path)
            qid, cid = row["query_id"], row["candidate_id"]
            public = dict(
                query={k: prior_queries[qid][k] for k in ["id", "text", "image_path"]},
                document=dict(
                    id=cid,
                    image_path=prior_docs[cid]["image_path"],
                    text=prior_docs[cid]["title"],
                ),
            )
            digest = hashlib.sha256(
                json.dumps(public, sort_keys=True).encode()
            ).hexdigest()
            assert digest == row["model_input_sha256"]
            if digest in cache:
                assert cache[digest]["logit"] == row["logit"]
            cache[digest] = dict(
                logit=row["logit"],
                cache_source=path.relative_to(PROJECT_ROOT).as_posix(),
                cache_source_sha256=sha256_file(path),
            )
    by_id = {r["id"]: r for r in records}
    pair_scores = {}
    origins: Counter = Counter()
    directory = ROOT / "pair_scores"
    directory.mkdir(exist_ok=True)
    reranker = None
    start = time.perf_counter()
    try:
        for number, query in enumerate(inputs["queries"], 1):
            qid = query["id"]
            for cid in pools[qid]:
                doc = dict(
                    id=cid,
                    image_path=by_id[cid]["image_path"],
                    text=by_id[cid]["title"],
                )
                public = dict(
                    query={k: query[k] for k in ["id", "text", "image_path"]},
                    document=doc,
                )
                input_hash = hashlib.sha256(
                    json.dumps(public, sort_keys=True).encode()
                ).hexdigest()
                path = directory / (key(qid + "|" + cid) + ".json")
                if path.exists():
                    scored = load(path)
                    assert scored["model_input_sha256"] == input_hash
                elif input_hash in cache:
                    scored = dict(
                        query_id=qid,
                        candidate_id=cid,
                        logit=cache[input_hash]["logit"],
                        cache_source=cache[input_hash]["cache_source"],
                        cache_source_sha256=cache[input_hash]["cache_source_sha256"],
                        model_input_sha256=input_hash,
                        origin="identical_frozen_input_cache",
                        seconds=0,
                    )
                    write_json(path, scored)
                else:
                    if reranker is None:
                        reranker = BusinessReranker()
                    begin = time.perf_counter()
                    logit = float(
                        reranker.score(
                            [doc],
                            query_text=query["text"],
                            query_image=(
                                Path(query["image_path"])
                                if query["image_path"]
                                else None
                            ),
                        )[0]
                    )
                    scored = dict(
                        query_id=qid,
                        candidate_id=cid,
                        logit=logit,
                        model_input_sha256=input_hash,
                        origin="actual_full_cross_encoder_inference",
                        seconds=time.perf_counter() - begin,
                    )
                    write_json(path, scored)
                pair_scores[qid, cid] = scored["logit"]
                origins[scored["origin"]] += 1
            print(f"Coverage development cross encoder {number}/18", flush=True)
        if reranker is not None:
            write_json(ROOT / "reranker_runtime.json", reranker.diagnostics())
    finally:
        if reranker is not None:
            reranker.close()
    for i, query in enumerate(inputs["queries"]):
        qid = query["id"]
        positions = np.flatnonzero(valid[i])
        current_ids = [ids[j] for j in positions]
        rrf, _ = recall_channels(current_ids, x[i, positions])
        offsets = np.array([evidence[qid][cid]["offset"] for cid in current_ids])
        for size in [20, 60]:
            pool = pools[qid][:size]
            logits = np.array([pair_scores[qid, cid] for cid in pool])
            for name, adjustment in [("cross", None), ("cross_evidence", offsets)]:
                rankings[qid][f"{name}_pool{size}"] = rerank_complete(
                    current_ids, rrf, pool, logits, adjustment
                )
    write_json(ROOT / "rankings.json", rankings)
    scored_labels = [
        r
        for r in read_rows(ROOT / "judgments.ai.jsonl")
        if r["candidate_id"] in set(ids)
    ]
    modes = {q["id"]: "text" if q["text"] else "image" for q in inputs["queries"]}
    write_json(
        ROOT / "internal_metrics.ai.development.json",
        metric_ranges(rankings, scored_labels, modes),
    )
    write_json(
        ROOT / "ranking_receipt.json",
        dict(
            methods=12,
            queries=18,
            full_rankings=216,
            pairs=len(pair_scores),
            pair_score_origins=dict(origins),
            cross_wall_seconds=time.perf_counter() - start,
            formal_acceptance=False,
            acceptance_predictions=0,
            unknown_labels_retained=int((valid & ~np.isfinite(y)).sum()),
        ),
    )
    write_json(
        ROOT / "public_status.json",
        dict(
            state="expanded_development_index_and_comparison_complete",
            development_catalog_products=len(records),
            new_products=protocol["new_products"],
            development_queries=18,
            methods=12,
            full_rankings=216,
            guard_passed=True,
            AI_labels_only=True,
            quality_metrics=None,
            formal_metrics=None,
            formal_acceptance=False,
            goal_met=False,
            acceptance_predictions=0,
            external_quality_claim_allowed=False,
            default_strategy_changed=False,
        ),
    )
    print(json.dumps(load(ROOT / "ranking_receipt.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "encode", "rank"])
    globals()[parser.parse_args().action]()
