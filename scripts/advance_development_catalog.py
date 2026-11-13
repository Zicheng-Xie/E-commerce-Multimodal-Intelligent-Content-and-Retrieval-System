"""Freeze, extend and compare the 344-product development catalog only."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.ai_review_metrics import metric_ranges
from multimodal_ecommerce.blind_evaluation import file_map, verify_files
from multimodal_ecommerce.business_retrieval import BusinessReranker, hybrid_scores
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.development_expansion import (
    POLICY,
    acceptance_guard,
    clean_document,
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
    RegionalIndex,
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

ROOT = PROJECT_ROOT / "artifacts/module2_expanded_development"
AI = PROJECT_ROOT / "artifacts/module2_ai_review"
BANK = PROJECT_ROOT / "artifacts/module2_machine_assist/semantic_ranker"
OLD = PROJECT_ROOT / "data/index_regions_products_v1"
RESERVED = (
    PROJECT_ROOT
    / "artifacts/module2_business_acceptance/dataset/evaluator/lineage.jsonl"
)
REGISTRY = PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl"


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def dump_rows(path: Path, rows: list[dict]) -> None:
    """Apply the dump rows workflow step.

    Args:
        path: Path of the input or output artifact used by this operation.
        rows: Records processed by this operation.

    Notes:
        Writes derived artifacts through path.write_text.
    """
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def key(value: str) -> str:
    """Return key for this workflow.

    Args:
        value: Input value to validate or transform.

    Returns:
        Text produced by this operation.
    """
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def check() -> Any:
    """Verify the prerequisites and integrity of the current workflow.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.

    Raises:
        ValueError: Changed executed source or documented alignment repair; Expanded
            development policy changed.
    """
    protocol = load(ROOT / "protocol.json")
    implementations = dict(protocol["implementation"])
    amendment_path = ROOT / "implementation_amendment.json"
    if amendment_path.exists():
        amendment = load(amendment_path)
        executed = ROOT / amendment["executed_source"]
        expected = implementations.pop(amendment["current_source"])
        if (
            sha256_file(executed) != expected
            or sha256_file(PROJECT_ROOT / amendment["current_source"])
            != amendment["current_sha256"]
        ):
            raise ValueError("Changed executed source or documented alignment repair")
    verify_files(PROJECT_ROOT, implementations)
    verify_files(PROJECT_ROOT, protocol["parents"])
    verify_files(PROJECT_ROOT, protocol["development_image_files"])
    verify_files(ROOT, protocol["prepared_files"])
    for cfg in [protocol["embedding"], protocol["reranker"]]:
        verify_files(Path(cfg["path"]), cfg["checkpoint_files"])
    if protocol["policy"] != POLICY:
        raise ValueError("Expanded development policy changed")
    return protocol


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        ValueError: Preserve frozen development protocol.

    Notes:
        Writes derived artifacts through ROOT.mkdir, np.savez_compressed, write_json.
    """
    if (ROOT / "protocol.json").exists():
        raise ValueError("Preserve frozen development protocol")
    ROOT.mkdir(exist_ok=True)
    verify_files(AI, load(AI / "completion_receipt.json")["files"])
    public = [clean_document(row) for row in read_rows(AI / "catalog.reviewed.jsonl")]
    assert len(public) == 344
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
    guard = acceptance_guard(public, raw, read_rows(RESERVED))
    # Drop evaluator fields before saving any runtime input.
    queries = [
        {k: q[k] for k in ["id", "text", "image_path", "exclude_id"]}
        for q in load(AI / "inputs.json")["queries"]
    ]
    qmeta = load(PROJECT_ROOT / "artifacts/stage2_predictions_v1/run.json")
    assert qmeta["query_order"] == [q["id"] for q in queries]
    with np.load(
        PROJECT_ROOT / "artifacts/stage2_predictions_v1/query_vectors.npz",
        allow_pickle=False,
    ) as bank:
        qvectors = bank["vectors"].copy()
    assert qvectors.shape == (18, 2048)
    np.savez_compressed(ROOT / "query_vectors.npz", vectors=qvectors)
    write_json(ROOT / "inputs.public.json", dict(queries=queries, catalog=public))
    write_json(ROOT / "guard.json", guard)
    folds = fold_metadata(public, queries, guard)
    write_json(ROOT / "folds.json", folds)
    old_index, _, _ = RegionalIndex.load(OLD)
    old_ids = {r["id"] for r in old_index.records}
    captions = read_rows(ROOT / "profiles.public.jsonl")
    assert {r["id"] for r in captions} == {r["id"] for r in public} - old_ids
    assert all(
        set(r) == {"id", "product_type", "appearance", "caption_origin", "image_sha256"}
        for r in captions
    )
    implementation = [
        "scripts/advance_development_catalog.py",
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
    parent_files = [
        "artifacts/module2_ai_review/completion_receipt.json",
        "artifacts/module2_ai_review/judgments.ai.jsonl",
        "artifacts/module2_ai_review/catalog.reviewed.jsonl",
        "data/module2_reserved_acceptance/records.jsonl",
        "artifacts/module2_business_acceptance/dataset/evaluator/lineage.jsonl",
        "data/module2_development_expansion/records.jsonl",
        "data/index_regions_products_v1/manifest.json",
        "artifacts/module2_machine_assist/semantic_ranker/feature_receipt.json",
        "artifacts/module2_machine_assist/semantic_ranker/feature_metadata.json",
        "artifacts/module2_machine_assist/semantic_ranker/profile_vectors.npz",
        "artifacts/module2_machine_assist/semantic_ranker/protocol.json",
        "artifacts/module2_precision_recall/inference_receipt.json",
        "configs/embedding_model.json",
        "configs/reranker_model.json",
        "configs/retrieval_regions_v1.json",
    ]
    checkpoints = []
    for cfg in [model_configuration(), reranker_configuration()]:
        path = Path(cfg["path"])
        checkpoints.append(dict(cfg, checkpoint_files=file_map(path)))
    write_json(
        ROOT / "protocol.json",
        dict(
            frozen_at=datetime.now(timezone.utc).isoformat(),
            policy=POLICY,
            implementation={p: sha256_file(PROJECT_ROOT / p) for p in implementation},
            parents={p: sha256_file(PROJECT_ROOT / p) for p in parent_files},
            prepared_files={
                p: sha256_file(ROOT / p)
                for p in [
                    "query_vectors.npz",
                    "inputs.public.json",
                    "guard.json",
                    "folds.json",
                    "profiles.public.jsonl",
                ]
            },
            development_image_files={
                str(Path(r["image_path"]).relative_to(PROJECT_ROOT)).replace(
                    "\\", "/"
                ): sha256_file(Path(r["image_path"]))
                for r in [*public, *[q for q in queries if q["image_path"]]]
            },
            embedding=checkpoints[0],
            reranker=checkpoints[1],
            region_policy=region_configuration(),
            training_policy=load(BANK / "protocol.json")["policy"],
            comparison="All methods evaluated on identical 344-product gallery and unchanged 18 exposed queries",
            attribution="Catalog expansion and same-gallery strategy comparison reported separately",
            scorer="AI labels are internal diagnostics, never formal acceptance; unknown labels preserved",
            historical_fashion_metrics_are_separate=True,
            query_reference_images_from_acceptance_allowed=False,
            retrieval_uses_judgment_table=False,
            hyperparameter_search=False,
            rerank_pool_sizes=[20, 60],
            acceptance_registry_sha256=sha256_file(REGISTRY),
        ),
    )
    write_json(
        ROOT / "public_status.json",
        dict(
            state="development_protocol_frozen",
            development_catalog_products=344,
            new_products=58,
            development_queries=18,
            guard_passed=True,
            quality_metrics=None,
            formal_metrics=None,
            formal_acceptance=False,
            goal_met=False,
            acceptance_predictions=0,
            external_quality_claim_allowed=False,
        ),
    )
    print(
        json.dumps(
            dict(
                frozen=True,
                catalog=344,
                new=58,
                reserved_overlap=0,
                acceptance_predictions=0,
            )
        )
    )


def encode() -> None:
    """Encode the configured inputs into model feature representations.

    Raises:
        ValueError: Expanded index already complete.

    Notes:
        Writes derived artifacts through (ROOT / 'embedding_cache').mkdir,
        np.savez_compressed, output.mkdir, write_json.
    """
    protocol = check()
    if (ROOT / "index/manifest.json").exists():
        raise ValueError("Expanded index already complete")
    inputs = load(ROOT / "inputs.public.json")
    old_index, old_titles, _ = RegionalIndex.load(OLD)
    assert old_titles is not None
    old_ids = {r["id"] for r in old_index.records}
    captions = {r["id"]: r for r in read_rows(ROOT / "profiles.public.jsonl")}
    output = ROOT / "index"
    output.mkdir(exist_ok=True)
    (ROOT / "embedding_cache").mkdir(exist_ok=True)
    pending = [
        r
        for r in inputs["catalog"]
        if r["id"] not in old_ids
        and not (ROOT / "embedding_cache" / (key(r["id"]) + ".json")).exists()
    ]
    encoder = QwenEmbedder() if pending else None
    try:
        for number, product in enumerate(
            [r for r in inputs["catalog"] if r["id"] not in old_ids], 1
        ):
            name = key(product["id"])
            cached = ROOT / "embedding_cache" / (name + ".json")
            if not cached.exists():
                assert encoder is not None
                destination = output / "views" / name
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
                profile = (
                    "商品主体："
                    + cap["product_type"]
                    + "。可见外观："
                    + cap["appearance"]
                    + "。"
                )
                profile_vector = encoder.encode(text=profile)
                asset = cached.with_suffix(".npz")
                np.savez_compressed(
                    asset, regions=vectors, title=title, profile=profile_vector
                )
                write_json(
                    cached,
                    dict(
                        id=product["id"],
                        title=product["title"],
                        image_path=product["image_path"],
                        image_sha256=sha256_file(Path(product["image_path"])),
                        views=views,
                        profile_text=profile,
                        vector_asset=asset.name,
                        vector_sha256=sha256_file(asset),
                    ),
                )
            print(f"Expanded development embeddings {number}/58", flush=True)
        if encoder is not None:
            runtime_path = ROOT / "embedding_runtime.json"
            if runtime_path.exists():
                runtime_path = ROOT / (
                    "embedding_resume_"
                    + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                    + ".json"
                )
            write_json(runtime_path, encoder.diagnostics())
    finally:
        if encoder is not None:
            encoder.close()
    # Preserve old feature bytes and reference immutable original crops in place.
    old_positions = {r["id"]: i for i, r in enumerate(old_index.records)}
    source_positions = [old_positions[r["id"]] for r in inputs["catalog"][:286]]
    records = [
        {k: row[k] for k in ["id", "title", "image_path", "image_sha256", "views"]}
        for row in [old_index.records[i] for i in source_positions]
    ]
    bank = np.zeros(
        (344, protocol["region_policy"]["max_views"], 2048), dtype=np.float32
    )
    bank[:286] = old_index.vectors[source_positions]
    titles = [*old_titles[source_positions]]
    old_meta = load(BANK / "feature_metadata.json")
    profile_positions = {cid: i for i, cid in enumerate(old_meta["gallery_ids"])}
    aligned_profile_positions = [profile_positions[r["id"]] for r in records]
    with np.load(BANK / "profile_vectors.npz", allow_pickle=False) as archive:
        profiles = [*archive["vectors"][aligned_profile_positions]]
    profile_texts = [old_meta["profile_texts"][i] for i in aligned_profile_positions]
    assert [r["id"] for r in records] == [r["id"] for r in inputs["catalog"][:286]]
    for i, product in enumerate(inputs["catalog"][286:], 286):
        cached = load(ROOT / "embedding_cache" / (key(product["id"]) + ".json"))
        asset = ROOT / "embedding_cache" / cached["vector_asset"]
        assert sha256_file(asset) == cached["vector_sha256"]
        with np.load(asset, allow_pickle=False) as archive:
            bank[i, : len(cached["views"])], title, profile = (
                archive["regions"],
                archive["title"],
                archive["profile"],
            )
        titles.append(title)
        profiles.append(profile)
        profile_texts.append(cached["profile_text"])
        records.append(
            {
                k: cached[k]
                for k in ["id", "title", "image_path", "image_sha256", "views"]
            }
        )
    np.savez_compressed(
        output / "vectors.npz",
        regions=bank,
        texts=np.stack(titles),
        profiles=np.stack(profiles),
    )
    write_json(output / "records.json", records)
    write_json(output / "profile_texts.json", profile_texts)
    write_json(
        output / "manifest.json",
        dict(
            role="exposed_AI_assessed_business_development_only",
            count=344,
            new_records=58,
            reused_records=286,
            model=protocol["embedding"],
            regions=protocol["region_policy"],
            files={
                str(p.relative_to(output)).replace("\\", "/"): sha256_file(p)
                for p in output.rglob("*")
                if p.is_file()
            },
            upstream_files={
                str((OLD / name).relative_to(PROJECT_ROOT)).replace("\\", "/"): checksum
                for name, checksum in load(OLD / "manifest.json")["files"].items()
            },
            formal_acceptance=False,
            acceptance_predictions=0,
        ),
    )
    load_expanded_index(output)
    print(
        json.dumps(
            dict(
                index_complete=True,
                count=344,
                new=58,
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
    valid = np.ones((18, 344), dtype=bool)
    rankings, pools, evidence, bases = {}, {}, {}, np.zeros((18, 344))
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
        for r in read_rows(AI / "judgments.ai.jsonl")
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
    cache = {}
    previous_inputs = load(
        PROJECT_ROOT / "artifacts/module2_precision_recall/inputs.json"
    )
    prior_queries = {q["id"]: q for q in previous_inputs["queries"]}
    prior_docs = {r["id"]: r for r in previous_inputs["catalog"]}
    for p in (PROJECT_ROOT / "artifacts/module2_precision_recall/pair_scores").glob(
        "*.json"
    ):
        row = load(p)
        qid, cid = row["query_id"], row["candidate_id"]
        public = dict(
            query=prior_queries[qid],
            document=dict(
                id=cid,
                image_path=prior_docs[cid]["image_path"],
                text=prior_docs[cid]["title"],
            ),
        )
        cache[
            hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
        ] = row["logit"]
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
                        logit=cache[input_hash],
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
            print(f"Expanded development cross encoder {number}/18", flush=True)
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
        r for r in read_rows(AI / "judgments.ai.jsonl") if r["candidate_id"] in set(ids)
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
            development_catalog_products=344,
            new_products=58,
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
    {"prepare": prepare, "encode": encode, "rank": rank}[parser.parse_args().action]()
