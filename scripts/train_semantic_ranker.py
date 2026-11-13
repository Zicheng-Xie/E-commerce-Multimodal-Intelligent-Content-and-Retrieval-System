"""Freeze, encode semantic profiles and cross-fit weakly supervised ranking."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import run  # noqa: F401,E402
from multimodal_ecommerce.blind_evaluation import verify_files  # noqa: E402
from multimodal_ecommerce.business_acceptance import product_family_keys  # noqa: E402
from multimodal_ecommerce.dataset_release import read_rows, write_rows  # noqa: E402
from multimodal_ecommerce.embedding import QwenEmbedder  # noqa: E402
from multimodal_ecommerce.precision_reranking import color_descriptor  # noqa: E402
from multimodal_ecommerce.regional_retrieval import RegionalIndex  # noqa: E402
from multimodal_ecommerce.safety import (  # noqa: E402
    PROJECT_ROOT,
    sha256_file,
    write_json,
)
from multimodal_ecommerce.semantic_ranker import (  # noqa: E402 - local path bootstrap must run first.; noqa: E402
    FEATURE_NAMES,
    SemanticRankerFeatureMatrixInputs,
    SemanticRankerFitRankerInputs,
    SemanticRankerTrainingMaskInputs,
    assign_folds,
    feature_matrix,
    fit_ranker,
    machine_topk_bounds,
    predict_ranker,
    ranked_ids,
    training_mask,
)

ROOT = PROJECT_ROOT / "artifacts/module2_machine_assist"
OUT = ROOT / "semantic_ranker"
SOURCE = PROJECT_ROOT / "artifacts/module2_precision60/business"
INDEX = PROJECT_ROOT / "data/index_regions_products_v1"
QUERIES = PROJECT_ROOT / "artifacts/stage2_predictions_v1"


def freeze() -> None:
    """Freeze the inputs and policy used by this workflow.

    Raises:
        FileExistsError: Preserve the ranking protocol.

    Notes:
        Writes derived artifacts through OUT.mkdir, write_json.
    """
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve the ranking protocol")
    OUT.mkdir(parents=True, exist_ok=True)
    parent = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    policy = parent["cross_fit_policy"]
    paths = [
        "scripts/train_semantic_ranker.py",
        "src/multimodal_ecommerce/semantic_ranker.py",
        "src/multimodal_ecommerce/embedding.py",
        "src/multimodal_ecommerce/regional_retrieval.py",
        "src/multimodal_ecommerce/precision_reranking.py",
    ]
    write_json(
        OUT / "protocol.json",
        dict(
            feature_names=FEATURE_NAMES,
            policy=policy,
            models=["balanced_logistic", "hard_negative_ranknet"],
            hard_negative_limit=8,
            semantic_blend={"attention": 0.5, "title": 0.2, "semantic_profile": 0.3},
            source_development_protocol_sha256=sha256_file(ROOT / "protocol.json"),
            implementation={p: sha256_file(PROJECT_ROOT / p) for p in paths},
            parents={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                for p in [
                    INDEX / "manifest.json",
                    QUERIES / "manifest.json",
                    SOURCE / "prediction_receipt.json",
                    PROJECT_ROOT / "data/stage2_taxonomy_v2/products.jsonl",
                    PROJECT_ROOT / "configs/embedding_model.json",
                ]
            },
            quality_source="machine_weak_labels_not_human_business_judgments",
            formal_acceptance=False,
            formal_metrics=None,
            human_labels=0,
            goal_met=False,
            default_promotion_allowed=False,
            cross_fit_definition="Each validation query/gallery-family cell is predicted by a model excluding its query fold and gallery-family fold from scaling, mining and gradients; source families of held-out image queries are excluded from both training roles",
            limitations=[
                "Machine teacher may be biased; teacher agreement is not independent quality evidence",
                "All material is exposed development data; group-masked internal validation is not formal heldout",
                "Full rankings stitch nine fold models; final production candidate trained separately on all known weak labels",
            ],
        ),
    )
    print(
        json.dumps(
            dict(
                frozen=True,
                features=len(FEATURE_NAMES),
                cross_fit_models=18,
                formal_acceptance=False,
            )
        )
    )


def protocol() -> Any:
    """Return protocol for this workflow.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.

    Raises:
        ValueError: Teacher protocol changed.
    """
    p = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    verify_files(PROJECT_ROOT, p["implementation"])
    verify_files(PROJECT_ROOT, p["parents"])
    if sha256_file(ROOT / "protocol.json") != p["source_development_protocol_sha256"]:
        raise ValueError("Teacher protocol changed")
    return p


def source_families(ids: list[str]) -> list:
    """Return source families for this workflow.

    Args:
        ids: Identifiers in the order expected by the associated arrays.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    records = {
        r["id"]: r
        for r in read_rows(PROJECT_ROOT / "data/stage2_taxonomy_v2/products.jsonl")
    }
    parent = {key: key for key in ids}

    def find(key: str) -> str:
        """Return find for this workflow.

        Args:
            key: Key used by this operation.

        Returns:
            Text produced by this operation.
        """
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    owners: dict = {}
    for key in ids:
        row = records[key]
        keys = product_family_keys(row) | {
            "byte:" + row["image_sha256"],
            "pixels:" + row["pixel_sha256"],
        }
        for evidence in keys:
            if evidence in owners:
                left, right = find(key), find(owners[evidence])
                parent[max(left, right)] = min(left, right)
            else:
                owners[evidence] = key
    return [find(key) for key in ids]


def encode() -> None:
    """Encode the configured inputs into model feature representations.

    Raises:
        FileExistsError: Preserve semantic features.
        ValueError: Machine weak labels changed; Original published title vectors are
            required; Cached query order changed.

    Notes:
        Writes derived artifacts through np.savez_compressed, write_json.
    """
    p = protocol()
    if (OUT / "feature_receipt.json").exists():
        raise FileExistsError("Preserve semantic features")
    receipt = json.loads((ROOT / "annotation_receipt.json").read_text(encoding="utf-8"))
    verify_files(ROOT, receipt["files"])
    if sha256_file(ROOT / "weak_labels.jsonl") != receipt["weak_labels_sha256"]:
        raise ValueError("Machine weak labels changed")
    index, titles, _ = RegionalIndex.load(INDEX)
    if titles is None:
        raise ValueError("Original published title vectors are required")
    ids = [r["id"] for r in index.records]
    profiles = {
        row["id"]: row
        for path in (ROOT / "profiles").glob("*.json")
        for row in [json.loads(path.read_text(encoding="utf-8"))]
    }
    texts = [
        "商品主体："
        + profiles[k]["product_type"]
        + "。可见外观："
        + profiles[k]["appearance"]
        + "。"
        for k in ids
    ]
    encoder = QwenEmbedder()
    vectors = []
    try:
        for i, text in enumerate(texts, 1):
            vectors.append(encoder.encode(text=text, query=False))
            if i % 30 == 0:
                print(
                    json.dumps(
                        dict(stage="profile_embeddings", completed=i, total=len(texts))
                    ),
                    flush=True,
                )
        write_json(OUT / "embedding_diagnostics.json", encoder.diagnostics())
    finally:
        encoder.close()
    bank = np.asarray(vectors, dtype=np.float32)
    np.savez_compressed(OUT / "profile_vectors.npz", vectors=bank)
    context = json.loads((ROOT / "development_inputs.json").read_text(encoding="utf-8"))
    queries = context["queries"]
    qids = [q["id"] for q in queries]
    query_manifest = json.loads((QUERIES / "manifest.json").read_text(encoding="utf-8"))
    verify_files(QUERIES, query_manifest["files"])
    metadata = json.loads((QUERIES / "run.json").read_text(encoding="utf-8"))
    if metadata["query_order"] != qids:
        raise ValueError("Cached query order changed")
    with np.load(QUERIES / "query_vectors.npz", allow_pickle=False) as archive:
        qvectors = archive["vectors"].copy()
    colors = np.stack([color_descriptor(Path(r["image_path"])) for r in index.records])
    features = np.stack(
        [
            feature_matrix(
                SemanticRankerFeatureMatrixInputs(
                    index,
                    titles,
                    bank,
                    texts,
                    qvectors[i],
                    text=q["text"],
                    image=Path(q["image_path"]) if q["image_path"] else None,
                    color_bank=colors,
                )
            )
            for i, q in enumerate(queries)
        ]
    )
    allowed = json.loads(
        (SOURCE / "allowed_candidates.json").read_text(encoding="utf-8")
    )
    valid = np.array([[cid in set(allowed[qid]) for cid in ids] for qid in qids])
    weak = {
        (r["query_id"], r["candidate_id"]): r["grade"]
        for r in read_rows(ROOT / "weak_labels.jsonl")
    }
    labels = np.array(
        [
            [np.nan if weak.get((qid, cid)) is None else weak[qid, cid] for cid in ids]
            for qid in qids
        ],
        dtype=float,
    )
    families = source_families(ids)
    gfolds = assign_folds(families, 3, p["policy"]["seed"])
    gallery_folds = np.array([gfolds[g] for g in families])
    qfamilies = []
    id_families = dict(zip(ids, families))
    image_families = {
        sha256_file(Path(r["image_path"])): family
        for r, family in zip(index.records, families)
    }
    for q in queries:
        if q["image_path"]:
            qfamilies.append(
                id_families.get(q["exclude_id"])
                or image_families.get(sha256_file(Path(q["image_path"])))
                or "outside-query:" + sha256_file(Path(q["image_path"]))
            )
        else:
            qfamilies.append(None)
    qfolds = np.zeros(len(queries), dtype=int)
    for mode in ["text", "image"]:
        positions = [
            i for i, q in enumerate(queries) if bool(q["text"]) == (mode == "text")
        ]
        keys = [qfamilies[i] or "text:" + queries[i]["text"] for i in positions]
        folds = assign_folds(keys, 3, p["policy"]["seed"])
        for i, key in zip(positions, keys):
            qfolds[i] = folds[key]
    oldrankings = json.loads((SOURCE / "rankings.json").read_text(encoding="utf-8"))
    base = np.zeros(valid.shape)
    for i, qid in enumerate(qids):
        ranks = {cid: rank for rank, cid in enumerate(oldrankings[qid]["hybrid"])}
        base[i] = [1 / (ranks.get(cid, len(ids)) + 1) for cid in ids]
    np.savez_compressed(
        OUT / "features.npz",
        features=features,
        labels=labels,
        valid=valid,
        query_folds=qfolds,
        gallery_folds=gallery_folds,
        base_scores=base,
        color_bank=colors,
    )
    write_json(
        OUT / "feature_metadata.json",
        dict(
            query_ids=qids,
            gallery_ids=ids,
            gallery_families=families,
            query_families=qfamilies,
            profile_texts=texts,
            feature_names=FEATURE_NAMES,
            query_vectors_reused=True,
            label_values="0/1 or NaN unknown; self/identical candidates excluded",
            human_labels=0,
        ),
    )
    paths = [
        "profile_vectors.npz",
        "features.npz",
        "feature_metadata.json",
        "embedding_diagnostics.json",
    ]
    write_json(
        OUT / "feature_receipt.json",
        dict(
            files={name: sha256_file(OUT / name) for name in paths},
            weak_labels_sha256=receipt["weak_labels_sha256"],
            formal_acceptance=False,
        ),
    )
    print(
        json.dumps(
            dict(
                queries=len(queries),
                gallery=len(ids),
                allowed_relations=int(valid.sum()),
                features=len(FEATURE_NAMES),
                profile_encodings=len(texts),
            )
        )
    )


def fit() -> None:
    """Apply the fit workflow step.

    Raises:
        FileExistsError: Preserve the trained candidate and internal comparisons.
        ValueError: Training weak labels changed; Cross-fit validation coverage
            incomplete.

    Notes:
        Writes derived artifacts through np.savez_compressed, write_json, write_rows.
    """
    p = protocol()
    if (OUT / "training_receipt.json").exists():
        raise FileExistsError("Preserve the trained candidate and internal comparisons")
    receipt = json.loads((OUT / "feature_receipt.json").read_text(encoding="utf-8"))
    verify_files(OUT, receipt["files"])
    if sha256_file(ROOT / "weak_labels.jsonl") != receipt["weak_labels_sha256"]:
        raise ValueError("Training weak labels changed")
    metadata = json.loads((OUT / "feature_metadata.json").read_text(encoding="utf-8"))
    with np.load(OUT / "features.npz", allow_pickle=False) as archive:
        x = archive["features"]
        y = archive["labels"]
        valid = archive["valid"]
        qfolds = archive["query_folds"]
        gfolds = archive["gallery_folds"]
        base = archive["base_scores"]
    models = {}
    audits = []
    predictions = {name: np.full(valid.shape, np.nan) for name in p["models"]}
    for qfold in range(3):
        for gfold in range(3):
            mask, evidence = training_mask(
                SemanticRankerTrainingMaskInputs(
                    metadata["query_ids"],
                    qfolds,
                    metadata["query_families"],
                    metadata["gallery_families"],
                    gfolds,
                    qfold,
                    gfold,
                )
            )
            mask &= valid
            validation = (qfolds[:, None] == qfold) & (gfolds[None, :] == gfold) & valid
            for name in p["models"]:
                model = fit_ranker(
                    SemanticRankerFitRankerInputs(
                        x,
                        y,
                        mask,
                        base,
                        p["policy"],
                        pairwise=name == "hard_negative_ranknet",
                    )
                )
                values = predict_ranker(x, model)
                predictions[name][validation] = values[validation]
                key = f"{name}_q{qfold}_g{gfold}"
                models[key] = model
                audits.append(
                    dict(
                        model=key,
                        **evidence,
                        training_cells=model["training_cells"],
                        training_positive=model["training_positive"],
                        training_negative=model["training_negative"],
                        validation_cells=int(validation.sum()),
                    )
                )
    if any(not np.isfinite(values[valid]).all() for values in predictions.values()):
        raise ValueError("Cross-fit validation coverage incomplete")
    rankings = json.loads((SOURCE / "rankings.json").read_text(encoding="utf-8"))
    for i, qid in enumerate(metadata["query_ids"]):
        positions = np.flatnonzero(valid[i])
        ids = [metadata["gallery_ids"][j] for j in positions]
        rankings[qid]["semantic_profile_blend"] = ranked_ids(
            ids, (0.5 * x[i, :, 1] + 0.2 * x[i, :, 3] + 0.3 * x[i, :, 4])[positions]
        )
        for name, values in predictions.items():
            rankings[qid]["weak_oof_" + name] = ranked_ids(ids, values[i, positions])
    weak = {
        (r["query_id"], r["candidate_id"]): r["grade"]
        for r in read_rows(ROOT / "weak_labels.jsonl")
    }
    proxy = {
        name: machine_topk_bounds(
            {qid: current[name] for qid, current in rankings.items()}, weak
        )
        for name in next(iter(rankings.values()))
    }
    candidates = {
        name: fit_ranker(
            SemanticRankerFitRankerInputs(
                x, y, valid, base, p["policy"], pairwise=name == "hard_negative_ranknet"
            )
        )
        for name in p["models"]
    }
    write_json(OUT / "cross_fit_models.json", models)
    write_json(OUT / "cross_fit_audit.json", audits)
    write_json(OUT / "candidate_models.json", candidates)
    write_json(OUT / "rankings.json", rankings)
    write_json(
        OUT / "machine_agreement_bounds.json",
        dict(
            label_source="machine_weak_development_only",
            not_business_quality_or_acceptance=True,
            methods=proxy,
            formal_metrics=None,
            goal_met=False,
            default_changed=False,
        ),
    )
    np.savez_compressed(OUT / "oof_scores.npz", allow_pickle=False, **predictions)
    priorities = []
    for i, qid in enumerate(metadata["query_ids"]):
        for j, cid in enumerate(metadata["gallery_ids"]):
            if not valid[i, j]:
                continue
            grade = weak[qid, cid]
            score = float(predictions["balanced_logistic"][i, j])
            priority = 2.0 if grade is None else abs(float(grade) - score)
            priorities.append(
                dict(
                    query_id=qid,
                    candidate_id=cid,
                    machine_grade=grade,
                    machine_score=score,
                    priority=priority,
                    role="coordinator_development_triage_not_blind_human_packet",
                    human_label=False,
                )
            )
    write_rows(
        OUT / "machine_review_priorities.jsonl",
        sorted(
            priorities, key=lambda r: (-r["priority"], r["query_id"], r["candidate_id"])
        ),
    )
    paths = [
        "cross_fit_models.json",
        "cross_fit_audit.json",
        "candidate_models.json",
        "rankings.json",
        "machine_agreement_bounds.json",
        "oof_scores.npz",
        "machine_review_priorities.jsonl",
    ]
    write_json(
        OUT / "training_receipt.json",
        dict(
            cross_fit_models=len(models),
            final_candidate_models=len(candidates),
            rankings=sum(len(v) for v in rankings.values()),
            human_labels=0,
            formal_acceptance=False,
            formal_metrics=None,
            goal_met=False,
            default_changed=False,
            files={name: sha256_file(OUT / name) for name in paths},
        ),
    )
    print(
        json.dumps(
            dict(
                cross_fit_models=len(models),
                candidate_models=len(candidates),
                complete_rankings=sum(len(v) for v in rankings.values()),
                human_labels=0,
                formal_acceptance=False,
                goal_met=False,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "encode", "fit"])
    args = parser.parse_args()
    {"freeze": freeze, "encode": encode, "fit": fit}[args.action]()
