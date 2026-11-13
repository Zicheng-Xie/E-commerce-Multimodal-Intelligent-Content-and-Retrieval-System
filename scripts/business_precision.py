"""Freeze and run a title/region/keyword business reranking development prototype."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import run  # noqa: F401,E402
from multimodal_ecommerce.blind_evaluation import file_map, verify_files  # noqa: E402
from multimodal_ecommerce.business_retrieval import (  # noqa: E402
    BUSINESS_POLICY,
    BusinessReranker,
    hybrid_scores,
)
from multimodal_ecommerce.dataset_release import read_rows, write_rows  # noqa: E402
from multimodal_ecommerce.holdout_data import fingerprints  # noqa: E402
from multimodal_ecommerce.precision_reranking import refine_pool  # noqa: E402
from multimodal_ecommerce.regional_retrieval import RegionalIndex  # noqa: E402
from multimodal_ecommerce.regional_retrieval import (  # noqa: E402 - local path bootstrap must run first.
    RegionalSearchRequest,
)
from multimodal_ecommerce.retrieval_strategies import ordered_indices  # noqa: E402
from multimodal_ecommerce.safety import (  # noqa: E402
    PROJECT_ROOT,
    sha256_file,
    write_json,
)

OUT = PROJECT_ROOT / "artifacts/module2_precision60/business"
INDEX = PROJECT_ROOT / "data/index_regions_products_v1"
QUERIES = PROJECT_ROOT / "artifacts/evaluation_stage2_v1/queries.jsonl"
OLD_VECTORS = PROJECT_ROOT / "artifacts/stage2_predictions_v1"


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        FileExistsError: Preserve this business prototype release.
        ValueError: Published title vectors are required; Reused query vector order
            mismatch; Reused query embeddings are invalid; Query pixels changed.

    Notes:
        Writes derived artifacts through OUT.mkdir, write_json.
    """
    if OUT.exists():
        raise FileExistsError("Preserve this business prototype release")
    OUT.mkdir()
    code = [
        "scripts/business_precision.py",
        "src/multimodal_ecommerce/business_retrieval.py",
    ]
    parents = [
        INDEX / "manifest.json",
        QUERIES,
        OLD_VECTORS / "manifest.json",
        PROJECT_ROOT / "configs/reranker_model.json",
    ]
    write_json(
        OUT / "protocol.json",
        {
            "date": "2026-10-22",
            "policy": BUSINESS_POLICY,
            "implementation": {p: sha256_file(PROJECT_ROOT / p) for p in code},
            "parents": {
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in parents
            },
            "scope": "Old exposed product development material only; final acceptance relevance definition remains pending user confirmation",
            "quality_metrics": None,
            "formal_acceptance": False,
            "consumed_same_item_blind_used": False,
            "queries": "All 18 old text/image queries retained; legacy category metadata is not a ranking input or validated reporting stratum",
            "catalog": "All 286 corrected product entries; published source titles only; no generated descriptions or query-answer category",
        },
    )
    index, texts, _ = RegionalIndex.load(INDEX)
    if texts is None:
        raise ValueError("Published title vectors are required")
    prior = json.loads((OLD_VECTORS / "manifest.json").read_text(encoding="utf-8"))
    verify_files(OLD_VECTORS, prior["files"])
    queries = read_rows(QUERIES)
    metadata = json.loads((OLD_VECTORS / "run.json").read_text(encoding="utf-8"))
    if metadata["query_order"] != [q["id"] for q in queries]:
        raise ValueError("Reused query vector order mismatch")
    with np.load(OLD_VECTORS / "query_vectors.npz", allow_pickle=False) as bank:
        vectors = bank["vectors"].copy()
    if (
        vectors.shape != (len(queries), 2048)
        or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Reused query embeddings are invalid")
    fingerprints_by_id = {
        r["id"]: fingerprints(Path(r["image_path"])) for r in index.records
    }
    inputs, rankings, allowed_by_query, queries_for_review = [], {}, {}, []
    for number, query in enumerate(queries):
        # Strip evaluator fields before building every ranking and cross-encoder input.
        public = {
            "id": query["id"],
            "text": query.get("text"),
            "image_path": query.get("image_path"),
            "exclude_id": query.get("exclude_id"),
        }
        excluded = {public["exclude_id"]} if public["exclude_id"] else set()
        if public["image_path"]:
            image = Path(public["image_path"])
            if sha256_file(image) != query["image_sha256"]:
                raise ValueError("Query pixels changed")
            signature = fingerprints(image)
            excluded.update(
                k
                for k, v in fingerprints_by_id.items()
                if v["sha256"] == signature["sha256"]
                or v["pixels"] == signature["pixels"]
            )
        included = [i for i, r in enumerate(index.records) if r["id"] not in excluded]
        records = [index.records[i] for i in included]
        ids = [r["id"] for r in records]
        allowed_by_query[public["id"]] = ids
        q = vectors[number]
        global_scores = index.vectors[included, 0] @ q
        regional = index.search(
            RegionalSearchRequest(
                q, top_k=len(index.records), strategy="attention", exclude_ids=excluded
            )
        )
        regional_map = {r["id"]: r["score"] for r in regional}
        visual = np.array([regional_map[k] for k in ids])
        title = texts[included] @ q
        hybrid = hybrid_scores(
            ids,
            visual,
            title,
            text=public["text"],
            titles=[r["title"] for r in records],
        )
        current = {
            "global": [ids[i] for i in ordered_indices(global_scores, ids)],
            "region_title": [
                ids[i] for i in ordered_indices(0.8 * visual + 0.2 * title, ids)
            ],
            "hybrid": [ids[i] for i in ordered_indices(hybrid, ids)],
        }
        rankings[public["id"]] = current
        pool = current["hybrid"][: BUSINESS_POLICY["rerank_pool"]]
        by_id = {r["id"]: r for r in records}
        inputs.append(
            {
                "query_id": public["id"],
                "query_text": public["text"],
                "query_image": public["image_path"],
                "documents": [
                    {
                        "id": k,
                        "image_path": by_id[k]["image_path"],
                        "text": by_id[k]["title"],
                    }
                    for k in pool
                ],
            }
        )
        queries_for_review.append(
            {
                **public,
                "relevance_rule": query["relevance_rule"],
                "legacy_category_unverified": query["category_for_reporting"],
            }
        )
    write_json(OUT / "model_inputs.json", inputs)
    write_json(OUT / "base_rankings.json", rankings)
    write_json(OUT / "allowed_candidates.json", allowed_by_query)
    write_json(
        OUT / "review_context.json",
        {
            "queries": queries_for_review,
            "catalog": [
                {k: r[k] for k in ["id", "title", "image_path"]} for r in index.records
            ],
        },
    )
    write_json(
        OUT / "input_manifest.json",
        {
            "files": file_map(OUT),
            "query_count": len(queries),
            "rerank_pairs": sum(len(r["documents"]) for r in inputs),
            "formal_acceptance": False,
        },
    )
    print(
        json.dumps(
            {
                "queries": len(queries),
                "catalog": len(index.records),
                "full_candidate_relations": sum(map(len, allowed_by_query.values())),
                "rerank_pairs": sum(len(r["documents"]) for r in inputs),
            }
        )
    )


def score() -> None:
    """Apply the score workflow step.

    Raises:
        FileExistsError: Preserve the frozen business predictions.
        ValueError: Business task or method changed.

    Notes:
        Writes derived artifacts through write_json, write_rows.
    """
    if (OUT / "prediction_receipt.json").exists() or (OUT / "rankings.json").exists():
        raise FileExistsError("Preserve the frozen business predictions")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    verify_files(PROJECT_ROOT, protocol["implementation"])
    verify_files(PROJECT_ROOT, protocol["parents"])
    if protocol["policy"] != BUSINESS_POLICY:
        raise ValueError("Business task or method changed")
    manifest = json.loads((OUT / "input_manifest.json").read_text(encoding="utf-8"))
    verify_files(OUT, manifest["files"])
    inputs = json.loads((OUT / "model_inputs.json").read_text(encoding="utf-8"))
    rankings = json.loads((OUT / "base_rankings.json").read_text(encoding="utf-8"))
    cached = []
    model = BusinessReranker()
    started = time.perf_counter()
    try:
        for number, row in enumerate(inputs, 1):
            scores = model.score(
                row["documents"],
                query_image=Path(row["query_image"]) if row["query_image"] else None,
                query_text=row["query_text"],
            )
            pool = [d["id"] for d in row["documents"]]
            base = rankings[row["query_id"]]["hybrid"]
            rankings[row["query_id"]]["business_reranker"] = refine_pool(
                base, pool, scores
            )
            rankings[row["query_id"]]["business_reranker_blend50_anchor2"] = (
                refine_pool(base, pool, scores, blend=0.5, anchors=2)
            )
            cached.append(
                {
                    "query_id": row["query_id"],
                    "candidate_ids": pool,
                    "logits": scores.tolist(),
                }
            )
            print(f"Business prototype {number}/{len(inputs)}", flush=True)
        diagnostics = model.diagnostics()
    finally:
        model.close()
    write_json(OUT / "rankings.json", rankings)
    write_rows(OUT / "pair_scores.jsonl", cached)
    write_json(
        OUT / "runtime.json",
        {
            "seconds": time.perf_counter() - started,
            "diagnostics": diagnostics,
            "e2e_acceptance": False,
        },
    )
    write_json(
        OUT / "prediction_receipt.json",
        {
            "files": file_map(OUT),
            "predictions_complete_before_labels": True,
            "quality_metrics": None,
            "formal_acceptance": False,
        },
    )
    print(
        json.dumps(
            {
                "completed": len(inputs),
                "pairs": sum(len(r["documents"]) for r in inputs),
                "quality_metrics": None,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "score"])
    args = parser.parse_args()
    prepare() if args.action == "prepare" else score()
