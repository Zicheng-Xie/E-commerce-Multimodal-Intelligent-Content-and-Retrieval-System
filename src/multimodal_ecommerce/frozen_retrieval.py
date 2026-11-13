"""Freeze unlabeled rankings once, then score reviewed labels without a model."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.embedding import QwenEmbedder, model_configuration
from multimodal_ecommerce.evaluation import score_rankings, validate_qrels, verify_files
from multimodal_ecommerce.retrieval import VectorIndex
from multimodal_ecommerce.safety import ensure_output_path, sha256_file, write_json


def load_kit(kit: Path) -> tuple[dict, list[dict], list[dict]]:
    """Check frozen evaluation inputs before inference or score calculation.

    Args:
        kit: Kit used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    manifest = json.loads((kit / "manifest.json").read_text(encoding="utf-8"))
    verify_files(kit, manifest)
    return (
        manifest,
        read_rows(kit / "queries.jsonl"),
        json.loads((kit / "catalog.json").read_text(encoding="utf-8")),
    )


def freeze_predictions(embedder: QwenEmbedder, kit: Path, output: Path) -> dict:
    """Retain every candidate, query vector, score and source/configuration hash.

    Args:
        embedder: Embedder used by this operation.
        kit: Kit used by this operation.
        output: Output used by this operation.

    Returns:
        Mapping with fields: queries, candidate_pairs, query_order, target,
        category_filter, kit, kit_manifest_sha256, index_manifest_sha256, model,
        diagnostics, quality_metrics, scope.

    Raises:
        FileExistsError: Use a new prediction directory.
        ValueError: Index changed since evaluation preparation; Embedding configuration
            differs from frozen index; Candidate IDs disagree; Query image changed.
    """
    output = ensure_output_path(output)
    if output.exists():
        raise FileExistsError("Use a new prediction directory")
    manifest, queries, catalog = load_kit(kit)
    index_root = Path(manifest["index"])
    if sha256_file(index_root / "manifest.json") != manifest["index_manifest_sha256"]:
        raise ValueError("Index changed since evaluation preparation")
    settings = json.loads((index_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        settings["model"] != model_configuration()
        or settings["model"] != embedder.config
    ):
        raise ValueError("Embedding configuration differs from frozen index")
    index = VectorIndex.load(index_root)
    if {row["id"] for row in catalog} != {row["id"] for row in index.records}:
        raise ValueError("Candidate IDs disagree")
    output.mkdir(parents=True)
    vectors, rankings, timings = [], {}, []
    for query in queries:
        image = Path(query["image_path"]) if query["image_path"] else None
        if image and sha256_file(image) != query["image_sha256"]:
            raise ValueError("Query image changed")
        started = time.perf_counter()
        vector = embedder.encode(text=query["text"], image=image, query=True)
        encoded = time.perf_counter()
        rows = index.search(
            vector, top_k=len(catalog), target="image", exclude_id=query["exclude_id"]
        )
        ended = time.perf_counter()
        vectors.append(vector)
        rankings[query["id"]] = [
            {
                key: row[key]
                for key in ("rank", "id", "score", "image_score", "text_score")
            }
            for row in rows
        ]
        timings.append(
            {
                "query_id": query["id"],
                "embedding_seconds": encoded - started,
                "search_seconds": ended - encoded,
                "total_seconds": ended - started,
            }
        )
        print(f"Frozen query {len(vectors)}/{len(queries)}: {query['id']}", flush=True)
    np.savez(output / "query_vectors.npz", vectors=np.stack(vectors))
    write_json(output / "rankings.json", rankings)
    write_json(output / "timings.json", timings)
    report = {
        "queries": len(queries),
        "candidate_pairs": sum(len(rows) for rows in rankings.values()),
        "query_order": [query["id"] for query in queries],
        "target": "image",
        "category_filter": None,
        "kit": str(kit.resolve()),
        "kit_manifest_sha256": sha256_file(kit / "manifest.json"),
        "index_manifest_sha256": sha256_file(index_root / "manifest.json"),
        "model": embedder.config,
        "diagnostics": embedder.diagnostics(),
        "quality_metrics": None,
        "scope": "Frozen before human labels. Complete non-self rankings; raw cosine, no threshold, category filtering or reranking. Timings include first-query warmup effects and are not the warmed performance benchmark.",
    }
    write_json(output / "run.json", report)
    write_json(
        output / "manifest.json",
        {
            "files": {
                path.name: sha256_file(path)
                for path in output.iterdir()
                if path.is_file()
            }
        },
    )
    return report


def score_predictions(predictions: Path, judgments: Path) -> dict:
    """Score fixed predictions only after all non-self candidates are reviewed.

    Args:
        predictions: Predictions used by this operation.
        judgments: Judgments used by this operation.

    Returns:
        Mapping with fields: metrics, predictions_manifest_sha256, judgments_sha256,
        target, model_inference_performed, scope.

    Raises:
        ValueError: Evaluation kit changed after prediction freeze; Prediction query IDs
            disagree; Predictions must cover each non-self candidate exactly once;
            Invalid saved ranking order.
    """
    verify_files(
        predictions,
        json.loads((predictions / "manifest.json").read_text(encoding="utf-8")),
    )
    run = json.loads((predictions / "run.json").read_text(encoding="utf-8"))
    kit = Path(run["kit"])
    if sha256_file(kit / "manifest.json") != run["kit_manifest_sha256"]:
        raise ValueError("Evaluation kit changed after prediction freeze")
    _, queries, catalog = load_kit(kit)
    raw = json.loads((predictions / "rankings.json").read_text(encoding="utf-8"))
    product_ids = {row["id"] for row in catalog}
    if set(raw) != {row["id"] for row in queries}:
        raise ValueError("Prediction query IDs disagree")
    rankings = {}
    for query in queries:
        rows = raw[query["id"]]
        ids = [row["id"] for row in rows]
        expected = product_ids - {query.get("exclude_id")}
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError(
                "Predictions must cover each non-self candidate exactly once"
            )
        if [row["rank"] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError("Invalid saved ranking order")
        rankings[query["id"]] = ids
    relevant = validate_qrels(queries, product_ids, read_rows(judgments))
    return {
        "metrics": score_rankings(queries, relevant, rankings),
        "predictions_manifest_sha256": sha256_file(predictions / "manifest.json"),
        "judgments_sha256": sha256_file(judgments),
        "target": run["target"],
        "model_inference_performed": False,
        "scope": "Same frozen predictions for every reviewer; human judgments must be exhaustive. No quality score is inferred from unreviewed entries.",
    }
