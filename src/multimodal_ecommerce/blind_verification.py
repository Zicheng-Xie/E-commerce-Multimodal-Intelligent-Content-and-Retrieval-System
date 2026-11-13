"""Recompute frozen rankings without reading relevance truth or re-encoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from multimodal_ecommerce.blind_evaluation import (
    SUITE,
    exclusive_json,
    now,
    read_json,
    validate_rankings,
    verify_files,
    verify_recipe,
)
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.postranking import selected_scores
from multimodal_ecommerce.regional_retrieval import RegionalIndex
from multimodal_ecommerce.retrieval_strategies import ordered_indices
from multimodal_ecommerce.safety import sha256_file


def replay_predictions(bundle: Path = SUITE) -> dict:
    """Validate values and ordering of every saved method/query, with no qrels.

    Args:
        bundle: Bundle used by this operation.

    Returns:
        Mapping with fields: verified_at, passed, quality_truth_parsed, queries,
        strategies, complete_rankings, max_absolute_score_error,
        prediction_receipt_sha256, verification_implementation_sha256.

    Raises:
        ValueError: Recipe and prediction receipt differ; Dataset and prediction receipt
            differ; Invalid saved query vector bank; Query vectors are not normalized;
            Replayed ranking order differs; Replayed score values differ.
    """
    recipe = verify_recipe(bundle)
    prediction = bundle / "predictions"
    receipt = read_json(bundle / "prediction_receipt.json")
    if receipt["recipe_sha256"] != sha256_file(bundle / "recipe.json"):
        raise ValueError("Recipe and prediction receipt differ")
    if receipt["dataset_manifest_sha256"] != sha256_file(
        bundle / "dataset/manifest.json"
    ):
        raise ValueError("Dataset and prediction receipt differ")
    verify_files(prediction, receipt["files"])
    verify_files(
        bundle / "dataset", read_json(bundle / "dataset/manifest.json")["files"]
    )
    index, _, _ = RegionalIndex.load(prediction / "index")
    ids = [row["id"] for row in index.records]
    rows = read_rows(bundle / "dataset/public/queries/products.jsonl")
    vectors = np.load(prediction / "query_vectors.npy", allow_pickle=False)
    if (
        vectors.shape != (len(rows), index.vectors.shape[-1])
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Invalid saved query vector bank")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError("Query vectors are not normalized")
    rankings = read_json(prediction / "rankings.json")
    validate_rankings(rankings, {r["id"] for r in rows}, set(ids), recipe["methods"])
    lookup = {(r["query_id"], r["strategy"]): r for r in rankings}
    max_error = 0.0
    for row, query in zip(rows, vectors):
        for method in recipe["methods"]:
            scores = (
                index.vectors[:, 0] @ query
                if method == "global"
                else selected_scores(index, query, method)[0]
            )
            order = ordered_indices(scores, ids)
            saved = lookup[(row["id"], method)]
            expected = [ids[i] for i in order]
            if expected != saved["ranked_ids"]:
                raise ValueError("Replayed ranking order differs")
            max_error = max(
                max_error, float(np.max(np.abs(scores[order] - saved["scores"])))
            )
    if max_error > 2e-6:
        raise ValueError("Replayed score values differ")
    result = {
        "verified_at": now(),
        "passed": True,
        "quality_truth_parsed": False,
        "queries": len(rows),
        "strategies": len(recipe["methods"]),
        "complete_rankings": len(rankings),
        "max_absolute_score_error": max_error,
        "prediction_receipt_sha256": sha256_file(bundle / "prediction_receipt.json"),
        "verification_implementation_sha256": sha256_file(Path(__file__)),
    }
    exclusive_json(bundle / "prediction_replay_verification.json", result)
    return result
