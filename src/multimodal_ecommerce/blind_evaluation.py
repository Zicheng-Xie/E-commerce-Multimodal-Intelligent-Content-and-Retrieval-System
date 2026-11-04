"""Frozen one-shot evaluation, with development evidence blocked from release."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.embedding import model_configuration
from multimodal_ecommerce.holdout_data import prepare_same_item
from multimodal_ecommerce.measurement import cluster_interval, distribution
from multimodal_ecommerce.postranking import selected_scores
from multimodal_ecommerce.regional_retrieval import (
    RegionalIndex,
    build_regional_index,
    region_configuration,
)
from multimodal_ecommerce.retrieval_strategies import (
    EXPLORATION_POLICY,
    extended_metrics,
    ordered_indices,
)
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)

SUITE = PROJECT_ROOT / "artifacts/module2_blind"
METHODS = ("global", "reciprocal_attention_fusion")
FROZEN_FILES = [
    "src/multimodal_ecommerce/" + name + ".py"
    for name in (
        "blind_evaluation",
        "holdout_data",
        "dataset_release",
        "embedding",
        "regional_retrieval",
        "retrieval_strategies",
        "postranking",
        "retrieval",
        "safety",
        "measurement",
    )
] + [
    "scripts/blind_retrieval.py",
    "scripts/run.py",
    "pyproject.toml",
    "configs/embedding_model.json",
    "configs/retrieval_regions_v1.json",
    "vendor/qwen3_vl_embedding/qwen3_vl_embedding.py",
]


def now() -> str:
    """Return the current UTC time as an ISO-formatted string.

    Returns:
        Text produced by this operation.
    """
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    """Read json.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8-sig"))


def exclusive_json(path: Path, value: dict) -> None:
    """Claim an operation once, even when a later step fails.

    Args:
        path: Path of the input or output artifact used by this operation.
        value: Input value to validate or transform.
    """
    path = ensure_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def file_map(root: Path) -> dict[str, str]:
    """Return file map for this workflow.

    Args:
        root: Root directory used to resolve the operation artifacts.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def verify_files(root: Path, files: dict[str, str]) -> None:
    """Verify files.

    Args:
        root: Root directory used to resolve the operation artifacts.
        files: Files used by this operation.

    Raises:
        ValueError: Empty integrity manifest; Escaped integrity manifest.
    """
    if not files:
        raise ValueError("Empty integrity manifest")
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Escaped integrity manifest")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Frozen file changed: {relative}")


def runtime_versions() -> dict:
    """Return runtime versions for this workflow.

    Returns:
        Mapping with fields: python.
    """
    return {
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in ("numpy", "pillow", "torch", "transformers", "qwen-vl-utils")
        },
    }


def freeze_recipe(bundle: Path = SUITE) -> dict:
    """Pre-register exactly one baseline and one candidate before data inference.

    Args:
        bundle: Bundle used by this operation.

    Returns:
        Mapping with fields: schema, frozen_at, role, formal_module2_acceptance,
        methods, candidate_chosen_on, primary_comparison, secondary,
        no_selection_after_unblinding, query_input, gallery_input, gallery_context,
        data_policy, regions, postrank, model, checkpoint_files, runtime,
        implementation, exposure_sha256, stop_rule, scope_limit.

    Raises:
        ValueError: Checkpoint weights missing.
    """
    bundle = ensure_output_path(bundle)
    exposure = read_json(bundle / "exposure.json")
    verify_files(PROJECT_ROOT, exposure["files"])
    model = model_configuration()
    checkpoint = Path(model["path"])
    model_files = {
        p.relative_to(checkpoint).as_posix(): sha256_file(p)
        for p in sorted(checkpoint.rglob("*"))
        if p.is_file()
        and ".cache" not in p.parts
        and p.suffix in {".json", ".safetensors", ".txt", ".jinja"}
    }
    if not any(name.endswith(".safetensors") for name in model_files):
        raise ValueError("Checkpoint weights missing")
    recipe = {
        "schema": 1,
        "frozen_at": now(),
        "role": "independent_same_item_holdout",
        "formal_module2_acceptance": False,
        "methods": list(METHODS),
        "candidate_chosen_on": "old development set only",
        "primary_comparison": "paired recall@10 and full-ranking mAP; report both directions",
        "secondary": "precision@10, NDCG@10, MRR, Hit@1, R/P/Hit@5/20/50; warm latency",
        "no_selection_after_unblinding": True,
        "query_input": "single original query image, no source crop/box/mask/title/category",
        "gallery_input": "original images plus pixel-derived/fixed regions only",
        "gallery_context": "one current query plus frozen gallery, no other heldout queries",
        "data_policy": {
            "source_split": "validation",
            "scan_limit": 32153,
            "seed": 20261022,
            "category_ids": [1, 2, 7, 8, 11],
            "queries_per_category": 8,
            "gallery_size": 400,
            "max_positive_shop_views": 10,
            "dhash_threshold": 4,
        },
        "regions": region_configuration(),
        "postrank": dict(EXPLORATION_POLICY),
        "model": model,
        "checkpoint_files": model_files,
        "runtime": runtime_versions(),
        "implementation": {p: sha256_file(PROJECT_ROOT / p) for p in FROZEN_FILES},
        "exposure_sha256": sha256_file(bundle / "exposure.json"),
        "stop_rule": "Complete the fixed sample once. No score-driven sampling, retries or retuning.",
        "scope_limit": "Source identity labels, clothing image queries only; not three-category human relevance acceptance",
    }
    exclusive_json(bundle / "recipe.json", recipe)
    exclusive_json(
        bundle / "freeze_receipt.json",
        {"recipe_sha256": sha256_file(bundle / "recipe.json"), "frozen_at": now()},
    )
    return recipe


def verify_recipe(bundle: Path, checkpoint: bool = True) -> dict:
    """Verify recipe.

    Args:
        bundle: Bundle used by this operation.
        checkpoint: Checkpoint used by this operation.

    Returns:
        Result mapping produced by the workflow.

    Raises:
        ValueError: Frozen recipe changed; Frozen comparison changed; Exposure snapshot
            changed; Frozen runtime changed.
    """
    recipe = read_json(bundle / "recipe.json")
    if (
        sha256_file(bundle / "recipe.json")
        != read_json(bundle / "freeze_receipt.json")["recipe_sha256"]
    ):
        raise ValueError("Frozen recipe changed")
    if recipe["methods"] != list(METHODS):
        raise ValueError("Frozen comparison changed")
    verify_files(PROJECT_ROOT, recipe["implementation"])
    if sha256_file(bundle / "exposure.json") != recipe["exposure_sha256"]:
        raise ValueError("Exposure snapshot changed")
    verify_files(PROJECT_ROOT, read_json(bundle / "exposure.json")["files"])
    if runtime_versions() != recipe["runtime"]:
        raise ValueError("Frozen runtime changed")
    if checkpoint:
        verify_files(Path(recipe["model"]["path"]), recipe["checkpoint_files"])
    return recipe


def prepare(bundle: Path = SUITE) -> dict:
    """Prepare the inputs and protocol for this development workflow.

    Args:
        bundle: Bundle used by this operation.

    Returns:
        Result mapping produced by the workflow.

    Notes:
        Writes derived artifacts through write_json.
    """
    recipe = verify_recipe(bundle)
    exclusive_json(bundle / "prepare.claim.json", {"started_at": now()})
    report = prepare_same_item(bundle, recipe, read_json(bundle / "exposure.json"))
    write_json(bundle / "public_status.json", release_status("prepared"))
    return report


def release_status(state: str) -> dict:
    """A shareable progress summary contains no development or substitute scores.

    Args:
        state: State used by this operation.

    Returns:
        Mapping with fields: updated_at, same_item_blind_state,
        formal_module2_acceptance, formal_acceptance_metrics,
        development_metrics_releasable, same_item_scores_are_formal_acceptance.
    """
    return {
        "updated_at": now(),
        "same_item_blind_state": state,
        "formal_module2_acceptance": "pending_independent_human_relevance_labels",
        "formal_acceptance_metrics": None,
        "development_metrics_releasable": False,
        "same_item_scores_are_formal_acceptance": False,
    }


def validate_public(rows: list[dict], root: Path) -> None:
    """Validate public.

    Args:
        rows: Records processed by this operation.
        root: Root directory used to resolve the operation artifacts.

    Raises:
        ValueError: Empty or duplicate public input; Labels or source metadata entered
            model input; Public image changed or escaped.
    """
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate public input")
    allowed = {
        "id",
        "image_path",
        "image_sha256",
        "evaluation_eligible",
        "text",
        "category",
        "attributes",
    }
    for row in rows:
        if (
            set(row) != allowed
            or row["text"] is not None
            or row["category"]
            or row["attributes"]
        ):
            raise ValueError("Labels or source metadata entered model input")
        image = (root / row["image_path"]).resolve()
        if (
            not image.is_relative_to(root.resolve())
            or sha256_file(image) != row["image_sha256"]
        ):
            raise ValueError("Public image changed or escaped")


def predict(bundle: Path = SUITE, embedder_factory: Any = None) -> dict:
    """Freeze all complete rankings before any evaluator truth is parsed.

    Args:
        bundle: Bundle used by this operation.
        embedder_factory: Embedder factory used by this operation.

    Returns:
        Mapping with fields: query_count, gallery_count, truth_read.
    """
    recipe = verify_recipe(bundle)
    dataset = bundle / "dataset"
    verify_files(dataset, read_json(dataset / "manifest.json")["files"])
    gallery = dataset / "public/gallery/products.jsonl"
    query_file = dataset / "public/queries/products.jsonl"
    queries, gallery_rows = read_rows(query_file), read_rows(gallery)
    validate_public(gallery_rows, gallery.parent)
    validate_public(queries, query_file.parent)
    exclusive_json(
        bundle / "predict.claim.json",
        {
            "started_at": now(),
            "recipe_sha256": sha256_file(bundle / "recipe.json"),
            "dataset_manifest_sha256": sha256_file(dataset / "manifest.json"),
        },
    )
    output = ensure_output_path(bundle / "predictions")
    output.mkdir()
    if embedder_factory is None:
        from multimodal_ecommerce.embedding import QwenEmbedder

        embedder_factory = QwenEmbedder
    embedder = embedder_factory()
    try:
        build_regional_index(embedder, gallery, output / "index")
        index, _, _ = RegionalIndex.load(output / "index")
        ids = [r["id"] for r in index.records]
        rankings, encoded, timings = [], [], []
        for number, row in enumerate(queries, 1):
            started = time.perf_counter()
            query = embedder.encode(
                image=query_file.parent / row["image_path"], query=True
            )
            encoding = time.perf_counter() - started
            encoded.append(query)
            for strategy in recipe["methods"]:
                before = time.perf_counter()
                scores = (
                    index.vectors[:, 0] @ query
                    if strategy == "global"
                    else selected_scores(index, query, strategy)[0]
                )
                order = ordered_indices(scores, ids)
                elapsed = time.perf_counter() - before
                rankings.append(
                    {
                        "query_id": row["id"],
                        "strategy": strategy,
                        "ranked_ids": [ids[i] for i in order],
                        "scores": [float(scores[i]) for i in order],
                    }
                )
                timings.append(
                    {
                        "query_id": row["id"],
                        "strategy": strategy,
                        "encoding_seconds": encoding,
                        "ranking_seconds": elapsed,
                        "warm_query_seconds": encoding + elapsed,
                    }
                )
            print(
                f"Blind predictions {number}/{len(queries)}; no quality scores read",
                flush=True,
            )
        np.save(output / "query_vectors.npy", np.stack(encoded), allow_pickle=False)
        write_json(output / "rankings.json", rankings)
        write_json(output / "timings.json", timings)
        write_json(
            output / "runtime.json",
            {"versions": runtime_versions(), "diagnostics": embedder.diagnostics()},
        )
    finally:
        embedder.close()
    receipt = {
        "completed_at": now(),
        "quality_truth_read": False,
        "recipe_sha256": sha256_file(bundle / "recipe.json"),
        "dataset_manifest_sha256": sha256_file(dataset / "manifest.json"),
        "files": file_map(output),
    }
    exclusive_json(bundle / "prediction_receipt.json", receipt)
    write_json(
        bundle / "public_status.json", release_status("predictions_frozen_truth_unread")
    )
    return {"query_count": len(queries), "gallery_count": len(ids), "truth_read": False}


def validate_rankings(
    rankings: list[dict], query_ids: set[str], gallery_ids: set[str], methods: list[str]
) -> None:
    """Validate rankings.

    Args:
        rankings: Candidate rankings keyed by query and retrieval method.
        query_ids: Query ids used by this operation.
        gallery_ids: Gallery ids used by this operation.
        methods: Methods used by this operation.

    Raises:
        ValueError: Missing or duplicate frozen rankings; Incomplete candidate ranking;
            Nonfinite or unsorted scores.
    """
    expected = {(q, s) for q in query_ids for s in methods}
    observed = [(r["query_id"], r["strategy"]) for r in rankings]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Missing or duplicate frozen rankings")
    for row in rankings:
        ids, scores = row["ranked_ids"], row["scores"]
        if (
            len(ids) != len(gallery_ids)
            or set(ids) != gallery_ids
            or len(scores) != len(ids)
        ):
            raise ValueError("Incomplete candidate ranking")
        if not np.isfinite(scores).all() or any(
            a < b for a, b in zip(scores, scores[1:])
        ):
            raise ValueError("Nonfinite or unsorted scores")


def score(bundle: Path = SUITE) -> dict:
    """Unblind once. Every result, including regressions, is retained internally.

    Args:
        bundle: Bundle used by this operation.

    Returns:
        Mapping with fields: role, unblinded_at, formal_module2_acceptance,
        external_release_as_formal_acceptance_allowed, scope, queries, gallery,
        query_groups, selected_after_results, metrics, paired_candidate_minus_baseline,
        per_query, precision_at_10_structural_ceiling, latency, latency_scope,
        uncertainty, future_use, recipe_sha256, dataset_manifest_sha256,
        prediction_receipt_sha256.

    Raises:
        ValueError: Prediction receipt does not match frozen data and recipe; Exposure
            audit failed; Truth coverage mismatch; holdout is consumed.
    """
    recipe = verify_recipe(bundle)
    dataset, predictions = bundle / "dataset", bundle / "predictions"
    receipt = read_json(bundle / "prediction_receipt.json")
    if receipt["recipe_sha256"] != sha256_file(bundle / "recipe.json") or receipt[
        "dataset_manifest_sha256"
    ] != sha256_file(dataset / "manifest.json"):
        raise ValueError("Prediction receipt does not match frozen data and recipe")
    verify_files(dataset, read_json(dataset / "manifest.json")["files"])
    verify_files(predictions, receipt["files"])
    audit = read_json(dataset / "audit.json")
    if not audit["recorded_exposure_audit_passed"]:
        raise ValueError("Exposure audit failed")
    queries = read_rows(dataset / "public/queries/products.jsonl")
    gallery = read_rows(dataset / "public/gallery/products.jsonl")
    rankings = read_json(predictions / "rankings.json")
    validate_rankings(
        rankings,
        {r["id"] for r in queries},
        {r["id"] for r in gallery},
        recipe["methods"],
    )
    exclusive_json(
        bundle / "score.claim.json",
        {
            "unblinded_at": now(),
            "prediction_receipt_sha256": sha256_file(
                bundle / "prediction_receipt.json"
            ),
            "consumed": True,
            "reuse_as_unseen_test_forbidden": True,
        },
    )
    # First truth parse happens only after all predictions are locked and claimed.
    truth = read_json(dataset / "evaluator/qrels.json")
    if len(truth) != len(queries) or {r["query_id"] for r in truth} != {
        r["id"] for r in queries
    }:
        raise ValueError("Truth coverage mismatch; holdout is consumed")
    judgments = {r["query_id"]: r for r in truth}
    per_query = []
    by_method: dict[str, dict[str, dict]] = {s: {} for s in recipe["methods"]}
    for row in rankings:
        label = judgments[row["query_id"]]
        values = extended_metrics(row["ranked_ids"], set(label["relevant_ids"]))
        by_method[row["strategy"]][row["query_id"]] = values
        per_query.append(
            {
                "query_id": row["query_id"],
                "strategy": row["strategy"],
                "pair_group": label["pair_group"],
                "metrics": values,
            }
        )
    order = [r["query_id"] for r in truth]
    groups = [judgments[q]["pair_group"] for q in order]
    keys = list(next(iter(by_method[METHODS[0]].values())))
    summary = {
        s: {
            k: cluster_interval([by_method[s][q][k] for q in order], groups)
            for k in keys
        }
        for s in recipe["methods"]
    }
    differences = {
        k: cluster_interval(
            [by_method[METHODS[1]][q][k] - by_method[METHODS[0]][q][k] for q in order],
            groups,
        )
        for k in keys
    }
    timing = read_json(predictions / "timings.json")
    report = {
        "role": "independent_same_item_holdout_consumed",
        "unblinded_at": now(),
        "formal_module2_acceptance": False,
        "external_release_as_formal_acceptance_allowed": False,
        "scope": recipe["scope_limit"],
        "queries": len(truth),
        "gallery": len(gallery),
        "query_groups": len(set(groups)),
        "selected_after_results": False,
        "metrics": summary,
        "paired_candidate_minus_baseline": differences,
        "per_query": per_query,
        "precision_at_10_structural_ceiling": float(
            np.mean([min(10, len(r["relevant_ids"])) / 10 for r in truth])
        ),
        "latency": {
            s: distribution(
                [r["warm_query_seconds"] for r in timing if r["strategy"] == s]
            )
            for s in recipe["methods"]
        },
        "latency_scope": "Serial warm image encoding plus ranking; excludes model load/index construction/network/UI",
        "uncertainty": "1000 paired group bootstrap draws, exploratory 95% intervals; secondary metrics are descriptive",
        "future_use": "Further tuning belongs to development; this consumed set cannot certify a later strategy",
        "recipe_sha256": sha256_file(bundle / "recipe.json"),
        "dataset_manifest_sha256": sha256_file(dataset / "manifest.json"),
        "prediction_receipt_sha256": sha256_file(bundle / "prediction_receipt.json"),
    }
    write_json(bundle / "internal_report.json", report)
    write_json(bundle / "public_status.json", release_status("unblinded_once_consumed"))
    return report
