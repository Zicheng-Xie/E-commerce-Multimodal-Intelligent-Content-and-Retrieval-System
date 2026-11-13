"""Verify the cheap development experiment and the sealed next-acceptance media."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.holdout_data import is_exposed
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.simple_baseline import (
    human_accuracy,
    rerank_head,
    sample_audit,
    stable_order,
)
from simple_baseline_cycle import DATA, ROOT, guard
from simple_cycle_holdout import EXPOSURE

FINAL = PROJECT_ROOT / "artifacts/module2_final_acceptance_20261105"


def load(path: Path) -> Any:
    """Decode a verification input without previewing protected media.

    Args:
        path: Existing local JSON receipt or metadata.

    Returns:
        Decoded JSON at this serialization boundary.
    """
    return json.loads(path.read_text())


def verify_reservation() -> dict:
    """Check frozen identities, family/image isolation and blocked inference.

    Returns:
        Reservation counts and current readiness; no image semantics or metrics.
    """
    release = load(FINAL / "release_receipt.json")
    for name, digest in release["files"].items():
        if sha256_file(FINAL / name) != digest:
            raise ValueError(f"Frozen reservation metadata changed: {name}")
    manifest = load(FINAL / "frozen_media_manifest.json")
    for name, digest in manifest["files"].items():
        if sha256_file(FINAL / name) != digest:
            raise ValueError(f"Frozen acceptance media changed: {name}")
    prior = load(EXPOSURE)
    families = set(prior["family_keys"])
    sources = set(prior["product_source_ids"])
    signatures = list(prior["image_fingerprints"])
    gallery = load(FINAL / "gallery.reserved.json")
    queries = load(FINAL / "image_queries.reserved.json")
    for row in gallery + queries:
        if row["source_id"] in sources or set(row["family_keys"]) & families:
            raise ValueError("Reserved identity/family overlap")
        if is_exposed(row["fingerprints"], signatures, 4):
            raise ValueError("Reserved byte, decoded-pixel or near-image overlap")
        families.update(row["family_keys"])
        sources.add(row["source_id"])
        signatures.append(row["fingerprints"])
    if (
        len(gallery) != 300
        or len(queries) != 30
        or manifest["model_predictions"] != 0
        or manifest["model_inference_allowed"] is not False
        or release["formal_acceptance_ready"] is not False
    ):
        raise ValueError("Reservation counts or readiness changed")
    registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
    for purpose in ["development", "training", "acceptance_inference"]:
        try:
            require_data_access([FINAL / "gallery.reserved.json"], purpose, registry)
        except PermissionError:
            logging.getLogger(__name__).debug("Reservation access correctly rejected")
        else:
            raise ValueError("Protected reservation access was allowed")
    return dict(
        gallery=300,
        image_queries=30,
        byte_pixel_dhash_family_source_disjoint=True,
        image_integrity_verified=len(manifest["files"]),
        model_predictions=0,
        semantic_previews=0,
        formal_acceptance_ready=False,
    )


def verify_human_packet() -> dict:
    """Check the fixed 20% sample and import only actual returned human answers.

    Returns:
        Audit completion and honest sample agreement status.
    """
    folder = ROOT / "human_audit"
    manifest = load(folder / "sampling_manifest.json")
    if sha256_file(folder / "sample.private.json") != manifest["sample_sha256"]:
        raise ValueError("Frozen human sample changed")
    for name, digest in manifest["source_sha256"].items():
        if sha256_file(PROJECT_ROOT / name) != digest:
            raise ValueError("AI judgments changed after sampling")
    tasks = load(folder / "sample.private.json")
    expected: set[tuple[str, str, str]] = set()
    for cohort in ["tune", "regression", "fresh"]:
        rows = [
            json.loads(line)
            for line in (DATA / cohort / "judgments.ai.jsonl").read_text().splitlines()
            if line
        ]
        expected.update(
            (cohort, r["query_id"], r["candidate_id"]) for r in sample_audit(rows)
        )
    if (
        expected != {(r["cohort"], r["query_id"], r["candidate_id"]) for r in tasks}
        or len(tasks) != 2520
    ):
        raise ValueError("The exact 20% human sample changed")
    answers: list[dict] = []
    for path in [
        folder / "human_answers.csv",
        *sorted((folder / "answers").glob("*.csv")),
    ]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            answers.extend(csv.DictReader(handle))
    return human_accuracy(tasks, answers)


def verify_predictions() -> dict:
    """Rebuild every ranking from frozen vectors and complete real Top20 scores.

    Returns:
        Independently replayed ranking, vector and actual pair-score counts.

    Raises:
        ValueError: A vector, cached model identity or full ranking changed.
    """
    receipt = load(ROOT / "encoding_receipt.json")
    for name, digest in receipt["files"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Encoded feature integrity failed")
    inputs = load(ROOT / "inputs.json")
    count, pairs = 0, 0
    for cohort, info in inputs["cohorts"].items():
        saved = load(ROOT / cohort / "predictions.json")
        ids = info["gallery"]
        banks = {}
        for kind in ["whole", "crop"]:
            banks[kind] = np.stack(
                [
                    np.load(
                        ROOT
                        / "vectors"
                        / f"{hashlib.sha256(cid.encode()).hexdigest()[:24]}_{kind}.npy",
                        allow_pickle=False,
                    )
                    for cid in ids
                ]
            )
        if set(saved) != {q["id"] for q in info["queries"]}:
            raise ValueError("Incomplete image-query predictions")
        for query in info["queries"]:
            vector = np.load(
                ROOT / "vectors" / f"{cohort}_{query['id']}.npy", allow_pickle=False
            )
            whole = stable_order(ids, banks["whole"] @ vector)
            crop = stable_order(ids, banks["crop"] @ vector)
            cache = load(ROOT / "pairs" / cohort / (query["id"] + ".json"))
            if cache["identity"] != dict(
                query_sha256=query["source_sha256"],
                candidate_ids=crop[:20],
                protocol_sha256=sha256_file(ROOT / "protocol.json"),
            ):
                raise ValueError("Precision score cache provenance changed")
            expected = dict(
                whole_cosine=whole,
                crop_cosine=crop,
                crop_top20_complete_reranker=rerank_head(crop, cache["scores"]),
            )
            if saved[query["id"]] != expected:
                raise ValueError("Ranking replay differs from frozen results")
            count += len(expected)
            pairs += len(cache["scores"])
    if count != 72 or pairs != 480 or len(receipt["files"]) != 846:
        raise ValueError("Expected 72 rankings, 480 pairs and 846 vectors")
    return dict(rankings=count, complete_real_precision_scores=pairs, vectors=846)


def main() -> None:
    """Verify current frozen sources, development paths and protected reservations."""
    guard()
    registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
    require_data_access(
        [DATA / c / "catalog.public.json" for c in ["tune", "regression", "fresh"]],
        "development",
        registry,
    )
    reservation = verify_reservation()
    human = verify_human_packet()
    result: dict[str, Any] = dict(
        frozen_development_sources_verified=True,
        reservation=reservation,
        human_audit=human,
        model_state=load(ROOT / "status.json"),
        formal_metrics=None,
    )
    complete = result["model_state"]["phase"] == "model_ablation_completed"
    if complete:
        result["prediction_replay"] = verify_predictions()
    write_json(
        ROOT / ("verification.json" if complete else "verification.pre_results.json"),
        result,
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
