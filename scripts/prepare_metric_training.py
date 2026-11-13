"""Prepare and encode a new group-separated module-two training release."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.embedding import QwenEmbedder, model_configuration
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.training_data import DATA_POLICY, prepare_training_data

OUT = PROJECT_ROOT / "artifacts/module2_training"
DATA = PROJECT_ROOT / "data/module2_training"
EXPOSURE = PROJECT_ROOT / "artifacts/module2_blind/consumed_exposure_snapshot.json"


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        FileExistsError: Preserve this release and its protocol.

    Notes:
        Writes derived artifacts through OUT.mkdir, write_json.
    """
    if OUT.exists() or DATA.exists():
        raise FileExistsError("Preserve this release and its protocol")
    exposure = json.loads(EXPOSURE.read_text(encoding="utf-8"))
    verify_files(PROJECT_ROOT, exposure["files"])
    OUT.mkdir(parents=True)
    write_json(
        OUT / "protocol.json",
        {
            "date": "2026-10-22",
            "role": "development_training_only",
            "formal_acceptance": False,
            "data_policy": DATA_POLICY,
            "embedding": model_configuration(),
            "exposure_sha256": sha256_file(EXPOSURE),
            "implementation": {
                p: sha256_file(PROJECT_ROOT / p)
                for p in [
                    "scripts/prepare_metric_training.py",
                    "src/multimodal_ecommerce/training_data.py",
                ]
            },
            "representation": "Pinned full-image Qwen features; learn a lightweight diagonal metric, freeze encoder",
        },
    )
    result = prepare_training_data(DATA, exposure)
    write_json(
        OUT / "data_receipt.json",
        {
            "manifest_sha256": sha256_file(DATA / "manifest.json"),
            "groups": result["groups"],
            "records": result["counts"],
        },
    )
    print(json.dumps(result["groups"]))


def encode() -> None:
    """Encode the configured inputs into model feature representations.

    Raises:
        FileExistsError: Preserve encoded training features.
        ValueError: Pinned embedding configuration changed; Training pixels changed.

    Notes:
        Writes derived artifacts through np.savez_compressed, write_json.
    """
    target = OUT / "features.npz"
    if target.exists():
        raise FileExistsError("Preserve encoded training features")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    verify_files(PROJECT_ROOT, protocol["implementation"])
    if protocol["embedding"] != model_configuration():
        raise ValueError("Pinned embedding configuration changed")
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    verify_files(DATA, manifest["files"])
    inputs = json.loads((DATA / "model_inputs.json").read_text(encoding="utf-8"))
    rows = {r["opaque_id"]: r for r in read_rows(DATA / "records.jsonl")}
    started = time.perf_counter()
    vectors = []
    embedder = QwenEmbedder()
    try:
        for number, row in enumerate(inputs, 1):
            image = Path(row["image_path"])
            if sha256_file(image) != rows[row["id"]]["image_sha256"]:
                raise ValueError("Training pixels changed")
            vectors.append(embedder.encode(image=image, query=row["query"]))
            if number % 50 == 0:
                print(f"Training features {number}/{len(inputs)}", flush=True)
        diagnostics = embedder.diagnostics()
    finally:
        embedder.close()
    np.savez_compressed(
        target, ids=np.array([r["id"] for r in inputs]), vectors=np.asarray(vectors)
    )
    write_json(
        OUT / "feature_manifest.json",
        {
            "files": {"features.npz": sha256_file(target)},
            "dataset_manifest_sha256": sha256_file(DATA / "manifest.json"),
            "records": len(inputs),
            "seconds": time.perf_counter() - started,
            "diagnostics": diagnostics,
        },
    )
    print(
        json.dumps({"encoded": len(inputs), "seconds": time.perf_counter() - started})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "encode"])
    args = parser.parse_args()
    prepare() if args.action == "prepare" else encode()
