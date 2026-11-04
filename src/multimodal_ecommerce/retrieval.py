"""Exact cosine retrieval with separate real image and text vectors."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.runtime_device import relocate_assets
from multimodal_ecommerce.safety import ensure_output_path, sha256_file, write_json


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Normalize finite, nonzero vectors and reject malformed index arrays.

    Args:
        vectors: Vectors used by this operation.

    Returns:
        Computed array in the order of the supplied records.

    Raises:
        ValueError: Expected a finite two-dimensional vector matrix; Zero vectors cannot
            be indexed.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or 0 in vectors.shape or not np.isfinite(vectors).all():
        raise ValueError("Expected a finite two-dimensional vector matrix")
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(lengths < 1e-8):
        raise ValueError("Zero vectors cannot be indexed")
    return vectors / lengths


class VectorIndex:
    """Small exact index, with filtering and explicit similarity scores."""

    def __init__(
        self, records: list[dict], images: np.ndarray, texts: np.ndarray
    ) -> None:
        """Initialize the component and validate its configured dependencies.

        Args:
            records: Records used by this operation.
            images: Images used by this operation.
            texts: Texts used by this operation.

        Raises:
            ValueError: Index vectors and metadata disagree; Duplicate index IDs.
        """
        self.records = records
        self.images = unit_rows(images)
        self.texts = unit_rows(texts)
        if self.images.shape != self.texts.shape or len(records) != len(images):
            raise ValueError("Index vectors and metadata disagree")
        if len({r["id"] for r in records}) != len(records):
            raise ValueError("Duplicate index IDs")

    def search(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        *,
        target: str = "image",
        category: str | None = None,
        exclude_id: str | None = None,
    ) -> list[dict]:
        """Rank by cosine similarity; category is an explicit filter, not a label
        boost.

        Args:
            vector: Vector used by this operation.
            top_k: Top k used by this operation.
            target: Target used by this operation.
            category: Category used by this operation.
            exclude_id: Exclude id used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.

        Raises:
            ValueError: Invalid top_k or target; Query dimension does not match index.
        """
        if top_k < 1 or target not in {"image", "text", "fused"}:
            raise ValueError("Invalid top_k or target")
        query = unit_rows(np.asarray(vector).reshape(1, -1))[0]
        if query.shape[0] != self.images.shape[1]:
            raise ValueError("Query dimension does not match index")
        image_scores = self.images @ query
        text_scores = self.texts @ query
        scores = {
            "image": image_scores,
            "text": text_scores,
            "fused": (image_scores + text_scores) / 2,
        }[target]
        candidates = [
            i
            for i, record in enumerate(self.records)
            if record["id"] != exclude_id
            and (category is None or record["category"] == category)
        ]
        ranked = sorted(
            candidates, key=lambda i: (-float(scores[i]), self.records[i]["id"])
        )[:top_k]
        return [
            {
                "rank": rank,
                **self.records[i],
                "score": float(scores[i]),
                "image_score": float(image_scores[i]),
                "text_score": float(text_scores[i]),
            }
            for rank, i in enumerate(ranked, 1)
        ]

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        """Load NumPy arrays without pickle and check their recorded checksum.

        Args:
            directory: Directory used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.

        Raises:
            ValueError: Index checksum manifest is incomplete; Index checksum mismatch.
        """
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if set(manifest.get("files", {})) != {"vectors.npz", "records.json"}:
            raise ValueError("Index checksum manifest is incomplete")
        for filename, checksum in manifest["files"].items():
            path = (directory / filename).resolve()
            if (
                not path.is_relative_to(directory.resolve())
                or sha256_file(path) != checksum
            ):
                raise ValueError("Index checksum mismatch")
        records = relocate_assets(
            json.loads((directory / "records.json").read_text(encoding="utf-8"))
        )
        with np.load(directory / "vectors.npz", allow_pickle=False) as archive:
            return cls(records, archive["images"], archive["texts"])


def build_index(embedder: QwenEmbedder, dataset: Path, output: Path) -> dict[str, Any]:
    """Encode valid local products and save an auditable immutable index.

    Args:
        embedder: Embedder used by this operation.
        dataset: Dataset used by this operation.
        output: Output used by this operation.

    Returns:
        Mapping with fields: model, count, dimension, dataset_sha256, build_seconds,
        files, default_target, fused_formula, diagnostics.

    Raises:
        ValueError: No eligible images in the dataset.
    """
    output = ensure_output_path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a fresh index directory: {output}")
    source = [
        json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()
    ]
    records = []
    images, texts = [], []
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    for row in source:
        if not row.get("image_path") or not row.get("evaluation_eligible"):
            continue
        image = (dataset.parent / row["image_path"]).resolve()
        if (
            not image.is_relative_to(dataset.parent.resolve())
            or sha256_file(image) != row["image_sha256"]
        ):
            raise ValueError(f"Changed or escaped image: {row['id']}")
        images.append(embedder.encode(image=image))
        texts.append(embedder.encode(text=row["text"]))
        records.append(
            {
                "id": row["id"],
                "split": row["split"],
                "category": row["category"],
                "title": row["text"],
                "image_path": str(image),
                "image_sha256": row["image_sha256"],
            }
        )
        print(f"Indexed {len(records)}: {row['id']}", flush=True)
    if not records:
        raise ValueError("No eligible images in the dataset")
    index = VectorIndex(records, np.stack(images), np.stack(texts))
    np.savez(output / "vectors.npz", images=index.images, texts=index.texts)
    write_json(output / "records.json", records)
    manifest = {
        "model": embedder.config,
        "count": len(records),
        "dimension": 2048,
        "dataset_sha256": sha256_file(dataset),
        "build_seconds": time.perf_counter() - started,
        "files": {
            name: sha256_file(output / name) for name in ("vectors.npz", "records.json")
        },
        "default_target": "image",
        "fused_formula": "0.5 * image_cosine + 0.5 * title_cosine",
        "diagnostics": embedder.diagnostics(),
    }
    write_json(output / "manifest.json", manifest)
    return manifest
