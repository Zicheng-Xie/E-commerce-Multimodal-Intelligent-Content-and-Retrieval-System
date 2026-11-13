"""Attach verified training image shards to live search without loading test data."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from multimodal_ecommerce.fashion_gallery import FashionGallery
from multimodal_ecommerce.selected_runtime_io import digest
from multimodal_ecommerce.showcase_engine import ROOT, ShowcaseEngine


class ExpandedShowcase(ShowcaseEngine):
    """Share one serial model adapter between live queries and resumable indexing."""

    def __init__(self, root: Path = ROOT, backend: Any = None) -> None:
        """Load the original gallery and only completed fashion vector shards.

        Args:
            root: Original owned public-gallery directory.
            backend: Actual local models, or a synthetic test backend.
        """
        super().__init__(root, backend)
        self.base_count = len(self.ids)
        self.base_index_id = self.manifest["content_id"]
        self.fashion_vectors: list[np.ndarray] = []
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.gallery = FashionGallery(
            self.manifest["model_policy"],
            self.protected,
            set(self.asset_hashes.values()),
        )
        for records, vectors in self.gallery.restore():
            self._attach(records, vectors)
        self.gallery.publish()

    def _attach(self, records: list[dict], vectors: np.ndarray) -> None:
        for row in records:
            if row["id"] in self.docs:
                raise ValueError("An expansion image ID is already indexed")
            self.tokens[row["id"]] = self._asset(row)
            self.ids.append(row["id"])
            self.records.append(row)
            self.docs[row["id"]] = row
        if len(records):
            self.fashion_vectors.append(vectors)
        self.manifest["content_id"] = digest(
            dict(
                base=self.base_index_id,
                last_commit=(
                    self.gallery.shards[-1]["commit_id"]
                    if self.gallery.shards
                    else None
                ),
            )
        )

    def _similarities(
        self, vector: np.ndarray, image: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        whole, titles = super()._similarities(vector, image)
        scores = [vector @ bank.T for bank in self.fashion_vectors]
        # Native fashion category captions are not invented retail title embeddings.
        # New fashion images use the real shared image/text embedding space directly.
        return np.concatenate([whole, *scores]), np.concatenate([titles, *scores])

    def card(self, cid: str, rank: int | None = None) -> dict:
        """Identify dataset photographs separately from original retail products.

        Args:
            cid: Indexed product/image identifier.
            rank: Optional search result position.

        Returns:
            Original card facts plus an explicit source badge.
        """
        return dict(
            super().card(cid, rank),
            source=(
                "DeepFashion2 训练图" if cid.startswith("deepfashion2/") else "电商商品"
            ),
        )

    def indexing_status(self) -> dict:
        """Expose ready/pending counts, never claim that pending vectors are searchable.

        Returns:
            Public progress and actual searchable total.
        """
        return dict(
            state=self.gallery.state,
            products=len(self.ids),
            base_products=self.base_count,
            fashion_indexed=self.gallery.ready,
            target_training_images=len(self.gallery.paths),
            processed=self.gallery.cursor,
            pending=len(self.gallery.paths) - self.gallery.cursor,
            excluded=self.gallery.excluded,
            failed=self.gallery.failed,
            error=self.gallery.error,
            validation_used=False,
            fashionai_test_a_used=False,
        )

    def start_indexing(self) -> None:
        """Start one bounded background worker sharing the query's MPS lock."""
        if self.worker is not None:
            raise RuntimeError("Indexing worker already started")
        self.worker = threading.Thread(
            target=self._build, daemon=True, name="fashion-index"
        )
        self.worker.start()

    def _build(self) -> None:
        def encode(paths: list[Path]) -> np.ndarray:
            """Compute actual fixed image embeddings in bounded batches.

            Args:
                paths: Verified owned training-image copies.

            Returns:
                Real vectors without zero/placeholder substitution.
            """
            encoder = self.models._switch("embedding")
            return np.concatenate(
                [
                    encoder.encode_images(paths[i : i + 4])
                    for i in range(0, len(paths), 4)
                ]
            )

        try:
            while not self.stop_event.is_set() and self.gallery.cursor < len(
                self.gallery.paths
            ):
                with self.lock:
                    try:
                        records, vectors = self.gallery.step(encode)
                        self._attach(records, vectors)
                    finally:
                        self.models.close()
                print(
                    f"服饰图库索引：{self.gallery.ready:,} 已就绪 / {len(self.gallery.paths):,} 训练图片；网页共 {len(self.ids):,} 张可检索。",
                    flush=True,
                )
                self.stop_event.wait(0.25)
        except Exception as error:
            logging.getLogger(__name__).exception(
                "Fashion indexing stopped; committed shards remain usable"
            )
            self.gallery.state = "failed"
            self.gallery.error = str(error)
        finally:
            if self.stop_event.is_set() and self.gallery.cursor < len(
                self.gallery.paths
            ):
                self.gallery.state = "stopped"
            self.gallery.publish()

    def stop_indexing(self) -> None:
        """Stop after the current bounded shard; completed vectors survive restarts."""
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join()
