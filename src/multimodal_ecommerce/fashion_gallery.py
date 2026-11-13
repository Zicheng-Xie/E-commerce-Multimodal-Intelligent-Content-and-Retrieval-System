"""Resumable real image embeddings for the authorized DeepFashion2 training split."""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.safety import PROJECT_ROOT, ensure_output_path, sha256_file
from multimodal_ecommerce.selected_runtime_io import atomic_json, digest, load

TRAIN = PROJECT_ROOT / "Alibaba阿里算法实习/服饰分割/train"
OUTPUT = PROJECT_ROOT / "data/module2_fashion_gallery_20261112"
CATEGORIES = (
    "短袖上衣",
    "长袖上衣",
    "短袖外套",
    "长袖外套",
    "背心",
    "吊带上衣",
    "短裤",
    "长裤",
    "半身裙",
    "短袖连衣裙",
    "长袖连衣裙",
    "无袖连衣裙",
    "吊带连衣裙",
)


def training_paths(root: Path) -> list[Path]:
    """Enumerate only the named training directory, never validation/test images.

    Args:
        root: Explicit DeepFashion2 training root.

    Returns:
        Sorted original training paths without metadata resource-fork files.
    """
    folder = root / "image"
    if root.is_symlink() or folder.is_symlink():
        raise ValueError("Training directory must not redirect to another split")
    paths = sorted(
        p for p in folder.glob("*.jpg") if p.stem.isdigit() and len(p.stem) == 6
    )
    if any(
        p.is_symlink() or not p.resolve().is_relative_to(folder.resolve())
        for p in paths
    ):
        raise ValueError("Training images must not escape the explicit source root")
    return paths


def public_title(annotation: dict) -> str:
    """Use native category names as metadata, without inventing a retail title.

    Args:
        annotation: One permitted training image's native annotation.

    Returns:
        Explicit dataset-photo caption using the native categories only.
    """
    kinds = sorted(
        {
            item["category_id"]
            for key, item in annotation.items()
            if key.startswith("item")
            and isinstance(item, dict)
            and type(item.get("category_id")) is int
            and 1 <= item["category_id"] <= 13
        }
    )
    return "DeepFashion2 训练图片 · " + (
        "、".join(CATEGORIES[k - 1] for k in kinds) or "服饰"
    )


def consumed_exclusions() -> tuple[set[str], list[dict]]:
    """Read existing exclusion registries without media, labels or test answers.

    Returns:
        Consumed source IDs and recorded image fingerprints to reject.
    """
    registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
    ids, signatures = set(), []
    for name in registry["exclusion_metadata_allowlist"]:
        if not name.endswith("records.jsonl"):
            continue
        path = PROJECT_ROOT / name
        if not path.exists():
            continue
        require_data_access([path], "exclusion_metadata", registry)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            ids.add(str(row.get("id", "")))
            if row.get("image_sha256"):
                ids.add("sha256/" + row["image_sha256"])
            pair = str(row.get("pair_id", ""))
            if (
                "deepfashion" in str(row.get("dataset", "")).lower()
                and pair.isdigit()
                and int(pair) > 0
            ):
                ids.add("deepfashion/pair/" + pair)
            signature = row.get("fingerprints")
            if (
                isinstance(signature, dict)
                and {"sha256", "pixels", "dhash"} <= signature.keys()
            ):
                signatures.append(signature)
    return ids, signatures


class FashionGallery:
    """Commit verified vector shards; interrupted uncommitted work is retried."""

    def __init__(
        self,
        identity: dict,
        protected: list[dict],
        existing_hashes: set[str],
        output: Path = OUTPUT,
        source: Path = TRAIN,
    ) -> None:
        """Bind original training paths, exact model identity and exposure exclusions.

        Args:
            identity: Actual fixed embedding model/preprocessing identity.
            protected: Final-reservation and consumed image signatures.
            existing_hashes: Images already present in the live gallery.
            output: Owned derived index directory, never the source directory.
            source: Explicit training root; validation/test are not accepted.
        """
        if source.name != "train":
            raise ValueError("Only the training split is authorized for expansion")
        self.output, self.source = ensure_output_path(output), source
        self.paths = training_paths(source)
        if not self.paths:
            raise ValueError("No local DeepFashion2 training images found")
        self.blocked_ids, consumed = consumed_exclusions()
        self.protected = protected + consumed
        self.known_hashes = set(existing_hashes)
        self.cursor = 0
        self.shards: list[dict] = []
        self.failed = self.excluded = self.ready = 0
        self.state = "indexing"
        self.error: str | None = None
        self.started = time.time()
        self.output.mkdir(parents=True, exist_ok=True)
        self.policy = dict(
            role="training_images_for_live_retrieval_not_independent_acceptance",
            source=str(source.relative_to(PROJECT_ROOT)),
            files=digest([p.name for p in self.paths]),
            total=len(self.paths),
            model=identity,
            preprocessing="original_rgb_qwen_document_embedding",
            official_validation_used=False,
            fashionai_test_a_used=False,
        )
        path = self.output / "policy.json"
        if path.exists() and load(path) != self.policy:
            raise ValueError(
                "Expansion model or source inventory changed; preserve the previous index"
            )
        atomic_json(path, self.policy)

    def restore(self) -> list[tuple[list[dict], np.ndarray]]:
        """Load only atomically committed, hash-verified complete vector shards.

        Returns:
            Previously committed public records and actual vectors.
        """
        restored = []
        for path in sorted((self.output / "shards").glob("*.json")):
            row = load(path)
            previous = (
                self.shards[-1]["commit_id"] if self.shards else digest(self.policy)
            )
            if (
                row["previous"] != previous
                or digest({k: v for k, v in row.items() if k != "commit_id"})
                != row["commit_id"]
            ):
                raise ValueError("Expansion commit chain changed")
            vectors_path = path.with_suffix(".npz")
            if sha256_file(vectors_path) != row["vectors_sha256"]:
                raise ValueError("Expansion vectors changed")
            with np.load(vectors_path, allow_pickle=False) as bank:
                vectors = bank["images"].copy()
            self._validate(row["records"], vectors)
            for record in row["records"]:
                image = PROJECT_ROOT / record["image_path"]
                if record["id"].replace(
                    "deepfashion2/", "deepfashion/"
                ) in self.blocked_ids or any(
                    record["image_sha256"] == p["sha256"] for p in self.protected
                ):
                    raise PermissionError(
                        "Restored image now belongs to a protected set"
                    )
                source_id = record["id"].rsplit("/", 1)[-1]
                annotation = load(self.source / "annos" / (source_id + ".json"))
                if "sha256/" + record["image_sha256"] in self.blocked_ids or (
                    "deepfashion/pair/" + str(annotation.get("pair_id"))
                    in self.blocked_ids
                ):
                    raise PermissionError(
                        "Restored source family belongs to consumed data"
                    )
                if sha256_file(image) != record["image_sha256"]:
                    raise ValueError("Expansion image changed")
                self.known_hashes.add(record["image_sha256"])
            self.shards.append(row)
            self.cursor = row["next_cursor"]
            self.failed += row["failed"]
            self.excluded += row["excluded"]
            self.ready += len(row["records"])
            restored.append((row["records"], vectors))
        return restored

    def _validate(self, records: list[dict], vectors: np.ndarray) -> None:
        if (
            vectors.shape != (len(records), 2048)
            or not np.isfinite(vectors).all()
            or (
                len(vectors)
                and not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3)
            )
        ):
            raise ValueError("Real finite normalized image vectors required")
        if any(
            not (PROJECT_ROOT / r["image_path"])
            .resolve()
            .is_relative_to((self.output / "media").resolve())
            for r in records
        ):
            raise ValueError(
                "Expansion records must reference owned read-only image copies"
            )

    def _prepare(self, source: Path) -> dict | None:
        if f"deepfashion/train/{source.stem}" in self.blocked_ids:
            return None
        annotation_path = self.source / "annos" / (source.stem + ".json")
        if annotation_path.is_symlink() or (self.source / "annos").is_symlink():
            raise ValueError(
                "Native training annotation must not redirect to another dataset"
            )
        annotation = load(annotation_path)
        if "deepfashion/pair/" + str(annotation.get("pair_id")) in self.blocked_ids:
            return None
        checksum = sha256_file(source)
        if "sha256/" + checksum in self.blocked_ids or any(
            checksum == r["sha256"] for r in self.protected
        ):
            return None
        signature = fingerprints(source)
        if signature["sha256"] in self.known_hashes or is_exposed(
            signature, self.protected, 4
        ):
            return None
        with Image.open(source) as image:
            if (
                image.width * image.height > 20_000_000
                or getattr(image, "n_frames", 1) != 1
            ):
                raise ValueError("Invalid single image size")
        destination = ensure_output_path(self.output / "media" / (source.stem + ".jpg"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
        if sha256_file(destination) != signature["sha256"]:
            raise ValueError("Derived image differs from the original training image")
        return dict(
            id="deepfashion2/train/" + source.stem,
            title=public_title(annotation),
            image_path=destination.relative_to(PROJECT_ROOT).as_posix(),
            image_sha256=signature["sha256"],
        )

    def step(
        self, encode: Callable[[list[Path]], np.ndarray], size: int = 32
    ) -> tuple[list[dict], np.ndarray]:
        """Prepare, actually encode and atomically commit one resumable shard.

        Args:
            encode: Real fixed image encoder; tests may inject a synthetic one.
            size: Maximum source images considered in this bounded GPU stage.

        Returns:
            Newly searchable records and vectors; skipped sources are documented.
        """
        if size < 1:
            raise ValueError("Positive shard size required")
        next_cursor = min(self.cursor + size, len(self.paths))
        records: list[dict] = []
        errors: list[dict] = []
        excluded = 0
        for source in self.paths[self.cursor : next_cursor]:
            try:
                row = self._prepare(source)
                if row is None or row["image_sha256"] in {
                    r["image_sha256"] for r in records
                }:
                    excluded += 1
                else:
                    records.append(row)
            except (ValueError, OSError) as error:
                logging.getLogger(__name__).warning(
                    "Training image excluded: %s", source, exc_info=True
                )
                errors.append(dict(source=source.stem, error=str(error)))
        vectors = (
            encode([PROJECT_ROOT / r["image_path"] for r in records])
            if records
            else np.empty((0, 2048), np.float32)
        )
        self._validate(records, vectors)
        path = self.output / "shards" / f"{len(self.shards):06}.npz"
        path.parent.mkdir(exist_ok=True)
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, images=vectors)
        temporary.replace(path)
        row = dict(
            previous=(
                self.shards[-1]["commit_id"] if self.shards else digest(self.policy)
            ),
            records=records,
            vectors_sha256=sha256_file(path),
            next_cursor=next_cursor,
            excluded=excluded,
            failed=len(errors),
            errors=errors,
        )
        row["commit_id"] = digest(row)
        atomic_json(path.with_suffix(".json"), row)
        self.shards.append(row)
        self.cursor = next_cursor
        self.ready += len(records)
        self.excluded += excluded
        self.failed += len(errors)
        self.known_hashes.update(r["image_sha256"] for r in records)
        self.publish()
        return records, vectors

    def publish(self) -> dict:
        """Publish honest counts; pending source images are never called searchable.

        Returns:
            Public progress metadata without any relevance judgments.
        """
        if self.cursor == len(self.paths):
            self.state = "complete"
        value = dict(
            state=self.state,
            target_training_images=len(self.paths),
            indexed=self.ready,
            processed=self.cursor,
            pending=len(self.paths) - self.cursor,
            excluded=self.excluded,
            failed=self.failed,
            error=self.error,
            elapsed_seconds=time.time() - self.started,
            validation_images_used=0,
            fashionai_test_a_images_used=0,
            formal_metrics=None,
        )
        atomic_json(self.output / "progress.json", value)
        return value
