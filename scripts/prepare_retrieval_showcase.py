"""Export the complete allowed development gallery into a public-only demo bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from multimodal_ecommerce.selected_release import verify_release
from multimodal_ecommerce.selected_runtime_io import (
    ContentCache,
    RuntimeAccess,
    atomic_json,
    digest,
    load,
)

ROOT = PROJECT_ROOT / "artifacts/module2_selected_runtime_20261110"
OUTPUT = ROOT / "showcase"
DEVELOPMENT = PROJECT_ROOT / "artifacts/module2_p80_r90_20261106/development_benchmark"


def prepare() -> dict:
    """Copy actual source-bound vectors and public fields; never export judgments.

    Returns:
        Verified public bundle manifest with its input and vector hashes.
    """
    access = RuntimeAccess()
    access.output(OUTPUT)
    inputs: dict = {}

    def read(path: Path, expected: str | None = None) -> Any:
        """Read a permitted development cache with optional frozen checksum.

        Args:
            path: Registered development metadata.
            expected: Frozen input checksum, when supplied.

        Returns:
            Decoded original metadata.
        """
        access.check(path)
        actual = sha256_file(path)
        if expected is not None and actual != expected:
            raise ValueError("Source cache changed: " + str(path))
        inputs[str(path.relative_to(PROJECT_ROOT))] = actual
        return load(path)

    release = verify_release(ROOT / "release.json")
    shared = read(DEVELOPMENT / "shared_gallery.json")
    records = sorted(
        [
            {k: r[k] for k in ("id", "title", "image_path", "image_sha256")}
            for r in shared["catalog"]
        ],
        key=lambda r: r["id"],
    )
    for row in records:
        image = access.check(PROJECT_ROOT / row["image_path"])
        if sha256_file(image) != row["image_sha256"]:
            raise ValueError("Development source image changed")
    folder = DEVELOPMENT / "subject_experiment/simple_crop_top20/index"
    source = read(folder / "manifest.json")
    if source["products"] != [r["id"] for r in records]:
        raise ValueError("Actual whole/crop vectors have another gallery order")
    if sha256_file(access.check(folder / "vectors.npz")) != source["vectors_sha256"]:
        raise ValueError("Actual image/crop vectors changed")
    inputs[str((folder / "vectors.npz").relative_to(PROJECT_ROOT))] = source[
        "vectors_sha256"
    ]
    with np.load(folder / "vectors.npz", allow_pickle=False) as bank:
        whole, crop = bank["whole"].copy(), bank["crop"].copy()
    focus = PROJECT_ROOT / "artifacts/module2_focus_coverage"
    pins = read(focus / "completion_receipt.json")["files"]
    old = read(focus / "index/records.json", pins["index/records.json"])
    old_manifest = read(focus / "index/manifest.json", pins["index/manifest.json"])
    protocol = read(ROOT / "subject_description/protocol.json")
    config = release["model_policy"]["model_configs"]["embedding"]
    for key in (
        "repository",
        "revision",
        "max_pixels",
        "max_length",
        "query_instruction",
        "document_instruction",
    ):
        if old_manifest["model"][key] != config[key]:
            raise ValueError("The title embedding model configuration differs")
    title_path = focus / "index/vectors.npz"
    if sha256_file(access.check(title_path)) != old_manifest["files"]["vectors.npz"]:
        raise ValueError("Actual title vectors changed")
    inputs[str(title_path.relative_to(PROJECT_ROOT))] = sha256_file(title_path)
    catalog = {r["id"]: r for r in records}
    titles = {}
    with np.load(title_path, allow_pickle=False) as bank:
        for row, vector in zip(old, bank["texts"], strict=True):
            if any(row[k] != catalog[row["id"]][k] for k in ("title", "image_sha256")):
                raise ValueError("Cached title has different original input")
            titles[row["id"]] = vector
    cache = ContentCache(
        ROOT / "subject_description/cache",
        dict(
            model_policy=release["model_policy"],
            experimental_sources=protocol["source_sha256"],
        ),
    )
    for row in records:
        if row["id"] not in titles:
            cache_key = cache.key(
                "title_embedding", dict(text=row["title"], query=False)
            )
            value = cache.get(cache_key)
            if value is None:
                raise ValueError("Missing actual title embedding; do not fabricate it")
            titles[row["id"]] = np.asarray(value, dtype=np.float32)
    banks = dict(
        whole=whole, crop=crop, titles=np.stack([titles[r["id"]] for r in records])
    )
    for bank in banks.values():
        if (
            bank.shape != (845, 2048)
            or not np.isfinite(bank).all()
            or not np.allclose(np.linalg.norm(bank, axis=1), 1, atol=1e-3)
        ):
            raise ValueError("845 real normalized 2048-dimensional vectors required")
    primary = read(DEVELOPMENT / "subject_experiment/primary_recall/protocol.json")
    caption_rows = read(
        DEVELOPMENT / "subject_experiment/subject_recall/query_captions.json"
    )["rows"]
    samples = []
    for key, query in shared["query_manifest"].items():
        if query["mode"] == "image":
            samples.append(
                dict(
                    id=key,
                    image_path=query["image_path"],
                    image_sha256=query["image_sha256"],
                    name=caption_rows[key]["value"]["product"],
                )
            )
    output = access.output(OUTPUT / "public_index")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        format="development_showcase_public_v1",
        products=records,
        samples=samples,
        supported_types=primary["candidate_type_support"],
        model_policy=release["model_policy"],
        release_id=release["release_id"],
        inputs_sha256=inputs,
        labels_exported=False,
        formal_metrics=None,
    )
    manifest["content_id"] = digest(manifest)
    path = output / "manifest.json"
    if path.exists():
        prior = load(path)
        if prior["content_id"] != manifest["content_id"]:
            raise ValueError("Preserve existing public bundle when inputs change")
        if sha256_file(output / "vectors.npz") != prior["vectors_sha256"]:
            raise ValueError("Published vectors changed")
        return prior
    np.savez_compressed(output / "vectors.npz", **banks)
    manifest["vectors_sha256"] = sha256_file(output / "vectors.npz")
    manifest["manifest_sha256"] = digest(manifest)
    atomic_json(path, manifest)
    print(
        f"Public-only showcase: {len(records)} products, {len(samples)} image examples."
    )
    return manifest


if __name__ == "__main__":
    prepare()
