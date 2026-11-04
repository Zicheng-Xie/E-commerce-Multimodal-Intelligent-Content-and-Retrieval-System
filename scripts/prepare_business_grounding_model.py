"""Download the complete official SAM 2.1 Base+ checkpoint; no package changes."""

from __future__ import annotations

import json
import time

import httpx

import run  # noqa: F401
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

ROOT = PROJECT_ROOT / "artifacts/module2_business_grounding_development"
REPOSITORY = "facebook/sam2.1-hiera-base-plus"
REVISION = "b7320756a13354e7530a63935656d35b2f91a290"
CACHE = PROJECT_ROOT / ".cache/open_source_models/sam2.1-hiera-base-plus" / REVISION
FILES = (
    "README.md",
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "model.safetensors",
)


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        ValueError: Official model revision differs; Downloaded model exceeds official
            size; Official checkpoint size/SHA256 mismatch; Cached checkpoint changed;
            Immutable model manifest differs.
        FileExistsError: Preserve partial download for diagnosis.

    Notes:
        Writes derived artifacts through CACHE.mkdir, write_json.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=90) as client:
        meta_path = ROOT / "sam2_metadata.json"
        if not meta_path.exists():
            response = client.get(
                f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}",
                params={"blobs": "true"},
            )
            response.raise_for_status()
            metadata = response.json()
            if metadata["sha"] != REVISION:
                raise ValueError("Official model revision differs")
            write_json(meta_path, metadata)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        siblings = {r["rfilename"]: r for r in metadata["siblings"]}
        hashes = {}
        for name in FILES:
            info = siblings[name]
            target = CACHE / name
            expected = info.get("lfs", {}).get("sha256")
            if not target.exists():
                partial = target.with_suffix(target.suffix + ".partial")
                if partial.exists():
                    raise FileExistsError("Preserve partial download for diagnosis")
                started = time.perf_counter()
                with client.stream(
                    "GET",
                    f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}",
                ) as response:
                    response.raise_for_status()
                    with partial.open("xb") as handle:
                        for block in response.iter_bytes(4 * 1024 * 1024):
                            handle.write(block)
                            if handle.tell() > info["size"]:
                                raise ValueError(
                                    "Downloaded model exceeds official size"
                                )
                if partial.stat().st_size != info["size"] or (
                    expected and sha256_file(partial) != expected
                ):
                    raise ValueError("Official checkpoint size/SHA256 mismatch")
                partial.rename(target)
                print(
                    json.dumps(
                        {
                            "downloaded": name,
                            "bytes": info["size"],
                            "seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
            if target.stat().st_size != info["size"] or (
                expected and sha256_file(target) != expected
            ):
                raise ValueError("Cached checkpoint changed")
            hashes[name] = sha256_file(target)
    value = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "path": CACHE.relative_to(PROJECT_ROOT).as_posix(),
        "files": hashes,
        "metadata_sha256": sha256_file(meta_path),
        "published_LFS_verified": True,
        "full_checkpoint": True,
        "implementation": "transformers==4.57.1 Sam2Model",
        "dtype": "float32",
        "processor": "official pinned defaults",
        "quality_results": None,
    }
    path = ROOT / "sam2_model_manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError("Immutable model manifest differs")
    write_json(path, value)
    print(
        json.dumps({"complete_official_SAM2_available": True, "quality_results": None}),
        flush=True,
    )


if __name__ == "__main__":
    main()
