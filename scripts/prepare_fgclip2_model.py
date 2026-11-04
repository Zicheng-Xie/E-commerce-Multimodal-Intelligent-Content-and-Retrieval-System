"""Download a pinned public FG-CLIP 2 checkpoint into this project's cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import httpx

import run  # noqa: F401
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

REPOSITORY = "qihoo360/fg-clip2-base"
REVISION = "430fbc8a912c86fd4de601381b6245a0edab22f0"
MODEL_ROOT = PROJECT_ROOT / ".cache/open_source_models/fg-clip2-base" / REVISION
OUTPUT = PROJECT_ROOT / "artifacts/module2_fgclip2_comparison"
NAMES = (
    "README.md",
    "config.json",
    "configuration_fgclip2.py",
    "modeling_fgclip2.py",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
)


def download() -> None:
    """Apply the download workflow step.

    Notes:
        Writes derived artifacts through MODEL_ROOT.mkdir, OUTPUT.mkdir, write_json.
    """
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata_path = OUTPUT / "model_metadata.json"
    with httpx.Client(follow_redirects=True, timeout=90) as client:
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            response = client.get(
                f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}",
                params={"blobs": "true"},
            )
            response.raise_for_status()
            metadata = response.json()
            assert metadata["sha"] == REVISION
            write_json(metadata_path, metadata)
        siblings = {r["rfilename"]: r for r in metadata["siblings"]}
        files = {}
        for name in NAMES:
            source = siblings[name]
            size = source["size"]
            assert 0 < size <= 1_600_000_000
            expected = source.get("lfs", {}).get("sha256")
            target = MODEL_ROOT / name
            if target.exists():
                digest = sha256_file(target)
                if target.stat().st_size != size or (expected and digest != expected):
                    raise RuntimeError(f"Preserve mismatching existing file: {name}")
                files[name] = digest
                continue
            partial = target.with_suffix(target.suffix + ".partial")
            if partial.exists():
                raise RuntimeError(
                    f"Preserve incomplete download for inspection: {name}"
                )
            started = time.perf_counter()
            hasher = hashlib.sha256()
            count, milestone = 0, 0
            with client.stream(
                "GET", f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
            ) as response:
                response.raise_for_status()
                with partial.open("xb") as handle:
                    for chunk in response.iter_bytes(4 * 1024 * 1024):
                        count += len(chunk)
                        if count > size:
                            raise RuntimeError(f"File exceeds published size: {name}")
                        hasher.update(chunk)
                        handle.write(chunk)
                        if count // (128 * 1024 * 1024) > milestone:
                            milestone = count // (128 * 1024 * 1024)
                            print(
                                json.dumps(
                                    {"file": name, "bytes": count, "total": size}
                                ),
                                flush=True,
                            )
            actual = hasher.hexdigest()
            if count != size or (expected and actual != expected):
                raise RuntimeError(f"File size or published LFS SHA mismatch: {name}")
            partial.rename(target)
            files[name] = actual
            with (OUTPUT / "download_events.jsonl").open("a", encoding="utf-8") as log:
                log.write(
                    json.dumps(
                        {
                            "file": name,
                            "bytes": count,
                            "sha256": actual,
                            "seconds": time.perf_counter() - started,
                        }
                    )
                    + "\n"
                )
            print(json.dumps({"downloaded": name, "bytes": count}), flush=True)
        write_json(
            OUTPUT / "model_manifest.json",
            dict(
                repository=REPOSITORY,
                revision=REVISION,
                path=MODEL_ROOT.relative_to(PROJECT_ROOT).as_posix(),
                files=files,
                metadata_sha256=sha256_file(metadata_path),
                published_lfs_checksums_verified=True,
                source=f"https://huggingface.co/{REPOSITORY}/tree/{REVISION}",
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["download"])
    parser.parse_args()
    download()
