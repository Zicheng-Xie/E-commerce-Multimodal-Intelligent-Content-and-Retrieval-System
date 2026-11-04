"""Fetch the pinned official GroundingDINO checkpoint into the project cache."""

from __future__ import annotations

import json
import time

import httpx

import run  # noqa: F401
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

REPOSITORY = "IDEA-Research/grounding-dino-tiny"
REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
ROOT = PROJECT_ROOT / "artifacts/module2_subject_roi"
MODEL_ROOT = PROJECT_ROOT / ".cache/open_source_models/grounding-dino-tiny" / REVISION
NAMES = [
    "README.md",
    "added_tokens.json",
    "config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "model.safetensors",
]


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        RuntimeError: Download exceeds published size; Changed cached model.

    Notes:
        Writes derived artifacts through MODEL_ROOT.mkdir, ROOT.mkdir, write_json.
    """
    ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        meta_path = ROOT / "detector_metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            response = client.get(
                f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}",
                params={"blobs": "true"},
            )
            response.raise_for_status()
            metadata = response.json()
            assert metadata["sha"] == REVISION
            write_json(meta_path, metadata)
        siblings = {r["rfilename"]: r for r in metadata["siblings"]}
        files = {}
        for name in NAMES:
            info = siblings[name]
            target = MODEL_ROOT / name
            expected = info.get("lfs", {}).get("sha256")
            if not target.exists():
                partial = target.with_suffix(target.suffix + ".partial")
                if partial.exists():
                    raise FileExistsError(f"Preserve incomplete download: {name}")
                start = time.perf_counter()
                size = 0
                with client.stream(
                    "GET",
                    f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}",
                ) as response:
                    response.raise_for_status()
                    with partial.open("xb") as output:
                        for chunk in response.iter_bytes(4 * 1024 * 1024):
                            size += len(chunk)
                            if size > info["size"]:
                                raise RuntimeError("Download exceeds published size")
                            output.write(chunk)
                if size != info["size"] or (
                    expected and sha256_file(partial) != expected
                ):
                    raise RuntimeError(f"Published size/SHA mismatch: {name}")
                partial.rename(target)
                with (ROOT / "download_events.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(
                        json.dumps(
                            dict(
                                file=name,
                                bytes=size,
                                seconds=time.perf_counter() - start,
                            )
                        )
                        + "\n"
                    )
                print(json.dumps(dict(downloaded=name, bytes=size)), flush=True)
            if target.stat().st_size != info["size"] or (
                expected and sha256_file(target) != expected
            ):
                raise RuntimeError("Changed cached model")
            files[name] = sha256_file(target)
    write_json(
        ROOT / "detector_manifest.json",
        dict(
            repository=REPOSITORY,
            revision=REVISION,
            path=MODEL_ROOT.relative_to(PROJECT_ROOT).as_posix(),
            files=files,
            published_lfs_checksums_verified=True,
            metadata_sha256=sha256_file(meta_path),
        ),
    )


if __name__ == "__main__":
    main()
