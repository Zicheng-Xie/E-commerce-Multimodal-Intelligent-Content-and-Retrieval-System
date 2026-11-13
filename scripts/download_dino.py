"""Fetch a compact pinned official visual checkpoint, preserving other models."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


def main() -> None:
    """Fetch a compact pinned official visual checkpoint, preserving other models."""
    root = Path(__file__).resolve().parents[1]
    meta = json.loads(
        (root / "artifacts/module2_precision/dino_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    target = Path("E:/LLM_for_Ollama/external/dinov2-small").resolve()
    target.mkdir(parents=True, exist_ok=True)
    files = {}
    for row in meta["files"]:
        if row["rfilename"] not in {
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
        }:
            continue
        p = target / row["rfilename"]
        if not p.exists():
            with urllib.request.urlopen(
                f"https://huggingface.co/{meta['repository']}/resolve/{meta['revision']}/{row['rfilename']}?download=true",
                timeout=120,
            ) as response:
                content = response.read()
            if len(content) != row["size"]:
                raise ValueError("Size mismatch")
            if (
                row.get("lfs")
                and hashlib.sha256(content).hexdigest() != row["lfs"]["sha256"]
            ):
                raise ValueError("Weight hash mismatch")
            p.write_bytes(content)
        content = p.read_bytes()
        if (
            row.get("lfs")
            and hashlib.sha256(content).hexdigest() != row["lfs"]["sha256"]
        ):
            raise ValueError("Existing weight differs")
        if (
            not row.get("lfs")
            and hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            != row["blobId"]
        ):
            raise ValueError("Configuration git hash mismatch")
        files[p.name] = hashlib.sha256(content).hexdigest()
        print("Verified", p.name, flush=True)
    config = {
        "repository": meta["repository"],
        "revision": meta["revision"],
        "path": str(target),
        "files": files,
        "dtype": "float32",
        "attn_implementation": "eager",
        "patch_top_k": 64,
        "processor": "pinned model default, EXIF-oriented RGB",
        "region_attention": "last-layer mean CLS-to-patch attention; top64 normalized patch descriptors",
    }
    (root / "configs/dino_model.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
