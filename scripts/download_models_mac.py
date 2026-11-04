"""Download the project's pinned external checkpoints on Apple Silicon."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import get_hf_file_metadata, hf_hub_url, snapshot_download


def main() -> None:
    """Download pinned checkpoints and verify their recorded hashes."""
    ROOT = Path(__file__).resolve().parents[1]
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface-mac"))

    for name in ("embedding_model.json", "reranker_model.json", "dino_model.json"):
        config = json.loads((ROOT / "configs" / name).read_text())
        target = ROOT / ".models" / config["repository"].split("/")[-1]
        snapshot_download(
            config["repository"], revision=config["revision"], local_dir=target
        )
        # LFS safetensors use SHA256 ETags; verify the full pinned download as well.
        for weight in target.glob("*.safetensors"):
            metadata = get_hf_file_metadata(
                hf_hub_url(
                    config["repository"], weight.name, revision=config["revision"]
                )
            )
            if len(metadata.etag or "") == 64:
                with weight.open("rb") as handle:
                    actual = hashlib.file_digest(handle, "sha256").hexdigest()
                if actual != metadata.etag:
                    raise ValueError(f"Downloaded weight checksum mismatch: {weight}")
        for filename, expected in config.get("files", {}).items():
            with (target / filename).open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Checkpoint checksum mismatch: {target / filename}")
        print(f"Verified {target}", flush=True)


if __name__ == "__main__":
    main()
