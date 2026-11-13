"""Pin official reranker metadata and reference source; download no weights."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/module2_precision"


def fetch(url: str) -> bytes:
    """Return fetch for this workflow.

    Args:
        url: Url used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": "module2-local-evaluation"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> None:
    """Pin official reranker metadata and reference source; download no weights."""
    OUT.mkdir(parents=True, exist_ok=True)
    model = json.loads(
        fetch("https://huggingface.co/api/models/Qwen/Qwen3-VL-Reranker-2B?blobs=true")
    )
    code = json.loads(
        fetch("https://api.github.com/repos/QwenLM/Qwen3-VL-Embedding/commits/main")
    )["sha"]
    reference = fetch(
        f"https://raw.githubusercontent.com/QwenLM/Qwen3-VL-Embedding/{code}/src/models/qwen3_vl_reranker.py"
    )
    folder = ROOT / "vendor/qwen3_vl_reranker"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "qwen3_vl_reranker.py").write_bytes(reference)
    (folder / "LICENSE").write_bytes(
        fetch(
            f"https://raw.githubusercontent.com/QwenLM/Qwen3-VL-Embedding/{code}/LICENSE"
        )
    )
    metadata = {
        "repository": "Qwen/Qwen3-VL-Reranker-2B",
        "revision": model["sha"],
        "code_revision": code,
        "files": model["siblings"],
    }
    (OUT / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "revision": model["sha"],
                "code_revision": code,
                "weight_bytes": sum(
                    x.get("size", 0)
                    for x in model["siblings"]
                    if x["rfilename"].endswith(".safetensors")
                ),
                "files": [x["rfilename"] for x in model["siblings"]],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
