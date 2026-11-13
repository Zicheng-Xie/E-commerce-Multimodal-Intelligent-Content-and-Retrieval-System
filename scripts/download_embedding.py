"""Download official, revision-pinned Qwen weights and reference code on E:."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["TEMP"] = str(ROOT / "tmp")
os.environ["TMP"] = str(ROOT / "tmp")

import httpx  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

from multimodal_ecommerce.ollama_client import OllamaClient  # noqa: E402
from multimodal_ecommerce.safety import write_json  # noqa: E402

REPOSITORY = "Qwen/Qwen3-VL-Embedding-2B"
MODEL_ROOT = Path("E:/LLM_for_Ollama/external/Qwen3-VL-Embedding-2B")


def main() -> None:
    """Download weights under the existing model root; code stays in this project.

    Raises:
        RuntimeError: Need 15 GiB free on E: before downloading.
    """
    client = OllamaClient()
    try:
        client.verify_model_storage(Path("E:/LLM_for_Ollama"))
    finally:
        client.close()
    if shutil.disk_usage("E:/").free < 15 * 2**30:
        raise RuntimeError("Need 15 GiB free on E: before downloading")
    manifest_path = ROOT / "configs/embedding_model.json"
    prior = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    with httpx.Client(timeout=60, follow_redirects=True, trust_env=True) as http:
        model_info = http.get(f"https://huggingface.co/api/models/{REPOSITORY}")
        model_info.raise_for_status()
        revision = prior.get("revision") or model_info.json()["sha"]
        upstream = http.get(
            "https://api.github.com/repos/QwenLM/Qwen3-VL-Embedding/commits/main"
        )
        upstream.raise_for_status()
        code_revision = prior.get("code_revision") or upstream.json()["sha"]
        vendor = ROOT / "vendor/qwen3_vl_embedding"
        vendor.mkdir(parents=True, exist_ok=True)
        checksums = {}
        for remote, filename in [
            ("src/models/qwen3_vl_embedding.py", "qwen3_vl_embedding.py"),
            ("LICENSE", "LICENSE"),
        ]:
            url = f"https://raw.githubusercontent.com/QwenLM/Qwen3-VL-Embedding/{code_revision}/{remote}"
            response = http.get(url)
            response.raise_for_status()
            (vendor / filename).write_bytes(response.content)
            checksums[filename] = hashlib.sha256(response.content).hexdigest()
    config = {
        "repository": REPOSITORY,
        "revision": revision,
        "path": str(MODEL_ROOT),
        "code_revision": code_revision,
        "vendor_sha256": checksums,
        "backend": "transformers",
        "dtype": "bfloat16",
        "max_pixels": 262144,
        "max_length": 2048,
        "batch_size": 1,
        "query_instruction": "Retrieve images or text relevant to the user's query.",
        "document_instruction": "Represent the user's input.",
        "source": "https://github.com/QwenLM/Qwen3-VL-Embedding",
    }
    write_json(manifest_path, config)
    snapshot_download(
        REPOSITORY,
        revision=revision,
        local_dir=str(MODEL_ROOT),
        max_workers=2,
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "*.txt",
            "*.jinja",
            "README.md",
            "LICENSE",
        ],
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
