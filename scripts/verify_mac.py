"""Run actual local generation and image/text embedding after model downloads."""

from __future__ import annotations

import json
import time

import numpy as np

import run  # noqa: F401 - establishes project paths and runtime caches
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.pair_reranker import PairReranker
from multimodal_ecommerce.safety import PROJECT_ROOT


def main() -> None:
    """Run local model checks and save a separate runtime report."""
    folder = PROJECT_ROOT / "artifacts/mac_migration"
    folder.mkdir(exist_ok=True)
    image = next((PROJECT_ROOT / "data/stage2_taxonomy_v2/images").glob("*"))
    report: dict[str, object] = {
        "scope": "Mac runtime smoke check; not a quality benchmark",
        "image": str(image),
    }
    client = OllamaClient(timeout=600)
    report["ollama"] = client.health()
    report["text"] = client.chat("请用中文说一句你好。")["message"]["content"]
    report["vision"] = client.chat(
        "简要描述图片中的物品。不要推测品牌、价格或材质。", image_path=image
    )["message"]["content"]
    client.close()
    embedder = QwenEmbedder()
    try:
        for name, vector in [
            ("text", embedder.encode(text="衣服", query=True)),
            ("image", embedder.encode(image=image)),
        ]:
            report["embedding_" + name] = {
                "shape": list(vector.shape),
                "finite": bool(np.isfinite(vector).all()),
                "norm": float(np.linalg.norm(vector)),
            }
        report["embedding_runtime"] = embedder.diagnostics()
    finally:
        embedder.close()
    reranker = PairReranker()
    try:
        scores = reranker.score(
            [{"id": "smoke", "image_path": str(image), "text": "商品图片"}],
            query_text="衣服",
        )
        report["reranker"] = {
            "scores": scores.tolist(),
            "finite": bool(np.isfinite(scores).all()),
            "runtime": reranker.diagnostics(),
        }
    finally:
        reranker.close()
    path = folder / f"smoke_{time.time_ns()}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps({"success": True, "report": str(path)}, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
