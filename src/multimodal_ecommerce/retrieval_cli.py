"""CLI for building and querying the genuine multimodal Qwen index."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from multimodal_ecommerce.embedding import QwenEmbedder, model_configuration
from multimodal_ecommerce.retrieval import VectorIndex, build_index
from multimodal_ecommerce.retrieval_eval import run_diagnostics, write_gallery
from multimodal_ecommerce.safety import PROJECT_ROOT, write_json


def main() -> None:
    """Run index construction, one search, or reproducible diagnostics.

    Raises:
        ValueError: Model/preprocessing config changed; rebuild the index.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "search", "check"])
    parser.add_argument("--index", type=Path, default=Path("data/index_qwen_v1"))
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/localized/products.jsonl")
    )
    parser.add_argument("--text")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--target", choices=["image", "text", "fused"], default="image")
    parser.add_argument("--category")
    parser.add_argument("--exclude-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = PROJECT_ROOT / args.index
    if args.command == "search" and ((args.text is None) == (args.image is None)):
        parser.error("search requires exactly one of --text or --image")
    if args.command != "build":
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["model"] != model_configuration():
            raise ValueError("Model/preprocessing config changed; rebuild the index")
    # Release only this project's generation model, not unrelated Ollama models.
    subprocess.run(
        ["ollama", "stop", "qwen3.5:4b-q4_K_M"], check=True, capture_output=True
    )
    embedder = QwenEmbedder()
    try:
        if args.command == "build":
            result = build_index(embedder, PROJECT_ROOT / args.dataset, directory)
            print(
                json.dumps(
                    {
                        key: result[key]
                        for key in ("count", "dimension", "build_seconds")
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "check":
            result = run_diagnostics(
                embedder,
                directory,
                PROJECT_ROOT / (args.output or Path("artifacts/retrieval_v1")),
            )
            print(
                json.dumps(
                    {
                        "indexed_products": result["indexed_products"],
                        "paired_diagnostic": result["paired_diagnostic"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            started = time.perf_counter()
            vector = embedder.encode(text=args.text, image=args.image, query=True)
            results = VectorIndex.load(directory).search(
                vector,
                args.top_k,
                target=args.target,
                category=args.category,
                exclude_id=args.exclude_id,
            )
            output = PROJECT_ROOT / (
                args.output or Path("artifacts/search_latest.json")
            )
            result = {
                "query": args.text or str(args.image),
                "query_image": str(args.image.resolve()) if args.image else None,
                "target": args.target,
                "category_filter": args.category,
                "exclude_id": args.exclude_id,
                "search_seconds_excluding_model_load": time.perf_counter() - started,
                "results": results,
                "diagnostics": embedder.diagnostics(),
            }
            write_json(output, result)
            write_gallery(output.with_suffix(".html"), [result])
            print(
                json.dumps(
                    {"query": result["query"], "results": results},
                    ensure_ascii=False,
                    indent=2,
                )
            )
    finally:
        embedder.close()
