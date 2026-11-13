"""Run phase-two same-item retrieval and real-gallery performance checks."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.retrieval_benchmark import benchmark
from multimodal_ecommerce.safety import PROJECT_ROOT


def main() -> None:
    """Keep measurement artifacts separate from frozen product benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/mentor_fashion_1000_v1/samples.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/mentor_retrieval_v1")
    )
    args = parser.parse_args()
    subprocess.run(
        ["ollama", "stop", "qwen3.5:4b-q4_K_M"], check=True, capture_output=True
    )
    embedder = QwenEmbedder()
    try:
        result = benchmark(
            embedder, PROJECT_ROOT / args.dataset, PROJECT_ROOT / args.output
        )
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "corpus_records",
                        "same_item",
                        "hard_negatives",
                        "cosine_distributions",
                    )
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        embedder.close()


if __name__ == "__main__":
    main()
