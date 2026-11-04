"""Run selected module-two post-ranking strategies on real or saved queries."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import run  # noqa: F401, E402
from multimodal_ecommerce.embedding import QwenEmbedder  # noqa: E402
from multimodal_ecommerce.postranking import (  # noqa: E402 - local path bootstrap must run first.; noqa: E402
    POSTRANK_METHODS,
    PostrankingSearchPostrankInputs,
    search_postrank,
)
from multimodal_ecommerce.regional_retrieval import (  # noqa: E402 - local path bootstrap must run first.; noqa: E402
    RegionalIndex,
    RegionalRetrievalEncodeViewsInputs,
    encode_views,
    write_regional_gallery,
)
from multimodal_ecommerce.safety import (  # noqa: E402
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Use a new query output directory.
        ValueError: Invalid saved query number.

    Notes:
        Writes derived artifacts through output.mkdir, write_json.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=Path, default=Path("data/index_regions_products_v1")
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--image", type=Path)
    source.add_argument(
        "--query-vectors",
        type=Path,
        help="Saved non-pickle NPZ containing vectors[queries,views,dimension] and counts",
    )
    parser.add_argument("--query-number", type=int, default=0)
    parser.add_argument("--roi", type=float, nargs=4)
    parser.add_argument(
        "--strategy", choices=POSTRANK_METHODS, default="reciprocal_attention_fusion"
    )
    parser.add_argument(
        "--title-fusion",
        action="store_true",
        help="Optional genuine title RRF; requires separate product relevance validation",
    )
    parser.add_argument("--exclude-id", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/module2_postrank_search")
    )
    args = parser.parse_args()
    if args.roi is not None and args.image is None:
        parser.error("ROI requires an image query")
    output = ensure_output_path(PROJECT_ROOT / args.output)
    if output.exists():
        raise FileExistsError("Use a new query output directory")
    started = time.perf_counter()
    index, texts, manifest = RegionalIndex.load(PROJECT_ROOT / args.index)
    index_load_seconds = time.perf_counter() - started
    output.mkdir(parents=True)
    embedder = None
    diagnostics = None
    started = time.perf_counter()
    try:
        if args.query_vectors:
            with np.load(args.query_vectors, allow_pickle=False) as archive:
                if not 0 <= args.query_number < len(archive["vectors"]):
                    raise ValueError("Invalid saved query number")
                query = archive["vectors"][
                    args.query_number, : int(archive["counts"][args.query_number])
                ].copy()
            query_description = f"Saved query {args.query_number}"
            query_encoding_seconds = None
        else:
            subprocess.run(
                ["ollama", "stop", "qwen3.5:4b-q4_K_M"], check=True, capture_output=True
            )
            embedder = QwenEmbedder()
            encoding_started = time.perf_counter()
            if args.image:
                query_description = str(args.image)
                if args.roi is None:
                    query = embedder.encode(
                        image=args.image.resolve(), query=True
                    ).reshape(1, -1)
                else:
                    query, _ = encode_views(
                        RegionalRetrievalEncodeViewsInputs(
                            embedder,
                            args.image.resolve(),
                            output / "query_roi",
                            index.config,
                            query=True,
                            roi=args.roi,
                        )
                    )
            else:
                query_description = args.text
                query = embedder.encode(text=args.text, query=True).reshape(1, -1)
            query_encoding_seconds = time.perf_counter() - encoding_started
        results, evidence = search_postrank(
            PostrankingSearchPostrankInputs(
                index,
                query,
                strategy=args.strategy,
                top_k=args.top_k,
                exclude_ids=set(args.exclude_id),
                texts=texts,
                title_fusion=args.title_fusion,
            )
        )
        if embedder:
            diagnostics = embedder.diagnostics()
        result = {
            "query": query_description,
            "query_image": str(args.image.resolve()) if args.image else None,
            "roi": args.roi,
            "strategy": args.strategy + ("_title_rrf" if args.title_fusion else ""),
            "index_manifest_sha256": sha256_file(
                PROJECT_ROOT / args.index / "manifest.json"
            ),
            "model": manifest["model"],
            "results": results,
            "evidence": evidence,
            "index_load_seconds": index_load_seconds,
            "query_encoding_seconds": query_encoding_seconds,
            "openblas_threads_requested": os.environ.get("OPENBLAS_NUM_THREADS"),
            "total_after_index_load_seconds_including_model_load": time.perf_counter()
            - started,
            "diagnostics": diagnostics,
            "formal_quality_evaluation": False,
        }
        write_json(output / "results.json", result)
        write_regional_gallery(output / "results.html", [result])
        print(
            json.dumps(
                {
                    "output": str(output),
                    "candidate_count": evidence["candidate_count"],
                    "scoring_seconds": evidence["scoring_seconds"],
                    "top_ids": [r["id"] for r in results],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        if embedder:
            embedder.close()


if __name__ == "__main__":
    main()
