"""Build and search module-two regional indexes independently of old baselines."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.regional_retrieval import (
    STRATEGIES,
    RegionalIndex,
    RegionalRetrievalEncodeViewsInputs,
    RegionalSearchRequest,
    build_regional_index,
    encode_views,
    write_regional_gallery,
)
from multimodal_ecommerce.safety import PROJECT_ROOT, ensure_output_path, write_json


def main() -> None:
    """Expose switchable attention, explicit metadata preferences and image ROI.

    Raises:
        FileExistsError: Use a new search output directory; Use a new index directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "search"])
    parser.add_argument(
        "--index", type=Path, default=Path("data/index_regions_products_v1")
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/stage2_taxonomy_v2/products.jsonl")
    )
    parser.add_argument("--text")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--roi", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--strategy", choices=STRATEGIES, default="attention_fused")
    parser.add_argument(
        "--query-views",
        choices=["global", "multi"],
        default="multi",
        help="Image query global-only versus multiview; ROI always encodes cropped multiviews",
    )
    parser.add_argument("--category")
    parser.add_argument(
        "--attributes",
        type=Path,
        help="JSON: explicit field names to desired string values",
    )
    parser.add_argument("--exclude-id", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/region_search_001")
    )
    args = parser.parse_args()
    index: RegionalIndex | None = None
    texts: np.ndarray | None = None
    manifest: dict = {}
    index_load = 0.0
    if args.command == "search":
        if (args.text is None) == (args.image is None):
            parser.error("Provide exactly one of --text or --image")
        if args.roi and not args.image:
            parser.error("--roi requires --image")
        output = ensure_output_path(PROJECT_ROOT / args.output)
        if output.exists():
            raise FileExistsError("Use a new search output directory")
        load_started = time.perf_counter()
        index, texts, manifest = RegionalIndex.load(PROJECT_ROOT / args.index)
        index_load = time.perf_counter() - load_started
    else:
        output = ensure_output_path(PROJECT_ROOT / args.index)
        if output.exists():
            raise FileExistsError("Use a new index directory")
    subprocess.run(
        ["ollama", "stop", "qwen3.5:4b-q4_K_M"], check=True, capture_output=True
    )
    embedder = QwenEmbedder()
    try:
        if args.command == "build":
            result = build_regional_index(embedder, PROJECT_ROOT / args.dataset, output)
            print(
                json.dumps(
                    {k: result[k] for k in ("count", "view_count", "build_seconds")},
                    indent=2,
                )
            )
            return
        assert index is not None
        output.mkdir(parents=True)
        started = time.perf_counter()
        if args.image and args.query_views == "global" and args.roi is None:
            query = embedder.encode(image=args.image.resolve(), query=True).reshape(
                1, -1
            )
            views: list = []
        elif args.image:
            query, views = encode_views(
                RegionalRetrievalEncodeViewsInputs(
                    embedder,
                    args.image.resolve(),
                    output / "query_views",
                    index.config,
                    query=True,
                    roi=args.roi,
                )
            )
        else:
            query = embedder.encode(text=args.text, query=True).reshape(1, -1)
            views = []
        attributes = (
            json.loads(args.attributes.read_text(encoding="utf-8-sig"))
            if args.attributes
            else {}
        )
        results = index.search(
            RegionalSearchRequest(
                query,
                args.top_k,
                strategy=args.strategy,
                text_vector=query[0] if texts is not None else None,
                text_bank=texts,
                category=args.category,
                attributes=attributes,
                exclude_ids=set(args.exclude_id),
            )
        )
        result = {
            "query": args.text or str(args.image),
            "query_image": str(args.image.resolve()) if args.image else None,
            "roi": args.roi,
            "query_views_mode": (
                "multi"
                if args.image and (args.roi or args.query_views == "multi")
                else "global"
            ),
            "query_views": views,
            "strategy": args.strategy,
            "explicit_category": args.category,
            "explicit_attributes": attributes,
            "model": manifest["model"],
            "regions": index.config,
            "results": results,
            "query_and_search_seconds_excluding_model_and_index_load": time.perf_counter()
            - started,
            "index_load_seconds_including_hash_verification": index_load,
            "diagnostics": embedder.diagnostics(),
        }
        write_json(output / "results.json", result)
        write_regional_gallery(output / "results.html", [result])
        print(
            json.dumps(
                {"output": str(output), "top_ids": [r["id"] for r in results]},
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        embedder.close()


if __name__ == "__main__":
    main()
