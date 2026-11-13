"""Experimental clothing-image precision reranking, isolated from default search."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import run  # noqa: F401,E402
from multimodal_ecommerce.embedding import QwenEmbedder  # noqa: E402
from multimodal_ecommerce.pair_reranker import PairReranker  # noqa: E402
from multimodal_ecommerce.postranking import search_postrank  # noqa: E402
from multimodal_ecommerce.postranking import (  # noqa: E402 - local path bootstrap must run first.
    PostrankingSearchPostrankInputs,
)
from multimodal_ecommerce.precision_boundary import boundary_refine  # noqa: E402
from multimodal_ecommerce.precision_reranking import refine_pool  # noqa: E402
from multimodal_ecommerce.regional_retrieval import (  # noqa: E402
    RegionalIndex,
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
        FileExistsError: Use a new query output folder.
        ValueError: Consumed blind assets cannot enter development precision search; The
            experimental same-item reranker requires a clothing-only index; Invalid
            cached development query; Changed cached query image; Experimental policy
            needs at least30 allowed candidates.

    Notes:
        Writes derived artifacts through output.mkdir, write_json.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--cached-query-number", type=int)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/module2_regions_benchmark_v1/index"),
    )
    parser.add_argument(
        "--mode", choices=("base", "boundary", "quality"), default="base"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = ensure_output_path(PROJECT_ROOT / args.output)
    if output.exists():
        raise FileExistsError("Use a new query output folder")
    index_path = (PROJECT_ROOT / args.index).resolve()
    if index_path.is_relative_to((PROJECT_ROOT / "artifacts/module2_blind").resolve()):
        raise ValueError(
            "Consumed blind assets cannot enter development precision search"
        )
    started = time.perf_counter()
    timing: dict[str, float | None] = {}
    index, _, _ = RegionalIndex.load(PROJECT_ROOT / args.index)
    timing["index_load_seconds"] = time.perf_counter() - started
    if args.mode != "base" and any(
        row.get("category", "") not in {"", "服饰穿搭"} for row in index.records
    ):
        raise ValueError(
            "The experimental same-item reranker requires a clothing-only index"
        )
    if args.cached_query_number is not None:
        experiment = PROJECT_ROOT / "artifacts/module2_regions_benchmark_v1"
        protocol = json.loads(
            (experiment / "protocol.json").read_text(encoding="utf-8")
        )["source_protocol"]
        number = args.cached_query_number
        if not 0 <= number < len(protocol["queries"]):
            raise ValueError("Invalid cached development query")
        source_row = protocol["queries"][number]["record"]
        image = Path(source_row["image_path"])
        if sha256_file(image) != source_row["image_sha256"]:
            raise ValueError("Changed cached query image")
        with np.load(experiment / "query_vectors.npz", allow_pickle=False) as archive:
            vector = archive["vectors"][number, :1].copy()
        timing["query_encoding_seconds"] = None
    else:
        image = args.image.resolve()
        begin = time.perf_counter()
        embedder = QwenEmbedder()
        timing["embedding_model_load_seconds"] = time.perf_counter() - begin
        try:
            begin = time.perf_counter()
            vector = embedder.encode(image=image, query=True).reshape(1, -1)
            timing["query_encoding_seconds"] = time.perf_counter() - begin
        finally:
            embedder.close()
    image_hash = sha256_file(image)
    excluded = {r["id"] for r in index.records if r["image_sha256"] == image_hash}
    begin = time.perf_counter()
    results, evidence = search_postrank(
        PostrankingSearchPostrankInputs(
            index,
            vector,
            strategy="reciprocal_attention_fusion",
            top_k=len(index.records),
            exclude_ids=excluded,
        )
    )
    timing["baseline_ranking_seconds"] = time.perf_counter() - begin
    ids = [r["id"] for r in results]
    source_order = {r["id"]: r for r in results}
    logits = {}
    if args.mode != "base":
        if len(ids) < 30:
            raise ValueError("Experimental policy needs at least30 allowed candidates")
        query_ids = ids[5:15] if args.mode == "boundary" else ids[:30]
        documents = [
            {"id": key, "image_path": source_order[key]["image_path"]}
            for key in query_ids
        ]
        begin = time.perf_counter()
        reranker = PairReranker()
        timing["reranker_load_and_integrity_seconds"] = time.perf_counter() - begin
        try:
            begin = time.perf_counter()
            values = reranker.score(documents, query_image=image)
            timing["pair_rerank_seconds"] = time.perf_counter() - begin
            diagnostics = reranker.diagnostics()
        finally:
            reranker.close()
        logits = dict(zip(query_ids, values.tolist()))
        if args.mode == "boundary":
            ids = boundary_refine(ids, values, anchors=5, count=10, blend=0.5)
        else:
            ids = refine_pool(ids, ids[:30], values, blend=0.5, anchors=2)
    else:
        diagnostics = None
    final = []
    for rank, key in enumerate(ids[:10], 1):
        final.append(
            {
                **source_order[key],
                "rank": rank,
                "baseline_rank": source_order[key]["rank"],
                "baseline_score": source_order[key]["score"],
                "score": -float(rank),
                "cross_encoder_logit": logits.get(key),
            }
        )
    timing["complete_command_before_export_seconds"] = time.perf_counter() - started
    output.mkdir(parents=True)
    write_json(
        output / "results.json",
        {
            "role": "experimental_clothing_image_same_item_search",
            "formal_acceptance": False,
            "default_production_strategy_changed": False,
            "mode": args.mode,
            "query_image": str(image),
            "query_image_sha256": image_hash,
            "timings": timing,
            "results": final,
            "score_type": "negative final rank; not calibrated relevance probability",
            "diagnostics": diagnostics,
            "baseline_evidence": evidence,
            "limits": "Development benefit below the +2pp precision target; full request latency is not accepted. Cached queries do not measure query encoding.",
        },
    )
    write_regional_gallery(
        output / "results.html",
        [
            {
                "query": str(image),
                "query_image": str(image),
                "strategy": "experimental_" + args.mode,
                "results": final,
            }
        ],
    )
    print(
        json.dumps(
            {"output": str(output), "timings": timing, "formal_acceptance": False},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
