"""Real same-item region ablation with the frozen 26-query/334-gallery protocol."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.measurement import cluster_interval, distribution
from multimodal_ecommerce.regional_retrieval import (
    RegionalIndex,
    RegionalRetrievalEncodeViewsInputs,
    RegionalSearchRequest,
    build_regional_index,
    encode_views,
    write_regional_gallery,
)
from multimodal_ecommerce.retrieval_eval import ranking_metrics
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)


def benchmark(embedder: QwenEmbedder, output: Path) -> dict:
    """Strip identity labels out of encoder inputs; label only saved predictions.

    Args:
        embedder: Embedder used by this operation.
        output: Output used by this operation.

    Returns:
        Mapping with fields: scope, gallery_count, query_count, query_groups,
        regions, model, summary, cases, query_multiview_latency, index_build_seconds,
        view_count, diagnostics, formal_acceptance.

    Raises:
        FileExistsError: Use a new experiment output directory.
        ValueError: Historical benchmark changed; Changed benchmark image; Changed query
            image.
    """
    output = ensure_output_path(output)
    if output.exists():
        raise FileExistsError("Use a new experiment output directory")
    prior = PROJECT_ROOT / "artifacts/mentor_retrieval_v1"
    protocol_path = prior / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads((prior / "manifest.json").read_text(encoding="utf-8"))
    for name in ("protocol.json", "records.json"):
        if sha256_file(prior / name) != manifest["files"][name]:
            raise ValueError("Historical benchmark changed")
    old_records = json.loads((prior / "records.json").read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in old_records}
    output.mkdir(parents=True)
    write_json(
        output / "protocol.json",
        {
            "source_sha256": sha256_file(protocol_path),
            "source_protocol": protocol,
            "policy": "Fixed before encoding. Same-item development only; source identities used for scoring only; no ground-truth ROI, class or attribute inputs.",
        },
    )
    source = output / "gallery_source"
    (source / "images").mkdir(parents=True)
    rows = []
    for key in protocol["gallery_ids"]:
        row = by_id[key]
        image = Path(row["image_path"])
        if sha256_file(image) != row["image_sha256"]:
            raise ValueError("Changed benchmark image")
        destination = source / "images" / (key.replace("/", "_") + image.suffix)
        shutil.copyfile(image, destination)
        rows.append(
            {
                "id": key,
                "image_path": destination.relative_to(source).as_posix(),
                "image_sha256": row["image_sha256"],
                "evaluation_eligible": True,
                "text": None,
                "category": "",
                "attributes": {},
            }
        )
    dataset = source / "products.jsonl"
    dataset.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    bank_manifest = build_regional_index(embedder, dataset, output / "index")
    index, _, _ = RegionalIndex.load(output / "index")
    variants = ["global", "mean", "attention"]
    predictions: dict
    metrics, predictions, sections, timings, query_bank = [], {}, [], [], []
    for number, query in enumerate(protocol["queries"], 1):
        row = query["record"]
        path = Path(row["image_path"])
        if sha256_file(path) != row["image_sha256"]:
            raise ValueError("Changed query image")
        started = time.perf_counter()
        vectors, views = encode_views(
            RegionalRetrievalEncodeViewsInputs(
                embedder,
                path,
                output / "queries" / f"{number:02}",
                index.config,
                query=True,
            )
        )
        encoding_seconds = time.perf_counter() - started
        query_bank.append(vectors)
        excluded = {
            item["id"]
            for item in index.records
            if item["image_sha256"] == row["image_sha256"]
        }
        predictions[row["id"]] = {}
        for strategy in variants:
            started = time.perf_counter()
            ranked = index.search(
                RegionalSearchRequest(
                    vectors, len(index.records), strategy=strategy, exclude_ids=excluded
                )
            )
            elapsed = time.perf_counter() - started
            ids = [item["id"] for item in ranked]
            predictions[row["id"]][strategy] = ids
            scores = ranking_metrics(ids, set(query["relevant_ids"]), 10)
            metrics.append(
                {
                    "query_id": row["id"],
                    "pair_group": row["pair_group"],
                    "strategy": strategy,
                    **scores,
                }
            )
            timings.append(
                {
                    "query_id": row["id"],
                    "strategy": strategy,
                    "encode_all_views_seconds": encoding_seconds,
                    "search_preloaded_seconds": elapsed,
                    "combined_region_pipeline_seconds": encoding_seconds + elapsed,
                }
            )
            if number <= 3:
                sections.append(
                    {
                        "query": row["id"],
                        "query_image": str(path),
                        "strategy": strategy,
                        "results": ranked[:10],
                    }
                )
        write_json(output / "queries" / f"{number:02}" / "views.json", views)
        print(
            f"Regional benchmark query {number}/{len(protocol['queries'])}", flush=True
        )
    max_views = index.config["max_views"]
    arrays = np.zeros(
        (len(query_bank), max_views, query_bank[0].shape[1]), dtype=np.float32
    )
    for i, vector in enumerate(query_bank):
        arrays[i, : len(vector)] = vector
    np.savez(
        output / "query_vectors.npz",
        vectors=arrays,
        counts=np.array([len(v) for v in query_bank]),
    )
    write_json(output / "rankings.json", predictions)
    write_json(output / "query_metrics.json", metrics)
    write_json(output / "timings.json", timings)
    write_regional_gallery(output / "comparison.html", sections)
    summary = {}
    for strategy in variants:
        selected = [r for r in metrics if r["strategy"] == strategy]
        summary[strategy] = {
            name: cluster_interval(
                [r[name] for r in selected], [r["pair_group"] for r in selected]
            )
            for name in ("precision@10", "recall@10")
        }
    cases = []
    for query in protocol["queries"]:
        query_id = query["record"]["id"]
        case_metrics = {r["strategy"]: r for r in metrics if r["query_id"] == query_id}
        delta = (
            case_metrics["attention"]["recall@10"] - case_metrics["global"]["recall@10"]
        )
        cases.append(
            {
                "query_id": query_id,
                "recall_delta": delta,
                "global_top10": predictions[query_id]["global"][:10],
                "attention_top10": predictions[query_id]["attention"][:10],
            }
        )
    report = {
        "scope": "Development source-label same-item experiment; not human-labeled three-category product acceptance",
        "gallery_count": len(index.records),
        "query_count": len(protocol["queries"]),
        "query_groups": len({q["record"]["pair_group"] for q in protocol["queries"]}),
        "regions": index.config,
        "model": embedder.config,
        "summary": summary,
        "cases": cases,
        "query_multiview_latency": distribution(
            [
                r["combined_region_pipeline_seconds"]
                for r in timings
                if r["strategy"] == "attention"
            ]
        ),
        "index_build_seconds": bank_manifest["build_seconds"],
        "view_count": bank_manifest["view_count"],
        "diagnostics": embedder.diagnostics(),
        "formal_acceptance": False,
    }
    write_json(output / "report.json", report)
    write_json(
        output / "manifest.json",
        {
            "files": {
                p.relative_to(output).as_posix(): sha256_file(p)
                for p in output.rglob("*")
                if p.is_file()
            },
            "script_sha256": sha256_file(Path(__file__)),
            "attention_implementation_sha256": sha256_file(
                PROJECT_ROOT / "src/multimodal_ecommerce/regional_retrieval.py"
            ),
        },
    )
    return report


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Use a new benchmark directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/module2_regions_benchmark_v1")
    )
    args = parser.parse_args()
    if (PROJECT_ROOT / args.output).exists():
        raise FileExistsError("Use a new benchmark directory")
    subprocess.run(
        ["ollama", "stop", "qwen3.5:4b-q4_K_M"], check=True, capture_output=True
    )
    embedder = QwenEmbedder()
    try:
        report = benchmark(embedder, PROJECT_ROOT / args.output)
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "query_count",
                        "gallery_count",
                        "summary",
                        "query_multiview_latency",
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
