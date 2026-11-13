"""Prepare, validate or run independently judged retrieval evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.embedding import QwenEmbedder, model_configuration
from multimodal_ecommerce.evaluation import (
    prepare_evaluation,
    score_rankings,
    validate_qrels,
    verify_files,
)
from multimodal_ecommerce.retrieval import VectorIndex
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)


def main() -> None:
    """Refuse scoring until all judgments are complete; retain timing evidence.

    Raises:
        FileExistsError: Use a new score report path.
        ValueError: Index changed after evaluation preparation; Model config changed;
            create a new frozen evaluation; Candidate IDs disagree; Query image changed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "validate", "run"])
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/stage2_v1/release/products.jsonl")
    )
    parser.add_argument("--index", type=Path, default=Path("data/index_stage2_v1"))
    parser.add_argument(
        "--kit", type=Path, default=Path("artifacts/evaluation_stage2_v1")
    )
    parser.add_argument("--qrels", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/evaluation_scored_v1.json")
    )
    parser.add_argument("--target", choices=["image", "text", "fused"], default="image")
    args = parser.parse_args()
    kit = PROJECT_ROOT / args.kit
    if args.command == "prepare":
        report = prepare_evaluation(
            PROJECT_ROOT / args.dataset, PROJECT_ROOT / args.index, kit
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    manifest = json.loads((kit / "manifest.json").read_text(encoding="utf-8"))
    verify_files(kit, manifest)
    queries = read_rows(kit / "queries.jsonl")
    catalog = json.loads((kit / "catalog.json").read_text(encoding="utf-8"))
    qrels = PROJECT_ROOT / args.qrels if args.qrels else kit / "qrels.template.jsonl"
    relevant = validate_qrels(queries, {row["id"] for row in catalog}, read_rows(qrels))
    if args.command == "validate":
        print(json.dumps({"passed": True, "queries": len(queries)}))
        return
    output = ensure_output_path(PROJECT_ROOT / args.output)
    if output.exists():
        raise FileExistsError("Use a new score report path")
    index_root = Path(manifest["index"])
    if sha256_file(index_root / "manifest.json") != manifest["index_manifest_sha256"]:
        raise ValueError("Index changed after evaluation preparation")
    settings = json.loads((index_root / "manifest.json").read_text(encoding="utf-8"))
    if settings["model"] != model_configuration():
        raise ValueError("Model config changed; create a new frozen evaluation")
    index = VectorIndex.load(index_root)
    if {row["id"] for row in index.records} != {row["id"] for row in catalog}:
        raise ValueError("Candidate IDs disagree")
    subprocess.run(
        ["ollama", "stop", "qwen3.5:4b-q4_K_M"], check=True, capture_output=True
    )
    embedder = QwenEmbedder()
    timings, rankings = {}, {}
    try:
        for query in queries:
            image = Path(query["image_path"]) if query["image_path"] else None
            if image and sha256_file(image) != query["image_sha256"]:
                raise ValueError("Query image changed")
            started = time.perf_counter()
            vector = embedder.encode(text=query["text"], image=image, query=True)
            result = index.search(
                vector, top_k=10, target=args.target, exclude_id=query["exclude_id"]
            )
            timings[query["id"]] = time.perf_counter() - started
            rankings[query["id"]] = [row["id"] for row in result]
        report = {
            "metrics": score_rankings(queries, relevant, rankings),
            "rankings": rankings,
            "query_seconds_excluding_model_load": timings,
            "model": settings["model"],
            "target": args.target,
            "qrels_sha256": sha256_file(qrels),
            "kit_manifest_sha256": sha256_file(kit / "manifest.json"),
            "diagnostics": embedder.diagnostics(),
        }
        write_json(output, report)
        print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    finally:
        embedder.close()


if __name__ == "__main__":
    main()
