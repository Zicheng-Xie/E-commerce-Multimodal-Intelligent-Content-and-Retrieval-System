"""Score exact-image pairs on the allowlisted development set, never heldout."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TypedDict

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import run  # noqa: F401,E402
import verify_regions  # noqa: E402
from multimodal_ecommerce.blind_evaluation import verify_files  # noqa: E402
from multimodal_ecommerce.pair_reranker import PairReranker  # noqa: E402
from multimodal_ecommerce.safety import (  # noqa: E402
    PROJECT_ROOT,
    sha256_file,
    write_json,
)

EXPERIMENT = PROJECT_ROOT / "artifacts/module2_regions_benchmark_v1"
PREVIOUS = PROJECT_ROOT / "artifacts/module2_strategy_exploration"
OUTPUT = PROJECT_ROOT / "artifacts/module2_precision"


class _PolicySpec(TypedDict):
    """Exact fields of the POLICY configuration and provenance record."""

    role: str
    formal_acceptance: bool
    candidate_pool: int
    rank_blend_weights: list[float]
    protected_anchors: int
    hard_negative_limit: int
    pairwise_steps: int
    pairwise_lr: float
    pairwise_l2: float
    distillation_mix: float
    distillation_temperature: float
    features: list[str]
    teacher_feature: str
    cv: str
    development_selection: str
    latency: str
    independent_validation: str
    metric_contract: str


POLICY: _PolicySpec = {
    "role": "internal_development_only",
    "formal_acceptance": False,
    "candidate_pool": 30,
    "rank_blend_weights": [0.25, 0.5, 0.75],
    "protected_anchors": 2,
    "hard_negative_limit": 8,
    "pairwise_steps": 500,
    "pairwise_lr": 0.1,
    "pairwise_l2": 0.1,
    "distillation_mix": 0.5,
    "distillation_temperature": 2.0,
    "features": [
        "global_cosine",
        "region_attention",
        "reciprocal_fusion",
        "color_similarity",
        "base_rank",
    ],
    "teacher_feature": "cross_encoder_logit",
    "cv": "leave-one-source-pair-group-out; exclude validation-group gallery candidates from supervised training pairs",
    "development_selection": "Require +2 percentage points Precision@10 and no mean regression in R@10,mAP,NDCG@10,MRR,Hit@1 versus frozen fusion; then maximize Precision@10, mAP, Recall@10",
    "latency": "Cross-encoder rerank alone above 5 seconds is unsuitable for the interactive path; student uses no online cross-encoder",
    "independent_validation": "No use of artifacts/module2_blind for tuning or repeated acceptance; any selected later version needs new unseen evaluation",
    "metric_contract": "P@10 denominator remains 10; no query dropping, positive duplication, relabeling or reduced-K substitution",
}


def prepare_queue() -> tuple[dict, list[dict]]:
    """Prepare queue.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Changed development query image; Query duplicate entered candidate
            pool.
    """
    verify_regions.verify(EXPERIMENT)
    verify_files(
        PREVIOUS,
        json.loads((PREVIOUS / "manifest.json").read_text(encoding="utf-8"))["files"],
    )
    protocol = json.loads((EXPERIMENT / "protocol.json").read_text(encoding="utf-8"))[
        "source_protocol"
    ]
    rows = json.loads((EXPERIMENT / "index/records.json").read_text(encoding="utf-8"))
    records = {r["id"]: r for r in rows}
    rankings = json.loads((PREVIOUS / "rankings.json").read_text(encoding="utf-8"))
    queue = []
    for entry in protocol["queries"]:
        query = entry["record"]
        candidates = rankings[query["id"]]["reciprocal_attention_fusion"][
            : POLICY["candidate_pool"]
        ]
        documents = [
            {"id": key, "image_path": records[key]["image_path"]} for key in candidates
        ]
        if sha256_file(Path(query["image_path"])) != query["image_sha256"]:
            raise ValueError("Changed development query image")
        if any(
            records[key]["image_sha256"] == query["image_sha256"] for key in candidates
        ):
            raise ValueError("Query duplicate entered candidate pool")
        queue.append(
            {
                "query_id": query["id"],
                "query_image": query["image_path"],
                "documents": documents,
            }
        )
    frozen = {
        "policy": POLICY,
        "development_manifest_sha256": sha256_file(EXPERIMENT / "manifest.json"),
        "old_scores_manifest_sha256": sha256_file(PREVIOUS / "manifest.json"),
        "reranker_configuration_sha256": sha256_file(
            PROJECT_ROOT / "configs/reranker_model.json"
        ),
        "adapter_sha256": sha256_file(
            PROJECT_ROOT / "src/multimodal_ecommerce/pair_reranker.py"
        ),
        "script_sha256": sha256_file(Path(__file__)),
        "query_count": len(queue),
        "pair_count": sum(len(q["documents"]) for q in queue),
    }
    return frozen, queue


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        ValueError: Development protocol changed; preserve previous evidence; Duplicate
            cached development query; Cached candidate pool changed.

    Notes:
        Writes derived artifacts through write_json.
    """
    frozen, queue = prepare_queue()
    path = OUTPUT / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != frozen:
            raise ValueError("Development protocol changed; preserve previous evidence")
    else:
        write_json(path, frozen)
        write_json(OUTPUT / "model_inputs.json", queue)
    cache = OUTPUT / "pair_scores.jsonl"
    finished = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["query_id"] in finished:
                raise ValueError("Duplicate cached development query")
            finished[row["query_id"]] = row
    if len(finished) == len(queue):
        print("All cached development pairs already complete")
        return
    reranker = PairReranker()
    try:
        for number, item in enumerate(queue, 1):
            if item["query_id"] in finished:
                if [d["id"] for d in item["documents"]] != finished[item["query_id"]][
                    "candidate_ids"
                ]:
                    raise ValueError("Cached candidate pool changed")
                continue
            started = time.perf_counter()
            values = reranker.score(
                item["documents"], query_image=Path(item["query_image"])
            )
            result = {
                "query_id": item["query_id"],
                "candidate_ids": [d["id"] for d in item["documents"]],
                "logits": values.tolist(),
                "rerank_seconds": time.perf_counter() - started,
            }
            with cache.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"Development pair scoring {number}/{len(queue)}; {len(values)} candidates",
                flush=True,
            )
        write_json(OUTPUT / "reranker_diagnostics.json", reranker.diagnostics())
        write_json(
            OUTPUT / "pair_cache_manifest.json",
            {
                "files": {
                    name: sha256_file(OUTPUT / name)
                    for name in (
                        "protocol.json",
                        "model_inputs.json",
                        "pair_scores.jsonl",
                        "reranker_diagnostics.json",
                    )
                },
                "role": "internal_development_only",
            },
        )
    finally:
        reranker.close()


if __name__ == "__main__":
    main()
