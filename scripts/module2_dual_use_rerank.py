"""Compare two complete rerankers on the same dual-use development Top20."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import paper_sid_rerank as worker
import run  # noqa: F401
from module2_dual_use_recall import REVIEW, ROOT
from module2_focus_boundary import FOCUS, FRESH, verified
from module2_poster_usage import ROOT as POSTER
from multimodal_ecommerce.business_retrieval import BUSINESS_INSTRUCTION
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.intent_retrieval import rerank_complete
from multimodal_ecommerce.interval_retrieval_metrics import query_bounds
from multimodal_ecommerce.runtime_device import relocate_assets
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.visual_query import DescribedImageReranker
from paper_fusion_experiment import load

BASE = "whole_and_description"
STANDARD = "original_business_instruction"
DUAL = "same_device_instruction"
DUAL_INSTRUCTION = (
    BUSINESS_INSTRUCTION
    + " When the described query is for a single game console that also functions as a power bank, "
    + "use the candidate image and title together. The same offered device must play games "
    + "and supply power to a separate device. Distinguish an internal rechargeable battery "
    + "from an external power output. A separate bundled power bank, holder, controller, "
    + "or ordinary portable charger does not satisfy both functions."
)


def plan(registry: dict) -> dict:
    """Freeze the same pool, original model settings, and reusable input scores.

    Args:
        registry: Evaluation data-role policy.

    Returns:
        Exact scoring inputs with candidate and source hashes.
    """
    recall_path = ROOT / "recall_results.json"
    protocol_path = ROOT / "recall_protocol.json"
    index_path = FOCUS / "index/records.json"
    require_data_access(
        [recall_path, protocol_path, index_path], "development", registry
    )
    recall, prior = load(recall_path), load(protocol_path)
    if recall["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError("Recall protocol changed")
    query = prior["query"]
    original_path = POSTER / (query["qid"] + "_replay.json")
    poster_path = POSTER / "experiment_protocol.json"
    require_data_access([original_path, poster_path], "development", registry)
    original = load(original_path)
    if original["protocol_sha256"] != sha256_file(poster_path):
        raise ValueError("The prior scoring receipt is not aligned")
    if load(poster_path)["jobs"] != [query]:
        raise ValueError("The archived standard prompt received different query data")
    archived_hashes = prior["files"]
    verified(
        index_path, archived_hashes[str(index_path.relative_to(PROJECT_ROOT))], registry
    )
    records = {r["id"]: r for r in relocate_assets(load(index_path))}
    records.update(
        {
            r["id"]: dict(title=r["title"], image_path=r["image"])
            for r in prior["candidates"]
        }
    )
    order = recall["rankings"][BASE]
    pool = order[:20]
    if len(set(pool)) != 20 or set(pool) != set(
        recall["rankings"]["conjunctive_support_then_fallback"][:20]
    ):
        raise ValueError("The compared methods must have the same complete Top20")
    docs, reuse, image_hashes = [], {}, {}
    for cid in pool:
        record = records[cid]
        image = Path(record["image_path"])
        name = str(image.relative_to(PROJECT_ROOT))
        old = original["image_hashes"].get(name)
        image_hashes[name] = verified(
            image, old or archived_hashes.get(name, ""), registry
        )
        docs.append(dict(id=cid, image_path=str(image), text=record["title"]))
        if cid in original["receipt"]["scores"]:
            if not old:
                raise ValueError("A reused Qwen score has no bound candidate original")
            reuse[cid] = original["receipt"]["scores"][cid]
    image = Path(query["image"])
    image_hashes[str(image.relative_to(PROJECT_ROOT))] = verified(
        image, query["image_sha256"], registry
    )
    return dict(
        query=query,
        order=order,
        documents=docs,
        reuse=reuse,
        original_model=original["receipt"]["reranker_runtime"]["model"],
        prompts={STANDARD: BUSINESS_INSTRUCTION, DUAL: DUAL_INSTRUCTION},
        inputs={
            str(p.relative_to(PROJECT_ROOT)): sha256_file(p)
            for p in (
                recall_path,
                protocol_path,
                index_path,
                original_path,
                poster_path,
            )
        },
        image_hashes=image_hashes,
        source_sha256=sha256_file(Path(__file__)),
        role="exposed_development",
    )


def infer(protocol: dict, digest: str) -> dict:
    """Run the unchanged Qwen checkpoint for both instructions and save real logits.

    Args:
        protocol: Label-free frozen scoring inputs.
        digest: Frozen protocol hash.

    Returns:
        Complete finite scores and runtime evidence for each instruction.
    """
    results = {}
    for method, instruction in protocol["prompts"].items():
        path = ROOT / (method + "_scores.json")
        reused = protocol["reuse"] if method == STANDARD else {}
        docs = [d for d in protocol["documents"] if d["id"] not in reused]
        ids = [d["id"] for d in docs]
        if path.exists():
            cached = load(path)
            if (
                cached["protocol_sha256"] != digest
                or set(cached["scores"]) != set(ids)
                or not np.isfinite(list(cached["scores"].values())).all()
            ):
                raise ValueError("Score cache differs from the frozen request")
        else:
            model = DescribedImageReranker()
            try:
                model.adapter.config = {
                    **model.adapter.config,
                    "instruction": instruction,
                }
                prior = protocol["original_model"]
                for key in (
                    "revision",
                    "files",
                    "vendor_sha256",
                    "max_length",
                    "max_pixels",
                ):
                    if model.adapter.config[key] != prior[key]:
                        raise ValueError(f"Reranker identity differs: {key}")
                query = protocol["query"]
                logits = model.score_described(
                    docs, query_image=Path(query["image"]), description=query["caption"]
                )
                runtime = model.diagnostics()
            finally:
                model.close()
            cached = dict(
                protocol_sha256=digest,
                scores=dict(zip(ids, logits.tolist(), strict=True)),
                runtime=runtime,
            )
            write_json(path, cached)
        results[method] = dict(
            scores={**reused, **cached["scores"]},
            reused=len(reused),
            new=len(ids),
            runtime=cached["runtime"],
        )
        print(method, "new scores:", len(ids), flush=True)
    return results


def evaluate(protocol: dict, scores: dict, registry: dict) -> dict:
    """Score human-reviewed development relations without changing their provenance.

    Args:
        protocol: Frozen complete Top20 replay.
        scores: Actual logits and receipts for both methods.
        registry: Data roles for reviewed input files.

    Returns:
        Full orders and development metrics under the mixed label scope.
    """
    review = REVIEW / "adjudication.json"
    old = FRESH / "judgments.ai.jsonl"
    require_data_access([review, old], "development", registry)
    audited = load(review)
    if audited["old_labels_sha256"] != sha256_file(old):
        raise ValueError("Old AI labels changed")
    if (
        audited["rankings_sha256"]
        != protocol["inputs"][
            str((ROOT / "recall_results.json").relative_to(PROJECT_ROOT))
        ]
    ):
        raise ValueError("Human adjudication used another recall version")
    qid = protocol["query"]["qid"]
    grades = {
        row["candidate_id"]: row["grade"]
        for row in map(json.loads, old.read_text(encoding="utf-8").splitlines())
        if row["query_id"] == qid
    }
    for cid, row in audited["rows"].items():
        if cid in grades:
            raise ValueError("New human data overlaps the old gallery")
        grades[cid] = None if row["grade"] == "uncertain" else int(row["grade"])
    base = protocol["order"]
    pool = base[:20]
    values = -np.arange(len(base), dtype=float)
    recall = load(ROOT / "recall_results.json")
    orders = recall["rankings"].copy()
    for name, result in scores.items():
        if set(result["scores"]) != set(pool):
            raise ValueError("Reranking must score all twenty candidates")
        logits = np.array([result["scores"][cid] for cid in pool])
        orders[name] = rerank_complete(base, values, pool, logits)
    metrics = {name: query_bounds(order, grades) for name, order in orders.items()}
    return dict(
        scope="One exposed development query. Original 418 AI grades and 16 new human grades.",
        human_review_sha256=sha256_file(review),
        metrics=metrics,
        rankings=orders,
        scores=scores,
        ideal_recall_at_10=min(10, sum(g == 1 for g in grades.values()))
        / sum(g == 1 for g in grades.values()),
        formal_metrics=None,
        frozen_acceptance_model_calls=0,
    )


def main() -> None:
    """Freeze, score, and evaluate the controlled full-model comparison."""
    worker.ROOT = ROOT
    with worker.worker_lock():
        registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
        protocol = plan(registry)
        path = ROOT / "rerank_protocol.json"
        if path.exists() and load(path) != protocol:
            raise ValueError("Frozen rerank experiment changed")
        if not path.exists():
            write_json(path, protocol)
        result = evaluate(protocol, infer(protocol, sha256_file(path)), registry)
        write_json(
            ROOT / "rerank_results.json",
            dict(**result, protocol_sha256=sha256_file(path)),
        )
        print("Completed dual-use rerank comparison.", flush=True)


if __name__ == "__main__":
    main()
