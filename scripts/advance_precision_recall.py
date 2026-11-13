"""Freeze, execute and analyze a multi-recall/cross-encoder development comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_retrieval import (
    BUSINESS_INSTRUCTION,
    BusinessReranker,
)
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.intent_retrieval import (
    POLICY,
    balanced_pool,
    normalized_ranks,
    public_evidence,
    recall_channels,
    rerank_complete,
)
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.semantic_ranker import machine_topk_bounds, ranked_ids

ROOT = PROJECT_ROOT / "artifacts/module2_precision_recall"
BASE = PROJECT_ROOT / "artifacts/module2_machine_assist"
BANK = BASE / "semantic_ranker"
OLD = PROJECT_ROOT / "artifacts/module2_precision60/business"
CONTINUATION = PROJECT_ROOT / "artifacts/module2_continuation"


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_inputs() -> tuple[Any, Any, Any]:
    """Return frozen inputs for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    protocol = load(ROOT / "protocol.json")
    verify_files(PROJECT_ROOT, protocol["implementation"])
    verify_files(PROJECT_ROOT, protocol["parents"])
    verify_files(PROJECT_ROOT, protocol["image_files"])
    assert sha256_file(ROOT / "inputs.json") == protocol["inputs_sha256"]
    assert sha256_file(ROOT / "pools.json") == protocol["pools_sha256"]
    return protocol, load(ROOT / "inputs.json"), load(ROOT / "pools.json")


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        FileExistsError: Preserve the frozen precision/recall experiment.

    Notes:
        Writes derived artifacts through ROOT.mkdir, write_json.
    """
    if (ROOT / "protocol.json").exists():
        raise FileExistsError("Preserve the frozen precision/recall experiment")
    for folder, receipt_name in [
        (BANK, "feature_receipt.json"),
        (OLD, "prediction_receipt.json"),
        (CONTINUATION / "constraints", "receipt.json"),
    ]:
        verify_files(folder, load(folder / receipt_name)["files"])
    context = load(BASE / "development_inputs.json")
    queries = [
        {k: q[k] for k in ["id", "text", "image_path"]} for q in context["queries"]
    ]
    catalog = context["catalog"]
    metadata = load(BANK / "feature_metadata.json")
    titles = {r["id"]: r["title"] for r in catalog}
    pools, evidence = {}, {}
    with np.load(BANK / "features.npz", allow_pickle=False) as arrays:
        x, valid = arrays["features"], arrays["valid"]
    by_query = {q["id"]: q for q in queries}
    for i, qid in enumerate(metadata["query_ids"]):
        positions = np.flatnonzero(valid[i])
        ids = [metadata["gallery_ids"][j] for j in positions]
        rrf, channels = recall_channels(ids, x[i, positions])
        why = [
            public_evidence(
                by_query[qid]["text"], titles[k], metadata["profile_texts"][j]
            )
            for k, j in zip(ids, positions)
        ]
        adjusted = normalized_ranks(ids, rrf) + np.array([r["offset"] for r in why])
        # Public-evidence channel supplements the visual/title channels; no filtering.
        channels.append(ranked_ids(ids, adjusted))
        pools[qid] = balanced_pool(channels, 60)
        evidence[qid] = {key: row for key, row in zip(ids, why)}
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / "inputs.json", dict(queries=queries, catalog=catalog))
    write_json(ROOT / "pools.json", pools)
    write_json(ROOT / "public_evidence.json", evidence)
    cached = load(BANK / "runtime_fingerprints.json")
    files = dict(cached["source_files"])
    for query in queries:
        if query["image_path"]:
            p = Path(query["image_path"])
            files[p.relative_to(PROJECT_ROOT).as_posix()] = sha256_file(p)
    reserved = read_rows(
        PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl"
    )
    assert not set(files.values()) & {r["image_sha256"] for r in reserved}
    names = [
        "scripts/advance_precision_recall.py",
        "src/multimodal_ecommerce/intent_retrieval.py",
        "src/multimodal_ecommerce/business_retrieval.py",
        "src/multimodal_ecommerce/pair_reranker.py",
    ]
    parent_paths = [
        "configs/module2_current_target.json",
        "configs/reranker_model.json",
        "data/module2_reserved_acceptance/records.jsonl",
        "artifacts/module2_machine_assist/development_inputs.json",
        "artifacts/module2_machine_assist/semantic_ranker/feature_metadata.json",
        "artifacts/module2_machine_assist/semantic_ranker/features.npz",
        "artifacts/module2_precision60/business/model_inputs.json",
        "artifacts/module2_precision60/business/pair_scores.jsonl",
        "artifacts/module2_precision60/business/runtime.json",
        "artifacts/module2_continuation/constraints/rankings.json",
        "artifacts/module2_continuation/pixel_refinement/refined_weak_labels.jsonl",
    ]
    write_json(
        ROOT / "protocol.json",
        dict(
            policy=POLICY,
            instruction=BUSINESS_INSTRUCTION,
            implementation={n: sha256_file(PROJECT_ROOT / n) for n in names},
            parents={n: sha256_file(PROJECT_ROOT / n) for n in parent_paths},
            image_files=files,
            inputs_sha256=sha256_file(ROOT / "inputs.json"),
            pools_sha256=sha256_file(ROOT / "pools.json"),
            public_evidence_sha256=sha256_file(ROOT / "public_evidence.json"),
            queries=18,
            broad_pool_pairs=sum(map(len, pools.values())),
            source_relevance_categories_or_qrels_used_as_features=False,
            full_gallery_rankings_required=True,
            human_labels=0,
            formal_acceptance=False,
            formal_metrics=None,
            goal_met=False,
            acceptance_model_predictions=0,
            default_changed=False,
        ),
    )
    print(
        json.dumps(
            dict(
                frozen=True,
                queries=18,
                pool_pairs=sum(map(len, pools.values())),
                formal_metrics=None,
            )
        )
    )


def infer() -> None:
    """Run model inference using the configured inputs.

    Raises:
        FileExistsError: Preserve completed cross-encoder predictions.

    Notes:
        Writes derived artifacts through folder.mkdir, write_json.
    """
    if (ROOT / "inference_receipt.json").exists():
        raise FileExistsError("Preserve completed cross-encoder predictions")
    protocol, inputs, pools = frozen_inputs()
    previous = {r["query_id"]: r for r in load(OLD / "model_inputs.json")}
    cached = {}
    for row in read_rows(OLD / "pair_scores.jsonl"):
        for cid, value in zip(row["candidate_ids"], row["logits"]):
            cached[row["query_id"], cid] = value
    old_model = load(OLD / "runtime.json")["diagnostics"]["model"]
    config = load(PROJECT_ROOT / "configs/reranker_model.json")
    assert (
        old_model["files"] == config["files"]
        and old_model["instruction"] == BUSINESS_INSTRUCTION
    )
    catalog = {r["id"]: r for r in inputs["catalog"]}
    tasks = []
    for q in inputs["queries"]:
        old = previous[q["id"]]
        assert q["text"] == old["query_text"] and q["image_path"] == old["query_image"]
        docs = {r["id"]: r for r in old["documents"]}
        for cid in pools[q["id"]]:
            doc = dict(
                id=cid,
                image_path=catalog[cid]["image_path"],
                text=catalog[cid]["title"],
            )
            if (q["id"], cid) in cached:
                assert docs[cid] == doc
            tasks.append((q, doc))
    runner = None
    reused, executed = 0, 0
    folder = ROOT / "pair_scores"
    folder.mkdir(exist_ok=True)
    started = time.perf_counter()
    try:
        for i, (query, doc) in enumerate(tasks, 1):
            key = (query["id"], doc["id"])
            name = hashlib.sha256("|".join(key).encode()).hexdigest()[:24] + ".json"
            path = folder / name
            if path.exists():
                row = load(path)
                assert (row["query_id"], row["candidate_id"]) == key
                reused += row["origin"] == "frozen_prior_identical_input"
                executed += row["origin"] == "new_cross_encoder_inference"
                continue
            if key in cached:
                value, elapsed, origin = (
                    cached[key],
                    0.0,
                    "frozen_prior_identical_input",
                )
                reused += 1
            else:
                if runner is None:
                    runner = BusinessReranker()
                before = time.perf_counter()
                value = float(
                    runner.score(
                        [doc],
                        query_text=query["text"],
                        query_image=(
                            Path(query["image_path"]) if query["image_path"] else None
                        ),
                    )[0]
                )
                elapsed, origin = (
                    time.perf_counter() - before,
                    "new_cross_encoder_inference",
                )
                executed += 1
            write_json(
                path,
                dict(
                    query_id=key[0],
                    candidate_id=key[1],
                    logit=value,
                    seconds=elapsed,
                    origin=origin,
                    model_input_sha256=hashlib.sha256(
                        json.dumps(
                            dict(query=query, document=doc), sort_keys=True
                        ).encode()
                    ).hexdigest(),
                ),
            )
            if i % 30 == 0:
                print(
                    json.dumps(
                        dict(
                            completed=i,
                            total=len(tasks),
                            new_inferences=executed,
                            cached=reused,
                        )
                    ),
                    flush=True,
                )
        diagnostic = runner.diagnostics() if runner else dict(model=old_model)
    finally:
        if runner is not None:
            runner.close()
    write_json(
        ROOT / "runtime.json",
        dict(
            wall_seconds=time.perf_counter() - started,
            new_inferences=executed,
            cached_pairs=reused,
            diagnostics=diagnostic,
            formal_performance_acceptance=False,
        ),
    )
    write_json(
        ROOT / "inference_receipt.json",
        dict(
            protocol_sha256=sha256_file(ROOT / "protocol.json"),
            files={
                p.relative_to(ROOT).as_posix(): sha256_file(p)
                for p in [*(folder.glob("*.json")), ROOT / "runtime.json"]
            },
        ),
    )
    print(
        json.dumps(
            dict(
                completed=len(tasks),
                new_inferences=executed,
                cached_pairs=reused,
                formal_metrics=None,
            )
        )
    )


def analyze() -> None:
    """Apply the analyze workflow step.

    Raises:
        FileExistsError: Preserve completed development analysis.

    Notes:
        Writes derived artifacts through write_json.
    """
    if (ROOT / "analysis_receipt.json").exists():
        raise FileExistsError("Preserve completed development analysis")
    protocol, inputs, pools = frozen_inputs()
    verify_files(ROOT, load(ROOT / "inference_receipt.json")["files"])
    assert (
        sha256_file(ROOT / "public_evidence.json") == protocol["public_evidence_sha256"]
    )
    evidence = load(ROOT / "public_evidence.json")
    metadata = load(BANK / "feature_metadata.json")
    with np.load(BANK / "features.npz", allow_pickle=False) as bank:
        x, valid = bank["features"], bank["valid"]
    scores = {
        (r["query_id"], r["candidate_id"]): r["logit"]
        for r in (load(p) for p in (ROOT / "pair_scores").glob("*.json"))
    }
    rankings = load(CONTINUATION / "constraints/rankings.json")
    machine = {
        (r["query_id"], r["candidate_id"]): r["grade"]
        for r in read_rows(CONTINUATION / "pixel_refinement/refined_weak_labels.jsonl")
    }
    coverage = []
    for i, qid in enumerate(metadata["query_ids"]):
        positions = np.flatnonzero(valid[i])
        ids = [metadata["gallery_ids"][j] for j in positions]
        rrf, _ = recall_channels(ids, x[i, positions])
        offsets = np.array([evidence[qid][key]["offset"] for key in ids])
        ranks = normalized_ranks(ids, rrf)
        rankings[qid]["multi_channel_rrf"] = ranked_ids(ids, rrf)
        rankings[qid]["public_evidence_soft"] = ranked_ids(ids, ranks + offsets)
        for size in POLICY["pool_sizes"]:
            pool = pools[qid][:size]
            logits = np.array([scores[qid, key] for key in pool])
            rankings[qid][f"broad_cross_pool{size}"] = rerank_complete(
                ids, rrf, pool, logits
            )
            rankings[qid][f"broad_cross_evidence_pool{size}"] = rerank_complete(
                ids, rrf, pool, logits, offsets
            )
            known_positive = {key for key in ids if machine.get((qid, key)) == 1}
            coverage.append(
                dict(
                    query_id=qid,
                    pool_size=size,
                    known_machine_positives_in_pool=len(known_positive & set(pool)),
                    known_machine_positives=len(known_positive),
                    coverage_among_known_machine_positives=(
                        len(known_positive & set(pool)) / len(known_positive)
                        if known_positive
                        else None
                    ),
                    not_true_recall=True,
                )
            )
        for order in rankings[qid].values():
            assert len(order) == len(set(order)) == len(ids) and set(order) == set(ids)
    bounds = {
        name: machine_topk_bounds({q: r[name] for q, r in rankings.items()}, machine)
        for name in next(iter(rankings.values()))
    }
    write_json(ROOT / "rankings.json", rankings)
    write_json(
        ROOT / "machine_pool_diagnostics.json",
        dict(rows=coverage, not_true_recall=True, human_metrics=None),
    )
    write_json(
        ROOT / "machine_agreement_bounds.json",
        dict(
            methods=bounds,
            source="machine_development_diagnostics_not_business_quality",
            formal_metrics=None,
            goal_met=False,
        ),
    )
    # Identify machine/public-source contradictions for review.
    # Relabeling requires a separate review decision.
    conflicts = [
        dict(
            query_id=q, candidate_id=c, machine_grade=machine.get((q, c)), evidence=row
        )
        for q, rows in evidence.items()
        for c, row in rows.items()
        if machine.get((q, c)) == 1
        and row["state"] in ["explicit_type_conflict", "explicit_attribute_conflict"]
    ]
    write_json(
        ROOT / "machine_conflicts_for_review.json",
        dict(rows=conflicts, automatic_relabeling=False, human_labels=0),
    )
    write_json(
        ROOT / "status.json",
        dict(
            queries=len(rankings),
            complete_rankings=sum(len(r) for r in rankings.values()),
            methods=len(next(iter(rankings.values()))),
            new_reranker_predictions=load(ROOT / "runtime.json")["new_inferences"],
            cached_prior_pairs=load(ROOT / "runtime.json")["cached_pairs"],
            public_machine_conflicts=len(conflicts),
            precision_required=0.6,
            recall_preferred=0.85,
            human_labels=0,
            formal_metrics=None,
            goal_met=False,
            default_changed=False,
            acceptance_model_predictions=0,
        ),
    )
    write_json(
        ROOT / "analysis_receipt.json",
        dict(
            files={
                n: sha256_file(ROOT / n)
                for n in [
                    "rankings.json",
                    "machine_pool_diagnostics.json",
                    "machine_agreement_bounds.json",
                    "machine_conflicts_for_review.json",
                    "status.json",
                ]
            }
        ),
    )
    print(json.dumps(load(ROOT / "status.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "infer", "analyze"])
    args = parser.parse_args()
    {"prepare": prepare, "infer": infer, "analyze": analyze}[args.action]()
