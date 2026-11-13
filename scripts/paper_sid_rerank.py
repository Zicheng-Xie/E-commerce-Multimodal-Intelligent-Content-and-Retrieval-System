"""Test HybSID candidate additions with real, resumable Qwen reranking."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce import platform_lock
from multimodal_ecommerce.business_retrieval import BusinessReranker
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.simple_baseline import stable_order
from paper_fusion_experiment import BASELINES, COHORTS, DATA, ROOT, SIMPLE, load
from querywise_paired_bounds import paired_querywise
from simple_baseline_cycle import guard

METHOD = "sid_cross_recall_complete_ce"
ANCHORED = "sid_cross_recall_consensus_anchor"


@contextmanager
def worker_lock() -> Iterator[None]:
    """Hold a process lock for the local inference and result files.

    Yields:
        Exclusive write access to this experiment's scoring artifacts.
    """
    with (ROOT / "sid_worker.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        platform_lock.locking(handle.fileno(), platform_lock.LK_NBLCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            platform_lock.locking(handle.fileno(), platform_lock.LK_UNLCK, 1)


def plan() -> list[dict]:
    """Freeze a bounded label-free pool expansion and exact original score reuse.

    Returns:
        One immutable scoring job per image query.
    """
    guard()
    inputs = load(SIMPLE / "inputs.json")
    pools = load(ROOT / "sid_candidates.json")
    old = load(ROOT / "rankings.json")
    digest = sha256_file(SIMPLE / "protocol.json")
    jobs = []
    for cohort in COHORTS:
        crop = load(SIMPLE / cohort / "predictions.json")
        for query in inputs["cohorts"][cohort]["queries"]:
            qid = query["id"]
            order = crop[qid]["crop_cosine"]
            head = order[:20]
            cached = load(SIMPLE / "pairs" / cohort / f"{qid}.json")
            if cached["identity"] != dict(
                query_sha256=query["source_sha256"],
                candidate_ids=head,
                protocol_sha256=digest,
            ) or set(cached["scores"]) != set(head):
                raise ValueError("Earlier Top20 score cache is not exact")
            control = old[cohort][qid]
            sid = [cid for cid in pools[cohort][qid]["sid_top20"] if cid not in head][
                :3
            ]
            cross = [cid for cid in control["cross20"][:20] if cid not in head + sid][
                :3
            ]
            anchor = control["global"][0]
            if anchor != control["region_title"][0]:
                anchor = None
            extra = sid + cross
            if anchor and anchor not in head + extra:
                extra.append(anchor)
            if len(set(head + extra)) != len(head + extra) or not set(extra) <= set(
                order
            ):
                raise ValueError("Expanded head changed gallery membership")
            jobs.append(
                dict(
                    cohort=cohort,
                    query=query,
                    head=head,
                    original_scores=cached["scores"],
                    sid_additions=sid,
                    cross_additions=cross,
                    extra=extra,
                    anchor=anchor,
                    documents=[inputs["documents"][cid] for cid in extra],
                )
            )
    return jobs


def inference(jobs: list[dict], registry: dict) -> dict:
    """Score all added candidates, checkpointing each verified query.

    Args:
        jobs: Frozen query identities and bounded candidate additions.
        registry: Project evaluation roles used before reading original images.

    Returns:
        Model runtime evidence for this invocation.
    """
    client = None
    try:
        for job in jobs:
            query = job["query"]
            cohort, qid = job["cohort"], query["id"]
            path = ROOT / "sid_pairs" / cohort / f"{qid}.json"
            identity = dict(
                protocol_sha256=sha256_file(ROOT / "sid_protocol.json"),
                query_sha256=query["source_sha256"],
                extra=job["extra"],
            )
            if path.exists():
                cached = load(path)
                if cached["identity"] != identity or set(cached["scores"]) != set(
                    job["extra"]
                ):
                    raise ValueError("Expanded score cache differs from frozen job")
                continue
            images = [query, *job["documents"]]
            require_data_access(
                [Path(row["image_path"]) for row in images], "development", registry
            )
            for row in images:
                if sha256_file(Path(row["image_path"])) != row["source_sha256"]:
                    raise ValueError("An original image changed after the first run")
            documents = [
                dict(id=row["id"], image_path=row["image_path"], text=row["title"])
                for row in job["documents"]
            ]
            scores = {}
            if documents:
                if client is None:
                    client = BusinessReranker()
                logits = client.score(documents, query_image=Path(query["image_path"]))
                if logits.shape != (len(documents),) or not np.isfinite(logits).all():
                    raise ValueError("Complete finite Qwen scores required")
                scores = dict(zip(job["extra"], logits.tolist(), strict=True))
            write_json(path, dict(identity=identity, scores=scores))
            print(f"Complete {cohort}/{qid}: {len(scores)} new pairs", flush=True)
        return client.diagnostics() if client else dict(new_model_calls=0)
    finally:
        if client is not None:
            client.close()


def evaluate(jobs: list[dict]) -> None:
    """Replay the scored head and evaluate all original image and text queries.

    Args:
        jobs: Fixed scoring jobs with original Top20 score maps.
    """
    previous = load(ROOT / "rankings.json")
    ranks: dict[str, dict[str, dict[str, list[str]]]] = {}
    for cohort in COHORTS:
        ranks[cohort] = {}
        for qid, variants in previous[cohort].items():
            ranks[cohort][qid] = {
                **{k: variants[k] for k in (*BASELINES, "simple_previous")},
                METHOD: variants["cross20"],
                ANCHORED: variants["cross20"],
            }
    for job in jobs:
        cohort, qid = job["cohort"], job["query"]["id"]
        new = load(ROOT / "sid_pairs" / cohort / f"{qid}.json")["scores"]
        scores = {**job["original_scores"], **new}
        head = job["head"] + job["extra"]
        order = stable_order(head, np.array([scores[cid] for cid in head]))
        tail = [
            cid for cid in previous[cohort][qid]["simple_previous"] if cid not in scores
        ]
        ranks[cohort][qid][METHOD] = order + tail
        anchor = job["anchor"]
        protected = (
            [anchor] + [cid for cid in order if cid != anchor] if anchor else order
        )
        ranks[cohort][qid][ANCHORED] = protected + tail
    write_json(ROOT / "sid_rerank_rankings.json", ranks)
    results = {}
    for cohort, group in ranks.items():
        labels = [
            json.loads(line)
            for line in (DATA / cohort / "judgments.ai.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        modes = {qid: "text" if qid.startswith("text_") else "image" for qid in group}
        absolute = evaluate_intervals(group, labels, modes)
        paired = paired_querywise(
            group,
            labels,
            modes,
            [(ANCHORED, base) for base in (*BASELINES, "simple_previous", METHOD)]
            + [(METHOD, "simple_previous")],
        )
        results[cohort] = dict(absolute=absolute, paired=paired)
    write_json(ROOT / "sid_rerank_results.json", results)


def main() -> None:
    """Run the bounded expansion using Apple MPS and immutable development inputs."""
    registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
    with worker_lock():
        paths = [SIMPLE / "inputs.json", SIMPLE / "protocol.json"]
        for cohort in COHORTS:
            paths.extend(
                [
                    SIMPLE / cohort / "predictions.json",
                    DATA / cohort / "judgments.ai.jsonl",
                ]
            )
            paths.extend(sorted((SIMPLE / "pairs" / cohort).glob("*.json")))
        require_data_access(paths, "development", registry)
        parent = load(ROOT / "protocol.json")
        for relative, digest in parent["files"].items():
            if sha256_file(PROJECT_ROOT / relative) != digest:
                raise ValueError(f"Parent fusion input changed: {relative}")
        jobs = plan()
        protocol: dict[str, Any] = dict(
            k=10,
            role="exposed_development",
            original_pairs=480,
            new_pairs=sum(len(job["extra"]) for job in jobs),
            source_sha256=sha256_file(Path(__file__)),
            inputs={str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in paths},
            parent_rankings_sha256=sha256_file(ROOT / "rankings.json"),
            parent_sid_sha256=sha256_file(ROOT / "sid_candidates.json"),
            jobs=jobs,
            formal_metrics=None,
        )
        path = ROOT / "sid_protocol.json"
        if path.exists() and load(path) != protocol:
            raise ValueError("Frozen expansion protocol changed")
        write_json(path, protocol)
        print(
            f"Frozen additional full-model pairs: {protocol['new_pairs']}", flush=True
        )
        runtime = inference(jobs, registry)
        if not (ROOT / "sid_reranker_runtime.json").exists() or runtime.get("timings"):
            write_json(ROOT / "sid_reranker_runtime.json", runtime)
        evaluate(jobs)


if __name__ == "__main__":
    main()
