"""Check original-image business reranking on the three reviewed subject mismatches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import run  # noqa: F401
from module2_development_benchmark import OUTPUT, REVIEW_GRADES, Evidence, load
from module2_shared_gallery_retrieval import REVIEW, _checked_media
from multimodal_ecommerce import platform_lock
from multimodal_ecommerce.business_retrieval import BUSINESS_INSTRUCTION
from multimodal_ecommerce.interval_retrieval_metrics import query_bounds
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.simple_baseline import human_accuracy, rerank_head
from multimodal_ecommerce.visual_query import DescribedImageReranker
from simple_cycle_review import FIELDS

ROOT = OUTPUT / "subject_experiment"
POOL = 20
CAPTIONS = (
    PROJECT_ROOT / "artifacts/module2_focus_followup_20261107/caption_model_comparison"
)


@contextmanager
def worker_lock() -> Iterator[None]:
    """Keep the local experiment caches single-writer across model runs.

    Yields:
        Exclusive access until the model worker closes.
    """
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "model_worker.lock").open("a+b") as handle:
        handle.seek(0)
        platform_lock.locking(handle.fileno(), platform_lock.LK_NBLCK, 1)
        try:
            yield
        finally:
            platform_lock.locking(handle.fileno(), platform_lock.LK_UNLCK, 1)


def reviewed_inputs(evidence: Evidence) -> tuple[dict, dict, dict]:
    """Bind the gallery and frozen ranks to the previously imported human decisions.

    Args:
        evidence: Reader restricted to registered development inputs.

    Returns:
        Gallery manifest with verified overrides, full rankings, and human decisions.
    """
    progress = evidence.json(REVIEW / "progress.json")
    pins = progress["inputs"]
    paths = [
        OUTPUT / "shared_gallery.json",
        OUTPUT / "shared_rankings.json",
        REVIEW / "tasks.private.json",
    ]
    manifest, rankings, tasks = [
        evidence.json(path, pins[path.relative_to(PROJECT_ROOT).as_posix()])
        for path in paths
    ]
    answers: list[dict] = []
    for name, digest in pins.items():
        path = PROJECT_ROOT / name
        if path.parent == REVIEW / "answers" and path.suffix == ".csv":
            evidence.digest(path, digest)
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != FIELDS:
                    raise ValueError(f"Unexpected human CSV fields: {path.name}")
                answers.extend(reader)
    audit = human_accuracy([dict(task, grade=None) for task in tasks], answers)
    if not audit["complete"] or audit["completed"] != progress["reviewed"]:
        raise ValueError("The original review batch must be complete")
    tasks_by_id = {task["task_id"]: task for task in tasks}
    grades = {}
    for row in answers:
        if not row["human_grade"].strip():
            continue
        task = tasks_by_id[row["task_id"]]
        qid, cid = task["query_id"], task["candidate_id"]
        grade = REVIEW_GRADES[row["human_grade"]]
        grades[qid, cid] = grade
        manifest["query_manifest"][qid]["judgments"][cid] = dict(
            grade=grade, human=True
        )
    return manifest, rankings, grades


def _plan(evidence: Evidence) -> tuple[dict, dict, dict]:
    manifest, rankings, reviewed = reviewed_inputs(evidence)
    queries = manifest["query_manifest"]
    catalog = {row["id"]: row for row in manifest["catalog"]}
    jobs = {}
    for (qid, cid), grade in reviewed.items():
        if grade != 0:
            continue
        query = queries[qid]
        head = rankings[qid][:POOL]
        if cid not in head[:10] or qid in jobs:
            raise ValueError(
                "Each reviewed negative must be in a distinct original Top10"
            )
        image = _checked_media(evidence, query) if query["mode"] == "image" else None
        jobs[qid] = dict(
            mode=query["mode"],
            query_image=str(image) if image else None,
            query_text=query.get("text"),
            reviewed_negative=cid,
            documents=[
                dict(
                    id=key,
                    image_path=str(_checked_media(evidence, catalog[key])),
                    text=catalog[key]["title"],
                )
                for key in head
            ],
        )
    if len(jobs) != 3:
        raise ValueError("Expected the three frozen human-confirmed mismatches")
    return jobs, queries, rankings


def _score_head(
    model: DescribedImageReranker, job: dict, identity: str, key: str
) -> dict:
    path = ROOT / "pair_cache" / (hashlib.sha256(key.encode()).hexdigest() + ".json")
    saved: dict = load(path) if path.is_file() else dict(identity=identity, scores={})
    ids = {row["id"] for row in job["documents"]}
    scores = saved["scores"]
    if (
        saved["identity"] != identity
        or not set(scores) <= ids
        or not all(math.isfinite(value) for value in scores.values())
    ):
        raise ValueError("Cached reranker inputs or scores differ")
    pending = [row for row in job["documents"] if row["id"] not in scores]
    for start in range(0, len(pending), 5):
        batch = pending[start : start + 5]
        image = Path(job["query_image"]) if job["query_image"] else None
        description = job.get("description")
        if description:
            if image is None:
                raise ValueError("Described scoring requires the original query image")
            values = model.score_described(
                batch, query_image=image, description=description
            )
        else:
            values = model.score(batch, query_image=image, query_text=job["query_text"])
        scores.update(
            (row["id"], float(value)) for row, value in zip(batch, values, strict=True)
        )
        temporary = path.with_suffix(".partial")
        write_json(temporary, saved)
        temporary.replace(path)
    return scores


def _run() -> None:
    """Score all twenty candidates per diagnostic query on the local full model."""
    config = PROJECT_ROOT / "configs"
    evidence = Evidence(load(config / "evaluation_registry.json"))
    jobs, queries, rankings = _plan(evidence)
    policy = dict(
        role="exposed_development_human_error_diagnostic",
        instruction=BUSINESS_INSTRUCTION,
        pool_size=POOL,
        inputs=dict(evidence.files),
        reranker_config_sha256=sha256_file(config / "reranker_model.json"),
        jobs=jobs,
    )
    path = ROOT / "precision_protocol.json"
    if path.is_file() and evidence.json(path) != policy:
        raise ValueError("Frozen precision protocol differs")
    write_json(path, policy)
    identity = sha256_file(path)
    finished = ROOT / "precision_results.json"
    if finished.is_file():
        if evidence.json(finished)["protocol_sha256"] != identity:
            raise ValueError("Completed precision results belong to another protocol")
        print("The complete Top20 results already match the frozen protocol.")
        return
    model = DescribedImageReranker()
    results = {}
    try:
        for qid, job in jobs.items():
            base = rankings[qid]
            order = rerank_head(base, _score_head(model, job, identity, qid), POOL)
            labels = queries[qid]["judgments"]
            grades = {cid: labels.get(cid, {}).get("grade") for cid in base}
            negative = job["reviewed_negative"]
            results[qid] = dict(
                mode=job["mode"],
                query_aliases=queries[qid]["sources"],
                scored_candidates=POOL,
                reviewed_negative=negative,
                rank_before=base.index(negative) + 1,
                rank_after=order.index(negative) + 1,
                base_top20=base[:POOL],
                reranked_top20=order[:POOL],
                top10_entering=sorted(set(order[:10]) - set(base[:10])),
                top10_leaving=sorted(set(base[:10]) - set(order[:10])),
                before=query_bounds(base, grades),
                after=query_bounds(order, grades),
            )
            print(
                f"Completed {len(results)}/{len(jobs)} exact Top20 reranks.", flush=True
            )
        runtime = model.diagnostics()
    finally:
        model.close()
    write_json(
        ROOT / "precision_results.json",
        dict(
            protocol_sha256=identity,
            source_sha256=sha256_file(Path(__file__)),
            query_results=results,
            scope="Three human-confirmed mismatch queries; no population estimate",
            total_model_scores=POOL * len(results),
            runtime=runtime,
            formal_metrics=None,
        ),
    )


def _chair_description(evidence: Evidence, job: dict) -> str:
    summary = evidence.json(CAPTIONS / "summary.json")
    digest = summary["protocol_sha256"]
    plan = evidence.json(CAPTIONS / "protocol.json", digest)
    original = next(row for row in plan["jobs"] if row["qid"] == "fresh_chair")
    if evidence.digest(Path(job["query_image"])) != original["image_sha256"]:
        raise ValueError("Caption was generated for a different query original")
    subjects = set()
    for name in ("4b", "9b"):
        relative = f"{name}/fresh_chair.json"
        record = evidence.json(
            CAPTIONS / relative, summary["prediction_sha256"][relative]
        )
        value = record["result"]["value"]
        if record["identity"]["protocol_sha256"] != digest or value["unclear"]:
            raise ValueError("Subject evidence is unclear or from another protocol")
        subjects.add(value["product"])
    if len(subjects) != 1:
        raise ValueError("The two query-only models disagree on the subject")
    return f"商品主体：{subjects.pop()}。"


def _chair() -> None:
    evidence = Evidence(load(PROJECT_ROOT / "configs/evaluation_registry.json"))
    jobs, queries, rankings = _plan(evidence)
    qid = next(key for key, job in jobs.items() if job["mode"] == "image")
    job = jobs[qid]
    description = _chair_description(evidence, job)
    archived = evidence.json(ROOT / "precision_results.json")
    prior_path = ROOT / "precision_protocol.json"
    prior = evidence.json(prior_path, archived["protocol_sha256"])
    if prior["jobs"][qid] != job:
        raise ValueError("The original-image rerank used a different pool or query")
    config = PROJECT_ROOT / "configs/reranker_model.json"
    if sha256_file(config) != prior["reranker_config_sha256"]:
        raise ValueError("Reranker configuration changed after the original comparison")
    policy = dict(
        prior_protocol_sha256=archived["protocol_sha256"], description=description
    )
    path = ROOT / "chair_subject_protocol.json"
    if path.is_file() and evidence.json(path) != policy:
        raise ValueError("Frozen chair subject protocol differs")
    write_json(path, policy)
    identity = sha256_file(path)
    model = DescribedImageReranker()
    try:
        scores = _score_head(
            model, dict(job, description=description), identity, qid + "/described"
        )
        runtime = model.diagnostics()
    finally:
        model.close()
    base = rankings[qid]
    order = rerank_head(base, scores, POOL)
    original = archived["query_results"][qid]
    negative = job["reviewed_negative"]
    labels = queries[qid]["judgments"]
    grades = {cid: labels.get(cid, {}).get("grade") for cid in base}
    write_json(
        ROOT / "chair_subject_results.json",
        dict(
            scope="One reviewed image query, original and described versions share Top20",
            query=qid,
            description=description,
            reviewed_negative=negative,
            raw_ranks=[original["rank_before"], original["rank_after"]],
            described_rank=order.index(negative) + 1,
            described_top20=order[:POOL],
            described_metrics=query_bounds(order, grades),
            top10_entering=sorted(
                set(order[:10]) - set(original["reranked_top20"][:10])
            ),
            top10_leaving=sorted(
                set(original["reranked_top20"][:10]) - set(order[:10])
            ),
            protocol_sha256=identity,
            inputs=evidence.files,
            source_sha256=sha256_file(Path(__file__)),
            runtime=runtime,
            formal_metrics=None,
        ),
    )
    print(
        f"Chair query with subject description, reviewed negative at rank {order.index(negative) + 1}."
    )


def main() -> None:
    """Run the three original-image reranks or the single-query subject ablation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chair-subject", action="store_true")
    with worker_lock():
        if parser.parse_args().chair_subject:
            _chair()
        else:
            _run()


if __name__ == "__main__":
    main()
