"""Retrieve original titles with query-only subject descriptions and rerank Top20."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np

import run  # noqa: F401
from module2_development_benchmark import FOCUS, Evidence, load
from module2_shared_evaluation import (
    merge_previous_reviews,
    precision_delta,
    recall_ceiling,
    remaining_subject_tasks,
)
from module2_shared_gallery_fusion_review import FUSION, _apply_answers
from module2_shared_gallery_precision import ROOT, worker_lock
from module2_shared_gallery_retrieval import _checked_media, _score
from module2_shared_gallery_route import CASE_REVIEW, _inputs
from multimodal_ecommerce.business_retrieval import BUSINESS_INSTRUCTION
from multimodal_ecommerce.embedding import QwenEmbedder, model_configuration
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.head_guard import protected_prefix
from multimodal_ecommerce.human_audit_ui import render_review
from multimodal_ecommerce.machine_teacher import MODEL
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.simple_baseline import rerank_head, stable_order
from multimodal_ecommerce.visual_query import (
    CAPTION_OPTIONS,
    CAPTION_PROMPT,
    CAPTION_SCHEMA,
    DescribedImageReranker,
    caption_text,
    describe_query,
)

OUTPUT = ROOT / "subject_recall"
POOL = 20
ANCHORS = 10


def _context(evidence: Evidence) -> tuple[dict, dict]:
    manifest, orders = _inputs(evidence)
    review = evidence.json(CASE_REVIEW / "progress.json")
    path = CASE_REVIEW / "tasks.private.json"
    tasks = evidence.json(
        path, review["inputs"][path.relative_to(PROJECT_ROOT).as_posix()]
    )
    for path in (CASE_REVIEW / "answers").glob("*.csv"):
        evidence.digest(
            path, review["inputs"][path.relative_to(PROJECT_ROOT).as_posix()]
        )
    if not _apply_answers(evidence, CASE_REVIEW, tasks, manifest["query_manifest"])[
        "complete"
    ]:
        raise ValueError("The previous chair review must be complete")
    manifest["historical_review_import"] = merge_previous_reviews(evidence, manifest)
    return manifest, orders


def _captions(evidence: Evidence, queries: dict) -> dict:
    images = {
        key: dict(path=str(_checked_media(evidence, row)), sha256=row["image_sha256"])
        for key, row in queries.items()
        if row["mode"] == "image"
    }
    client = OllamaClient(model=MODEL)
    try:
        health = client.health()
        models = [row for row in health["models"] if row["name"] == MODEL]
        if len(models) != 1:
            raise ValueError("The pinned local vision model must be installed")
        identity = dict(
            model=MODEL,
            digest=models[0]["digest"],
            images=images,
            prompt=CAPTION_PROMPT,
            schema=CAPTION_SCHEMA,
            options=CAPTION_OPTIONS,
        )
        path = OUTPUT / "query_captions.json"
        require_data_access([path], "development", evidence.registry)
        saved: dict = load(path) if path.exists() else dict(identity=identity, rows={})
        if saved["identity"] != identity or not set(saved["rows"]) <= set(images):
            raise ValueError("The query-only caption cache has different inputs")
        pending = [key for key in images if key not in saved["rows"]]
        for number, key in enumerate(pending, 1):
            result = describe_query(
                client, Path(images[key]["path"]), last=number == len(pending)
            )
            saved["rows"][key] = dict(
                value=result["value"], seconds=result["wall_seconds"]
            )
            temporary = path.with_suffix(".partial")
            write_json(temporary, saved)
            temporary.replace(path)
            if number % 6 == 0 or number == len(pending):
                print(
                    f"Query-only subject descriptions: {len(saved['rows'])}/{len(images)}.",
                    flush=True,
                )
        evidence.digest(path)
    finally:
        client.close()
    return {key: caption_text(row["value"]) for key, row in saved["rows"].items()}


def _title_vectors(
    evidence: Evidence, catalog: dict, texts: dict
) -> tuple[np.ndarray, np.ndarray]:
    index = FOCUS / "index"
    receipt = evidence.json(FOCUS / "completion_receipt.json")["files"]
    manifest = evidence.json(index / "manifest.json", receipt["index/manifest.json"])
    records = evidence.json(index / "records.json", manifest["files"]["records.json"])
    config = model_configuration()
    identity = (
        "repository",
        "revision",
        "vendor_sha256",
        "max_pixels",
        "max_length",
        "document_instruction",
        "query_instruction",
    )
    if any(manifest["model"][key] != config[key] for key in identity):
        raise ValueError(
            "Title embedding specification differs from the archived model"
        )
    path = index / "vectors.npz"
    evidence.digest(path, manifest["files"][path.name])
    docs = {}
    with np.load(path, allow_pickle=False) as bank:
        for row, vector in zip(records, bank["texts"], strict=True):
            if row["title"] != catalog[row["id"]]["title"]:
                raise ValueError("Archived title changed")
            docs[row["id"]] = vector
    model = QwenEmbedder()
    try:
        missing = sorted(set(catalog) - set(docs))
        for number, cid in enumerate(missing, 1):
            docs[cid] = model.encode(text=catalog[cid]["title"])
            if number % 100 == 0 or number == len(missing):
                print(f"New title encodings: {number}/{len(missing)}.", flush=True)
        vectors = {
            key: model.encode(text=text, query=True) if text else np.zeros(2048)
            for key, text in texts.items()
        }
        write_json(OUTPUT / "embedding_runtime.json", model.diagnostics())
    finally:
        model.close()
    return np.stack([docs[cid] for cid in sorted(catalog)]), np.stack(
        [vectors[key] for key in texts]
    )


def _prepare(evidence: Evidence) -> None:
    manifest, orders = _context(evidence)
    queries = manifest["query_manifest"]
    catalog = {row["id"]: row for row in manifest["catalog"]}
    captions = _captions(evidence, queries)
    texts = {
        key: row["text"] if row["mode"] == "text" else captions[key]
        for key, row in queries.items()
    }
    document_vectors, query_vectors = _title_vectors(evidence, catalog, texts)
    if not np.isfinite(document_vectors).all() or not np.isfinite(query_vectors).all():
        raise ValueError("Nonfinite title vectors")
    ids = sorted(catalog)
    scores = query_vectors @ document_vectors.T
    jobs = {}
    for (key, query), values in zip(queries.items(), scores, strict=True):
        baseline = orders[FUSION if query["mode"] == "text" else "whole"][key]
        excluded = set(query["exclusions"])
        title = [cid for cid in stable_order(ids, values) if cid not in excluded]
        pool = baseline[:ANCHORS]
        pool += [cid for cid in title if cid not in set(pool)][: POOL - ANCHORS]
        if not texts[key]:
            pool = baseline[:POOL]
        proposed = pool + [cid for cid in baseline if cid not in set(pool)]
        if (
            len(pool) != POOL
            or set(proposed) != set(baseline)
            or len(proposed) != len(baseline)
        ):
            raise ValueError("Subject recall must preserve the eligible gallery")
        jobs[key] = dict(
            query_text=query.get("text"),
            description=captions.get(key, ""),
            query_image=(
                str(_checked_media(evidence, query))
                if query["mode"] == "image"
                else None
            ),
            baseline=baseline,
            proposed=proposed,
            documents=[
                dict(
                    id=cid,
                    image_path=str(_checked_media(evidence, catalog[cid])),
                    text=catalog[cid]["title"],
                )
                for cid in pool
            ],
        )
    protocol = dict(
        role="exposed_development_query_subject_title_recall",
        pool_size=POOL,
        original_head_reserved=ANCHORS,
        selection="Keep baseline Top10, add ten distinct title-embedding candidates.",
        instruction=BUSINESS_INSTRUCTION,
        model_sha256=sha256_file(PROJECT_ROOT / "configs/reranker_model.json"),
        jobs=jobs,
        files=evidence.files,
    )
    write_json(OUTPUT / "protocol.json", protocol)
    write_json(
        OUTPUT / "recall_results.json",
        dict(
            protocol_sha256=sha256_file(OUTPUT / "protocol.json"),
            baseline=_score(
                {key: job["baseline"] for key, job in jobs.items()}, queries
            ),
            subject_recall=_score(
                {key: job["proposed"] for key, job in jobs.items()}, queries
            ),
            formal_metrics=None,
        ),
    )
    print(
        f"Frozen all {len(jobs)} subject-recall jobs, {POOL} candidates each.",
        flush=True,
    )


def _scores(
    evidence: Evidence,
    model: DescribedImageReranker,
    job: dict,
    binding: tuple[Path, str],
    key: str,
) -> dict:
    directory, identity = binding
    path = directory / (hashlib.sha256(key.encode()).hexdigest() + ".json")
    require_data_access([path], "development", evidence.registry)
    saved: dict = load(path) if path.exists() else dict(identity=identity, scores={})
    ids = {row["id"] for row in job["documents"]}
    if (
        saved["identity"] != identity
        or not set(saved["scores"]) <= ids
        or not all(math.isfinite(value) for value in saved["scores"].values())
    ):
        raise ValueError("Top20 score cache changed")
    pending = [row for row in job["documents"] if row["id"] not in saved["scores"]]
    for start in range(0, len(pending), 5):
        batch = pending[start : start + 5]
        image = Path(job["query_image"]) if job["query_image"] else None
        if image is not None and job["description"]:
            values = model.score_described(
                batch, query_image=image, description=job["description"]
            )
        else:
            values = model.score(batch, query_image=image, query_text=job["query_text"])
        saved["scores"].update(
            (row["id"], float(score)) for row, score in zip(batch, values, strict=True)
        )
        temporary = path.with_suffix(".partial")
        write_json(temporary, saved)
        temporary.replace(path)
    return saved["scores"]


def _evaluate(evidence: Evidence, protocol: dict, scores: dict) -> None:
    manifest, _ = _context(evidence)
    queries = manifest["query_manifest"]
    raw = {
        key: rerank_head(job["proposed"], scores[key], POOL)
        for key, job in protocol["jobs"].items()
    }
    baseline = {key: job["baseline"] for key, job in protocol["jobs"].items()}
    protected = {
        str(keep): {
            key: protected_prefix(baseline[key], order, keep)
            for key, order in raw.items()
        }
        for keep in (1, 2)
    }
    rankings = {
        key: protected["2"][key] if query["mode"] == "text" else baseline[key]
        for key, query in queries.items()
    }
    path = OUTPUT / "rankings.json"
    write_json(path, rankings)
    write_json(
        OUTPUT / "results.json",
        dict(
            role="exposed_development_complete_top20_subject_recall_and_rerank",
            protocol_sha256=sha256_file(OUTPUT / "protocol.json"),
            source_sha256=sha256_file(Path(__file__)),
            rankings_sha256=sha256_file(path),
            queries=len(rankings),
            products=len(manifest["catalog"]),
            scored_pairs=sum(len(value) for value in scores.values()),
            baseline=_score(baseline, queries),
            recall_pool=_score(
                {key: job["proposed"] for key, job in protocol["jobs"].items()}, queries
            ),
            candidate=_score(rankings, queries),
            unprotected=_score(raw, queries),
            prefix_controls={
                keep: _score(order, queries) for keep, order in protected.items()
            },
            selected_policy={
                "text": "subject Top20 rerank with original first two protected",
                "image": "baseline",
            },
            paired_precision=precision_delta(baseline, rankings, queries),
            optimistic_ideal_recall_at_10=recall_ceiling(queries),
            historical_review_import=manifest["historical_review_import"],
            inputs=evidence.files,
            formal_metrics=None,
        ),
    )


def _read_scores(evidence: Evidence, protocol: dict, identity: str) -> dict:
    scores = {}
    for key, job in protocol["jobs"].items():
        path = OUTPUT / "scores" / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        row = evidence.json(path)
        expected = {doc["id"] for doc in job["documents"]}
        if (
            row["identity"] != identity
            or set(row["scores"]) != expected
            or not all(math.isfinite(value) for value in row["scores"].values())
        ):
            raise ValueError("The fixed Top20 model scores must be complete")
        scores[key] = row["scores"]
    return scores


def _rescore(evidence: Evidence) -> None:
    path = OUTPUT / "protocol.json"
    protocol = evidence.json(path)
    for name, digest in protocol["files"].items():
        evidence.digest(PROJECT_ROOT / name, digest)
    _evaluate(evidence, protocol, _read_scores(evidence, protocol, sha256_file(path)))


def _review(evidence: Evidence, score_only: bool, remaining: bool = False) -> None:
    manifest, _ = _context(evidence)
    queries = manifest["query_manifest"]
    catalog = {row["id"]: row for row in manifest["catalog"]}
    result = evidence.json(OUTPUT / "results.json")
    ranks = evidence.json(OUTPUT / "rankings.json", result["rankings_sha256"])
    protocol = evidence.json(OUTPUT / "protocol.json", result["protocol_sha256"])
    before = {key: job["baseline"] for key, job in protocol["jobs"].items()}
    tasks = []
    for mode in ("image", "text"):
        for incoming in (True, False):
            rows = []
            for key, query in queries.items():
                if query["mode"] != mode:
                    continue
                a, b = (ranks, before) if incoming else (before, ranks)
                for rank, cid in enumerate(a[key][:10], 1):
                    label = query["judgments"].get(cid, {})
                    if cid not in b[key][:10] and not label.get("human"):
                        rows.append((rank, key, cid))
                        break
            for _, key, cid in sorted(rows)[:6]:
                query, doc = queries[key], catalog[cid]
                image = (
                    str(_checked_media(evidence, query)) if mode == "image" else None
                )
                tasks.append(
                    dict(
                        task_id=hashlib.sha256(
                            f"{key}\n{cid}\nsubject-recall".encode()
                        ).hexdigest()[:24],
                        query_id=key,
                        candidate_id=cid,
                        query=dict(text=query.get("text"), image_path=image),
                        candidate=dict(
                            title=doc["title"],
                            image_path=str(_checked_media(evidence, doc)),
                        ),
                    )
                )
    tasks.sort(key=lambda row: row["task_id"])
    folder = OUTPUT / ("text_complete/human_review" if remaining else "human_review")
    if remaining:
        tasks = remaining_subject_tasks(evidence, manifest, before, ranks)
    path = folder / "tasks.private.json"
    if path.exists() and evidence.json(path) != tasks:
        raise ValueError("Preserve the frozen subject-recall review packet")
    if score_only and not path.exists():
        raise ValueError("Prepare the subject-recall review packet first")
    if not score_only:
        render_review(
            tasks,
            folder,
            title=(
                "模块二 · 文搜全部剩余换位复核"
                if remaining
                else "模块二 · 主体召回与精排复核"
            ),
            instructions=(
                "按原标准核对商品主体、用途与明确属性。"
                + "选相关、不相关或证据不足，写一句理由，CSV 导出到本目录 answers 文件夹。"
            ),
        )
        if not path.exists():
            write_json(path, tasks)
            evidence.digest(path)
    progress = _apply_answers(evidence, folder, tasks, queries)
    write_json(
        folder / "progress.json",
        dict(
            reviewed=progress["completed"],
            expected=progress["expected"],
            baseline=_score(before, queries),
            candidate=_score(ranks, queries),
            paired_precision=precision_delta(before, ranks, queries),
            inputs=evidence.files,
            formal_metrics=None,
        ),
    )
    print(
        f"Subject-recall human review: {progress['completed']}/{progress['expected']}."
    )


def _rerank(evidence: Evidence) -> None:
    path = OUTPUT / "protocol.json"
    protocol = evidence.json(path)
    identity = sha256_file(path)
    if (
        protocol["model_sha256"]
        != sha256_file(PROJECT_ROOT / "configs/reranker_model.json")
        or protocol["instruction"] != BUSINESS_INSTRUCTION
    ):
        raise ValueError("Reranker configuration changed")
    for name, digest in protocol["files"].items():
        evidence.digest(PROJECT_ROOT / name, digest)
    scores = {}
    model = DescribedImageReranker()
    try:
        for key, job in protocol["jobs"].items():
            scores[key] = _scores(
                evidence, model, job, (OUTPUT / "scores", identity), key
            )
            write_json(
                OUTPUT / "status.json",
                dict(completed=len(scores), total=len(protocol["jobs"])),
            )
            print(
                f"Full-model Top20 complete: {len(scores)}/{len(protocol['jobs'])}.",
                flush=True,
            )
        runtime = model.diagnostics()
    finally:
        model.close()
    write_json(OUTPUT / "reranker_runtime.json", runtime)
    _read_scores(evidence, protocol, identity)
    _evaluate(evidence, protocol, scores)


def main() -> None:
    """Prepare or run fixed-pool subject recall across every shared development query."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "prepare",
            "rerank",
            "evaluate",
            "review",
            "score-review",
            "remaining-review",
            "score-remaining",
        ),
    )
    args = parser.parse_args()
    evidence = Evidence(load(PROJECT_ROOT / "configs/evaluation_registry.json"))
    with worker_lock():
        if args.phase == "prepare":
            if (OUTPUT / "protocol.json").exists():
                raise FileExistsError("Preserve the frozen subject-recall protocol")
            _prepare(evidence)
        elif args.phase == "rerank":
            _rerank(evidence)
        elif args.phase == "evaluate":
            _rescore(evidence)
        else:
            _review(
                evidence,
                args.phase in ("score-review", "score-remaining"),
                args.phase in ("remaining-review", "score-remaining"),
            )


if __name__ == "__main__":
    main()
