"""Run coarse Qwen retrieval on one development gallery and select missing labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

import run  # noqa: F401
from module2_development_benchmark import FOCUS, OUTPUT, REVIEW_GRADES, Evidence, load
from multimodal_ecommerce.ai_review_metrics import optional_mean_bounds
from multimodal_ecommerce.business_evaluation import METRICS
from multimodal_ecommerce.embedding import QwenEmbedder, model_configuration
from multimodal_ecommerce.human_audit_ui import render_review
from multimodal_ecommerce.interval_retrieval_metrics import query_bounds
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.simple_baseline import human_accuracy, stable_order
from simple_cycle_review import FIELDS

SIMPLE = PROJECT_ROOT / "artifacts/module2_simple_baseline_20261105"
INDEX = FOCUS / "index"
REVIEW = OUTPUT / "human_review"
IDENTITY = (
    "repository",
    "revision",
    "code_revision",
    "vendor_sha256",
    "backend",
    "dtype",
    "max_pixels",
    "max_length",
    "query_instruction",
    "document_instruction",
)
DEPTHS = (10, 20, 30, 50, 100)


def _checked_media(evidence: Evidence, row: dict) -> Path:
    path = PROJECT_ROOT / row["image_path"]
    evidence.digest(path, row["image_sha256"])
    return path


def _vector(evidence: Evidence, files: dict, filename: str) -> np.ndarray:
    relative = f"vectors/{filename}.npy"
    path = SIMPLE / relative
    evidence.digest(path, files[relative])
    return np.load(path, allow_pickle=False)


def _vector_caches(
    evidence: Evidence, catalog: dict, queries: dict
) -> tuple[dict, dict]:
    pins = evidence.json(FOCUS / "completion_receipt.json")["files"]
    manifest = evidence.json(INDEX / "manifest.json", pins["index/manifest.json"])
    runtime = evidence.json(SIMPLE / "embedding_runtime.json")
    config = model_configuration()
    for existing in (manifest["model"], runtime["config"]):
        if any(existing[name] != config[name] for name in IDENTITY):
            raise ValueError(
                "Archive and current embedding model specifications differ"
            )
    records = evidence.json(INDEX / "records.json", manifest["files"]["records.json"])
    bank_path = INDEX / "vectors.npz"
    evidence.digest(bank_path, manifest["files"][bank_path.name])
    docs = {}
    with np.load(bank_path, allow_pickle=False) as bank:
        for row, vector in zip(records, bank["regions"][:, 0, :], strict=True):
            cid = row["id"]
            if (
                catalog[cid]["image_sha256"] != row["image_sha256"]
                or row["views"][0]["kind"] != "global"
            ):
                raise ValueError(f"Historical global view is inconsistent: {cid}")
            docs[cid] = vector
    source = evidence.json(SIMPLE / "inputs.json")
    files = evidence.json(SIMPLE / "encoding_receipt.json")["files"]
    for cid, row in source["documents"].items():
        if cid in docs or catalog[cid]["image_sha256"] != row["source_sha256"]:
            raise ValueError(f"Conflicting whole-image vector: {cid}")
        digest = hashlib.sha256(cid.encode()).hexdigest()[:24]
        docs[cid] = _vector(evidence, files, f"{digest}_whole")
    aliases = {
        alias: key for key, query in queries.items() for alias in query["sources"]
    }
    query_vectors = {}
    for cohort, group in source["cohorts"].items():
        for row in group["queries"]:
            key = aliases[f"{cohort}/{row['id']}"]
            if queries[key]["image_sha256"] != row["source_sha256"]:
                raise ValueError(f"Query image differs from the cached original: {key}")
            query_vectors[key] = _vector(evidence, files, f"{cohort}_{row['id']}")
    return docs, query_vectors


def _retrieve(evidence: Evidence, catalog: dict, queries: dict) -> tuple[dict, dict]:
    docs, query_vectors = _vector_caches(evidence, catalog, queries)
    missing_docs = sorted(set(catalog) - set(docs))
    missing_queries = sorted(set(queries) - set(query_vectors))
    gallery_images = [_checked_media(evidence, catalog[cid]) for cid in missing_docs]
    query_images = {
        key: _checked_media(evidence, queries[key])
        for key in missing_queries
        if queries[key]["mode"] == "image"
    }
    model = QwenEmbedder()
    try:
        for start in range(0, len(missing_docs), 4):
            vectors = model.encode_images(gallery_images[start : start + 4])
            docs.update(zip(missing_docs[start : start + 4], vectors, strict=True))
        print(f"Gallery vectors ready for {len(docs)} products.", flush=True)
        for key in missing_queries:
            query_vectors[key] = model.encode(
                image=query_images.get(key), text=queries[key].get("text"), query=True
            )
        runtime = model.diagnostics()
    finally:
        model.close()
    ids = sorted(catalog)
    matrix = np.stack([docs[cid] for cid in ids])
    vectors = np.stack([query_vectors[key] for key in queries])
    if (
        matrix.shape != (len(ids), 2048)
        or vectors.shape != (len(queries), 2048)
        or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-3)
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3)
    ):
        raise ValueError("Whole-gallery embeddings must be finite unit vectors")
    rankings = {}
    for key, scores in zip(queries, vectors @ matrix.T, strict=True):
        exclusions = set(queries[key]["exclusions"])
        rankings[key] = [
            cid for cid in stable_order(ids, scores) if cid not in exclusions
        ]
    return rankings, dict(
        reused_product_vectors=len(catalog) - len(missing_docs),
        encoded_products=len(missing_docs),
        reused_query_vectors=len(queries) - len(missing_queries),
        encoded_queries=len(missing_queries),
        model_runtime=runtime,
    )


def _score(rankings: dict, queries: dict) -> dict:
    rows = {}
    for key, order in rankings.items():
        observed = queries[key]["judgments"]
        grades = {cid: observed.get(cid, {}).get("grade") for cid in order}
        bounds = query_bounds(order, grades)
        relevant = {cid for cid, grade in grades.items() if grade == 1}
        rows[key] = dict(
            **bounds,
            missing_top10=sum(cid not in observed for cid in order[:10]),
            missing_top20=sum(cid not in observed for cid in order[:20]),
            documented_hits_by_depth={
                k: len(relevant.intersection(order[:k])) for k in DEPTHS
            },
        )
    scopes = {}
    for scope in ("all", "image", "text"):
        current = [
            row for key, row in rows.items() if scope in ("all", queries[key]["mode"])
        ]
        ranges = {
            metric: [
                sum(row["ranges"][metric][index] for row in current) / len(current)
                for index in range(2)
            ]
            for metric in METRICS
            if metric != "recall@10"
        }
        recall = [
            tuple(row["ranges"]["recall@10"])
            for row in current
            if row["known_positive"]
        ]
        optional = [
            tuple(row["ranges"]["recall@10"])
            for row in current
            if row["unknown"] and not row["known_positive"]
        ]
        ranges["recall@10"] = list(optional_mean_bounds(recall, optional))
        positives = [row for row in current if row["known_positive"]]
        scopes[scope] = dict(
            ranges=ranges,
            queries=len(current),
            missing_top10=sum(row["missing_top10"] for row in current),
            missing_top20=sum(row["missing_top20"] for row in current),
            documented_positive_queries=len(positives),
            queries_with_over_ten_documented_positives=sum(
                row["known_positive"] > 10 for row in positives
            ),
            documented_positive_recall_ceiling_at_10=(
                sum(
                    min(10, row["known_positive"]) / row["known_positive"]
                    for row in positives
                )
                / len(positives)
                if positives
                else None
            ),
            documented_positive_coverage_by_depth={
                k: (
                    sum(
                        row["documented_hits_by_depth"][k] / row["known_positive"]
                        for row in positives
                    )
                    / len(positives)
                    if positives
                    else None
                )
                for k in DEPTHS
            },
        )
    return dict(scopes=scopes, query_metrics=rows)


def _review_tasks(
    evidence: Evidence, rankings: dict, catalog: dict, queries: dict
) -> list[dict]:
    tasks = []
    for mode in ("image", "text"):
        candidates = []
        for key, row in queries.items():
            if row["mode"] == mode:
                for rank, cid in enumerate(rankings[key][:10], 1):
                    if cid not in row["judgments"]:
                        candidates.append((rank, key, cid))
                        break
        for _, key, cid in sorted(candidates)[:12]:
            row, product = queries[key], catalog[cid]
            image = _checked_media(evidence, row) if mode == "image" else None
            candidate = _checked_media(evidence, product)
            task_id = hashlib.sha256(f"{key}\n{cid}".encode()).hexdigest()[:24]
            tasks.append(
                dict(
                    task_id=task_id,
                    query_id=key,
                    candidate_id=cid,
                    query=dict(
                        text=row.get("text"), image_path=str(image) if image else None
                    ),
                    candidate=dict(title=product["title"], image_path=str(candidate)),
                )
            )
    tasks.sort(key=lambda task: task["task_id"])
    return tasks


def _review(evidence: Evidence, rankings: dict, catalog: dict, queries: dict) -> int:
    tasks = _review_tasks(evidence, rankings, catalog, queries)
    frozen = REVIEW / "tasks.private.json"
    if frozen.exists() and load(frozen) != tasks:
        raise ValueError("Existing review tasks differ; preserve the frozen packet")
    render_review(
        tasks,
        REVIEW,
        title="模块二 · 统一图库高排名候选复核",
        instructions=(
            "按原标准判断候选与查询是否相关。结合商品主体、用途与明确属性。"
            + "请选相关、不相关或证据不足，写一句理由，导出 CSV 放到 answers 文件夹。"
        ),
    )
    write_json(frozen, tasks)
    return len(tasks)


def _score_review() -> None:
    evidence = Evidence(load(PROJECT_ROOT / "configs/evaluation_registry.json"))
    result = evidence.json(OUTPUT / "shared_retrieval.json")
    manifest = OUTPUT / "shared_gallery.json"
    shared = evidence.json(
        manifest, result["files"][manifest.relative_to(PROJECT_ROOT).as_posix()]
    )
    rankings = evidence.json(OUTPUT / "shared_rankings.json", result["rankings_sha256"])
    tasks = evidence.json(REVIEW / "tasks.private.json")
    queries = shared["query_manifest"]
    catalog = {row["id"]: row for row in shared["catalog"]}
    if tasks != _review_tasks(evidence, rankings, catalog, queries):
        raise ValueError("Review tasks differ from the frozen full-gallery ranking")
    answers: list[dict] = []
    for path in sorted((REVIEW / "answers").glob("*.csv")):
        evidence.digest(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                raise ValueError(f"Unexpected human CSV fields: {path.name}")
            answers.extend(reader)
    progress = human_accuracy([dict(task, grade=None) for task in tasks], answers)
    by_id = {task["task_id"]: task for task in tasks}
    counts = dict.fromkeys(REVIEW_GRADES, 0)
    for row in answers:
        grade = row["human_grade"]
        if grade.strip():
            task = by_id[row["task_id"]]
            queries[task["query_id"]]["judgments"][task["candidate_id"]] = dict(
                grade=REVIEW_GRADES[grade], human=True
            )
            counts[grade] += 1
    write_json(
        REVIEW / "progress.json",
        dict(
            reviewed=progress["completed"],
            expected=progress["expected"],
            human_counts=counts,
            metrics=_score(rankings, queries),
            inputs=evidence.files,
            source_sha256=sha256_file(Path(__file__)),
            formal_metrics=None,
        ),
    )
    print(f"Human review: {progress['completed']}/{progress['expected']}.")


def _run() -> None:
    registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
    evidence = Evidence(registry)
    manifest = OUTPUT / "shared_gallery.json"
    receipt = evidence.json(OUTPUT / "results.json")
    shared = evidence.json(manifest, receipt["shared_gallery_sha256"])
    catalog = {row["id"]: row for row in shared["catalog"]}
    queries = shared["query_manifest"]
    rankings, runtime = _retrieve(evidence, catalog, queries)
    metrics = _score(rankings, queries)
    tasks = _review(evidence, rankings, catalog, queries)
    path = OUTPUT / "shared_rankings.json"
    write_json(path, rankings)
    write_json(
        OUTPUT / "shared_retrieval.json",
        dict(
            role="exposed_development_coarse_retrieval",
            products=len(catalog),
            queries=len(queries),
            method="Qwen global cosine, fixed full gallery, no reranking",
            metrics=metrics,
            runtime=runtime,
            human_review_tasks=tasks,
            rankings_sha256=sha256_file(path),
            files=evidence.files,
            source_sha256=sha256_file(Path(__file__)),
            formal_metrics=None,
        ),
    )
    print(
        f"Searched {len(catalog)} products with {len(queries)} queries. Review pairs: {tasks}."
    )


def main() -> None:
    """Run the full-gallery retrieval or import submitted blind review decisions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-review", action="store_true")
    if parser.parse_args().score_review:
        _score_review()
    else:
        _run()


if __name__ == "__main__":
    main()
