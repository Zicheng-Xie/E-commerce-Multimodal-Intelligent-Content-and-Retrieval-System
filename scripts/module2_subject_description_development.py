"""Run a fixed-model subject-description ablation using development data only."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import run as _bootstrap  # noqa: F401
from module2_calibrated_development import development_inputs
from module2_development_benchmark import FOCUS, Evidence, load
from module2_mainline import REVISION
from module2_shared_evaluation import precision_delta
from module2_shared_gallery_retrieval import _score
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.human_retrieval_evaluation import evaluate_human_retrieval
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.selected_release import verify_release
from multimodal_ecommerce.selected_runtime_io import ContentCache, RuntimeAccess
from multimodal_ecommerce.simple_baseline import stable_order
from multimodal_ecommerce.subject_description import (
    clean_caption,
    prop_conflict_ids,
    subject_swap,
)
from multimodal_ecommerce.visual_query import DescribedImageReranker, caption_text

OUTPUT = (
    PROJECT_ROOT / "artifacts/module2_selected_runtime_20261110/subject_description"
)
SOURCES = (
    "src/multimodal_ecommerce/subject_description.py",
    "scripts/module2_subject_description_development.py",
)


def public_jobs(evidence: Evidence, manifest: dict, baseline: dict) -> dict:
    """Gate inference using explicit foreign props in public query descriptions alone.

    Args:
        evidence: Development-only reader.
        manifest: Development public catalog and query metadata.
        baseline: Fixed selected model rankings.

    Returns:
        Jobs with public query/image/title inputs and no relevance answers.
    """
    captions = evidence.json(REVISION.parent / "subject_recall/query_captions.json")[
        "rows"
    ]
    titles = {r["id"]: r["title"] for r in manifest["catalog"]}
    jobs = {}
    for key, query in manifest["query_manifest"].items():
        if query["mode"] != "image":
            continue
        observation = clean_caption(captions[key]["value"])
        if not prop_conflict_ids(baseline[key], titles, observation):
            continue
        image = PROJECT_ROOT / query["image_path"]
        evidence.digest(image, query["image_sha256"])
        jobs[key] = dict(
            image_path=str(image),
            image_sha256=query["image_sha256"],
            observation=observation,
            order=baseline[key],
        )
    return jobs


def title_vectors(
    evidence: Evidence, catalog: dict, cache: ContentCache, model: QwenEmbedder
) -> dict:
    """Reuse source-bound title embeddings and encode only missing public titles.

    Args:
        evidence: Development cache reader.
        catalog: Public product catalog.
        cache: Full model/policy identity cache.
        model: Actual fixed multimodal embedding model.

    Returns:
        Candidate ID to actual normalized title embedding.
    """
    pins = evidence.json(FOCUS / "completion_receipt.json")["files"]
    folder = FOCUS / "index"
    manifest = evidence.json(folder / "manifest.json", pins["index/manifest.json"])
    records = evidence.json(folder / "records.json", manifest["files"]["records.json"])
    evidence.digest(folder / "vectors.npz", manifest["files"]["vectors.npz"])
    for key in (
        "repository",
        "revision",
        "max_pixels",
        "max_length",
        "query_instruction",
        "document_instruction",
        "vendor_sha256",
    ):
        if model.config[key] != manifest["model"][key]:
            raise ValueError(
                "The fixed embedding identity differs from the title cache"
            )
    result = {}
    with np.load(folder / "vectors.npz", allow_pickle=False) as bank:
        for row, vector in zip(records, bank["texts"], strict=True):
            cid = row["id"]
            if (
                row["title"] != catalog[cid]["title"]
                or row["image_sha256"] != catalog[cid]["image_sha256"]
            ):
                raise ValueError("Public title cache belongs to another product")
            result[cid] = vector
    missing = sorted(set(catalog) - set(result))
    for number, cid in enumerate(missing, 1):
        cache_key = cache.key(
            "title_embedding", dict(text=catalog[cid]["title"], query=False)
        )
        value = cache.get(cache_key)
        if value is None:
            value = model.encode(text=catalog[cid]["title"]).tolist()
            cache.put(cache_key, value)
        result[cid] = np.asarray(value, dtype=np.float32)
        if number % 100 == 0 or number == len(missing):
            print(f"Missing title encodings {number}/{len(missing)}", flush=True)
    if any(
        v.shape != (2048,)
        or not np.isfinite(v).all()
        or not np.isclose(np.linalg.norm(v), 1, atol=1e-3)
        for v in result.values()
    ):
        raise ValueError("Actual finite unit title vectors required")
    return result


def retrieve(
    evidence: Evidence, catalog: dict, jobs: dict, cache: ContentCache
) -> tuple[dict, dict]:
    """Compare original and cleaned expansion using the same exact title index.

    Args:
        evidence: Source-bound development reader.
        catalog: Public catalog.
        jobs: Public inference jobs.
        cache: Full identity cache.

    Returns:
        Two fixed Top20 pools per job and actual embedding diagnostics.
    """
    pools: dict = {"full_caption_control": {}, "subject_caption": {}}
    model = QwenEmbedder()
    try:
        documents = title_vectors(evidence, catalog, cache, model)
        for key, job in jobs.items():
            ids = sorted(job["order"])
            matrix = np.stack([documents[c] for c in ids])
            for method, field in (
                ("full_caption_control", "original"),
                ("subject_caption", "cleaned"),
            ):
                description = caption_text(job["observation"][field])
                identity = cache.key("query_text", dict(text=description, query=True))
                value = cache.get(identity)
                if value is None:
                    value = model.encode(text=description, query=True).tolist()
                    cache.put(identity, value)
                vector = np.asarray(value, dtype=np.float32)
                if (
                    vector.shape != (2048,)
                    or not np.isfinite(vector).all()
                    or not np.isclose(np.linalg.norm(vector), 1, atol=1e-3)
                ):
                    raise ValueError("Actual finite query unit embedding required")
                ranking = stable_order(ids, vector @ matrix.T)
                anchors = job["order"][:10]
                head = anchors + [c for c in ranking if c not in set(anchors)][:10]
                pools[method][key] = dict(pool=head, description=description)
        diagnostics = model.diagnostics()
    finally:
        model.close()
    return pools, diagnostics


def infer(
    catalog: dict, jobs: dict, pools: dict, cache: ContentCache
) -> tuple[dict, dict]:
    """Score real query/candidate images after fully releasing the embedding stage.

    Args:
        catalog: Public product originals.
        jobs: Public query originals.
        pools: Source-bound query descriptions and Top20 candidate sets.
        cache: Full identity cache.

    Returns:
        Actual logits per condition and reranker diagnostics.
    """
    model = DescribedImageReranker()
    scores: dict = {}
    completed = 0
    try:
        for method, rows in pools.items():
            scores[method] = {}
            for key, row in rows.items():
                scores[method][key] = {}
                for cid in row["pool"]:
                    doc = catalog[cid]
                    identity = cache.key(
                        "cross_logit",
                        dict(
                            query_image_sha256=jobs[key]["image_sha256"],
                            description=row["description"],
                            document_image_sha256=doc["image_sha256"],
                            title=doc["title"],
                        ),
                    )
                    value = cache.get(identity)
                    if value is None:
                        value = float(
                            model.score_described(
                                [
                                    dict(
                                        id="document",
                                        text=doc["title"],
                                        image_path=str(
                                            PROJECT_ROOT / doc["image_path"]
                                        ),
                                    )
                                ],
                                query_image=Path(jobs[key]["image_path"]),
                                description=row["description"],
                            )[0]
                        )
                        cache.put(identity, value)
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        raise ValueError(
                            "A real numeric cross-encoder logit is required"
                        )
                    scores[method][key][cid] = float(value)
                    completed += 1
                    if completed % 10 == 0:
                        print(
                            f"Real cross-encoder pairs {completed}/{40 * len(jobs)}",
                            flush=True,
                        )
        diagnostics = model.diagnostics()
    finally:
        model.close()
    return scores, diagnostics


def change_evidence(before: dict, after: dict, queries: dict) -> dict:
    """Check each complete position swap against existing human answers after inference.

    Args:
        before: Full fixed baseline.
        after: Full new rankings.
        queries: Existing development human judgments, never model inputs.

    Returns:
        Known-negative removal evidence and unresolved incoming counts.
    """
    rows = []
    for key, order in before.items():
        changed = [i for i, c in enumerate(order) if c != after[key][i]]
        if not changed:
            continue
        if len(changed) != 2:
            raise ValueError("Only one full position swap is allowed")
        i, j = changed
        if (
            i < 2
            or i >= 10
            or j < 10
            or order[i] != after[key][j]
            or order[j] != after[key][i]
        ):
            raise ValueError(
                "Protected prefix and the complete eligible universe required"
            )
        labels = queries[key]["judgments"]
        grades = [
            labels.get(c, {}).get("grade") if labels.get(c, {}).get("human") else None
            for c in (order[i], order[j])
        ]
        rows.append(
            dict(
                query_id=key,
                outgoing=order[i],
                incoming=order[j],
                outgoing_human=grades[0],
                incoming_human=grades[1],
                incoming_old_rank=j + 1,
                target_rank=i + 1,
            )
        )
    safe = bool(rows) and all(r["outgoing_human"] == 0 for r in rows)
    return dict(
        rows=rows,
        known_negative_removals=sum(r["outgoing_human"] == 0 for r in rows),
        known_positive_incoming=sum(r["incoming_human"] == 1 for r in rows),
        unresolved_incoming=sum(r["incoming_human"] is None for r in rows),
        all_six_nondecreasing_for_every_human_label_completion=safe,
        default_changed=False,
    )


def execute(output: Path) -> dict:
    """Run the predeclared model-only description ablation on all development queries.

    Args:
        output: Separate exposed-development result directory.

    Returns:
        Honest same-label metric ranges and actual inference evidence.
    """
    output = RuntimeAccess().output(output)
    started = time.perf_counter()
    evidence = Evidence(load(PROJECT_ROOT / "configs/evaluation_registry.json"))
    manifest, baseline, selected = development_inputs(evidence)
    catalog = {r["id"]: r for r in manifest["catalog"]}
    jobs = public_jobs(evidence, manifest, baseline)
    release = verify_release(
        PROJECT_ROOT / "artifacts/module2_selected_runtime_20261110/release.json"
    )
    protocol = dict(
        role="exposed_development_subject_description_control",
        products=845,
        queries=55,
        model_policy=release["model_policy"],
        baseline=selected["candidate_id"],
        policy="Remove only explicit foreign prop clauses; preserve all original Top10 as scored anchors; retrieve ten new title candidates; at most one public-type- and model-supported prop swap; keep first two.",
        pool_size=20,
        gated_queries=list(jobs),
        human_labels_used_in_public_gate=False,
        jobs=jobs,
        inputs=dict(evidence.files),
        source_sha256={n: sha256_file(PROJECT_ROOT / n) for n in SOURCES},
        new_human_tasks=0,
        acceptance_data_used=False,
    )
    path = output / "protocol.json"
    if path.exists() and load(path) != protocol:
        raise ValueError(
            "Preserve completed protocol; use a separate experiment output"
        )
    write_json(path, protocol)
    cache = ContentCache(
        output / "cache",
        dict(
            model_policy=release["model_policy"],
            experimental_sources=protocol["source_sha256"],
        ),
    )
    pools, embedding = retrieve(evidence, catalog, jobs, cache)
    logits, reranker = infer(catalog, jobs, pools, cache)
    variants = dict(baseline=baseline)
    traces: dict = {}
    titles = {c: row["title"] for c, row in catalog.items()}
    for method, group in pools.items():
        variants[method], traces[method] = dict(baseline), {}
        for key, row in group.items():
            order, trace = subject_swap(
                dict(
                    order=baseline[key],
                    pool=row["pool"],
                    titles=titles,
                    scores=logits[method][key],
                    observation=jobs[key]["observation"],
                )
            )
            variants[method][key], traces[method][key] = order, trace
    queries = manifest["query_manifest"]
    human_queries = {
        k: dict(
            q, judgments={c: r for c, r in q["judgments"].items() if r.get("human")}
        )
        for k, q in queries.items()
    }
    target = load(PROJECT_ROOT / "configs/module2_target_p80_r90.json")
    result = dict(
        role=protocol["role"],
        products=845,
        queries=55,
        gated_queries=len(jobs),
        human_metrics={
            n: evaluate_human_retrieval(r, human_queries, target)
            for n, r in variants.items()
        },
        human_only_six_metric_bounds={
            n: _score(r, human_queries) for n, r in variants.items()
        },
        paired_precision={
            n: precision_delta(baseline, r, human_queries) for n, r in variants.items()
        },
        change_evidence={
            n: change_evidence(baseline, r, queries)
            for n, r in variants.items()
            if n != "baseline"
        },
        traces=traces,
        embedding_diagnostics=embedding,
        reranker_diagnostics=reranker,
        cache_hits=cache.hits,
        elapsed_seconds=time.perf_counter() - started,
        new_human_tasks=0,
        default_changed=False,
        acceptance_inference_started=False,
        formal_metrics=None,
        formal_acceptance=False,
    )
    for name, payload in (
        ("pools.json", pools),
        ("scores.json", logits),
        ("rankings.json", variants),
        ("results.json", result),
    ):
        write_json(output / name, payload)
    for name, metrics in result["human_metrics"].items():
        print(name, metrics["scopes"]["all"]["precision_at_10"])
    print("Subject-only changes:", result["change_evidence"]["subject_caption"])
    return result


def main() -> None:
    """Run the isolated fixed-model description experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    execute(parser.parse_args().output)


if __name__ == "__main__":
    main()
