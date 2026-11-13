"""Run only whole images, uniform subject crops, and complete native Top20 reranking."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np

import run  # noqa: F401
from module2_development_benchmark import Evidence, load
from module2_shared_evaluation import precision_delta
from module2_shared_gallery_crop import DETECTOR
from module2_shared_gallery_precision import ROOT, worker_lock
from module2_shared_gallery_retrieval import SIMPLE, _checked_media, _score
from module2_shared_primary_recall import _data
from module2_shared_subject_recall import _scores
from module2_simple_image_assets import (
    EXTRA_SUBJECTS,
    build_index,
    detect_all,
    title_prompt,
)
from multimodal_ecommerce.business_retrieval import BUSINESS_INSTRUCTION
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.simple_baseline import rerank_head, stable_order
from multimodal_ecommerce.subject_roi import POLICY, SUBJECTS
from multimodal_ecommerce.visual_query import DescribedImageReranker

OUTPUT = ROOT / "simple_crop_top20"
METHODS = ("whole", "crop", "crop_top20")
POOL = 20


def _prepare(evidence: Evidence) -> None:
    manifest, _, _ = _data(evidence)
    documents = {}
    for row in manifest["catalog"]:
        _checked_media(evidence, row)
        documents[row["id"]] = dict(row, prompt=title_prompt(row["title"]))
    for query in manifest["query_manifest"].values():
        if query["mode"] == "image":
            _checked_media(evidence, query)
    evidence.digest(DETECTOR)
    configuration = {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
        for path in (
            PROJECT_ROOT / "configs/embedding_model.json",
            PROJECT_ROOT / "configs/reranker_model.json",
            Path(__file__),
            Path(__file__).with_name("module2_simple_image_assets.py"),
            PROJECT_ROOT / "src/multimodal_ecommerce/subject_roi.py",
            PROJECT_ROOT / "src/multimodal_ecommerce/business_retrieval.py",
            PROJECT_ROOT / "src/multimodal_ecommerce/pair_reranker.py",
        )
    }
    write_json(
        OUTPUT / "protocol.json",
        dict(
            role="exposed_development_simple_all_gallery_crop_top20",
            products=len(documents),
            image_queries=sum(
                q["mode"] == "image" for q in manifest["query_manifest"].values()
            ),
            total_queries=len(manifest["query_manifest"]),
            documents=documents,
            crop_policy=POLICY,
            noun_translations=[*SUBJECTS, *EXTRA_SUBJECTS],
            unknown_title_prompt="object.",
            crop_failure="keep exact original image",
            query_policy="unchanged original image, no caption",
            rerank_policy="all twenty candidates; original product image and title; sort all twenty by model score; no protected prefix",
            instruction=BUSINESS_INSTRUCTION,
            text_policy="retain the previously human-verified text branch in all methods",
            pool_size=POOL,
            configuration_sha256=configuration,
            files=evidence.files,
        ),
    )


def _protocol(evidence: Evidence) -> dict:
    plan = evidence.json(OUTPUT / "protocol.json")
    for name, digest in plan["files"].items():
        evidence.digest(PROJECT_ROOT / name, digest)
    for name, digest in plan["configuration_sha256"].items():
        if sha256_file(PROJECT_ROOT / name) != digest:
            raise ValueError(f"Frozen source or model configuration changed: {name}")
    if plan["instruction"] != BUSINESS_INSTRUCTION:
        raise ValueError("The frozen business reranker instruction changed")
    return plan


def _retrieve(evidence: Evidence, manifest: dict, baseline: dict) -> dict:
    folder = OUTPUT / "index"
    receipt = evidence.json(folder / "manifest.json")
    if receipt["protocol_sha256"] != sha256_file(OUTPUT / "protocol.json"):
        raise ValueError("The image index belongs to a different crop protocol")
    evidence.digest(folder / "vectors.npz", receipt["vectors_sha256"])
    result = {name: dict(baseline) for name in METHODS[:2]}
    with np.load(folder / "vectors.npz", allow_pickle=False) as bank:
        for method in METHODS[:2]:
            scores = bank["queries"] @ bank[method].T
            for key, values in zip(receipt["queries"], scores, strict=True):
                query = manifest["query_manifest"][key]
                result[method][key] = [
                    cid
                    for cid in stable_order(receipt["products"], values)
                    if cid not in query["exclusions"]
                ]
    for key, query in manifest["query_manifest"].items():
        if any(
            len(orders[key]) != query["expected_pairs"]
            or set(orders[key]) != set(baseline[key])
            for orders in result.values()
        ):
            raise ValueError(
                "Every method must retain the same complete eligible gallery"
            )
    return result


def _reuse_native(evidence: Evidence, manifest: dict) -> dict:
    old = evidence.json(SIMPLE / "protocol.json")
    for name in (
        "configs/reranker_model.json",
        "src/multimodal_ecommerce/business_retrieval.py",
        "src/multimodal_ecommerce/pair_reranker.py",
    ):
        if sha256_file(PROJECT_ROOT / name) != old["files"][name]:
            raise ValueError(
                "Native cache uses a different model or scoring implementation"
            )
    path = SIMPLE / "inputs.json"
    inputs = evidence.json(
        path, old["files"][path.relative_to(PROJECT_ROOT).as_posix()]
    )
    queries = {
        q["image_sha256"]: key
        for key, q in manifest["query_manifest"].items()
        if q["mode"] == "image"
    }
    catalog = {r["id"]: r for r in manifest["catalog"]}
    reused: dict = {}
    for cohort, group in inputs["cohorts"].items():
        for query in group["queries"]:
            key = queries.get(query["source_sha256"])
            if key is None:
                continue
            path = SIMPLE / "pairs" / cohort / (query["id"] + ".json")
            cached = evidence.json(path)
            ids = cached["identity"]["candidate_ids"]
            if cached["identity"] != dict(
                query_sha256=query["source_sha256"],
                candidate_ids=ids,
                protocol_sha256=sha256_file(SIMPLE / "protocol.json"),
            ) or set(cached["scores"]) != set(ids):
                raise ValueError("The archived native score input identity differs")
            for cid, value in cached["scores"].items():
                previous, current = inputs["documents"][cid], catalog[cid]
                if (
                    previous["source_sha256"] != current["image_sha256"]
                    or previous["title"] != current["title"]
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        "Native reuse requires identical original image, title and model"
                    )
                reused.setdefault(key, {}).setdefault(cid, value)
    return reused


def _freeze_rerank(evidence: Evidence) -> dict:
    manifest, baseline, _ = _data(evidence)
    ranks = _retrieve(evidence, manifest, baseline)
    reuse = _reuse_native(evidence, manifest)
    catalog = {row["id"]: row for row in manifest["catalog"]}
    jobs: dict = {}
    for key, query in manifest["query_manifest"].items():
        if query["mode"] != "image":
            continue
        jobs[key] = dict(
            query_image=str(_checked_media(evidence, query)),
            query_text=None,
            description="",
            documents=[
                dict(
                    id=cid,
                    image_path=str(_checked_media(evidence, catalog[cid])),
                    text=catalog[cid]["title"],
                )
                for cid in ranks["crop"][key][:POOL]
            ],
            reuse={
                cid: reuse.get(key, {})[cid]
                for cid in ranks["crop"][key][:POOL]
                if cid in reuse.get(key, {})
            },
        )
    write_json(OUTPUT / "retrieval_rankings.json", ranks)
    write_json(
        OUTPUT / "retrieval_results.json",
        dict(
            methods={
                name: _score(order, manifest["query_manifest"])
                for name, order in ranks.items()
            },
            paired_precision=precision_delta(
                ranks["whole"], ranks["crop"], manifest["query_manifest"]
            ),
            inputs=evidence.files,
            formal_metrics=None,
        ),
    )
    plan = dict(
        jobs=jobs,
        retrieval_rankings_sha256=sha256_file(OUTPUT / "retrieval_rankings.json"),
        protocol_sha256=sha256_file(OUTPUT / "protocol.json"),
        files=dict(evidence.files),
    )
    path = OUTPUT / "rerank_protocol.json"
    if path.exists():
        if evidence.json(path) != plan:
            raise ValueError("Preserve the frozen native Top20 experiment")
    else:
        write_json(path, plan)
    print(
        f"Frozen {len(jobs)} native Top20 jobs, reusable scores: {sum(len(j['reuse']) for j in jobs.values())}.",
        flush=True,
    )
    return plan


def _run_rerank(evidence: Evidence) -> None:
    path = OUTPUT / "rerank_protocol.json"
    plan = evidence.json(path) if path.exists() else _freeze_rerank(evidence)
    for name, digest in plan["files"].items():
        evidence.digest(PROJECT_ROOT / name, digest)
    identity = sha256_file(path)
    model = DescribedImageReranker()
    completed = 0
    try:
        for key, job in plan["jobs"].items():
            path = (
                OUTPUT / "scores" / (hashlib.sha256(key.encode()).hexdigest() + ".json")
            )
            if not path.exists():
                write_json(path, dict(identity=identity, scores=job["reuse"]))
            _scores(evidence, model, job, (OUTPUT / "scores", identity), key)
            completed += 1
            write_json(
                OUTPUT / "status.json",
                dict(
                    completed=completed,
                    total=len(plan["jobs"]),
                    formal_metrics=None,
                ),
            )
            print(f"Native Top20 complete: {key}.", flush=True)
        write_json(OUTPUT / "reranker_runtime.json", model.diagnostics())
    finally:
        model.close()
    _evaluate(evidence)


def _evaluate(evidence: Evidence) -> None:
    manifest, _, _ = _data(evidence)
    plan = evidence.json(OUTPUT / "rerank_protocol.json")
    ranks = evidence.json(
        OUTPUT / "retrieval_rankings.json", plan["retrieval_rankings_sha256"]
    )
    ranks["crop_top20"] = dict(ranks["crop"])
    for key, job in plan["jobs"].items():
        path = OUTPUT / "scores" / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        cached = evidence.json(path)
        if (
            cached["identity"] != sha256_file(OUTPUT / "rerank_protocol.json")
            or set(cached["scores"]) != {d["id"] for d in job["documents"]}
            or not all(math.isfinite(v) for v in cached["scores"].values())
        ):
            raise ValueError(
                "Complete bound Top20 scores required for every image query"
            )
        ranks["crop_top20"][key] = rerank_head(
            ranks["crop"][key], cached["scores"], POOL
        )
    write_json(OUTPUT / "rankings.json", ranks)
    write_json(
        OUTPUT / "results.json",
        dict(
            role="exposed_development_simple_crop_and_complete_top20",
            products=len(manifest["catalog"]),
            total_queries=len(manifest["query_manifest"]),
            image_queries=len(plan["jobs"]),
            scored_pairs=POOL * len(plan["jobs"]),
            reused_scores=sum(len(j["reuse"]) for j in plan["jobs"].values()),
            crop_summary=evidence.json(OUTPUT / "crop_summary.json"),
            methods={
                name: _score(order, manifest["query_manifest"])
                for name, order in ranks.items()
            },
            paired_precision={
                "crop_minus_whole": precision_delta(
                    ranks["whole"], ranks["crop"], manifest["query_manifest"]
                ),
                "rerank_minus_crop": precision_delta(
                    ranks["crop"], ranks["crop_top20"], manifest["query_manifest"]
                ),
            },
            protocol_sha256=sha256_file(OUTPUT / "protocol.json"),
            rerank_protocol_sha256=sha256_file(OUTPUT / "rerank_protocol.json"),
            rankings_sha256=sha256_file(OUTPUT / "rankings.json"),
            source_sha256=sha256_file(Path(__file__)),
            inputs=evidence.files,
            formal_metrics=None,
        ),
    )
    print("Completed same-gallery whole / crop / native Top20 comparison.", flush=True)


def main() -> None:
    """Execute one resumable simple image-retrieval ablation stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("prepare", "crop", "index", "rerank", "evaluate")
    )
    phase = parser.parse_args().phase
    evidence = Evidence(load(PROJECT_ROOT / "configs/evaluation_registry.json"))
    with worker_lock():
        if phase == "prepare":
            if (OUTPUT / "protocol.json").exists():
                _protocol(evidence)
            else:
                _prepare(evidence)
            return
        protocol = _protocol(evidence)
        if phase == "crop":
            detect_all(evidence, OUTPUT, protocol)
        elif phase == "index":
            manifest, _, _ = _data(evidence)
            build_index(evidence, OUTPUT, manifest)
            _freeze_rerank(evidence)
        elif phase == "rerank":
            _run_rerank(evidence)
        else:
            _evaluate(evidence)


if __name__ == "__main__":
    main()
