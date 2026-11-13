"""Compare full-image retrieval with subject crops on the unified development gallery."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageOps

import run  # noqa: F401
from module2_development_benchmark import Evidence, load
from module2_shared_gallery_precision import ROOT, reviewed_inputs, worker_lock
from module2_shared_gallery_retrieval import (
    SIMPLE,
    _checked_media,
    _score,
    _vector,
    _vector_caches,
)
from multimodal_ecommerce.embedding import QwenEmbedder, model_configuration
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.simple_baseline import stable_order
from multimodal_ecommerce.subject_roi import POLICY, SubjectDetector, subject_prompt

DETECTOR = PROJECT_ROOT / "artifacts/module2_subject_roi/detector_manifest.json"
MIX = 0.5
METHODS = ("whole", "subject_with_whole_fallback", "whole_subject_equal_fusion")


def _plan(evidence: Evidence, catalog: dict) -> tuple[dict, dict]:
    path = SIMPLE / "inputs.json"
    receipt = evidence.json(SIMPLE / "protocol.json")["files"]
    archived = evidence.json(path, receipt[path.relative_to(PROJECT_ROOT).as_posix()])
    files = evidence.json(SIMPLE / "encoding_receipt.json")["files"]
    crops, candidates = {}, {}
    for cid, product in catalog.items():
        old = archived["documents"].get(cid)
        if old is not None and old["source_sha256"] != product["image_sha256"]:
            raise ValueError(f"Archived crop has another source image: {cid}")
        if old is not None and old["crop_applied"]:
            slug = hashlib.sha256(cid.encode()).hexdigest()[:24]
            crops[cid] = _vector(evidence, files, f"{slug}_crop")
        elif old is None:
            prompt = subject_prompt(product["title"])
            if prompt:
                candidates[cid] = dict(
                    prompt=prompt,
                    image=str(_checked_media(evidence, product)),
                    image_sha256=product["image_sha256"],
                )
    evidence.digest(DETECTOR)
    return crops, candidates


def _detect(candidates: dict, identity: str) -> tuple[dict, dict]:
    path = ROOT / "crop_detections.json"
    saved: dict = (
        load(path) if path.exists() else dict(protocol_sha256=identity, rows={})
    )
    decisions = saved["rows"]
    if saved["protocol_sha256"] != identity or not set(decisions) <= set(candidates):
        raise ValueError("Saved detections belong to a different crop experiment")
    pending = [cid for cid in candidates if cid not in decisions]
    if not pending:
        return decisions, dict(new_detections=0)
    detector = SubjectDetector(DETECTOR)
    try:
        for number, cid in enumerate(pending, 1):
            job = candidates[cid]
            result = detector.detect(Path(job["image"]), job["prompt"])
            selected = result["decision"]["selected"]
            decisions[cid] = dict(
                status=result["decision"]["status"],
                box=selected["padded_box"] if selected else None,
                score=selected["score"] if selected else None,
                source_size=result["source_size"],
            )
            if number % 20 == 0 or number == len(pending):
                temporary = path.with_suffix(".partial")
                write_json(temporary, saved)
                temporary.replace(path)
                print(
                    f"Checked subject boxes for {len(decisions)}/{len(candidates)} products.",
                    flush=True,
                )
        runtime = detector.diagnostics()
    finally:
        detector.close()
    return decisions, runtime


def _new_crops(model: QwenEmbedder, jobs: dict, decisions: dict) -> dict:
    vectors: dict[str, np.ndarray] = {}
    ids = [cid for cid, value in decisions.items() if value["box"]]
    with TemporaryDirectory(prefix="module2-crops-") as folder:
        paths = [Path(folder) / f"{i}.png" for i in range(4)]
        for start in range(0, len(ids), len(paths)):
            group = ids[start : start + len(paths)]
            outputs = paths[: len(group)]
            for cid, output in zip(group, outputs, strict=True):
                decision = decisions[cid]
                with Image.open(jobs[cid]["image"]) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                box = decision["box"]
                if list(image.size) != decision["source_size"] or not (
                    0 <= box[0] < box[2] <= image.width
                    and 0 <= box[1] < box[3] <= image.height
                ):
                    raise ValueError(f"Invalid original size or crop: {cid}")
                image.crop(tuple(box)).save(output)
            vectors.update(zip(group, model.encode_images(outputs), strict=True))
    return vectors


def _features(
    evidence: Evidence, catalog: dict, queries: dict, plan: dict
) -> tuple[dict, dict, dict]:
    whole, query_vectors = _vector_caches(evidence, catalog, queries)
    new_ids = sorted(set(catalog) - set(whole))
    missing_queries = sorted(set(queries) - set(query_vectors))
    paths = [_checked_media(evidence, catalog[cid]) for cid in new_ids]
    query_paths = {
        key: _checked_media(evidence, queries[key])
        for key in missing_queries
        if queries[key]["mode"] == "image"
    }
    model = QwenEmbedder()
    try:
        for start in range(0, len(paths), 4):
            vectors = model.encode_images(paths[start : start + 4])
            whole.update(zip(new_ids[start : start + 4], vectors, strict=True))
        for key in missing_queries:
            query_vectors[key] = model.encode(
                image=query_paths.get(key), text=queries[key].get("text"), query=True
            )
        new = _new_crops(model, plan["jobs"], plan["detections"])
        runtime = model.diagnostics()
    finally:
        model.close()
    return whole, query_vectors, dict(new_crops=new, model=runtime)


def _rank(whole: dict, crops: dict, vectors: dict, queries: dict) -> dict:
    ids = sorted(whole)
    originals = np.stack([whole[cid] for cid in ids])
    views = np.stack([crops.get(cid, whole[cid]) for cid in ids])
    queries_matrix = np.stack([vectors[key] for key in queries])
    if any(
        bank.shape[1:] != (2048,)
        or not np.allclose(np.linalg.norm(bank, axis=1), 1, atol=1e-3)
        for bank in (originals, views, queries_matrix)
    ):
        raise ValueError("Invalid model vector shape or norm")
    scores = queries_matrix @ originals.T
    crop_scores = queries_matrix @ views.T
    ranking: dict[str, dict] = {method: {} for method in METHODS}
    for i, (key, query) in enumerate(queries.items()):
        excluded = set(query["exclusions"])
        weights = (
            scores[i],
            crop_scores[i],
            MIX * scores[i] + (1 - MIX) * crop_scores[i],
        )
        for method, values in zip(METHODS, weights, strict=True):
            ranking[method][key] = [
                cid for cid in stable_order(ids, values) if cid not in excluded
            ]
    return ranking


def _changes(ranking: dict, queries: dict) -> dict:
    original = ranking["whole"]
    changes = {}
    for method in METHODS[1:]:
        rows = {}
        for key, order in ranking[method].items():
            before, after = set(original[key][:10]), set(order[:10])
            labels = queries[key]["judgments"]
            rows[key] = dict(
                entering=[
                    dict(id=cid, grade=labels.get(cid, {}).get("grade"))
                    for cid in sorted(after - before)
                ],
                leaving=[
                    dict(id=cid, grade=labels.get(cid, {}).get("grade"))
                    for cid in sorted(before - after)
                ],
            )
        changes[method] = rows
    return changes


def _run() -> None:
    """Run deterministic same-gallery comparisons and save complete development results."""
    evidence = Evidence(load(PROJECT_ROOT / "configs/evaluation_registry.json"))
    manifest, baseline, _ = reviewed_inputs(evidence)
    queries = manifest["query_manifest"]
    catalog = {row["id"]: row for row in manifest["catalog"]}
    archived, jobs = _plan(evidence, catalog)
    protocol = ROOT / "crop_protocol.json"
    policy = dict(
        role="exposed_development_subject_crop_ablation",
        methods=METHODS,
        whole_weight_in_fusion=MIX,
        detector_policy=POLICY,
        embedding=model_configuration(),
        archived_crops=sorted(archived),
        new_detector_jobs=jobs,
        files=dict(evidence.files),
    )
    if protocol.is_file():
        if evidence.json(protocol) != dict(policy, methods=list(METHODS)):
            raise ValueError("Frozen crop protocol differs")
    else:
        write_json(protocol, policy)
    identity = sha256_file(protocol)
    result_path = ROOT / "crop_results.json"
    path = ROOT / "crop_rankings.json"
    if result_path.is_file():
        existing = evidence.json(result_path)
        if existing["protocol_sha256"] != identity:
            raise ValueError("Completed crop results belong to another protocol")
        evidence.digest(path, existing["rankings_sha256"])
        print("The complete crop comparison already matches the frozen protocol.")
        return
    decisions, runtime = _detect(jobs, identity)
    whole, vectors, embedding = _features(
        evidence, catalog, queries, dict(jobs=jobs, detections=decisions)
    )
    new_crops = embedding.pop("new_crops")
    crops = {**archived, **new_crops}
    rankings = _rank(whole, crops, vectors, queries)
    for key, order in rankings["whole"].items():
        if len(order) != queries[key]["expected_pairs"] or set(order) != set(
            baseline[key]
        ):
            raise ValueError(f"Gallery/query eligibility differs: {key}")
    write_json(path, rankings)
    write_json(
        result_path,
        dict(
            protocol_sha256=identity,
            rankings_sha256=sha256_file(path),
            source_sha256=sha256_file(Path(__file__)),
            products=len(catalog),
            queries=len(queries),
            reused_crops=len(archived),
            new_detections=len(jobs),
            new_crops=len(new_crops),
            whole_fallback=len(catalog) - len(crops),
            detector_decisions=dict(
                Counter(row["status"] for row in decisions.values())
            ),
            old_control_exact_queries=sum(
                order == rankings["whole"][key] for key, order in baseline.items()
            ),
            old_control_top10_exact_queries=sum(
                order[:10] == rankings["whole"][key][:10]
                for key, order in baseline.items()
            ),
            methods={name: _score(order, queries) for name, order in rankings.items()},
            top10_changes=_changes(rankings, queries),
            detector_runtime=runtime,
            embedding_runtime=embedding["model"],
            inputs=evidence.files,
            formal_metrics=None,
        ),
    )
    print(
        f"Compared {len(catalog)} products, {len(queries)} queries, {len(crops)} cropped products."
    )


def main() -> None:
    """Run the unified-gallery ablation with exclusive access to the model caches."""
    with worker_lock():
        _run()


if __name__ == "__main__":
    main()
