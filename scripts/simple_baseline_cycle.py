"""Frozen whole-image, gallery-crop and complete Top20 development ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

import run  # noqa: F401
from multimodal_ecommerce import platform_lock
from multimodal_ecommerce.business_retrieval import BusinessReranker
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.holdout_data import fingerprints
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.runtime_device import project_path
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.simple_baseline import rerank_head, stable_order
from querywise_paired_bounds import paired_querywise

ROOT = PROJECT_ROOT / "artifacts/module2_simple_baseline_20261105"
DATA = PROJECT_ROOT / "artifacts/module2_balanced_gallery"
VIEWS = PROJECT_ROOT / "artifacts/module2_business_grounding_development"
CONFIG = PROJECT_ROOT / "configs/module2_simple_baseline.json"
CONSUMED = [
    PROJECT_ROOT / "data" / name / "records.jsonl"
    for name in ("module2_business_blind_consumed", "module2_grounded_blind_consumed")
]


def load(path: Path) -> Any:
    """Read one workflow JSON artifact.

    Args:
        path: Existing local JSON file.

    Returns:
        Decoded workflow value.
    """
    return json.loads(path.read_text())


def guard() -> dict:
    """Check immutable configuration, source and input identities before resuming.

    Returns:
        The frozen development protocol.

    Raises:
        ValueError: A frozen dependency changed or the cohort is not development.
    """
    protocol = load(ROOT / "protocol.json")
    if protocol["role"] != "exposed_development_only":
        raise ValueError("Only exposed development input is permitted")
    for name, digest in protocol["files"].items():
        if sha256_file(PROJECT_ROOT / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    return protocol


def prepare() -> None:
    """Freeze both cheap interventions before computing new model scores.

    Raises:
        ValueError: Inputs overlap consumed identities or source/crop checks fail.
    """
    if (ROOT / "protocol.json").exists():
        guard()
        return
    policy = load(CONFIG)
    ROOT.mkdir(parents=True, exist_ok=True)
    consumed = [
        json.loads(line)
        for p in CONSUMED
        for line in p.read_text().splitlines()
        if line
    ]
    banned = {str(r.get("source_id") or r["id"].rsplit("/", 1)[-1]) for r in consumed}
    banned_hashes = {
        r.get("image_sha256") or r.get("fingerprints", {}).get("sha256")
        for r in consumed
    }
    view_inputs = load(VIEWS / "views_inputs.public.json")
    view_map = {r["key"]: r for r in view_inputs if r["kind"] == "gallery"}
    files = [
        CONFIG,
        *CONSUMED,
        VIEWS / "views_inputs.public.json",
        PROJECT_ROOT / "artifacts/module2_subject_roi/detector_manifest.json",
        PROJECT_ROOT / "scripts/simple_baseline_cycle.py",
        PROJECT_ROOT / "src/multimodal_ecommerce/simple_baseline.py",
        PROJECT_ROOT / "src/multimodal_ecommerce/embedding.py",
        PROJECT_ROOT / "src/multimodal_ecommerce/business_retrieval.py",
        PROJECT_ROOT / "src/multimodal_ecommerce/pair_reranker.py",
        PROJECT_ROOT / "configs/embedding_model.json",
        PROJECT_ROOT / "configs/reranker_model.json",
    ]
    documents: dict = {}
    cohorts: dict = {}
    for cohort in policy["cohorts"]:
        folder = DATA / cohort
        for name in [
            "catalog.public.json",
            "queries.public.json",
            "judgments.ai.jsonl",
        ]:
            files.append(folder / name)
        catalog = load(folder / "catalog.public.json")
        queries = load(folder / "queries.public.json")
        cohorts[cohort] = dict(gallery=[r["id"] for r in catalog], queries=[])
        for row in catalog:
            image = project_path(row["image_path"])
            signature = fingerprints(image)
            if (
                row["id"].rsplit("/", 1)[-1] in banned
                or signature["sha256"] in banned_hashes
            ):
                raise ValueError("Consumed gallery overlap; no model is loaded")
            if row["id"] in documents:
                if documents[row["id"]]["source_sha256"] != signature["sha256"]:
                    raise ValueError("A shared product has different image bytes")
                continue
            own = view_map[row["id"]]
            detection = VIEWS / "subject_views" / own["opaque"] / "detection.json"
            files.append(detection)
            result = load(detection)
            if result["source_sha256"] != signature["sha256"]:
                raise ValueError("Cached detector input identity differs")
            box = result["box"]
            selected = image
            if box is not None:
                with Image.open(image) as raw:
                    picture = ImageOps.exif_transpose(raw).convert("RGB")
                if not (
                    len(box) == 4
                    and 0 <= box[0] < box[2] <= picture.width
                    and 0 <= box[1] < box[3] <= picture.height
                ):
                    raise ValueError("Frozen detector box is geometrically invalid")
                selected = (
                    ROOT
                    / "crops"
                    / (hashlib.sha256(row["id"].encode()).hexdigest()[:24] + ".png")
                )
                selected.parent.mkdir(exist_ok=True)
                picture.crop(tuple(box)).save(selected)
            documents[row["id"]] = dict(
                id=row["id"],
                title=row["title"],
                image_path=str(image),
                crop_path=str(selected),
                source_sha256=signature["sha256"],
                crop_sha256=sha256_file(selected),
                detection=str(detection.relative_to(PROJECT_ROOT)),
                crop_applied=box is not None,
                fingerprint=signature,
            )
        for query in queries:
            if not query["image_path"]:
                continue
            image = project_path(query["image_path"])
            signature = fingerprints(image)
            if signature["sha256"] in banned_hashes:
                raise ValueError("Consumed query image overlap")
            cohorts[cohort]["queries"].append(
                dict(query, image_path=str(image), source_sha256=signature["sha256"])
            )
    write_json(ROOT / "inputs.json", dict(documents=documents, cohorts=cohorts))
    files.append(ROOT / "inputs.json")
    write_json(
        ROOT / "protocol.json",
        dict(
            role=policy["role"],
            policy=policy,
            files={str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in files},
            consumed_registries_only=True,
            forbidden_consumed_access=["media", "judgments", "predictions", "metrics"],
            same_MPS_reencoding_for_whole_and_crop=True,
            no_quality_based_box_selection=True,
            human_validation_required=True,
            formal_metrics=None,
        ),
    )
    write_json(
        ROOT / "status.json",
        dict(
            phase="prepared",
            unique_gallery=len(documents),
            cropped=sum(r["crop_applied"] for r in documents.values()),
            image_queries=sum(len(c["queries"]) for c in cohorts.values()),
            formal_metrics=None,
        ),
    )


def encode() -> None:
    """Rebuild whole and cropped feature banks on the same backend, resumably."""
    protocol = guard()
    inputs = load(ROOT / "inputs.json")
    folder = ROOT / "vectors"
    folder.mkdir(exist_ok=True)
    encoder = QwenEmbedder()
    pending: list[Path] = []
    targets: list[Path] = []

    def flush() -> None:
        """Persist one real batch, never manufacturing an uncomputed vector."""
        if not pending:
            return
        values = encoder.encode_images(pending)
        for path, vector in zip(targets, values):
            np.save(path, vector, allow_pickle=False)
        pending.clear()
        targets.clear()

    try:
        for number, row in enumerate(inputs["documents"].values(), 1):
            key = hashlib.sha256(row["id"].encode()).hexdigest()[:24]
            for kind, field, digest_field in [
                ("whole", "image_path", "source_sha256"),
                ("crop", "crop_path", "crop_sha256"),
            ]:
                path = folder / f"{key}_{kind}.npy"
                if path.exists():
                    continue
                if sha256_file(Path(row[field])) != row[digest_field]:
                    raise ValueError("Model image changed")
                pending.append(Path(row[field]))
                targets.append(path)
                if len(pending) == protocol["policy"]["embedding_batch_size"]:
                    flush()
            if number % 10 == 0:
                flush()
                write_json(
                    ROOT / "status.json",
                    dict(
                        phase="gallery_encoding",
                        visited=number,
                        total=len(inputs["documents"]),
                        formal_metrics=None,
                    ),
                )
                print(f"Gallery {number}/{len(inputs['documents'])}", flush=True)
        flush()
        for cohort, info in inputs["cohorts"].items():
            for query in info["queries"]:
                path = folder / f"{cohort}_{query['id']}.npy"
                if not path.exists():
                    image = Path(query["image_path"])
                    if sha256_file(image) != query["source_sha256"]:
                        raise ValueError("Query original changed")
                    np.save(
                        path,
                        encoder.encode_images([image], query=True)[0],
                        allow_pickle=False,
                    )
        write_json(ROOT / "embedding_runtime.json", encoder.diagnostics())
    finally:
        encoder.close()
    write_json(
        ROOT / "encoding_receipt.json",
        dict(
            files={
                str(p.relative_to(ROOT)): sha256_file(p)
                for p in sorted(folder.glob("*.npy"))
            }
        ),
    )


def predict() -> None:
    """Run complete, uniform Top20 original-image/title precision scoring."""
    guard()
    inputs = load(ROOT / "inputs.json")
    receipt = load(ROOT / "encoding_receipt.json")
    for name, digest in receipt["files"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Encoded vector changed")
    encoder_folder = ROOT / "vectors"
    client = BusinessReranker()
    pair_count = 0
    try:
        for cohort, info in inputs["cohorts"].items():
            ids = info["gallery"]
            banks = {}
            for kind in ["whole", "crop"]:
                banks[kind] = np.stack(
                    [
                        np.load(
                            encoder_folder
                            / f"{hashlib.sha256(cid.encode()).hexdigest()[:24]}_{kind}.npy",
                            allow_pickle=False,
                        )
                        for cid in ids
                    ]
                )
            rankings = {}
            for query in info["queries"]:
                q = np.load(
                    encoder_folder / f"{cohort}_{query['id']}.npy", allow_pickle=False
                )
                whole = stable_order(ids, banks["whole"] @ q)
                crop = stable_order(ids, banks["crop"] @ q)
                cache = ROOT / "pairs" / cohort / (query["id"] + ".json")
                head = crop[:20]
                identity = dict(
                    query_sha256=query["source_sha256"],
                    candidate_ids=head,
                    protocol_sha256=sha256_file(ROOT / "protocol.json"),
                )
                if cache.exists():
                    saved = load(cache)
                    if saved["identity"] != identity:
                        raise ValueError("Precision cache input identity differs")
                    scores = saved["scores"]
                else:
                    scores = {}
                if sha256_file(Path(query["image_path"])) != query["source_sha256"]:
                    raise ValueError("Query image changed before precision scoring")
                for cid in head:
                    row = inputs["documents"][cid]
                    if sha256_file(Path(row["image_path"])) != row["source_sha256"]:
                        raise ValueError(
                            "Candidate image changed before precision scoring"
                        )
                missing = [cid for cid in head if cid not in scores]
                if missing:
                    documents = [
                        dict(
                            id=cid,
                            image_path=inputs["documents"][cid]["image_path"],
                            text=inputs["documents"][cid]["title"],
                        )
                        for cid in missing
                    ]
                    values = client.score(
                        documents, query_image=Path(query["image_path"])
                    )
                    scores.update({cid: float(v) for cid, v in zip(missing, values)})
                    write_json(cache, dict(identity=identity, scores=scores))
                    pair_count += len(missing)
                rankings[query["id"]] = dict(
                    whole_cosine=whole,
                    crop_cosine=crop,
                    crop_top20_complete_reranker=rerank_head(crop, scores),
                )
                write_json(
                    ROOT / "status.json",
                    dict(
                        phase="precision",
                        cohort=cohort,
                        query=query["id"],
                        new_pairs=pair_count,
                        formal_metrics=None,
                    ),
                )
                print(f"Top20 complete: {cohort}/{query['id']}", flush=True)
            write_json(ROOT / cohort / "predictions.json", rankings)
        write_json(ROOT / "reranker_runtime.json", client.diagnostics())
    finally:
        client.close()


def evaluate() -> None:
    """Report crop/CE increments on development only, keeping full denominators."""
    guard()
    inputs = load(ROOT / "inputs.json")
    results = {}
    for cohort, info in inputs["cohorts"].items():
        rankings = load(ROOT / cohort / "predictions.json")
        modes = {q["id"]: "image" for q in info["queries"]}
        labels = [
            json.loads(line)
            for line in (DATA / cohort / "judgments.ai.jsonl").read_text().splitlines()
            if line
        ]
        labels = [r for r in labels if r["query_id"] in modes]
        result = evaluate_intervals(rankings, labels, modes)
        paired = paired_querywise(
            rankings,
            labels,
            modes,
            [
                ("crop_cosine", "whole_cosine"),
                ("crop_top20_complete_reranker", "crop_cosine"),
            ],
        )
        write_json(ROOT / cohort / "AI_diagnostic_metrics.json", result)
        write_json(ROOT / cohort / "paired_increments.json", paired)
        results[cohort] = result["scopes"]["all"]
    write_json(
        ROOT / "summary.json",
        dict(
            results=results,
            meaning="AI development diagnostics; label-completion bounds omit AI mislabeling",
            human_audit_status="pending",
            formal_metrics=None,
        ),
    )
    write_json(
        ROOT / "status.json",
        dict(
            phase="model_ablation_completed",
            formal_metrics=None,
            human_review_pending=True,
        ),
    )


@contextmanager
def worker_lock() -> Iterator[None]:
    """Prevent duplicate model jobs for this experiment.

    Yields:
        Exclusive access released even if inference fails.
    """
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "worker.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        platform_lock.locking(handle.fileno(), platform_lock.LK_NBLCK, 1)
        try:
            write_json(ROOT / "worker_identity.json", dict(pid=os.getpid()))
            yield
        finally:
            handle.seek(0)
            platform_lock.locking(handle.fileno(), platform_lock.LK_UNLCK, 1)


def main() -> None:
    """Execute the frozen simple cycle or one resumable stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["prepare", "encode", "predict", "evaluate", "worker"]
    )
    action = parser.parse_args().action
    if action == "worker":
        with worker_lock():
            prepare()
            encode()
            predict()
            evaluate()
    else:
        {
            "prepare": prepare,
            "encode": encode,
            "predict": predict,
            "evaluate": evaluate,
        }[action]()


if __name__ == "__main__":
    main()
