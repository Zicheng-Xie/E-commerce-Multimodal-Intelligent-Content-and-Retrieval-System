"""Measure a corroborated poster-use cue on the frozen pressure development group."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

import paper_sid_rerank as worker
import run  # noqa: F401
from module2_focus_boundary import FOCUS, FRESH, FRESH_METHOD, verified
from module2_focus_followup import ROOT as PARENT
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.intent_retrieval import rerank_complete
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.query_fusion import (
    QueryFusionVisualChannelsInputs,
    visual_channels,
)
from multimodal_ecommerce.regional_retrieval import RegionalIndex
from multimodal_ecommerce.runtime_device import relocate_assets
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.semantic_ranker import (
    SemanticRankerFeatureMatrixInputs,
    feature_matrix,
)
from multimodal_ecommerce.visual_query import DescribedImageReranker
from paper_fusion_experiment import load
from querywise_paired_bounds import paired_querywise

CAPTIONS = PARENT / "caption_model_comparison"
ROOT = CAPTIONS / "poster_usage"
BASE = "archived_top20"
FIXED = "poster_usage_fixed_top20"
FUSED = "poster_usage_recalled_top20"
TERMS = {"游戏机", "充电宝"}


def supplemental_use(description: str, lines: list[dict], transcription: dict) -> str:
    """Append a dual-use cue only when two readings agree and the poster corroborates it.

    Args:
        description: Archived query-only image description.
        lines: Original-image Vision observations.
        transcription: Local transcription response with no supplied product names.

    Returns:
        Original or supplemented query description.
    """
    seen = {line["text"].strip() for line in lines}
    heading = description.split("。", 1)[0].removeprefix("商品主体：").strip()
    if (
        heading != "游戏机"
        or "充电宝" in description
        or not TERMS <= seen
        or not any(
            line["confidence"] >= 0.7
            and re.fullmatch(r"[0-9]{3,6}\s*mAh", line["text"].strip(), re.I)
            for line in lines
        )
    ):
        return description
    read = json.loads(transcription["raw"]["message"]["content"])["lines"]
    if TERMS <= {line.strip() for line in read}:
        return description + "海报标注的另一用途：充电宝。"
    return description


def prepare(registry: dict) -> dict:
    """Verify every frozen source and identify qualifying poster-use changes.

    Args:
        registry: Current development and protected data-role policy.

    Returns:
        Immutable experiment input, selected query captions, and source hashes.
    """
    paths = [
        CAPTIONS / "protocol.json",
        CAPTIONS / "summary.json",
        ROOT / "observations.json",
        ROOT / "local_transcription.json",
        FOCUS / "completion_receipt.json",
        FRESH / "inference_protocol.json",
        FRESH / "evaluation_receipt.json",
    ]
    require_data_access(paths, "development", registry)
    caption_path, summary_path, ocr_path, reading_path, focus, fresh, evaluation = paths
    captions, summary, ocr, reading = map(load, paths[:4])
    digest = sha256_file(caption_path)
    if (
        digest != summary["protocol_sha256"]
        or digest != ocr["source_sha256"]
        or digest != reading["protocol_sha256"]
        or reading["ocr_sha256"] != sha256_file(ocr_path)
    ):
        raise ValueError("OCR input identities differ")
    files = {path: sha256_file(path) for path in paths}
    focus_hashes = load(focus)["files"]
    for name in (
        "index/manifest.json",
        "index/records.json",
        "index/vectors.npz",
        "index/profile_texts.json",
        "features.npz",
    ):
        path = FOCUS / name
        files[path] = verified(path, focus_hashes[name], registry)
    rank_path = FRESH / "rankings.json"
    files[rank_path] = verified(
        rank_path, load(evaluation)["files"][rank_path.name], registry
    )
    labels = FRESH / "judgments.ai.jsonl"
    files[labels] = verified(
        labels, load(fresh)["files"][str(labels.relative_to(PROJECT_ROOT))], registry
    )
    jobs = []
    for item in captions["jobs"]:
        qid = item["qid"]
        if qid not in reading["results"]:
            continue
        proof = reading["results"][qid]
        if (
            proof["image_sha256"] != item["image_sha256"]
            or proof["raw"]["model"] != reading["model"]
            or not proof["raw"]["done"]
        ):
            raise ValueError("Poster transcription is not aligned")
        if item["cohort"] != "stress":
            raise ValueError("This bounded replay is registered for the stress group")
        folder = FRESH / "runs" / qid
        for name in ("query_fusion.json", "query_vectors.npz", "cross_encoder.json"):
            files[folder / name] = verified(folder / name, "", registry)
        fusion = load(folder / "query_fusion.json")
        caption = supplemental_use(
            fusion["description"], ocr["observations"][f"stress/{qid}"], proof
        )
        if caption == fusion["description"]:
            continue
        image = Path(item["image"])
        files[image] = verified(image, item["image_sha256"], registry)
        jobs.append(dict(**item, caption=caption))
    return dict(
        role="exposed_development",
        jobs=jobs,
        examined_queries=len(captions["jobs"]),
        files={str(p.relative_to(PROJECT_ROOT)): h for p, h in files.items()},
        source_sha256=sha256_file(Path(__file__)),
        formal_metrics=None,
    )


def rerun(
    job: dict, baseline: list[str], registry: dict, digest: str
) -> tuple[dict, dict]:
    """Replay the original path, then encode and score the poster-corroborated description.

    Args:
        job: Verified original development image and added poster use.
        baseline: Full original gallery ranking.
        registry: Data-role policy for candidate originals.
        digest: New frozen protocol digest.

    Returns:
        Revised gallery rankings and the exact additional scoring receipt.
    """
    qid = job["qid"]
    path = ROOT / (qid + "_replay.json")
    if path.exists():
        cached = load(path)
        if cached["protocol_sha256"] != digest:
            raise ValueError("Saved OCR experiment belongs to a different freeze")
        require_data_access(
            [PROJECT_ROOT / p for p in cached["image_hashes"]], "development", registry
        )
        if any(
            sha256_file(PROJECT_ROOT / p) != h
            for p, h in cached["image_hashes"].items()
        ):
            raise ValueError("Scored candidate images changed")
        return cached["rankings"], cached["receipt"]
    folder = FRESH / "runs" / qid
    original = load(folder / "query_fusion.json")
    original_cross = load(folder / "cross_encoder.json")
    records = relocate_assets(load(FOCUS / "index/records.json"))
    regions = load(FOCUS / "index/manifest.json")["regions"]
    with np.load(FOCUS / "index/vectors.npz", allow_pickle=False) as bank:
        index = RegionalIndex(records, bank["regions"], regions)
        titles, profiles = bank["texts"].copy(), bank["profiles"].copy()
    with np.load(FOCUS / "features.npz", allow_pickle=False) as bank:
        colors = bank["color_bank"].copy()
    with np.load(folder / "query_vectors.npz", allow_pickle=False) as bank:
        views, caption = bank["views"].copy(), bank["caption"].copy()
    image = Path(job["image"])
    texts = load(FOCUS / "index/profile_texts.json")
    features = feature_matrix(
        SemanticRankerFeatureMatrixInputs(
            index, titles, profiles, texts, views[0], image=image, color_bank=colors
        )
    )
    indices = np.arange(len(records))
    anchor = visual_channels(
        QueryFusionVisualChannelsInputs(
            index,
            titles,
            profiles,
            features,
            views,
            caption,
            original["description"],
            indices,
        )
    )
    ids, pool = anchor["ids"], original_cross["pool"]
    if (
        ids != original["ids"]
        or anchor["pool"][:20] != pool
        or original_cross["description"] != original["description"]
        or rerank_complete(
            ids,
            np.array(anchor["fused_scores"]),
            pool,
            np.array(original_cross["logits"]),
        )
        != baseline
    ):
        raise ValueError("Archived Top20 pipeline did not replay exactly")
    encoder = QwenEmbedder()
    try:
        replacement = encoder.encode(text=job["caption"], query=True)
        embedding_runtime = encoder.diagnostics()
    finally:
        encoder.close()
    updated = visual_channels(
        QueryFusionVisualChannelsInputs(
            index,
            titles,
            profiles,
            features,
            views,
            replacement,
            job["caption"],
            indices,
        )
    )
    new_pool = updated["pool"][:20]
    union = list(dict.fromkeys(pool + new_pool))
    by_id = {r["id"]: r for r in records}
    images = [Path(by_id[cid]["image_path"]) for cid in union]
    require_data_access(images, "development", registry)
    image_hashes = {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in images}
    documents = [
        dict(id=cid, image_path=str(p), text=by_id[cid]["title"])
        for cid, p in zip(union, images, strict=True)
    ]
    model = DescribedImageReranker()
    try:
        values = model.score_described(
            documents, query_image=image, description=job["caption"]
        )
        runtime = model.diagnostics()
    finally:
        model.close()
    previous = original_cross["diagnostics"]["model"]
    for key in (
        "revision",
        "files",
        "vendor_sha256",
        "instruction",
        "max_length",
        "max_pixels",
    ):
        if previous[key] != runtime["model"][key]:
            raise ValueError(f"Reranker configuration differs: {key}")
    scores = dict(zip(union, values.tolist(), strict=True))
    rankings = {
        BASE: baseline,
        FIXED: rerank_complete(
            ids,
            np.array(anchor["fused_scores"]),
            pool,
            np.array([scores[c] for c in pool]),
        ),
        FUSED: rerank_complete(
            ids,
            np.array(updated["fused_scores"]),
            new_pool,
            np.array([scores[c] for c in new_pool]),
        ),
    }
    receipt = dict(
        archived_pool=pool,
        poster_pool=new_pool,
        added_candidates=[c for c in new_pool if c not in pool],
        removed_candidates=[c for c in pool if c not in new_pool],
        scores=scores,
        embedding_runtime=embedding_runtime,
        reranker_runtime=runtime,
    )
    write_json(
        path,
        dict(
            protocol_sha256=digest,
            image_hashes=image_hashes,
            rankings=rankings,
            receipt=receipt,
        ),
    )
    return rankings, receipt


def main() -> None:
    """Run the frozen local experiment and evaluate the complete six-query stress group."""
    worker.ROOT = ROOT
    with worker.worker_lock():
        registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
        protocol = prepare(registry)
        path = ROOT / "experiment_protocol.json"
        if path.exists() and load(path) != protocol:
            raise ValueError("Frozen poster-use experiment changed")
        if not path.exists():
            write_json(path, protocol)
        original = load(FRESH / "rankings.json")
        rankings = {
            q: dict.fromkeys((BASE, FIXED, FUSED), row[FRESH_METHOD])
            for q, row in original.items()
        }
        scores = {}
        for job in protocol["jobs"]:
            qid = job["qid"]
            print(f"Running poster-use replay for {qid}", flush=True)
            rankings[qid], scores[qid] = rerun(
                job, rankings[qid][BASE], registry, sha256_file(path)
            )
        labels = [
            json.loads(line)
            for line in (FRESH / "judgments.ai.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        modes = dict.fromkeys(rankings, "image")
        metrics = evaluate_intervals(rankings, labels, modes)
        paired = paired_querywise(
            rankings, labels, modes, [(FIXED, BASE), (FUSED, BASE), (FUSED, FIXED)]
        )
        write_json(
            ROOT / "results.json",
            dict(
                examined_queries=protocol["examined_queries"],
                changed_descriptions=len(scores),
                new_qwen_pair_scores=sum(len(row["scores"]) for row in scores.values()),
                score_receipts=scores,
                metrics=metrics,
                paired=paired,
                rankings=rankings,
                frozen_acceptance_model_calls=0,
                formal_metrics=None,
            ),
        )
        print("Poster-use comparison complete.", flush=True)


if __name__ == "__main__":
    main()
