"""Fresh-query inference on the frozen visual-coverage development index."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import run  # noqa: F401
from advance_visual_coverage import ROOT, check, load
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_retrieval import BusinessReranker
from multimodal_ecommerce.development_expansion import load_expanded_index
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.intent_retrieval import (
    balanced_pool,
    normalized_ranks,
    public_evidence,
    recall_channels,
    rerank_complete,
)
from multimodal_ecommerce.regional_retrieval import pixel_roi
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)
from multimodal_ecommerce.semantic_ranker import (
    SemanticRankerFeatureMatrixInputs,
    feature_matrix,
    predict_ranker,
    ranked_ids,
)


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        ValueError: Reserved acceptance image path blocked before opening pixels or
            loading model; Reserved acceptance pixels/near duplicate blocked before
            model loading.
        FileExistsError: Preserve previous query output.

    Notes:
        Writes derived artifacts through crop.save, output.mkdir, write_json.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--roi", type=float, nargs=4)
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "global",
            "region_title",
            "semantic_blend",
            "public_evidence_soft",
            "balanced_logistic",
            "hard_negative_ranknet",
            "cross_evidence_pool60",
        ],
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if (args.text is None) == (args.image is None) or args.roi and args.image is None:
        parser.error("Exactly one query modality; ROI requires an image")
    if args.text is not None and (not args.text.strip() or len(args.text) > 512):
        parser.error("Text must contain 1-512 characters")
    start = time.perf_counter()
    image = args.image.resolve() if args.image else None
    if image and any(
        part in {"module2_business_acceptance", "module2_reserved_acceptance"}
        for part in image.parts
    ):
        raise ValueError(
            "Reserved acceptance image path blocked before opening pixels or loading model"
        )
    signature = fingerprints(image) if image else None
    if signature:
        reserved = [
            r["fingerprints"]
            for r in map(
                json.loads,
                (
                    PROJECT_ROOT
                    / "artifacts/module2_business_acceptance/dataset/evaluator/lineage.jsonl"
                )
                .read_text(encoding="utf-8")
                .splitlines(),
            )
        ]
        if is_exposed(signature, reserved, 4):
            raise ValueError(
                "Reserved acceptance pixels/near duplicate blocked before model loading"
            )
    runtime_protocol = load(ROOT / "query_runtime_protocol.json")
    verify_files(PROJECT_ROOT, runtime_protocol["files"])
    check()
    output = ensure_output_path(PROJECT_ROOT / args.output)
    if output.exists():
        raise FileExistsError("Preserve previous query output")
    output.mkdir(parents=True)
    target = image
    if args.roi:
        if image is None:
            raise ValueError("ROI requires a query image")
        with Image.open(image) as source:
            picture = ImageOps.exif_transpose(source).convert("RGB")
            crop = picture.crop(pixel_roi(args.roi, picture.size))
            scale = 768 / max(crop.size)
            crop = crop.resize(
                (max(2, round(crop.width * scale)), max(2, round(crop.height * scale))),
                Image.Resampling.LANCZOS,
            )
        target = output / "query_roi.png"
        crop.save(target)
    index, titles = load_expanded_index(ROOT / "index")
    with np.load(ROOT / "index/vectors.npz", allow_pickle=False) as bank:
        profiles = bank["profiles"].copy()
    profile_texts = load(ROOT / "index/profile_texts.json")
    with np.load(ROOT / "features.npz", allow_pickle=False) as bank:
        colors = bank["color_bank"].copy()
    encoder = QwenEmbedder()
    try:
        vector = encoder.encode(text=args.text, image=target, query=True)
        diagnostics = encoder.diagnostics()
    finally:
        encoder.close()
    features = feature_matrix(
        SemanticRankerFeatureMatrixInputs(
            index,
            titles,
            profiles,
            profile_texts,
            vector,
            text=args.text,
            image=target,
            color_bank=colors,
        )
    )
    signatures = load(ROOT / "guard.json")["signatures"]
    included = [
        i
        for i, row in enumerate(index.records)
        if not signature
        or (
            signatures[row["id"]]["sha256"] != signature["sha256"]
            and signatures[row["id"]]["pixels"] != signature["pixels"]
        )
    ]
    ids = [index.records[i]["id"] for i in included]
    current = features[included]
    if args.method == "global":
        scores = current[:, 0]
    elif args.method == "region_title":
        scores = 0.8 * current[:, 1] + 0.2 * current[:, 3]
    elif args.method == "semantic_blend":
        scores = 0.5 * current[:, 1] + 0.2 * current[:, 3] + 0.3 * current[:, 4]
    elif args.method == "public_evidence_soft":
        rrf, _ = recall_channels(ids, current)
        scores = normalized_ranks(ids, rrf) + np.array(
            [
                public_evidence(args.text, index.records[i]["title"], profile_texts[i])[
                    "offset"
                ]
                for i in included
            ]
        )
    elif args.method == "cross_evidence_pool60":
        rrf, channels = recall_channels(ids, current)
        offsets = np.array(
            [
                public_evidence(args.text, index.records[i]["title"], profile_texts[i])[
                    "offset"
                ]
                for i in included
            ]
        )
        adjusted = normalized_ranks(ids, rrf) + offsets
        channels.append(ranked_ids(ids, adjusted))
        pool = balanced_pool(channels, min(60, len(ids)))
        records_by_id = {row["id"]: row for row in index.records}
        documents = [
            dict(
                id=cid,
                image_path=records_by_id[cid]["image_path"],
                text=records_by_id[cid]["title"],
            )
            for cid in pool
        ]
        reranker = BusinessReranker()
        try:
            logits = np.array(
                reranker.score(documents, query_text=args.text, query_image=target)
            )
            cross_diagnostics = reranker.diagnostics()
        finally:
            reranker.close()
        write_json(
            output / "cross_encoder.json",
            dict(pool=pool, logits=logits.tolist(), diagnostics=cross_diagnostics),
        )
        complete_order = rerank_complete(ids, rrf, pool, logits, offsets)
        priority = {cid: len(ids) - i for i, cid in enumerate(complete_order)}
        scores = np.array([priority[cid] for cid in ids], dtype=float)
    else:
        model = load(ROOT / "trained_candidates.json")[args.method]
        scores = predict_ranker(current, model)
    order = ranked_ids(ids, scores)
    by_id = {row["id"]: row for row in index.records}
    score_by_id = dict(zip(ids, scores))
    results = [
        dict(
            rank=n,
            id=cid,
            title=by_id[cid]["title"],
            image_path=by_id[cid]["image_path"],
            score=float(score_by_id[cid]),
            score_kind=(
                "ordinal_position_after_cross_reranking"
                if args.method == "cross_evidence_pool60"
                else "uncalibrated_model_or_fusion_score"
            ),
        )
        for n, cid in enumerate(order[:10], 1)
    ]
    write_json(
        output / "results.json",
        dict(
            role="fresh_query_exposed_development_demo_only",
            method=args.method,
            text=args.text,
            source_image=str(image) if image else None,
            source_image_sha256=signature["sha256"] if signature else None,
            roi=args.roi,
            catalog_size=len(index.records),
            candidate_count=len(ids),
            results=results,
            historical_query_rankings_used=False,
            relevance_table_used_as_runtime_feature=False,
            formal_metrics=None,
            formal_acceptance=False,
            acceptance_predictions=0,
            wall_seconds_including_integrity_and_model_loading=time.perf_counter()
            - start,
            implementation_sha256=sha256_file(
                PROJECT_ROOT / "scripts/retrieve_visual_coverage.py"
            ),
            diagnostics=diagnostics,
        ),
    )
    print(
        json.dumps(
            dict(
                results=10,
                catalog_size=len(index.records),
                excluded=len(index.records) - len(ids),
                method=args.method,
                formal_acceptance=False,
                acceptance_predictions=0,
            )
        )
    )


if __name__ == "__main__":
    main()
