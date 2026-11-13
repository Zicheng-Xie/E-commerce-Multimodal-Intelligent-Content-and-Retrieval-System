"""Run the new development candidate on a fresh query, never a saved-query lookup."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_retrieval import BusinessReranker
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.intent_retrieval import (
    balanced_pool,
    normalized_ranks,
    public_evidence,
    recall_channels,
    rerank_complete,
)
from multimodal_ecommerce.regional_retrieval import (
    RegionalIndex,
    RegionalSearchRequest,
    pixel_roi,
    write_regional_gallery,
)
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

BANK = PROJECT_ROOT / "artifacts/module2_machine_assist/semantic_ranker"


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve existing query output.
        ValueError: Reserved acceptance image or duplicate risk blocked before model
            load; Ordered normalized ROI required.

    Notes:
        Writes derived artifacts through crop.save, output.mkdir, write_json.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--text")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--roi", nargs=4, type=float)
    parser.add_argument(
        "--method",
        choices=[
            "multi_channel_rrf",
            "public_evidence_soft",
            "cross_pool20",
            "cross_pool60",
            "cleaned_logistic",
            "cleaned_ranknet",
        ],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.text is None) == (args.image is None) or args.roi and args.image is None:
        parser.error("Exactly one query modality; ROI needs an image")
    if args.text is not None and (not args.text.strip() or len(args.text) > 512):
        parser.error("Text must contain 1-512 characters")
    started = time.perf_counter()
    output = ensure_output_path(PROJECT_ROOT / args.output)
    if output.exists():
        raise FileExistsError("Preserve existing query output")
    for name in ["feature_receipt.json", "runtime_receipt.json"]:
        verify_files(BANK, load(BANK / name)["files"])
    runtime = load(BANK / "runtime_fingerprints.json")
    verify_files(PROJECT_ROOT, runtime["source_files"])
    image = args.image.resolve() if args.image else None
    signature = fingerprints(image) if image else None
    if image:
        assert signature is not None
        image.relative_to(PROJECT_ROOT)
        dataset = PROJECT_ROOT / "artifacts/module2_business_acceptance/dataset"
        manifest = load(dataset / "manifest.json")
        registry = PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl"
        assert sha256_file(registry) == manifest["reserved_registry_sha256"]
        verify_files(
            dataset,
            {"evaluator/lineage.jsonl": manifest["files"]["evaluator/lineage.jsonl"]},
        )
        if is_exposed(
            signature,
            [r["fingerprints"] for r in read_rows(dataset / "evaluator/lineage.jsonl")],
            4,
        ):
            raise ValueError(
                "Reserved acceptance image or duplicate risk blocked before model load"
            )
    output.mkdir(parents=True)
    if args.roi:
        if image is None:
            raise ValueError("ROI requires a query image")
        if (
            not 0 <= args.roi[0] < args.roi[2] <= 1
            or not 0 <= args.roi[1] < args.roi[3] <= 1
        ):
            raise ValueError("Ordered normalized ROI required")
        with Image.open(image) as source:
            crop = ImageOps.exif_transpose(source).convert("RGB")
            crop = crop.crop(pixel_roi(args.roi, crop.size))
            image = output / "query_roi.png"
            crop.save(image)
    index, titles, _ = RegionalIndex.load(
        PROJECT_ROOT / "data/index_regions_products_v1"
    )
    metadata = load(BANK / "feature_metadata.json")
    assert titles is not None and metadata["gallery_ids"] == [
        r["id"] for r in index.records
    ]
    with np.load(BANK / "profile_vectors.npz", allow_pickle=False) as bank:
        profiles = bank["vectors"]
    with np.load(BANK / "features.npz", allow_pickle=False) as bank:
        colors = bank["color_bank"]
    encoder = QwenEmbedder()
    try:
        query = encoder.encode(text=args.text, image=image, query=True)
        x = feature_matrix(
            SemanticRankerFeatureMatrixInputs(
                index,
                titles,
                profiles,
                metadata["profile_texts"],
                query,
                text=args.text,
                image=image,
                color_bank=colors,
            )
        )
    finally:
        encoder.close()
    excluded = {
        key
        for key, sig in runtime["records"].items()
        if signature
        and (
            signature["sha256"] == sig["sha256"] or signature["pixels"] == sig["pixels"]
        )
    }
    positions = np.array(
        [i for i, r in enumerate(index.records) if r["id"] not in excluded]
    )
    ids = [index.records[j]["id"] for j in positions]
    rrf, channels = recall_channels(ids, x[positions])
    why = [
        public_evidence(
            args.text, index.records[j]["title"], metadata["profile_texts"][j]
        )
        for j in positions
    ]
    offsets = np.array([r["offset"] for r in why])
    adjusted = normalized_ranks(ids, rrf) + offsets
    display_scores = rrf if args.method == "multi_channel_rrf" else adjusted
    order = ranked_ids(ids, display_scores)
    if args.method.startswith("cleaned_"):
        folder = PROJECT_ROOT / "artifacts/module2_precision_recall/adjudicated_ranker"
        verify_files(folder, load(folder / "receipt.json")["files"])
        models = load(folder / "candidate_models.json")
        name = (
            "balanced_logistic"
            if args.method == "cleaned_logistic"
            else "hard_negative_ranknet"
        )
        display_scores = predict_ranker(x[positions], models[name])
        order = ranked_ids(ids, display_scores)
    pool = []
    if args.method.startswith("cross_pool"):
        channels.append(ranked_ids(ids, adjusted))
        size = int(args.method.removeprefix("cross_pool"))
        pool = balanced_pool(channels, size)
        catalog = {r["id"]: r for r in index.records}
        runner = BusinessReranker()
        try:
            logits = runner.score(
                [
                    dict(
                        id=k,
                        image_path=catalog[k]["image_path"],
                        text=catalog[k]["title"],
                    )
                    for k in pool
                ],
                query_text=args.text,
                query_image=image,
            )
        finally:
            runner.close()
        order = rerank_complete(ids, rrf, pool, logits, offsets)
        blend = (
            0.65 * normalized_ranks(pool, logits)
            + 0.35 * normalized_ranks(ids, rrf)[[ids.index(k) for k in pool]]
            + offsets[[ids.index(k) for k in pool]]
        )
        display_scores = display_scores.copy()
        display_scores[[ids.index(k) for k in pool]] = blend
    rows = index.search(
        RegionalSearchRequest(
            query, top_k=len(index.records), strategy="attention", exclude_ids=excluded
        )
    )
    by_id = {r["id"]: r for r in rows}
    evidence = dict(zip(ids, why))
    results = []
    for rank, key in enumerate(order[:10], 1):
        row = {
            **by_id[key],
            "rank": rank,
            "score": float(display_scores[ids.index(key)]),
        }
        row["public_evidence"] = evidence[key]
        results.append(row)
    value = dict(
        results=results,
        rankings=order,
        excluded_ids=sorted(excluded),
        method=args.method,
        rerank_pool=pool,
        query=dict(
            text=args.text, image=str(args.image) if args.image else None, roi=args.roi
        ),
        embedding_calls=1,
        result_cache_used=False,
        processing_seconds=time.perf_counter() - started,
        formal_acceptance=False,
        formal_metrics=None,
        goal_met=False,
        score_semantics="Uncalibrated ranking score from the selected development method, not measured Precision or calibrated relevance probability",
        implementation_sha256=sha256_file(Path(__file__)),
        default_promoted=False,
    )
    write_json(output / "results.json", value)
    write_regional_gallery(
        output / "results.html",
        [
            dict(
                query=args.text or str(args.image),
                query_image=str(image) if image else None,
                strategy=args.method,
                results=results,
            )
        ],
    )
    print(
        json.dumps(
            dict(
                method=args.method,
                results=len(results),
                seconds=value["processing_seconds"],
                formal_metrics=None,
            )
        )
    )


if __name__ == "__main__":
    main()
