"""Search the exposed development catalog using semantic profiles and optional ROI."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
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
)

ROOT = PROJECT_ROOT / "artifacts/module2_machine_assist/semantic_ranker"


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve prior query results.
        ValueError: Reserved acceptance image or duplicate risk cannot enter this
            development search command; Semantic vectors no longer align with the
            published-title index; Runtime fingerprints no longer align with the
            gallery.

    Notes:
        Writes derived artifacts through crop.save, output.mkdir, write_json.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--roi", type=float, nargs=4)
    parser.add_argument(
        "--method",
        choices=[
            "semantic_profile_blend",
            "balanced_logistic",
            "hard_negative_ranknet",
        ],
        default="semantic_profile_blend",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        (args.text is None) == (args.image is None)
        or (args.roi and args.image is None)
        or args.top_k < 1
    ):
        parser.error(
            "Exactly one query modality is required; ROI requires an image; top-k must be positive"
        )
    start = time.perf_counter()
    output = ensure_output_path(PROJECT_ROOT / args.output)
    if output.exists():
        raise FileExistsError("Preserve prior query results")
    receipt = json.loads((ROOT / "training_receipt.json").read_text(encoding="utf-8"))
    verify_files(ROOT, receipt["files"])
    receipt = json.loads((ROOT / "feature_receipt.json").read_text(encoding="utf-8"))
    verify_files(ROOT, receipt["files"])
    registry = read_rows(
        PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl"
    )
    image = args.image.resolve() if args.image else None
    original_signature = fingerprints(image) if image is not None else None
    if image is not None:
        assert original_signature is not None
        signatures = read_rows(
            PROJECT_ROOT
            / "artifacts/module2_business_acceptance/dataset/evaluator/lineage.jsonl"
        )
        if sha256_file(image) in {r["image_sha256"] for r in registry} or is_exposed(
            original_signature, [r["fingerprints"] for r in signatures], 4
        ):
            raise ValueError(
                "Reserved acceptance image or duplicate risk cannot enter this development search command"
            )
    if args.roi:
        if image is None:
            raise ValueError("ROI requires a query image")
        with Image.open(image) as source:
            crop = ImageOps.exif_transpose(source).convert("RGB")
            box = pixel_roi(args.roi, crop.size)
            crop = crop.crop(box)
        output.mkdir(parents=True)
        image = output / "query_roi.png"
        crop.save(image)
    else:
        output.mkdir(parents=True)
    index, titles, manifest = RegionalIndex.load(
        PROJECT_ROOT / "data/index_regions_products_v1"
    )
    metadata = json.loads((ROOT / "feature_metadata.json").read_text(encoding="utf-8"))
    if metadata["gallery_ids"] != [r["id"] for r in index.records] or titles is None:
        raise ValueError(
            "Semantic vectors no longer align with the published-title index"
        )
    with np.load(ROOT / "profile_vectors.npz", allow_pickle=False) as bank:
        profiles = bank["vectors"].copy()
    with np.load(ROOT / "features.npz", allow_pickle=False) as bank:
        colors = bank["color_bank"].copy()
    encoder = QwenEmbedder()
    try:
        inference_start = time.perf_counter()
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
        if args.method == "semantic_profile_blend":
            scores = 0.5 * x[:, 1] + 0.2 * x[:, 3] + 0.3 * x[:, 4]
        else:
            models = json.loads(
                (ROOT / "candidate_models.json").read_text(encoding="utf-8")
            )
            scores = predict_ranker(x, models[args.method])
        excluded = set()
        if args.image:
            runtime_receipt = json.loads(
                (ROOT / "runtime_receipt.json").read_text(encoding="utf-8")
            )
            verify_files(ROOT, runtime_receipt["files"])
            cached = json.loads(
                (ROOT / "runtime_fingerprints.json").read_text(encoding="utf-8")
            )
            verify_files(PROJECT_ROOT, cached["source_files"])
            if set(cached["records"]) != set(metadata["gallery_ids"]):
                raise ValueError(
                    "Runtime fingerprints no longer align with the gallery"
                )
            sig = original_signature
            assert sig is not None
            for identifier, other in cached["records"].items():
                if sig["sha256"] == other["sha256"] or sig["pixels"] == other["pixels"]:
                    excluded.add(identifier)
        results = index.search(
            RegionalSearchRequest(
                query,
                top_k=len(index.records),
                strategy="attention",
                exclude_ids=excluded,
            )
        )
        by_id = {r["id"]: i for i, r in enumerate(index.records)}
        for row in results:
            i = by_id[row["id"]]
            row.update(
                score=float(scores[i]),
                text_score=float(x[i, 3]),
                semantic_profile_score=float(x[i, 4]),
                machine_product_profile=metadata["profile_texts"][i],
            )
        results.sort(key=lambda r: (-r["score"], r["id"]))
        results = results[: args.top_k]
        for rank, row in enumerate(results, 1):
            row["rank"] = rank
        elapsed = time.perf_counter() - inference_start
        result = dict(
            query=args.text or str(args.image),
            query_image=str(image) if image else None,
            strategy=args.method,
            roi=args.roi,
            results=results,
            model=manifest["model"],
            query_and_search_seconds_excluding_model_and_index_load=elapsed,
            total_seconds_including_model_and_index_load=time.perf_counter() - start,
            diagnostics=encoder.diagnostics(),
            formal_acceptance=False,
            human_quality_validated=False,
            score_semantics="Ranking scores from an unvalidated development candidate, not calibrated relevance probabilities",
            reserved_acceptance_used=False,
        )
        write_json(output / "results.json", result)
        write_regional_gallery(output / "results.html", [result])
        print(
            json.dumps(
                dict(
                    output=str(output),
                    top_ids=[r["id"] for r in results],
                    query_and_search_seconds=elapsed,
                    formal_acceptance=False,
                ),
                ensure_ascii=False,
            )
        )
    finally:
        encoder.close()


if __name__ == "__main__":
    main()
