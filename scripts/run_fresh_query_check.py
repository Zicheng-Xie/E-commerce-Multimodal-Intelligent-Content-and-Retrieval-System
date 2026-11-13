"""Run a fixed six-query development challenge after native AI labels are frozen."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import run  # noqa: F401
from advance_rank_stability import NAME, PARENT
from advance_rank_stability import ROOT as STAGE
from multimodal_ecommerce.ai_review_metrics import metric_ranges
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.development_expansion import load_expanded_index
from multimodal_ecommerce.head_guard import protected_prefix
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.intent_retrieval import rerank_complete
from multimodal_ecommerce.rank_stability import stable_tail
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.semantic_ranker import (
    SemanticRankerFeatureMatrixInputs,
    feature_matrix,
    ranked_ids,
)
from prepare_fresh_query_check import (
    QUEUE,
    REGISTRY,
    ROOT,
    check,
    font,
)
from report_focus_coverage import paired_differences


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def finish_acquisition() -> None:
    """Preserve the frozen preparer's schema error; finish visualization separately.

    Raises:
        FileExistsError: Preserve acquisition evidence.
    """
    check()
    if (ROOT / "acquisition_receipt.json").exists():
        raise FileExistsError("Preserve acquisition evidence")
    queries = read_rows(ROOT / "queries.evaluator.jsonl")
    registry = read_rows(REGISTRY)
    assert len(queries) == len(registry) == 6
    public = load(ROOT / "queries.public.json")
    assert public == [
        {k: q[k] for k in ["id", "text", "image_path", "exclude_id"]} for q in queries
    ]
    prior = load(PARENT / "post_development_exclusion_snapshot.json")
    signatures = list(prior["image_fingerprints"])
    metadata = {r["id"]: r for r in read_rows(QUEUE)}
    for q in queries:
        assert q["source_product_id"] in metadata
        assert not set(metadata[q["source_product_id"]]["family_keys"]) & set(
            prior["family_keys"]
        )
        actual = fingerprints(Path(q["image_path"]))
        assert actual == q["image_fingerprints"]
        assert not is_exposed(actual, signatures, 4)
        signatures.append(actual)
    gallery = load(PARENT / "inputs.public.json")["catalog"]
    folder = ROOT / "native_review"
    write_rows(
        folder / "gallery_mapping.jsonl",
        [dict(position=i, **r) for i, r in enumerate(gallery, 1)],
    )
    for start in range(0, len(gallery), 36):
        panel = Image.new("RGB", (1800, 1800), "white")
        drawing = ImageDraw.Draw(panel)
        for offset, row in enumerate(gallery[start : start + 36]):
            x, y = (offset % 6) * 300, (offset // 6) * 300
            with Image.open(row["image_path"]) as im:
                thumb = ImageOps.contain(
                    ImageOps.exif_transpose(im).convert("RGB"), (288, 235)
                )
            panel.paste(thumb, (x + (300 - thumb.width) // 2, y))
            drawing.text(
                (x + 5, y + 237), str(start + offset + 1), fill="black", font=font(20)
            )
            drawing.text(
                (x + 5, y + 260), row["title"][:16], fill="black", font=font(16)
            )
            drawing.text(
                (x + 5, y + 280), row["title"][16:32], fill="black", font=font(14)
            )
        number = start // 36 + 1
        panel.save(folder / f"gallery_{number:02d}.jpg", quality=95)
    write_json(
        ROOT / "acquisition_repair.json",
        dict(
            original_preparer_unchanged=True,
            error="After all six downloads and registry writes, contact-sheet generation used documents instead of catalog in the frozen parent public input schema (KeyError).",
            repair="Separate finalizer verifies acquired queries and creates gallery contact sheets using catalog. No resampling, extra downloads, or changed strategy/labels.",
            frozen_preparer_sha256=sha256_file(
                PROJECT_ROOT / "scripts/prepare_fresh_query_check.py"
            ),
            finalizer_sha256=sha256_file(Path(__file__)),
            parent_public_sha256=sha256_file(PARENT / "inputs.public.json"),
        ),
    )
    write_json(
        ROOT / "acquisition_receipt.json",
        dict(
            requested=6,
            selected=6,
            metadata=18,
            fetched=6,
            files={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                for p in [
                    REGISTRY,
                    ROOT / "queries.public.json",
                    ROOT / "queries.evaluator.jsonl",
                    ROOT / "acquisition_repair.json",
                    *sorted((ROOT / "native_review").glob("*.jpg")),
                    ROOT / "native_review/gallery_mapping.jsonl",
                ]
            },
            model_inferences=0,
            acceptance_predictions=0,
        ),
    )
    print(json.dumps(dict(acquisition_finished=True, selected=6, gallery_sheets=12)))


def materialize_review() -> None:
    """Expand a complete native visual review; unmentioned mismatched bodies are 0.

    Raises:
        FileExistsError: Preserve challenge review.
    """
    check()
    verify_files(PROJECT_ROOT, load(ROOT / "acquisition_receipt.json")["files"])
    if (ROOT / "judgments.ai.jsonl").exists():
        raise FileExistsError("Preserve challenge review")
    observations = load(ROOT / "native_review/observations.json")
    gallery = read_rows(ROOT / "native_review/gallery_mapping.jsonl")
    query_ids = {q["id"] for q in load(ROOT / "queries.public.json")}
    assert set(observations["queries"]) == query_ids
    assert observations["all_gallery_sheets_viewed"] == list(range(1, 13))
    rows = []
    for qid, review in observations["queries"].items():
        positive = set(review["positive_positions"])
        unknown = set(int(p) for p in review["unknown_positions"])
        assert not positive & unknown
        assert positive | unknown <= set(range(1, 419))
        for document in gallery:
            position = document["position"]
            grade = 1 if position in positive else None if position in unknown else 0
            reason = (
                review["positive_reason"]
                if grade == 1
                else (
                    review["unknown_positions"][str(position)]
                    if grade is None
                    else review["negative_reason"]
                )
            )
            rows.append(
                dict(
                    query_id=qid,
                    candidate_id=document["id"],
                    grade=grade,
                    reason=reason,
                    native_position=position,
                    evidence="native_review/observations.json",
                    review_origin="model",
                    human_label=False,
                )
            )
    write_rows(ROOT / "judgments.ai.jsonl", rows)
    print(
        json.dumps(
            dict(
                relations=len(rows),
                unknown=sum(r["grade"] is None for r in rows),
                human_labels=0,
            )
        )
    )


def freeze_inference() -> None:
    """Freeze inference.

    Raises:
        FileExistsError: Preserve inference freeze.

    Notes:
        Writes derived artifacts through write_json.
    """
    check()
    if (ROOT / "inference_protocol.json").exists():
        raise FileExistsError("Preserve inference freeze")
    verify_files(PROJECT_ROOT, load(ROOT / "acquisition_receipt.json")["files"])
    files = [
        Path(__file__),
        ROOT / "queries.public.json",
        ROOT / "judgments.ai.jsonl",
        ROOT / "native_review/observations.json",
        ROOT / "review_policy.json",
        STAGE / "protocol.json",
        PROJECT_ROOT / "scripts/retrieve_focus_development.py",
        PROJECT_ROOT / "src/multimodal_ecommerce/rank_stability.py",
        PARENT / "query_runtime_protocol.json",
        PARENT / "index/manifest.json",
        PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl",
    ]
    files += [Path(q["image_path"]) for q in load(ROOT / "queries.public.json")]
    write_json(
        ROOT / "inference_protocol.json",
        dict(
            files={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in files
            },
            methods=[
                "global",
                "region_title",
                "described_cross_pool20",
                "region_top2_described_cross_pool20",
                NAME,
            ],
            queries=6,
            expected_gallery=418,
            cross_pairs_per_query=20,
            caption_entry="fixed existing query-only local Qwen3.5",
            inference_entry="scripts/retrieve_focus_development.py --method described_cross_pool20",
            replaces_for_low_scores=False,
            parameter_search=False,
            independent_final_acceptance=False,
            labels_frozen_before_predictions=True,
            human_labels=0,
            acceptance_predictions=0,
            caveat="Fresh query families, fixed exposed development gallery, same AI evaluator. This is a development stress check, not formal heldout or independent annotation validation.",
        ),
    )
    print(
        json.dumps(dict(inference_frozen=True, queries=6, planned_full_cross_pairs=120))
    )


def infer() -> None:
    """Run model inference using the configured inputs.

    Raises:
        FileExistsError: Preserve interrupted runtime output for audit before any retry.

    Notes:
        Writes derived artifacts through logs.mkdir, write_json.
    """
    protocol = load(ROOT / "inference_protocol.json")
    verify_files(PROJECT_ROOT, protocol["files"])
    queries = load(ROOT / "queries.public.json")
    logs = ROOT / "process_logs"
    logs.mkdir(exist_ok=True)
    for q in queries:
        output = ROOT / "runs" / q["id"]
        if (output / "results.json").exists():
            continue
        if output.exists():
            raise FileExistsError(
                "Preserve interrupted runtime output for audit before any retry"
            )
        command = [
            sys.executable,
            "scripts/retrieve_focus_development.py",
            "--image",
            q["image_path"],
            "--method",
            "described_cross_pool20",
            "--output",
            str(output),
        ]
        start = time.perf_counter()
        with (logs / (q["id"] + ".log")).open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT
            )
        write_json(
            logs / (q["id"] + ".json"),
            dict(
                returncode=process.returncode,
                wall_seconds=time.perf_counter() - start,
                command=command,
            ),
        )
        if process.returncode:
            raise RuntimeError(f"Fresh query failed; preserve logs: {q['id']}")
        print(
            json.dumps(dict(query=q["id"], completed=True, formal_acceptance=False)),
            flush=True,
        )


def reconstruct() -> dict:
    """Replay real fresh request vectors/logits; no evaluation query lookup.

    Returns:
        Result mapping produced by the workflow.
    """
    protocol = load(ROOT / "inference_protocol.json")
    verify_files(PROJECT_ROOT, protocol["files"])
    index, titles = load_expanded_index(PARENT / "index")
    with np.load(PARENT / "index/vectors.npz", allow_pickle=False) as bank:
        profiles = bank["profiles"].copy()
    with np.load(PARENT / "features.npz", allow_pickle=False) as bank:
        colors = bank["color_bank"].copy()
    profile_texts = load(PARENT / "index/profile_texts.json")
    ids = [r["id"] for r in index.records]
    rankings = {}
    for q in load(ROOT / "queries.public.json"):
        output = ROOT / "runs" / q["id"]
        result = load(output / "results.json")
        assert result["candidate_count"] == len(ids) == 418
        assert result["source_image_sha256"] == sha256_file(Path(q["image_path"]))
        assert not result["historical_query_rankings_used"]
        assert not result["relevance_table_used_as_runtime_feature"]
        with np.load(output / "query_vectors.npz", allow_pickle=False) as bank:
            vector = bank["views"][0].copy()
        features = feature_matrix(
            SemanticRankerFeatureMatrixInputs(
                index,
                titles,
                profiles,
                profile_texts,
                vector,
                image=Path(q["image_path"]),
                color_bank=colors,
            )
        )
        fusion = load(output / "query_fusion.json")
        cross = load(output / "cross_encoder.json")
        assert fusion["ids"] == ids and cross["pool"] == fusion["pool"][:20]
        assert len(cross["logits"]) == 20 and all(np.isfinite(cross["logits"]))
        order = rerank_complete(
            ids,
            np.array(fusion["fused_scores"]),
            cross["pool"],
            np.array(cross["logits"]),
        )
        assert order[:10] == [r["id"] for r in result["results"]]
        anchor = ranked_ids(ids, 0.8 * features[:, 1] + 0.2 * features[:, 3])
        protected = protected_prefix(anchor, order, 2)
        rankings[q["id"]] = {
            "global": ranked_ids(ids, features[:, 0]),
            "region_title": anchor,
            "described_cross_pool20": order,
            "region_top2_described_cross_pool20": protected,
            NAME: stable_tail(anchor, protected),
        }
    return rankings


def report() -> None:
    """Summarize the recorded results of this development workflow.

    Raises:
        FileExistsError: Preserve fresh query evaluation.

    Notes:
        Writes derived artifacts through write_json.
    """
    if (ROOT / "evaluation_receipt.json").exists():
        raise FileExistsError("Preserve fresh query evaluation")
    rankings = reconstruct()
    labels = read_rows(ROOT / "judgments.ai.jsonl")
    modes = {q: "image" for q in rankings}
    metrics = metric_ranges(rankings, labels, modes)
    paired = paired_differences(rankings, labels, modes)
    write_json(ROOT / "rankings.json", rankings)
    write_json(ROOT / "internal_metrics.ai.development.json", metrics)
    write_json(ROOT / "internal_paired_differences.json", paired)
    write_json(
        ROOT / "evaluation_receipt.json",
        dict(
            files={
                n: sha256_file(ROOT / n)
                for n in [
                    "rankings.json",
                    "internal_metrics.ai.development.json",
                    "internal_paired_differences.json",
                ]
            },
            query_count=6,
            full_rankings=30,
            actual_full_cross_pairs=120,
            review_origin="model",
            human_labels=0,
            formal_acceptance=False,
            acceptance_predictions=0,
        ),
    )
    print(
        json.dumps(
            dict(evaluated=True, queries=6, full_rankings=30, formal_acceptance=False)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "finish_acquisition",
            "materialize_review",
            "freeze_inference",
            "infer",
            "report",
        ],
    )
    args = parser.parse_args()
    globals()[args.action]()
