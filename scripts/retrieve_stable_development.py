"""Fresh image retrieval with a protected head and baseline-ordered tail."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import run  # noqa: F401
from advance_rank_stability import NAME, PARENT, ROOT, check, load
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.development_expansion import load_expanded_index
from multimodal_ecommerce.head_guard import protected_prefix
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.intent_retrieval import rerank_complete
from multimodal_ecommerce.rank_stability import stable_tail
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)
from multimodal_ecommerce.semantic_ranker import (
    SemanticRankerFeatureMatrixInputs,
    feature_matrix,
    ranked_ids,
)


def guard_source(source: Path) -> dict:
    """Return guard source for this workflow.

    Args:
        source: Source used by this operation.

    Returns:
        Result mapping produced by the workflow.

    Raises:
        ValueError: Reserved acceptance path blocked before pixels or model loading;
            Reserved/near-duplicate query blocked before model loading.
    """
    if {"module2_business_acceptance", "module2_reserved_acceptance"} & set(
        source.parts
    ):
        raise ValueError(
            "Reserved acceptance path blocked before pixels or model loading"
        )
    signature = fingerprints(source)
    reserved = [
        r["fingerprints"]
        for r in read_rows(
            PROJECT_ROOT
            / "artifacts/module2_business_acceptance/dataset/evaluator/lineage.jsonl"
        )
    ]
    if is_exposed(signature, reserved, 4):
        raise ValueError("Reserved/near-duplicate query blocked before model loading")
    return signature


def project_request(request: Path, output: Path) -> None:
    """Apply the project request workflow step.

    Args:
        request: Request used by this operation.
        output: Output used by this operation.

    Raises:
        ValueError: Need actual full described-image model request; Evaluation lookup is
            not a fresh request; Original query changed; Cached model scores do not
            align with this public index/request; Need at least twenty actual finite
            model scores.

    Notes:
        Writes derived artifacts through write_json.
    """
    raw = load(request / "results.json")
    if raw["method"] not in {"described_cross_pool20", "described_cross_pool60"}:
        raise ValueError("Need actual full described-image model request")
    if (
        raw["historical_query_rankings_used"]
        or raw["relevance_table_used_as_runtime_feature"]
    ):
        raise ValueError("Evaluation lookup is not a fresh request")
    source = Path(raw["source_image"]).resolve()
    signature = guard_source(source)
    if signature["sha256"] != raw["source_image_sha256"]:
        raise ValueError("Original query changed")
    index, titles = load_expanded_index(PARENT / "index")
    with np.load(PARENT / "index/vectors.npz", allow_pickle=False) as bank:
        profiles = bank["profiles"].copy()
    with np.load(PARENT / "features.npz", allow_pickle=False) as bank:
        colors = bank["color_bank"].copy()
    with np.load(request / "query_vectors.npz", allow_pickle=False) as bank:
        query_vector = bank["views"][0].copy()
    target = request / "query_roi.png" if raw["roi"] else source
    features = feature_matrix(
        SemanticRankerFeatureMatrixInputs(
            index,
            titles,
            profiles,
            load(PARENT / "index/profile_texts.json"),
            query_vector,
            image=target,
            color_bank=colors,
        )
    )
    signatures = load(PARENT / "guard.json")["signatures"]
    included = [
        i
        for i, r in enumerate(index.records)
        if signatures[r["id"]]["sha256"] != signature["sha256"]
        and signatures[r["id"]]["pixels"] != signature["pixels"]
    ]
    ids = [index.records[i]["id"] for i in included]
    fusion, cross = load(request / "query_fusion.json"), load(
        request / "cross_encoder.json"
    )
    if fusion["ids"] != ids or cross["pool"] != fusion["pool"][: len(cross["pool"])]:
        raise ValueError(
            "Cached model scores do not align with this public index/request"
        )
    if (
        len(cross["pool"]) < 20
        or len(cross["pool"]) != len(cross["logits"])
        or not all(np.isfinite(cross["logits"]))
    ):
        raise ValueError("Need at least twenty actual finite model scores")
    joint = rerank_complete(
        ids,
        np.array(fusion["fused_scores"]),
        cross["pool"][:20],
        np.array(cross["logits"][:20]),
    )
    current = features[included]
    anchor = ranked_ids(ids, 0.8 * current[:, 1] + 0.2 * current[:, 3])
    protected = protected_prefix(anchor, joint, 2)
    order = stable_tail(anchor, protected)
    assert order[:10] == protected[:10]
    by_id = {r["id"]: r for r in index.records}
    write_json(
        output / "results.json",
        dict(
            role="experimental_fresh_query_development_only",
            method=NAME,
            candidate_count=len(ids),
            catalog_size=len(index.records),
            roi=raw["roi"],
            source_image=str(source),
            source_image_sha256=signature["sha256"],
            results=[dict(rank=i, **by_id[cid]) for i, cid in enumerate(order[:10], 1)],
            full_order=order,
            anchor_order=anchor,
            protected_order=protected,
            underlying_joint_order=joint,
            source_query_run=request.relative_to(PROJECT_ROOT).as_posix(),
            source_files={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                for p in [
                    request / "results.json",
                    request / "query_vectors.npz",
                    request / "query_fusion.json",
                    request / "cross_encoder.json",
                    target,
                    source,
                ]
            },
            actual_cached_full_cross_pairs_reused=20,
            new_model_inferences_in_projection=0,
            historical_query_rankings_used=False,
            relevance_table_used_as_runtime_feature=False,
            default_strategy_changed=False,
            formal_acceptance=False,
            formal_metrics=None,
            acceptance_predictions=0,
            implementation_sha256=sha256_file(Path(__file__)),
        ),
    )


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve fresh query output.
        RuntimeError: Full fresh image inference failed; preserve process.log.

    Notes:
        Writes derived artifacts through output.mkdir.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path)
    group.add_argument("--query-run", type=Path)
    parser.add_argument("--roi", type=float, nargs=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.roi and args.image is None:
        parser.error("ROI belongs to a new image query")
    if args.image:
        guard_source(args.image.resolve())
    else:
        request = ensure_output_path(PROJECT_ROOT / args.query_run)
        guard_source(Path(load(request / "results.json")["source_image"]).resolve())
    check()
    verify_files(PROJECT_ROOT, load(ROOT / "runtime_protocol.json")["files"])
    output = ensure_output_path(PROJECT_ROOT / args.output)
    if output.exists():
        raise FileExistsError("Preserve fresh query output")
    output.mkdir(parents=True)
    if args.image:
        request = output / "raw_request"
        command = [
            sys.executable,
            "scripts/retrieve_focus_development.py",
            "--image",
            str(args.image.resolve()),
            "--method",
            "described_cross_pool20",
            "--output",
            str(request),
        ]
        if args.roi:
            command += ["--roi", *map(str, args.roi)]
        with (output / "process.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT
            )
        if process.returncode:
            raise RuntimeError(
                "Full fresh image inference failed; preserve process.log"
            )
    project_request(request, output)
    print(
        json.dumps(
            dict(
                results=10,
                method=NAME,
                formal_acceptance=False,
                acceptance_predictions=0,
            )
        )
    )


if __name__ == "__main__":
    main()
