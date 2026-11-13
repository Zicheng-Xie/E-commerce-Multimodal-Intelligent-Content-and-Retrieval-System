"""One fixed default-region prefix follow-up, with a cached fresh-query entry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

import numpy as np

import run  # noqa: F401
from advance_focus_coverage import ROOT, check, load
from multimodal_ecommerce.ai_review_metrics import metric_ranges
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.development_expansion import load_expanded_index
from multimodal_ecommerce.head_guard import protected_prefix
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.intent_retrieval import rerank_complete
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
from report_focus_coverage import paired_differences

DESTINATION = ROOT / "region_guard"
NAME = "region_top2_described_cross_pool20"


class _PolicySpec(TypedDict):
    """Exact fields of the POLICY configuration and provenance record."""

    anchor: str
    candidate: str
    keep: int
    role: str
    runtime_uses_labels: bool
    changes_queries_or_relevance_rules: bool
    further_variants_in_this_stage: bool
    formal_acceptance_allowed: bool
    applies_to: str
    text_strategy: str


POLICY: _PolicySpec = dict(
    anchor="region_title",
    candidate="described_cross_pool20",
    keep=2,
    role="single_adaptive_development_followup_after_cross_anchor_MRR_limit",
    runtime_uses_labels=False,
    changes_queries_or_relevance_rules=False,
    further_variants_in_this_stage=False,
    formal_acceptance_allowed=False,
    applies_to="image",
    text_strategy="cross_evidence_pool60",
)


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        FileExistsError: Preserve fixed region-prefix follow-up.

    Notes:
        Writes derived artifacts through DESTINATION.mkdir, write_json.
    """
    if DESTINATION.exists():
        raise FileExistsError("Preserve fixed region-prefix follow-up")
    DESTINATION.mkdir()
    files = [
        "scripts/compare_region_prefix.py",
        "src/multimodal_ecommerce/head_guard.py",
        "scripts/report_focus_coverage.py",
        "src/multimodal_ecommerce/ai_review_metrics.py",
        "src/multimodal_ecommerce/semantic_ranker.py",
        "artifacts/module2_focus_coverage/head_guard/rankings.json",
        "artifacts/module2_focus_coverage/head_guard/internal_paired_differences.json",
        "artifacts/module2_focus_coverage/judgments.ai.jsonl",
    ]
    write_json(
        DESTINATION / "protocol.json",
        dict(
            policy=POLICY,
            files={p: sha256_file(PROJECT_ROOT / p) for p in files},
            reason="Cross-encoder prefix preserved its own Hit@1 but did not establish MRR relative to the original region baseline. Freeze a single region-top2 anchor comparison; preserve earlier failures.",
            original_results_preserved=True,
            new_model_inferences_for_comparison=0,
            independent_validation=False,
        ),
    )
    print(json.dumps(dict(frozen=True, new_methods=1)))


def route_guard(rankings: dict, image_modes: dict) -> dict:
    """Return route guard for this workflow.

    Args:
        rankings: Candidate rankings keyed by query and retrieval method.
        image_modes: Image modes used by this operation.

    Returns:
        Result mapping produced by the workflow.
    """
    result = {qid: dict(methods) for qid, methods in rankings.items()}
    for qid, methods in result.items():
        methods[NAME] = (
            protected_prefix(
                methods[POLICY["anchor"]], methods[POLICY["candidate"]], POLICY["keep"]
            )
            if image_modes[qid]
            else list(methods[POLICY["text_strategy"]])
        )
    return result


def evaluate() -> None:
    """Apply the evaluate workflow step.

    Raises:
        FileExistsError: Preserve follow-up evidence.

    Notes:
        Writes derived artifacts through write_json.
    """
    protocol = load(DESTINATION / "protocol.json")
    verify_files(PROJECT_ROOT, protocol["files"])
    assert protocol["policy"] == POLICY
    if (DESTINATION / "receipt.json").exists():
        raise FileExistsError("Preserve follow-up evidence")
    rankings = load(ROOT / "head_guard/rankings.json")
    image_modes = {
        q["id"]: bool(q["image_path"])
        for q in load(ROOT / "inputs.public.json")["queries"]
    }
    rankings = route_guard(rankings, image_modes)
    labels = read_rows(ROOT / "judgments.ai.jsonl")
    modes = {
        q["id"]: "text" if q["text"] else "image"
        for q in load(ROOT / "inputs.public.json")["queries"]
    }
    metrics = metric_ranges(rankings, labels, modes)
    paired = paired_differences(rankings, labels, modes)
    write_json(DESTINATION / "rankings.json", rankings)
    write_json(DESTINATION / "internal_metrics.ai.development.json", metrics)
    write_json(DESTINATION / "internal_paired_differences.json", paired)
    assessment = dict(
        precision_60_AI_lower_bound_all_modes=all(
            metrics["scopes"][s][NAME]["precision_60_lower_bound_met"]
            for s in ["all", "text", "image"]
        ),
        nonregression_all_metrics_vs_region=all(
            v["nonregression_proven_for_shared_unknown_labels"]
            for s in ["all", "text", "image"]
            for v in paired["scopes"][s][NAME]["region_title"].values()
        ),
        nonregression_all_metrics_vs_global=all(
            v["nonregression_proven_for_shared_unknown_labels"]
            for s in ["all", "text", "image"]
            for v in paired["scopes"][s][NAME]["global"].values()
        ),
        formal_metrics=None,
        goal_met=False,
        default_promotion_allowed=False,
    )
    write_json(DESTINATION / "internal_decision.json", assessment)
    write_json(
        DESTINATION / "receipt.json",
        dict(
            new_methods=1,
            new_full_rankings=18,
            total_methods=21,
            total_full_rankings=378,
            model_inferences=0,
            adaptive_development_only=True,
            formal_acceptance=False,
            files={
                p.name: sha256_file(p) for p in DESTINATION.iterdir() if p.is_file()
            },
        ),
    )
    print(json.dumps(dict(compared=True, new_methods=1, formal_acceptance=False)))


def query(parent_run: Path, output: Path) -> None:
    """Reuse actual raw scores for this exact fresh request, not saved eval queries.

    Args:
        parent_run: Parent run used by this operation.
        output: Output used by this operation.

    Raises:
        FileExistsError: Preserve guarded query output.
        ValueError: Need an actual fresh described-cross60 request; Reserved acceptance
            paths blocked before pixels; Reserved or changed original query; Fresh
            request and frozen index no longer align.
    """
    verify_files(PROJECT_ROOT, load(DESTINATION / "protocol.json")["files"])
    parent_run = ensure_output_path(PROJECT_ROOT / parent_run)
    output = ensure_output_path(PROJECT_ROOT / output)
    if output.exists():
        raise FileExistsError("Preserve guarded query output")
    run_result = load(parent_run / "results.json")
    if (
        run_result["method"] != "described_cross_pool60"
        or run_result["historical_query_rankings_used"]
    ):
        raise ValueError("Need an actual fresh described-cross60 request")
    source = Path(run_result["source_image"]).resolve()
    if {"module2_business_acceptance", "module2_reserved_acceptance"} & set(
        source.parts
    ):
        raise ValueError("Reserved acceptance paths blocked before pixels")
    signature = fingerprints(source)
    reserved = [
        r["fingerprints"]
        for r in read_rows(
            PROJECT_ROOT
            / "artifacts/module2_business_acceptance/dataset/evaluator/lineage.jsonl"
        )
    ]
    if (
        is_exposed(signature, reserved, 4)
        or signature["sha256"] != run_result["source_image_sha256"]
    ):
        raise ValueError("Reserved or changed original query")
    check()
    verify_files(PROJECT_ROOT, load(ROOT / "query_runtime_protocol.json")["files"])
    index, titles = load_expanded_index(ROOT / "index")
    with np.load(ROOT / "index/vectors.npz", allow_pickle=False) as bank:
        profiles = bank["profiles"].copy()
    with np.load(ROOT / "features.npz", allow_pickle=False) as bank:
        colors = bank["color_bank"].copy()
    with np.load(parent_run / "query_vectors.npz", allow_pickle=False) as bank:
        vector = bank["views"][0].copy()
    target = parent_run / "query_roi.png" if run_result["roi"] else source
    features = feature_matrix(
        SemanticRankerFeatureMatrixInputs(
            index,
            titles,
            profiles,
            load(ROOT / "index/profile_texts.json"),
            vector,
            image=target,
            color_bank=colors,
        )
    )
    signatures = load(ROOT / "guard.json")["signatures"]
    included = [
        i
        for i, row in enumerate(index.records)
        if signatures[row["id"]]["sha256"] != signature["sha256"]
        and signatures[row["id"]]["pixels"] != signature["pixels"]
    ]
    ids = [index.records[i]["id"] for i in included]
    fusion = load(parent_run / "query_fusion.json")
    cross = load(parent_run / "cross_encoder.json")
    if fusion["ids"] != ids or cross["pool"] != fusion["pool"]:
        raise ValueError("Fresh request and frozen index no longer align")
    candidate = rerank_complete(
        ids,
        np.array(fusion["fused_scores"]),
        cross["pool"][:20],
        np.array(cross["logits"][:20]),
    )
    current = features[included]
    anchor = ranked_ids(ids, 0.8 * current[:, 1] + 0.2 * current[:, 3])
    order = protected_prefix(anchor, candidate, 2)
    by_id = {r["id"]: r for r in index.records}
    output.mkdir(parents=True)
    write_json(
        output / "results.json",
        dict(
            role="cached_actual_fresh_query_prefix_demo_only",
            method=NAME,
            catalog_size=len(index.records),
            candidate_count=len(ids),
            results=[
                dict(
                    rank=i,
                    id=cid,
                    title=by_id[cid]["title"],
                    image_path=by_id[cid]["image_path"],
                )
                for i, cid in enumerate(order[:10], 1)
            ],
            full_order=order,
            anchor_order=anchor,
            underlying_joint_order=candidate,
            source_query_run=parent_run.relative_to(PROJECT_ROOT).as_posix(),
            source_files={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                for p in [
                    parent_run / "results.json",
                    parent_run / "query_vectors.npz",
                    parent_run / "query_fusion.json",
                    parent_run / "cross_encoder.json",
                    target,
                ]
            },
            actual_cached_full_cross_pairs_reused=20,
            new_model_inferences=0,
            historical_evaluation_query_rankings_used=False,
            relevance_labels_used_as_runtime_features=False,
            formal_metrics=None,
            formal_acceptance=False,
            acceptance_predictions=0,
        ),
    )
    print(
        json.dumps(
            dict(
                results=10,
                cached_real_pairs=20,
                new_model_inferences=0,
                formal_acceptance=False,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "evaluate", "query"])
    parser.add_argument("--query-run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "query":
        if not args.query_run or not args.output:
            parser.error("query needs --query-run and --output")
        query(args.query_run, args.output)
    else:
        globals()[args.action]()
