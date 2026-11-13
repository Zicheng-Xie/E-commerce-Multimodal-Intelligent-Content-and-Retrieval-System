"""Compile direct AI judgments, refit fixed group-masked rankers and report ranges."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.ai_review_metrics import metric_ranges
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.semantic_ranker import ranked_ids
from multimodal_ecommerce.supervised_pipeline import (
    SupervisedPipelineFitSupervisedBundleInputs,
    fit_supervised_bundle,
)

ROOT = PROJECT_ROOT / "artifacts/module2_ai_review"
BANK = PROJECT_ROOT / "artifacts/module2_machine_assist/semantic_ranker"
PREVIOUS = PROJECT_ROOT / "artifacts/module2_precision_recall"
CAT = PROJECT_ROOT / "artifacts/module2_catalog_review"


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path, values: list[dict]) -> None:
    """Apply the rows workflow step.

    Args:
        path: Path of the input or output artifact used by this operation.
        values: Values used by this operation.

    Notes:
        Writes derived artifacts through path.write_text.
    """
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values),
        encoding="utf-8",
    )


def compile_reviews() -> tuple[Any, list, list, list[dict]]:
    """Return compile reviews for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Need all original visual observations in order; Expansion visual
            audit coverage mismatch; Inconsistent native query decisions; AI evidence
            missing; Native AI provenance required; Inconsistent expansion AI decisions;
            Full unchanged relation matrix required.

    Notes:
        Writes derived artifacts through write_json.
    """
    inputs = load(ROOT / "inputs.json")
    original = [p for p in inputs["catalog"] if p["catalog_source"] == "original286"]
    expansion = [p for p in inputs["catalog"] if p["catalog_source"] == "expansion64"]
    audit = ROOT / "native_audit"
    with (audit / "original/observations.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        observations = list(csv.DictReader(handle, delimiter="\t"))
    if len(observations) != 286 or [int(r["position"]) for r in observations] != list(
        range(1, 287)
    ):
        raise ValueError("Need all original visual observations in order")
    native = read_rows(audit / "catalog_observations.ai.jsonl")
    if len(native) != 64 or [r["candidate_id"] for r in native] != [
        p["id"] for p in expansion
    ]:
        raise ValueError("Expansion visual audit coverage mismatch")
    labels = []
    for product, observation in zip(original, observations):
        positive = set(filter(None, observation["positive_queries"].split(",")))
        unknown = set(filter(None, observation["unknown_queries"].split(",")))
        if positive & unknown or not positive | unknown <= {
            q["id"] for q in inputs["queries"]
        }:
            raise ValueError("Inconsistent native query decisions")
        if not observation["reason"].strip():
            raise ValueError("AI evidence missing")
        for query in inputs["queries"]:
            if query["exclude_id"] == product["id"]:
                continue
            labels.append(
                dict(
                    query_id=query["id"],
                    candidate_id=product["id"],
                    grade=(
                        None if query["id"] in unknown else int(query["id"] in positive)
                    ),
                    reason=observation["reason"],
                    native_position=int(observation["position"]),
                    evidence="native_audit/original/observations.tsv",
                    review_origin="model",
                    human_label=False,
                )
            )
    # Direct visual relevance decisions, separate from acquisition buckets and QA.
    with (audit / "expansion_relations.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        expansion_decisions = list(csv.DictReader(handle, delimiter="\t"))
    assert [int(r["position"]) for r in expansion_decisions] == list(range(1, 65))
    for product, observation, decision in zip(expansion, native, expansion_decisions):
        if (
            observation["review_origin"] != "model"
            or observation["human_label"] is not False
        ):
            raise ValueError("Native AI provenance required")
        positive = set(filter(None, decision["positive_queries"].split(",")))
        unknown = set(filter(None, decision["unknown_queries"].split(",")))
        if positive & unknown or not positive | unknown <= {
            q["id"] for q in inputs["queries"]
        }:
            raise ValueError("Inconsistent expansion AI decisions")
        for query in inputs["queries"]:
            grade = None if query["id"] in unknown else int(query["id"] in positive)
            labels.append(
                dict(
                    query_id=query["id"],
                    candidate_id=product["id"],
                    grade=grade,
                    reason=observation["reason"]
                    + (
                        " 手机背壳矩形轮廓及镜头开孔与image_04相似。"
                        if query["id"] == "image_04" and grade == 1
                        else ""
                    ),
                    native_position=observation["position"],
                    evidence="native_audit/catalog_observations.ai.jsonl",
                    review_origin="model",
                    human_label=False,
                )
            )
    if (
        len(labels) != 6295
        or len({(r["query_id"], r["candidate_id"]) for r in labels}) != 6295
    ):
        raise ValueError("Full unchanged relation matrix required")
    rows(ROOT / "judgments.ai.jsonl", labels)
    rows(ROOT / "catalog_QA.ai.jsonl", native)
    usable = {r["candidate_id"] for r in native if r["verdict"] == "usable"}
    approved = [
        dict(
            p,
            review_origin="model",
            human_approved=False,
            role="AI_reviewed_development_catalog",
        )
        for p in inputs["catalog"]
        if p["catalog_source"] == "original286" or p["id"] in usable
    ]
    rows(ROOT / "catalog.reviewed.jsonl", approved)
    write_json(
        ROOT / "catalog.reviewed.manifest.json",
        dict(
            products=len(approved),
            original_products=286,
            new_AI_usable=len(usable),
            new_rejected=sum(r["verdict"] == "reject" for r in native),
            new_uncertain=sum(r["verdict"] == "uncertain" for r in native),
            labels_are="AI judgments from direct images/titles, not independent human truth",
            active_index_updated=False,
            query_relevance_not_in_runtime_features=True,
            formal_metrics=None,
            acceptance_predictions=0,
        ),
    )
    conflicts = load(PREVIOUS / "machine_conflicts_for_review.json")["rows"]
    pairs = {(r["query_id"], r["candidate_id"]) for r in conflicts}
    priority = [r for r in labels if (r["query_id"], r["candidate_id"]) in pairs]
    assert len(priority) == 59
    rows(ROOT / "priority59.ai.jsonl", priority)
    capacity = {}
    for name, ids in [
        ("original286", {p["id"] for p in original}),
        ("reviewed344", {p["id"] for p in approved}),
        (
            "all350_including_rejected_and_uncertain",
            {p["id"] for p in inputs["catalog"]},
        ),
    ]:
        counts = []
        for q in inputs["queries"]:
            selected = [
                r
                for r in labels
                if r["query_id"] == q["id"] and r["candidate_id"] in ids
            ]
            p = sum(r["grade"] == 1 for r in selected)
            u = sum(r["grade"] is None for r in selected)
            counts.append(
                dict(
                    query_id=q["id"],
                    mode="text" if q["text"] else "image",
                    known_positive=p,
                    unknown=u,
                    precision_ceiling_lower=min(10, p) / 10,
                    precision_ceiling_upper=min(10, p + u) / 10,
                )
            )
        scopes = {}
        for scope in ["all", "text", "image"]:
            selected = [r for r in counts if scope == "all" or r["mode"] == scope]
            scopes[scope] = [
                sum(r[f"precision_ceiling_{side}"] for r in selected) / len(selected)
                for side in ["lower", "upper"]
            ]
        capacity[name] = dict(
            by_scope=scopes,
            query_counts=counts,
            observed_precision=None,
            note="Oracle capacity under AI judgments; no expanded retrieval rankings or observed gains implied",
        )
    write_json(ROOT / "capacity.json", capacity)
    return inputs, labels, priority, native


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        ValueError: Refit protocol changed.

    Notes:
        Writes derived artifacts through destination.mkdir, write_json.
    """
    prior = load(ROOT / "protocol.json")
    verify_files(PROJECT_ROOT, prior["guards"])
    verify_files(PROJECT_ROOT, prior["image_files"])
    inputs, labels, priority, native = compile_reviews()
    protocol_path = ROOT / "ranker_protocol.json"
    specification = dict(
        role="AI_development_group_masked_refit",
        label_origin="machine_pixel_refined_development",
        label_sha256=sha256_file(ROOT / "judgments.ai.jsonl"),
        recipe=load(BANK / "protocol.json")["policy"],
        implementation={
            p: sha256_file(PROJECT_ROOT / p)
            for p in [
                "scripts/finalize_module2_ai_review.py",
                "src/multimodal_ecommerce/ai_review_metrics.py",
                "src/multimodal_ecommerce/supervised_pipeline.py",
                "src/multimodal_ecommerce/semantic_ranker.py",
            ]
        },
        bank_features_sha256=sha256_file(BANK / "features.npz"),
        bank_metadata_sha256=sha256_file(BANK / "feature_metadata.json"),
        native_observation_files={
            str(p.relative_to(ROOT)): sha256_file(p)
            for p in [
                ROOT / "native_audit/original/observations.tsv",
                ROOT / "native_audit/catalog_observations.ai.jsonl",
                ROOT / "native_audit/expansion_relations.tsv",
                ROOT / "native_audit/full_review_protocol.json",
            ]
        },
        changed_hyperparameters=False,
        query_and_product_family_masks=True,
        default_promotion=False,
        acceptance_inference=False,
    )
    if protocol_path.exists():
        if specification != load(protocol_path):
            raise ValueError("Refit protocol changed")
    else:
        write_json(protocol_path, specification)
    metadata = load(BANK / "feature_metadata.json")
    lookup = {(r["query_id"], r["candidate_id"]): r["grade"] for r in labels}
    y = np.array(
        [
            [
                np.nan if lookup.get((q, c)) is None else lookup[q, c]
                for c in metadata["gallery_ids"]
            ]
            for q in metadata["query_ids"]
        ],
        dtype=float,
    )
    with np.load(BANK / "features.npz", allow_pickle=False) as bank:
        x, valid, qfold, gfold, base = [
            bank[k]
            for k in [
                "features",
                "valid",
                "query_folds",
                "gallery_folds",
                "base_scores",
            ]
        ]
    fitted = fit_supervised_bundle(
        SupervisedPipelineFitSupervisedBundleInputs(
            x,
            y,
            valid,
            qfold,
            gfold,
            base,
            metadata,
            specification["recipe"],
            "machine_pixel_refined_development",
        )
    )
    if not fitted["trained"]:
        raise ValueError(fitted)
    destination = ROOT / "ranker"
    destination.mkdir(exist_ok=True)
    for name, key in [
        ("cross_fit_models", "models"),
        ("candidate_models", "candidates"),
        ("cross_fit_audit", "audits"),
    ]:
        write_json(destination / (name + ".json"), fitted[key])
    rankings = load(PREVIOUS / "adjudicated_ranker/rankings.json")
    for i, qid in enumerate(metadata["query_ids"]):
        positions = np.flatnonzero(valid[i])
        ids = [metadata["gallery_ids"][j] for j in positions]
        for name, scores in fitted["predictions"].items():
            rankings[qid]["native_AI_oof_" + name] = ranked_ids(
                ids, scores[i, positions]
            )
    write_json(destination / "rankings.json", rankings)
    old = {p["id"] for p in inputs["catalog"] if p["catalog_source"] == "original286"}
    modes = {q["id"]: "text" if q["text"] else "image" for q in inputs["queries"]}
    original_labels = [r for r in labels if r["candidate_id"] in old]
    metrics = metric_ranges(rankings, original_labels, modes)
    write_json(ROOT / "metrics.ai.development.json", metrics)
    raw_profiles = list((ROOT / "profiles").glob("*.json"))
    status = dict(
        state="AI_review_complete_fixed_recipe_refit_complete",
        reviewed_products=350,
        original_products=286,
        expansion_candidates=64,
        reviewed_relations=6295,
        original_relations=5143,
        labels=dict(Counter(str(r["grade"]) for r in labels)),
        priority_pairs_reviewed=59,
        priority_results=dict(Counter(str(r["grade"]) for r in priority)),
        catalog_QA=dict(Counter(r["verdict"] for r in native)),
        reviewed_development_catalog_products=344,
        active_index_updated=False,
        partial_local_teacher_profiles_retained=len(raw_profiles),
        local_teacher_run_stopped="Cross-query responses confused sold subjects with background/props; replaced by complete native AI visual adjudication, partial outputs retained.",
        cross_fit_models=18,
        candidate_models=2,
        methods_compared=21,
        full_rankings=378,
        review_origin="model",
        human_labels=0,
        human_assignment_pending=False,
        formal_metrics=None,
        formal_acceptance=False,
        goal_met=False,
        acceptance_prediction_count=0,
        default_strategy_changed=False,
    )
    write_json(ROOT / "status.json", status)
    current = load(CAT / "public_status.json")
    current.update(
        state="AI_review_completed_user_delegated_no_human_wait",
        active_review_status="artifacts/module2_ai_review/status.json",
        human_assignment_pending=False,
        AI_reviewed_relations=6295,
        AI_catalog_usable=58,
        AI_catalog_rejected=3,
        AI_catalog_uncertain=3,
        active_index_updated=False,
        formal_metrics=None,
    )
    write_json(CAT / "public_status.json", current)
    print(json.dumps(status, ensure_ascii=False))
    for scope, value in load(ROOT / "capacity.json").items():
        print(scope, value["by_scope"])
    for name in [
        "global",
        "region_title",
        "native_AI_oof_balanced_logistic",
        "native_AI_oof_hard_negative_ranknet",
    ]:
        print(name, metrics["scopes"]["all"][name]["ranges"])


if __name__ == "__main__":
    main()
