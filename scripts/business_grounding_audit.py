"""Read-only development QA before new grounding/metric-learning experiments.

Consumed blind registries are used only as exclusion identities/fingerprints.
No blind media, judgments, predictions, or metrics are loaded here.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import run  # noqa: F401
from multimodal_ecommerce.holdout_data import fingerprints
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from primary_subject_ranker import signature

ROOT = PROJECT_ROOT / "artifacts/module2_business_grounding_development"
DATA = PROJECT_ROOT / "artifacts/module2_balanced_gallery"
OLD = PROJECT_ROOT / "artifacts/module2_primary_subjects"
PARENT = PROJECT_ROOT / "artifacts/module2_grounded_subject_repair"
COHORTS = ("tune", "regression", "fresh")
EXCLUSION = (
    PROJECT_ROOT
    / "artifacts/module2_grounded_independent_blind/post_blind_exclusion_snapshot.json"
)
CONSUMED = (
    PROJECT_ROOT / "data/module2_business_blind_consumed/records.jsonl",
    PROJECT_ROOT / "data/module2_grounded_blind_consumed/records.jsonl",
)


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list:
    """Return rows for this workflow.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]


def once(path: Path, value: object) -> None:
    """Persist a new result without replacing an existing artifact.

    Args:
        path: Path of the input or output artifact used by this operation.
        value: Input value to validate or transform.

    Notes:
        Writes derived artifacts through write_json.
    """
    if path.exists():
        if load(path) != value:
            raise ValueError(f"Immutable evidence changed: {path.name}")
    else:
        write_json(path, value)


def relative(path: Path) -> str:
    """Represent a workspace path relative to the project root.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Text produced by this operation.
    """
    return path.relative_to(PROJECT_ROOT).as_posix()


def capacity(positive: int, unknown: int, k: int = 10) -> dict:
    """Joint attainable points across unknown-positive counts, not model scores.

    Args:
        positive: Positive used by this operation.
        unknown: Unknown used by this operation.
        k: K used by this operation.

    Returns:
        Mapping with fields: known_positive, unknown, joint_capacity_points,
        precision_upper, recall_upper_defined, zero_positive_possible.

    Raises:
        ValueError: Invalid counts.
    """
    if min(positive, unknown) < 0 or k <= 0:
        raise ValueError("Invalid counts")
    possibilities: list[dict] = []
    for extra in range(unknown + 1):
        total = positive + extra
        possibilities.append(
            {
                "positive_count": total,
                "precision_ceiling": min(k, total) / k,
                "recall_ceiling": min(k, total) / total if total else None,
            }
        )
    return {
        "known_positive": positive,
        "unknown": unknown,
        "joint_capacity_points": possibilities,
        "precision_upper": max(x["precision_ceiling"] for x in possibilities),
        "recall_upper_defined": max(
            (
                x["recall_ceiling"]
                for x in possibilities
                if x["recall_ceiling"] is not None
            ),
            default=None,
        ),
        "zero_positive_possible": positive == 0,
    }


def exclusion_identities(
    registries: Sequence[Path],
) -> tuple[set[str], set[str], set[str], set[str], set[int]]:
    """Do not derive training examples from exclusions or inspect blind images.

    Args:
        registries: Registries used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    ids, sources, hashes, pixels, dhashes = set(), set(), set(), set(), set()
    for path in registries:
        for r in rows(path):
            ids.add(r["id"])
            sources.add(str(r.get("source_id") or r["id"].rsplit("/", 1)[-1]))
            fp = r.get("fingerprints") or {}
            if digest := fp.get("sha256") or r.get("image_sha256"):
                hashes.add(digest)
            if fp.get("pixels"):
                pixels.add(fp["pixels"])
            if fp.get("dhash") is not None:
                dhashes.add(fp["dhash"])
    return ids, sources, hashes, pixels, dhashes


def check_exclusion(
    identifier: str,
    fp: dict,
    excluded: tuple[set[str], set[str], set[str], set[str], set[int]],
) -> dict:
    """Return check exclusion for this workflow.

    Args:
        identifier: Identifier used by this operation.
        fp: Fp used by this operation.
        excluded: Excluded used by this operation.

    Returns:
        Mapping with fields: exact_consumed_overlap, near_consumed_risk.
    """
    ids, sources, hashes, pixels, dhashes = excluded
    source = identifier.rsplit("/", 1)[-1]
    exact = (
        identifier in ids
        or source in sources
        or fp["sha256"] in hashes
        or fp["pixels"] in pixels
    )
    near = any((fp["dhash"] ^ old).bit_count() <= 4 for old in dhashes)
    return {"exact_consumed_overlap": exact, "near_consumed_risk": near}


def inputs() -> dict:
    """Return inputs for this workflow.

    Returns:
        Result mapping produced by the workflow.
    """
    paths = [
        Path(__file__),
        PROJECT_ROOT / "tests/test_business_grounding_audit.py",
        PROJECT_ROOT / "scripts/primary_subject_ranker.py",
        PROJECT_ROOT / "configs/module2_current_target.json",
        EXCLUSION,
        *CONSUMED,
        PARENT / "completion_receipt.json",
        PARENT / "grounding_results.json",
        ROOT / "audit_protocol.initial.json",
        ROOT / "initial_audit_failure.json",
        ROOT / "source_history/business_grounding_audit.initial.py",
        PROJECT_ROOT
        / "artifacts/module2_balanced_supplement_features/profiles.candidate_only.jsonl",
    ]
    for c in COHORTS:
        paths.extend(
            DATA / c / name
            for name in (
                "catalog.public.json",
                "queries.public.json",
                "judgments.ai.jsonl",
                "runtime_scores.json",
            )
        )
        paths += [OLD / c / "profiles.public.jsonl", PARENT / c / "predictions.json"]
        paths += list((OLD / "query_captions" / c).glob("*.json"))
        paths += list((DATA / c / "runs").glob("*/features.json"))
    paths += sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py"))
    return {relative(p): sha256_file(p) for p in sorted(set(paths))}


def run_audit() -> None:
    """Apply the run audit workflow step.

    Raises:
        ValueError: Complete unchanged gallery/query relation matrix required; AI
            provenance missing; Duplicate or invalid relation; Missing full-gallery
            relevance cells; Development overlaps consumed blind identities or pixels.

    Notes:
        Writes derived artifacts through ROOT.mkdir, once, write_json.
    """
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "files": inputs(),
        "role": "exposed AI development only",
        "stage_order": [
            "data_quality_coverage",
            "subject_boxes_masks",
            "business_hard_negatives",
            "STIR_image_pair_attention",
            "fusion",
        ],
        "current_stage": "data_quality_coverage",
        "no_new_blind_opened": True,
        "original_labels_unchanged": True,
        "no_query_deletion": True,
        "k": 10,
        "capacity_not_performance": True,
        "unknowns_not_negatives": True,
        "human_labels": 0,
        "formal_metrics": None,
        "default_strategy_changed": False,
    }
    once(ROOT / "audit_protocol.json", protocol)
    excluded = exclusion_identities(CONSUMED)
    gallery, query_items, diagnosis = [], [], []
    fingerprints_by_path = {}
    supplement = rows(
        PROJECT_ROOT
        / "artifacts/module2_balanced_supplement_features/profiles.candidate_only.jsonl"
    )
    for cohort in COHORTS:
        profiles = {r["id"]: r for r in rows(OLD / cohort / "profiles.public.jsonl")}
        profiles.update({r["id"]: r for r in supplement})
        docs = load(DATA / cohort / "catalog.public.json")
        queries = load(DATA / cohort / "queries.public.json")
        labels = rows(DATA / cohort / "judgments.ai.jsonl")
        ids = {d["id"] for d in docs}
        if len(ids) != 315 or len(labels) != len(docs) * len(queries):
            raise ValueError(
                "Complete unchanged gallery/query relation matrix required"
            )
        if any(
            r["review_origin"] != "model" or r["human_label"] is not False
            for r in labels
        ):
            raise ValueError("AI provenance missing")
        by_query: defaultdict = defaultdict(dict)
        for r in labels:
            key = r["candidate_id"]
            if key in by_query[r["query_id"]] or r["grade"] not in (0, 1, None):
                raise ValueError("Duplicate or invalid relation")
            by_query[r["query_id"]][key] = r["grade"]
        rankings = load(PARENT / cohort / "predictions.json")
        runtime = load(DATA / cohort / "runtime_scores.json")
        for d in docs:
            p = Path(d["image_path"])
            if p not in fingerprints_by_path:
                fingerprints_by_path[p] = fingerprints(p)
            fp = fingerprints_by_path[p]
            profile = profiles[d["id"]]
            subject = signature(profile["product_type"], profile["appearance"])
            title = signature(d["title"])
            flags = []
            if profile.get("primary_unclear", False) or subject["family"] is None:
                flags.append("primary_unresolved")
            if (
                title["family"]
                and subject["family"]
                and title["family"] != subject["family"]
            ):
                flags.append("title_profile_family_disagreement_signal_not_label_error")
            gallery.append(
                {
                    "cohort": cohort,
                    "id": d["id"],
                    "image_path": relative(p),
                    "title": d["title"],
                    "AI_profile": profile,
                    "subject": subject,
                    "title_signal": title,
                    "QA_flags": flags,
                    "fingerprints": fp,
                    **check_exclusion(d["id"], fp, excluded),
                }
            )
        for q in queries:
            grades = by_query[q["id"]]
            if set(grades) != ids:
                raise ValueError("Missing full-gallery relevance cells")
            known = sum(x == 1 for x in grades.values())
            unknown = sum(x is None for x in grades.values())
            primary = rankings[q["id"]]["grounded_whole_zoom_union"]
            # Legacy text routes preserve the actual pool in features, not runtime.
            pool = (
                runtime[q["id"]]["pool"]
                if "pool" in runtime[q["id"]]
                else load(DATA / cohort / "runs" / q["id"] / "features.json")["pool"]
            )
            cap = capacity(known, unknown)
            diagnosis.append(
                {
                    "cohort": cohort,
                    "query_id": q["id"],
                    "modality": "image" if q["image_path"] else "text",
                    **cap,
                    "known_positive_in_old_pool": sum(grades[x] == 1 for x in pool),
                    "known_positive_outside_old_pool": sum(
                        grades[x] == 1 for x in ids - set(pool)
                    ),
                    "known_positive_in_parent_top10": sum(
                        grades[x] == 1 for x in primary[:10]
                    ),
                    "parent_top10_known_negative": sum(
                        grades[x] == 0 for x in primary[:10]
                    ),
                    "parent_top10_unknown": sum(
                        grades[x] is None for x in primary[:10]
                    ),
                }
            )
            if q["image_path"]:
                p = Path(q["image_path"])
                if p not in fingerprints_by_path:
                    fingerprints_by_path[p] = fingerprints(p)
                fp = fingerprints_by_path[p]
                query_items.append(
                    {
                        "cohort": cohort,
                        "query_id": q["id"],
                        "image_path": relative(p),
                        "fingerprints": fp,
                        **check_exclusion(q["id"], fp, excluded),
                    }
                )
    if any(r["exact_consumed_overlap"] for r in [*gallery, *query_items]):
        raise ValueError("Development overlaps consumed blind identities or pixels")
    aggregates: dict = {}
    for c in COHORTS:
        aggregates[c] = {}
        for scope in ("all", "image", "text"):
            values = [
                r
                for r in diagnosis
                if r["cohort"] == c and (scope == "all" or r["modality"] == scope)
            ]
            if not values:
                continue
            pcap = sum(x["precision_upper"] for x in values) / len(values)
            # Undefined zero-positive recall is optimistically 1 only for a bound.
            rcap = sum(
                (
                    x["recall_upper_defined"]
                    if x["recall_upper_defined"] is not None
                    else 1.0
                )
                for x in values
            ) / len(values)
            aggregates[c][scope] = {
                "queries": len(values),
                "optimistic_precision_capacity": pcap,
                "optimistic_recall_capacity": rcap,
                "precision_target_capacity_possible": pcap >= 0.60,
                "recall_preference_capacity_possible": rcap >= 0.85,
            }
    by_pixel = defaultdict(set)
    for r in gallery:
        by_pixel[r["fingerprints"]["pixels"]].add(r["id"])
    duplicate_groups = [sorted(v) for v in by_pixel.values() if len(v) > 1]
    result = {
        "gallery_rows_including_shared_cohorts": len(gallery),
        "unique_gallery_ids": len({r["id"] for r in gallery}),
        "unique_read_images": len(fingerprints_by_path),
        "AI_relations": 12600,
        "queries": len(diagnosis),
        "public_QA_flag_counts": dict(
            Counter(flag for r in gallery for flag in r["QA_flags"])
        ),
        "exact_duplicate_gallery_id_groups": duplicate_groups,
        "near_consumed_risk_rows": [
            {k: r[k] for k in ("cohort", "image_path")}
            for r in [*gallery, *query_items]
            if r["near_consumed_risk"]
        ],
        "catalog": gallery,
        "query_images": query_items,
        "query_diagnosis": diagnosis,
        "capacity_by_scope": aggregates,
        "capacity_is_not_model_performance": True,
        "QA_signals_are_not_relevance_truth": True,
        "all_original_records_retained": True,
        "formal_metrics": None,
    }
    once(ROOT / "internal_data_quality_audit.json", result)
    once(
        ROOT / "audit_receipt.json",
        {
            "files": {
                relative(ROOT / n): sha256_file(ROOT / n)
                for n in ("audit_protocol.json", "internal_data_quality_audit.json")
            },
            "input_files": protocol["files"],
            "completed": True,
            "formal_metrics": None,
        },
    )
    write_json(
        ROOT / "public_status.json",
        {
            "phase": "data_quality_audit_completed",
            "completed": False,
            "completed_stages": ["data_quality_coverage_audit"],
            "pending_stages": [
                "coverage_QA_review",
                "subject_boxes_masks",
                "business_hard_negatives",
                "STIR_image_pair_attention",
                "fusion_comparison",
            ],
            "development_queries": 40,
            "AI_relations_unchanged": 12600,
            "quality_metrics": None,
            "formal_metrics": None,
            "goal_met": False,
            "default_strategy_changed": False,
            "new_independent_AI_blind_predictions": 0,
        },
    )
    print(
        json.dumps(
            {
                "audit_completed": True,
                "queries": 40,
                "gallery_rows": len(gallery),
                "unique_read_images": len(fingerprints_by_path),
                "QA_flag_rows": sum(bool(r["QA_flags"]) for r in gallery),
                "consumed_exact_overlap": False,
                "formal_metrics": None,
            }
        )
    )


def verify() -> None:
    """Verify the recorded artifacts and their consistency."""
    receipt = load(ROOT / "audit_receipt.json")
    for key in ("files", "input_files"):
        for name, digest in receipt[key].items():
            if sha256_file(PROJECT_ROOT / name) != digest:
                raise ValueError(f"Audit evidence changed: {name}")
    print(json.dumps({"audit_replay_inputs_verified": True, "formal_metrics": None}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "verify"])
    action = parser.parse_args().action
    run_audit() if action == "run" else verify()
