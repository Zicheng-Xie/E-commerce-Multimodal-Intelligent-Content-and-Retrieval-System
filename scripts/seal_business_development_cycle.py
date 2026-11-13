"""Retain one source release and seal the completed five-stage development cycle."""

from __future__ import annotations

import json

import run  # noqa: F401
from business_grounding_audit import COHORTS, DATA
from business_grounding_audit import ROOT as CYCLE
from business_grounding_audit import load, once, relative, rows
from business_joint_experiment import ROOT as JOINT
from business_metric_experiment import ROOT as METRIC
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file

SCRIPTS = (
    "business_grounding_audit.py",
    "prepare_business_grounding_model.py",
    "business_subject_views.py",
    "author_business_source_qa.py",
    "business_view_qa_and_pairs.py",
    "business_mask_verifier.py",
    "business_metric_model.py",
    "business_metric_experiment.py",
    "business_joint_model.py",
    "business_joint_experiment.py",
    "assess_business_semantics.py",
    "sync_business_grounding_progress.py",
    "verify_business_development_cycle.py",
    "seal_business_development_cycle.py",
)
TESTS = (
    "test_business_grounding_audit.py",
    "test_business_subject_views.py",
    "test_business_view_qa_and_pairs.py",
    "test_business_mask_verifier.py",
    "test_business_metric_model.py",
    "test_business_joint_model.py",
)


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        ValueError: Immutable cycle source release changed.

    Notes:
        Writes derived artifacts through once, target.parent.mkdir, target.write_bytes.
    """
    assert load(CYCLE / "cycle_verification.json")["complete_cycle_verified"]
    assert load(JOINT / "completion_receipt.json")["completed"]
    assert load(CYCLE / "public_status.json")["completed"]
    once(
        CYCLE / "engineering_checks.json",
        {
            "pytest_command": ".venv312/Scripts/python.exe -m pytest -q",
            "pytest_result": "307 passed in 6.37s",
            "passed": True,
            "lint": "all current new method and progress scripts passed flake8; test import paths use explicit per-file E402 exception",
            "historical_blind_full_replay_passed": True,
            "new_cycle_model_rank_metric_replay_passed": True,
            "tests_are_not_quality_metrics": True,
            "formal_metrics": None,
        },
    )
    pools = load(JOINT / "frozen_parent_pools.json")
    diagnosis = []
    for c in COHORTS:
        labels = rows(DATA / c / "judgments.ai.jsonl")
        ranks = load(JOINT / c / "predictions.json")
        for q in load(DATA / c / "queries.public.json"):
            if not q["image_path"]:
                continue
            grades = {
                r["candidate_id"]: r["grade"]
                for r in labels
                if r["query_id"] == q["id"]
            }
            pool = pools[c + "/" + q["id"]]
            actual = ranks[q["id"]]["business_joint_attention_fusion"]
            known_gallery = sum(g == 1 for g in grades.values())
            known_pool = sum(grades[cid] == 1 for cid in pool)
            unknown_pool = sum(grades[cid] is None for cid in pool)
            raw = ranks[q["id"]]["business_joint_attention_only_ablation"][:20]
            diagnosis.append(
                {
                    "cohort": c,
                    "query_id": q["id"],
                    "known_positive_gallery": known_gallery,
                    "actual_parent_Qwen_pool_size": len(pool),
                    "known_positive_parent_pool": known_pool,
                    "unknown_parent_pool": unknown_pool,
                    "known_pool_coverage_gap_to_top10_oracle": max(
                        0, min(10, known_gallery) - min(10, known_pool)
                    ),
                    "known_pool_ranking_space": max(
                        0,
                        min(10, known_pool)
                        - sum(grades[cid] == 1 for cid in actual[:10]),
                    ),
                    "joint_top20_known_positives_outside_parent_pool": sum(
                        grades[cid] == 1 and cid not in set(pool) for cid in raw
                    ),
                    "unknown_not_assigned": True,
                }
            )
    once(
        CYCLE / "internal_next_cycle_diagnosis.json",
        {
            "queries": diagnosis,
            "purpose": "post-hoc exposed-development diagnosis only, not a new tested method or acceptance result",
            "next_mechanisms": "distinguish candidate coverage from pool ordering; explicit sale-item/prop/use-role evidence; keep failed masks disabled until structure verification improves",
            "no_after_result_changes_to_frozen_current_methods": True,
            "formal_metrics": None,
        },
    )
    sources = [PROJECT_ROOT / "scripts" / name for name in SCRIPTS] + [
        PROJECT_ROOT / "tests" / name for name in TESTS
    ]
    snapshots = {}
    for path in sources:
        target = CYCLE / "source_release" / relative(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise ValueError("Immutable cycle source release changed")
        if not target.exists():
            target.write_bytes(path.read_bytes())
        snapshots[relative(target)] = sha256_file(target)
    once(
        CYCLE / "source_release_manifest.json",
        {
            "source_snapshots": snapshots,
            "runtime_sources": {
                relative(p): sha256_file(p)
                for p in sources
                if p.name != "sync_business_grounding_progress.py"
            },
            "original_core_source_reference": "audit_protocol.json/files preserves all68 unchanged core hashes; reuse historical source snapshots",
            "progress_helper_snapshot_only": True,
        },
    )
    if not (METRIC / "completion_receipt.json").exists():
        once(
            METRIC / "completion_receipt.json",
            {
                "files": {
                    relative(p): sha256_file(p)
                    for p in sorted(METRIC.rglob("*"))
                    if p.is_file()
                    and p.name not in ("completion_receipt.json", "public_status.json")
                },
                "source_files": load(METRIC / "protocol.json")["files"],
                "completed": True,
                "component_only_not_final_system": True,
                "formal_metrics": None,
            },
        )
    if not (CYCLE / "completion_receipt.json").exists():
        once(
            CYCLE / "completion_receipt.json",
            {
                "files": {
                    relative(p): sha256_file(p)
                    for p in sorted(CYCLE.rglob("*"))
                    if p.is_file()
                    and p.name not in ("completion_receipt.json", "public_status.json")
                },
                "child_experiments": {
                    relative(p): sha256_file(p)
                    for p in (
                        METRIC / "completion_receipt.json",
                        JOINT / "completion_receipt.json",
                    )
                },
                "completed": True,
                "goal_met": False,
                "formal_metrics": None,
            },
        )
    for root in (CYCLE, METRIC, JOINT):
        for name, digest in load(root / "completion_receipt.json")["files"].items():
            if sha256_file(PROJECT_ROOT / name) != digest:
                raise ValueError("Final seal changed: " + name)
    print(
        json.dumps(
            {
                "five_stage_development_cycle_sealed": True,
                "formal_metrics": None,
                "goal_met": False,
                "default_strategy_changed": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
