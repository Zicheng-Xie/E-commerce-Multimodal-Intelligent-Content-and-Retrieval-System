"""Check automatic view choices against the already frozen native development QA."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import run  # noqa: F401
from business_grounding_audit import ROOT, load, once, relative
from multimodal_ecommerce.safety import sha256_file


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through once.
    """
    automatic = load(ROOT / "semantic_results.json")
    native = load(ROOT / "QA/query_views.native_AI.json")["observations"]
    by = {r["opaque"]: r for r in automatic}
    critical = []
    unresolved = []
    for row in native:
        selected = by[row["opaque"]]["selected_view"]
        bad = (
            selected == "masked_native"
            and (row["mask_decision"] == "reject" or row["box_decision"] == "reject")
        ) or (selected == "box_native" and row["box_decision"] == "reject")
        unknown = (
            selected == "masked_native"
            and (
                row["mask_decision"] == "uncertain"
                or row["box_decision"] == "uncertain"
            )
        ) or (selected == "box_native" and row["box_decision"] == "uncertain")
        item = dict(
            query=row["query_key"],
            automatic_view=selected,
            native_box=row["box_decision"],
            native_mask=row["mask_decision"],
            native_evidence=row["native_visual_evidence"],
        )
        if bad:
            critical.append(item)
        elif unknown:
            unresolved.append(item)
    decision = {
        "reviewed_records": len(automatic),
        "actual_native_model_calls": sum(r["model_call"] for r in automatic),
        "automatic_view_counts": dict(Counter(r["selected_view"] for r in automatic)),
        "native_query_QA_records": len(native),
        "critical_QA_disagreements": critical,
        "unresolved_QA_disagreements": unresolved,
        "automatic_subject_quality_gate_passed": not critical and not unresolved,
        "new_masks_in_primary_training_allowed": not critical and not unresolved,
        "policy_for_next_control": "Keep every original image and every query. When this gate fails, disable this new mask branch globally; never select a list of favored query IDs.",
        "old_masks_and_raw_model_outputs_unchanged": True,
        "AI_not_independent_human_validation": True,
        "formal_metrics": None,
    }
    once(ROOT / "semantic_quality_assessment.json", decision)
    paths = [
        Path(__file__),
        ROOT / "semantic_results.json",
        ROOT / "semantic_receipt.json",
        ROOT / "QA/query_views.native_AI.json",
        ROOT / "semantic_quality_assessment.json",
    ]
    once(
        ROOT / "semantic_assessment_receipt.json",
        {
            "files": {relative(p): sha256_file(p) for p in paths},
            "completed": True,
            "formal_metrics": None,
        },
    )
    print(
        json.dumps(
            {
                "semantic_assessment_completed": True,
                "automatic_subject_quality_gate_passed": decision[
                    "automatic_subject_quality_gate_passed"
                ],
                "new_masks_enabled_in_primary_training": decision[
                    "new_masks_in_primary_training_allowed"
                ],
                "formal_metrics": None,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
