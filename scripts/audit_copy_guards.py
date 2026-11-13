"""Replay new targeted guards over a frozen run; never invent human scores."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.content_guard import (
    GUARD_VERSION,
    check_claim_support,
    check_source_consistency,
)
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.evaluation import verify_files
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Use a new audit directory.

    Notes:
        Writes derived artifacts through write_json, write_rows.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=Path("artifacts/copy_v3_heldout_001")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/copy_guard_audit_v2")
    )
    args = parser.parse_args()
    source = PROJECT_ROOT / args.run
    output = ensure_output_path(PROJECT_ROOT / args.output, source)
    if output.exists():
        raise FileExistsError("Use a new audit directory")
    verify_files(
        source, json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    )
    verify_files(
        PROJECT_ROOT,
        json.loads(
            (PROJECT_ROOT / "configs/copy_recipe_v3.json").read_text(encoding="utf-8")
        ),
    )
    results = []
    for row in read_rows(source / "results.jsonl"):
        copy = row.get("copy")
        results.append(
            {
                "id": row["id"],
                "previous_category": row["category"],
                "source": row["source"],
                "historical_copy": copy,
                "legacy_status": (
                    copy.get("_meta", {}).get("release_status")
                    if copy
                    else row["status"]
                ),
                "source_guard": check_source_consistency(row["source"]),
                "claim_guard": (
                    check_claim_support(row["source"], copy) if copy else None
                ),
                "review_origin": "automated_rule_replay",
                "human_reviewed": False,
            }
        )
    source_flagged = [r for r in results if r["source_guard"]["issues"]]
    claim_flagged = [
        r for r in results if r["claim_guard"] and r["claim_guard"]["issues"]
    ]
    newly_flagged = [
        r["id"]
        for r in results
        if r["legacy_status"] == "draft_requires_human_review"
        and (
            r["source_guard"]["issues"]
            or (r["claim_guard"] and r["claim_guard"]["issues"])
        )
    ]
    report = {
        "guard_version": GUARD_VERSION,
        "source_run": str(source),
        "source_results_sha256": sha256_file(source / "results.jsonl"),
        "records": len(results),
        "source_conflict_records": len(source_flagged),
        "unsupported_claim_records": len(claim_flagged),
        "unique_flagged_records": len(
            {r["id"] for r in source_flagged + claim_flagged}
        ),
        "previously_rule_clean_flagged_ids": newly_flagged,
        "claim_issue_counts": dict(
            Counter(
                i["code"] for r in claim_flagged for i in r["claim_guard"]["issues"]
            )
        ),
        "model_inference": False,
        "human_scores_written": False,
        "quality_rate": None,
        "limitations": [
            "Replay of existing outputs only; no new-model performance estimate.",
            "Rules were developed using these inspected cases; this is a regression audit, not independent evaluation.",
            "Flag counts are review workload, not an error rate or guard precision/recall.",
            "Targeted lexical rules can miss or falsely flag semantic claims; all drafts still require review.",
        ],
    }
    write_rows(output / "results.jsonl", results)
    write_json(output / "report.json", report)
    write_json(
        output / "manifest.json",
        {
            "files": {
                name: sha256_file(output / name)
                for name in ("results.jsonl", "report.json")
            },
            "code_sha256": {
                str(p.relative_to(PROJECT_ROOT).as_posix()): sha256_file(p)
                for p in (
                    Path(__file__),
                    PROJECT_ROOT / "src/multimodal_ecommerce/content_guard.py",
                    PROJECT_ROOT / "src/multimodal_ecommerce/content_contract.py",
                )
            },
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
