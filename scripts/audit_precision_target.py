"""Audit the 60% goal against existing complete same-item development labels."""

from __future__ import annotations

import json

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.precision_target import (
    partial_precision_bounds,
    precision_capacity,
)
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

OUT = PROJECT_ROOT / "artifacts/module2_precision60"


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve the target audit.

    Notes:
        Writes derived artifacts through write_json.
    """
    path = OUT / "capacity_audit.json"
    if path.exists():
        raise FileExistsError("Preserve the target audit")
    old = PROJECT_ROOT / "artifacts/module2_regions_benchmark_v1/protocol.json"
    source = json.loads(old.read_text(encoding="utf-8"))["source_protocol"]
    old_counts = [len(set(q["relevant_ids"])) for q in source["queries"]]
    data = PROJECT_ROOT / "data/module2_training"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    verify_files(data, manifest["files"])
    rows = read_rows(data / "records.jsonl")
    new_counts = [
        sum(
            g["project_role"] == "development_validation"
            and g["source"] == "shop"
            and g["identity"] == q["identity"]
            for g in rows
        )
        for q in rows
        if q["project_role"] == "development_validation" and q["source"] == "user"
    ]
    blank = (
        PROJECT_ROOT / "artifacts/module2_precision/review_packet/judgments.blank.jsonl"
    )
    labels = read_rows(blank)
    queries = sorted({r["query_id"] for r in labels})
    packet_bounds = partial_precision_bounds(
        [[r["grade"] for r in labels if r["query_id"] == q] for q in queries]
    )
    report = {
        "target": {
            "precision@10": 0.6,
            "recall@10": 0.7,
            "definition": "Fixed denominator 10; all queries retained",
        },
        "old_development": precision_capacity(old_counts),
        "new_group_development": precision_capacity(new_counts),
        "human_training_packet_partial_gallery": packet_bounds,
        "consumed_blind_not_reopened_or_rescored": True,
        "formal_acceptance": False,
        "formal_metrics": None,
        "required_next_evidence": "Confirm relevance definition; obtain real labels on group-disjoint development and independently held-out queries before measuring acceptance",
        "source_hashes": {
            str(p.relative_to(PROJECT_ROOT)): sha256_file(p)
            for p in [old, data / "manifest.json", blank]
        },
    }
    write_json(path, report)
    print(
        json.dumps({k: report[k] for k in ["old_development", "new_group_development"]})
    )


if __name__ == "__main__":
    main()
