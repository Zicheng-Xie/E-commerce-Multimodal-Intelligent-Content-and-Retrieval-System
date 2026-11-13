"""Import actual human development labels and fit a group-isolated ranker if ready."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import run  # noqa: F401
from analyze_precision import load_entries
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.precision_labels import (
    human_training_entries,
    validate_precision_labels,
)
from multimodal_ecommerce.precision_reranking import fit_pairwise
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)
from precision_experiment import POLICY

PACKET = PROJECT_ROOT / "artifacts/module2_precision/review_packet"


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: This exact human label revision is already imported.

    Notes:
        Writes derived artifacts through target.mkdir, write_json, write_rows.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviewed",
        type=Path,
        default=PACKET / "judgments.blank.jsonl",
    )
    args = parser.parse_args()
    manifest = json.loads((PACKET / "manifest.json").read_text(encoding="utf-8"))
    verify_files(PACKET, manifest["files"])
    source = args.reviewed.resolve()
    target = ensure_output_path(
        PROJECT_ROOT
        / "artifacts/module2_training/human_review"
        / sha256_file(source)[:20]
    )
    if target.exists():
        raise FileExistsError("This exact human label revision is already imported")
    rows = read_rows(source)
    validation = validate_precision_labels(
        read_rows(PACKET / "judgments.blank.jsonl"), rows
    )
    completed = validation.pop("completed_rows")
    target.mkdir(parents=True)
    write_rows(target / "judgments.jsonl", rows)
    entries = human_training_entries(load_entries(), completed)
    groups = {e["group"] for e in entries}
    status = {
        **validation,
        "label_file_sha256": sha256_file(source),
        "source_file": str(source),
        "packet_manifest_sha256": sha256_file(PACKET / "manifest.json"),
        "eligible_training_queries": len(entries),
        "eligible_query_groups": len(groups),
        "training_ready": len(groups) >= 3,
        "training_rule": "Only explicit human 0/1; pending relations excluded from loss; both positive and negative needed per query",
        "label_semantics": "business relevance; separate from source same-item qrels",
        "default_strategy_changed": False,
        "formal_metrics": None,
    }
    if status["training_ready"]:
        folds = []
        for group in sorted(groups):
            try:
                model = fit_pairwise(entries, POLICY, excluded_groups={group})
            except ValueError:
                logging.getLogger(__name__).debug(
                    "Handled exception in main", exc_info=True
                )
                folds.append(
                    {
                        "validation_group": group,
                        "state": "insufficient_disjoint_training_pairs",
                    }
                )
            else:
                folds.append(
                    {"validation_group": group, "state": "trained", "model": model}
                )
        final = fit_pairwise(entries, POLICY, excluded_groups=set())
        write_json(
            target / "model.json",
            {
                **final,
                "label_origin": "declared_human_business_relevance",
                "formal_acceptance": False,
            },
        )
        write_json(target / "fold_models.json", folds)
        status["model_state"] = (
            "development_ranker_trained_requires_independent_validation"
        )
    else:
        status["model_state"] = "awaiting_real_positive_and_negative_human_judgments"
    write_json(target / "status.json", status)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
