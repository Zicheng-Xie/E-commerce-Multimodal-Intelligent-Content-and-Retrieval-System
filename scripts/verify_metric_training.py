"""Verify source-group isolation and replay complete development metrics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.retrieval_strategies import extended_metrics
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    PROTECTED_ROOT,
    sha256_file,
    write_json,
)
from prepare_metric_training import DATA, EXPOSURE, OUT
from train_retrieval_metric import METRICS, OLD


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve the completed training verification.
        ValueError: Source group leakage; Duplicate or previously exposed source ID;
            Known-image overlap or changed copied pixels; Incorrect protected-source
            lineage; Read-only source lineage no longer matches; Official training
            identity mismatch; Cross-group duplicate risk; Training image encoder
            received non-image label fields; Trained model group evidence mismatch;
            Query adapter anchors include another data role; Incomplete gallery ranking;
            Metric replay mismatch; Reported macro mean mismatch.

    Notes:
        Writes derived artifacts through write_json.
    """
    output = OUT / "verification_full.json"
    if output.exists():
        raise FileExistsError("Preserve the completed training verification")
    exposure = json.loads(EXPOSURE.read_text(encoding="utf-8"))
    records = read_rows(DATA / "records.jsonl")
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    verify_files(DATA, manifest["files"])
    old_pairs = set(exposure["deepfashion_pair_ids_conservative_cross_split"])
    train = {r["pair_id"] for r in records if r["project_role"] == "training"}
    development = {
        r["pair_id"] for r in records if r["project_role"] == "development_validation"
    }
    if train & development or (train | development) & old_pairs:
        raise ValueError("Source group leakage")
    if len({r["id"] for r in records}) != len(records) or set(
        r["id"] for r in records
    ) & set(exposure["ids"]):
        raise ValueError("Duplicate or previously exposed source ID")
    fresh = []
    for row in records:
        image = DATA / row["image_path"]
        actual = fingerprints(image)
        if actual != row["fingerprints"] or is_exposed(
            actual, exposure["image_fingerprints"]
        ):
            raise ValueError("Known-image overlap or changed copied pixels")
        annotation = Path(row["annotation_path"])
        original = Path(row["source_image"])
        if not annotation.resolve().is_relative_to(
            PROTECTED_ROOT
        ) or not original.resolve().is_relative_to(PROTECTED_ROOT):
            raise ValueError("Incorrect protected-source lineage")
        if (
            sha256_file(annotation) != row["annotation_sha256"]
            or sha256_file(original) != row["image_sha256"]
        ):
            raise ValueError("Read-only source lineage no longer matches")
        truth = json.loads(annotation.read_text(encoding="utf-8"))
        items = [v for k, v in truth.items() if k.startswith("item")]
        key = f"train/{truth['pair_id']}/{items[0]['style']}/{items[0]['category_id']}"
        if (
            len(items) != 1
            or key != row["identity"]
            or str(truth["pair_id"]) != row["pair_id"]
        ):
            raise ValueError("Official training identity mismatch")
        fresh.append((row["pair_id"], actual))
    for group, signature in fresh:
        if is_exposed(signature, [s for g, s in fresh if g != group]):
            raise ValueError("Cross-group duplicate risk")
    public = json.loads((DATA / "model_inputs.json").read_text(encoding="utf-8"))
    if len(public) != len(records) or any(
        set(r) != {"id", "image_path", "query"} for r in public
    ):
        raise ValueError("Training image encoder received non-image label fields")
    for folder in ["module2_blind", "module2_precision"]:
        root = PROJECT_ROOT / "artifacts" / folder
        receipt = json.loads(
            (root / "completion_receipt.json").read_text(encoding="utf-8")
        )
        verify_files(root, receipt["files"])
        if "implementation" in receipt:
            verify_files(PROJECT_ROOT, receipt["implementation"])
    metric_model = json.loads((OUT / "analysis/model.json").read_text(encoding="utf-8"))
    attention_model = json.loads(
        (OUT / "region_attention/model.json").read_text(encoding="utf-8")
    )
    for model in [metric_model, attention_model]:
        if (
            set(model["training_query_groups"]) != train
            or set(model["training_gallery_groups"]) != train
            or set(model["excluded_development_groups"]) != development
        ):
            raise ValueError("Trained model group evidence mismatch")
    with np.load(OUT / "features.npz", allow_pickle=False) as bank:
        by_id = {r["opaque_id"]: r for r in records}
        ordered = [by_id[i] for i in bank["ids"].tolist()]
        train_queries = [
            i
            for i, r in enumerate(ordered)
            if r["source"] == "user" and r["project_role"] == "training"
        ]
        if not np.allclose(
            bank["vectors"][train_queries], np.array(attention_model["ridge_anchors"])
        ):
            raise ValueError("Query adapter anchors include another data role")
    old_protocol = json.loads((OLD / "protocol.json").read_text(encoding="utf-8"))[
        "source_protocol"
    ]
    truth_by_cohort = {
        "old_development": {
            q["record"]["id"]: set(q["relevant_ids"]) for q in old_protocol["queries"]
        },
        "new_group_development": {
            q["id"]: {
                g["id"]
                for g in records
                if g["source"] == "shop"
                and g["project_role"] == "development_validation"
                and g["identity"] == q["identity"]
            }
            for q in records
            if q["source"] == "user" and q["project_role"] == "development_validation"
        },
    }
    old_ids = {
        r["id"]
        for r in json.loads((OLD / "index/records.json").read_text(encoding="utf-8"))
    }
    new_ids = {
        r["id"]
        for r in records
        if r["project_role"] == "development_validation" and r["source"] == "shop"
    }
    checks = 0
    for target in [OUT / "analysis", OUT / "region_attention", OUT / "rank_correction"]:
        report = json.loads((target / "report.json").read_text(encoding="utf-8"))
        for cohort, slot in [
            ("old_development", "old"),
            ("new_group_development", "new"),
        ]:
            rankings = json.loads(
                (target / f"{cohort}_rankings.json").read_text(encoding="utf-8")
            )
            saved = json.loads(
                (target / f"{cohort}_query_metrics.json").read_text(encoding="utf-8")
            )
            candidates = old_ids if slot == "old" else new_ids
            by_method: dict = {}
            for row in saved:
                ranked = rankings[row["query_id"]][row["method"]]
                if len(ranked) != len(candidates) or set(ranked) != candidates:
                    raise ValueError("Incomplete gallery ranking")
                scores = extended_metrics(
                    ranked, truth_by_cohort[cohort][row["query_id"]]
                )
                if any(abs(scores[k] - row[k]) > 1e-12 for k in scores):
                    raise ValueError("Metric replay mismatch")
                by_method.setdefault(row["method"], []).append(scores)
                checks += 1
            for method, rows in by_method.items():
                for key in METRICS:
                    if (
                        abs(
                            np.mean([r[key] for r in rows])
                            - report[
                                cohort if target.name == "rank_correction" else slot
                            ]["summary"][method][key]
                        )
                        > 1e-12
                    ):
                        raise ValueError("Reported macro mean mismatch")
    write_json(
        output,
        {
            "records": len(records),
            "groups": {"training": len(train), "development": len(development)},
            "previous_groups_excluded": len(old_pairs),
            "prior_or_cross_split_group_overlap": 0,
            "prior_byte_pixel_or_dhash_overlap": 0,
            "new_cross_group_duplicate_risks": 0,
            "record_roles": dict(Counter(r["project_role"] for r in records)),
            "complete_ranking_metric_sets_replayed": checks,
            "trained_model_group_audits": 2,
            "consumed_blind_and_previous_precision_files_unchanged": True,
            "formal_acceptance": False,
        },
    )
    print(
        json.dumps(
            {"records_verified": len(records), "ranking_metric_sets_replayed": checks}
        )
    )


if __name__ == "__main__":
    main()
