"""Validate actual dual-human consensus, then train/score only available evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_evaluation import score_business_development
from multimodal_ecommerce.business_review_jobs import merge_values
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)
from multimodal_ecommerce.semantic_ranker import ranked_ids
from multimodal_ecommerce.supervised_pipeline import (
    SupervisedPipelineFitSupervisedBundleInputs,
    fit_supervised_bundle,
)

REVIEW = PROJECT_ROOT / "artifacts/module2_business_review/development"
BASE = PROJECT_ROOT / "artifacts/module2_machine_assist/semantic_ranker"
OUT = PROJECT_ROOT / "artifacts/module2_continuation/human_pipeline"


def validate_consensus(
    manifest: dict, templates: list[dict], provenance: dict, actual: list[dict]
) -> dict:
    """Validate consensus.

    Args:
        manifest: Manifest used by this operation.
        templates: Templates used by this operation.
        provenance: Provenance used by this operation.
        actual: Actual used by this operation.

    Returns:
        Result mapping produced by the workflow.

    Raises:
        ValueError: Need original A/B/adjudication provenance; Submitted human labels do
            not match the independently recomputed consensus.
    """
    if set(provenance) != {"primary", "adjudications"}:
        raise ValueError("Need original A/B/adjudication provenance")
    result = merge_values(
        manifest, templates, provenance["primary"], provenance["adjudications"]
    )
    if result["rows"] != actual:
        raise ValueError(
            "Submitted human labels do not match the independently recomputed consensus"
        )
    return result


def run_human_pipeline(merge_dir: Path) -> dict:
    """Return run human pipeline for this workflow.

    Args:
        merge_dir: Merge dir used by this operation.

    Returns:
        Mapping with fields: state, human_completed, human_pending, human_positive,
        human_negative, quality_metrics, formal_metrics, formal_acceptance, goal_met,
        source_merge, training_executed.

    Notes:
        Writes derived artifacts through target.mkdir, write_json, write_rows.
    """
    merge_dir = merge_dir.resolve()
    merge_dir.relative_to((REVIEW / "merges").resolve())
    verify_files(
        REVIEW,
        json.loads((REVIEW / "release_receipt.json").read_text(encoding="utf-8"))[
            "files"
        ],
    )
    manifest = json.loads((REVIEW / "manifest.json").read_text(encoding="utf-8"))
    template = read_rows(REVIEW / "judgments.blank.jsonl")
    rows = read_rows(merge_dir / "judgments.jsonl")
    provenance = json.loads(
        (merge_dir / "input_provenance.json").read_text(encoding="utf-8")
    )
    consensus = validate_consensus(manifest, template, provenance, rows)
    implementation = {
        name: sha256_file(PROJECT_ROOT / name)
        for name in [
            "src/multimodal_ecommerce/human_business_pipeline.py",
            "src/multimodal_ecommerce/supervised_pipeline.py",
            "scripts/human_business_pipeline.py",
        ]
    }
    source = PROJECT_ROOT / "artifacts/module2_continuation/refined_ranker"
    constraints = PROJECT_ROOT / "artifacts/module2_continuation/constraints"
    if (constraints / "receipt.json").exists():
        source = constraints
    elif not (source / "receipt.json").exists():
        source = BASE
    receipt_name = "receipt.json" if source != BASE else "training_receipt.json"
    verify_files(
        source, json.loads((source / receipt_name).read_text(encoding="utf-8"))["files"]
    )
    key = hashlib.sha256(
        json.dumps(
            dict(
                labels=sha256_file(merge_dir / "judgments.jsonl"),
                provenance=sha256_file(merge_dir / "input_provenance.json"),
                rankings=sha256_file(source / "rankings.json"),
                implementation=implementation,
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()[:24]
    target = ensure_output_path(OUT / "revisions" / key)
    if (target / "receipt.json").exists():
        verify_files(
            target,
            json.loads((target / "receipt.json").read_text(encoding="utf-8"))["files"],
        )
        return json.loads((target / "status.json").read_text(encoding="utf-8"))
    target.mkdir(parents=True, exist_ok=True)
    write_rows(target / "human_consensus.jsonl", rows)
    progress = consensus["status"]
    status = dict(
        state="awaiting_real_human_consensus",
        human_completed=progress["completed"],
        human_pending=progress["pending"],
        human_positive=progress["positive"],
        human_negative=progress["negative"],
        quality_metrics=None,
        formal_metrics=None,
        formal_acceptance=False,
        goal_met=False,
        source_merge=merge_dir.relative_to(PROJECT_ROOT).as_posix(),
        training_executed=False,
    )
    rankings = json.loads((source / "rankings.json").read_text(encoding="utf-8"))
    if progress["completed"]:
        verify_files(
            BASE,
            json.loads((BASE / "feature_receipt.json").read_text(encoding="utf-8"))[
                "files"
            ],
        )
        metadata = json.loads(
            (BASE / "feature_metadata.json").read_text(encoding="utf-8")
        )
        mapping = {(r["query_id"], r["candidate_id"]): r["grade"] for r in rows}
        labels = np.array(
            [
                [
                    np.nan if mapping.get((qid, cid)) is None else mapping[qid, cid]
                    for cid in metadata["gallery_ids"]
                ]
                for qid in metadata["query_ids"]
            ],
            dtype=float,
        )
        with np.load(BASE / "features.npz", allow_pickle=False) as bank:
            x = bank["features"]
            valid = bank["valid"]
            qfolds = bank["query_folds"]
            gfolds = bank["gallery_folds"]
            base = bank["base_scores"]
        policy = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))[
            "policy"
        ]
        result = fit_supervised_bundle(
            SupervisedPipelineFitSupervisedBundleInputs(
                x,
                labels,
                valid,
                qfolds,
                gfolds,
                base,
                metadata,
                policy,
                "declared_human_business_development",
            )
        )
        status["training_state"] = "trained" if result["trained"] else result["state"]
        if result["trained"]:
            status["training_executed"] = True
            write_json(target / "candidate_models.json", result["candidates"])
            write_json(target / "cross_fit_models.json", result["models"])
            write_json(target / "cross_fit_audit.json", result["audits"])
            for i, qid in enumerate(metadata["query_ids"]):
                positions = np.flatnonzero(valid[i])
                ids = [metadata["gallery_ids"][j] for j in positions]
                for name, scores in result["predictions"].items():
                    rankings[qid]["human_oof_" + name] = ranked_ids(
                        ids, scores[i, positions]
                    )
            write_json(target / "rankings.json", rankings)
            status["state"] = "partial_human_labels_trained_full_metrics_pending"
    if progress["pending"] == 0:
        allowed = json.loads(
            (
                PROJECT_ROOT
                / "artifacts/module2_precision60/business/allowed_candidates.json"
            ).read_text(encoding="utf-8")
        )
        queries = json.loads(
            (
                PROJECT_ROOT
                / "artifacts/module2_precision60/business/review_context.json"
            ).read_text(encoding="utf-8")
        )["queries"]
        modes = {q["id"]: "image" if q["image_path"] else "text" for q in queries}
        scoring = score_business_development(rankings, allowed, rows, modes)
        write_json(target / "query_metrics.json", scoring.pop("query_metrics"))
        write_json(target / "human_development_metrics.json", scoring)
        status.update(
            state="complete_human_development_labels_scored",
            quality_metrics=scoring["quality_metrics"],
            development_gates=scoring["development_gates"],
        )
    write_json(target / "status.json", status)
    write_json(
        target / "receipt.json",
        dict(
            implementation=implementation,
            source_label_sha256=sha256_file(merge_dir / "judgments.jsonl"),
            source_provenance_sha256=sha256_file(merge_dir / "input_provenance.json"),
            files={
                p.relative_to(target).as_posix(): sha256_file(p)
                for p in target.rglob("*")
                if p.is_file()
            },
        ),
    )
    write_json(
        OUT / "current_status.json",
        {**status, "revision": target.relative_to(PROJECT_ROOT).as_posix()},
    )
    return status
