"""Whole-image business metric control; masks and final fusion remain separate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import run  # noqa: F401
from business_grounding_audit import COHORTS, DATA, PARENT
from business_grounding_audit import ROOT as CYCLE
from business_grounding_audit import load, once, relative, rows
from business_metric_model import POLICY, fit
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.semantic_ranker import (
    SemanticRankerTrainingMaskInputs,
    training_mask,
)
from querywise_paired_bounds import paired_querywise

ROOT = PROJECT_ROOT / "artifacts/module2_business_metric_learning"
SOURCE = PROJECT_ROOT / "artifacts/module2_subject_recall_learning"
NEW = ("whole_Qwen_image_metric_ablation", "business_metric_projection_oof_ablation")


def query_bank(q: np.ndarray) -> Path:
    """Return query bank for this workflow.

    Args:
        q: Q used by this operation.

    Returns:
        Numeric result computed from the supplied values.
    """
    if q["cohort"] == "fresh":
        return (
            PROJECT_ROOT
            / "artifacts/module2_primary_form_coverage/fresh/queries"
            / q["id"]
        )
    group = "tune" if q["cohort"] == "tune" else "validation"
    return (
        PROJECT_ROOT
        / "artifacts/module2_structure_development"
        / group
        / "queries"
        / q["id"]
    )


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        ValueError: Same gallery identity has different whole-image vectors; Complete
            full-dimensional source vectors required.

    Notes:
        Writes derived artifacts through ROOT.mkdir, np.savez_compressed, once.
    """
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "protocol.json").exists():
        guard()
        return
    meta = load(SOURCE / "feature_metadata.json")
    galleries: dict = {}
    inputs = [
        SOURCE / "feature_metadata.json",
        SOURCE / "feature_bank.npz",
        CYCLE / "audit_receipt.json",
        CYCLE / "QA_and_pairs_receipt.json",
        Path(__file__),
        PROJECT_ROOT / "scripts/business_metric_model.py",
        PROJECT_ROOT / "scripts/business_grounding_audit.py",
        PROJECT_ROOT / "scripts/querywise_paired_bounds.py",
        PROJECT_ROOT / "tests/test_business_metric_model.py",
        PROJECT_ROOT / "configs/module2_current_target.json",
    ]
    inputs += sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py"))
    for cohort in ("tune", "regression"):
        folder = (
            PROJECT_ROOT
            / "artifacts/module2_balanced_supplement_features"
            / cohort
            / "index"
        )
        records = load(folder / "records.json")
        with np.load(folder / "vectors.npz", allow_pickle=False) as bank:
            vectors = bank["regions"][:, 0, :].copy()
        for record, vector in zip(records, vectors):
            if record["id"] in galleries and not np.array_equal(
                galleries[record["id"]], vector
            ):
                raise ValueError(
                    "Same gallery identity has different whole-image vectors"
                )
            galleries[record["id"]] = vector
        inputs += [
            folder / "records.json",
            folder / "vectors.npz",
            folder / "manifest.json",
        ]
    gv = np.stack([galleries[c] for c in meta["gallery_ids"]])
    query_vectors = []
    for q in meta["queries"]:
        path = query_bank(q) / "vectors.npz"
        with np.load(path, allow_pickle=False) as bank:
            query_vectors.append(bank["views"][0].copy())
        inputs.append(path)
    qv = np.stack(query_vectors)
    if (
        qv.shape != (24, 2048)
        or gv.shape != (411, 2048)
        or not np.isfinite(qv).all()
        or not np.isfinite(gv).all()
    ):
        raise ValueError("Complete full-dimensional source vectors required")
    for c in COHORTS:
        inputs += [
            DATA / c / n
            for n in (
                "catalog.public.json",
                "queries.public.json",
                "judgments.ai.jsonl",
            )
        ]
        inputs.append(PARENT / c / "predictions.json")
    np.savez_compressed(ROOT / "full_embedding_bank.npz", queries=qv, gallery=gv)
    inputs.append(ROOT / "full_embedding_bank.npz")
    once(
        ROOT / "protocol.json",
        {
            "files": {relative(p): sha256_file(p) for p in inputs},
            "policy": POLICY,
            "source_idea": "https://github.com/KevinMusgrave/pytorch-metric-learning",
            "source_models": "unchanged complete Qwen embedding/reranker; source cached actual 2048d vectors only",
            "no_new_backbone_model_calls": True,
            "new_module": "2048->256 GELU LayerNorm ->256 normalized projection; all its parameters train",
            "mask_dependency": "none; this whole-image control does not consume unverified masks",
            "methods": ["global", "cross20", "grounded_whole_zoom_union", *NEW],
            "evaluation": "all 315 per-cohort products and all40 queries, unchanged12600 AI labels; text routes remain original cross20",
            "fit": "nine original query/gallery source-family masks, 400 steps each, no search/early stopping; one separate all-development candidate",
            "OOF_not_independent_acceptance": True,
            "component_ablation_not_final_system": True,
            "STIR_training": False,
            "final_fusion": False,
            "human_labels": 0,
            "formal_metrics": None,
        },
    )
    print(
        json.dumps(
            {
                "whole_metric_protocol_frozen": True,
                "input_dimensions": 2048,
                "new_quality_metrics": None,
            }
        ),
        flush=True,
    )


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    for name, digest in load(ROOT / "protocol.json")["files"].items():
        if sha256_file(PROJECT_ROOT / name) != digest:
            raise ValueError("Metric source/input changed: " + name)


def train() -> None:
    """Fit the configured model on the allowed training data.

    Raises:
        ValueError: Projection checkpoint changed; OOF incomplete/leaking.

    Notes:
        Writes derived artifacts through (ROOT / 'models').mkdir, np.save,
        np.savez_compressed, once.
    """
    guard()
    if (ROOT / "training_receipt.json").exists():
        return
    from safetensors.torch import save_file

    meta = load(SOURCE / "feature_metadata.json")
    with np.load(SOURCE / "feature_bank.npz", allow_pickle=False) as bank:
        y, valid, qf, gf = [
            bank[k].copy() for k in ("labels", "valid", "query_folds", "gallery_folds")
        ]
    with np.load(ROOT / "full_embedding_bank.npz", allow_pickle=False) as bank:
        q, g = bank["queries"].copy(), bank["gallery"].copy()
    oof = np.full(valid.shape, np.nan, dtype=np.float32)
    audits = []
    (ROOT / "models").mkdir(exist_ok=True)
    for hq in range(3):
        for hg in range(3):
            key = f"q{hq}_g{hg}"
            allowed, evidence = training_mask(
                SemanticRankerTrainingMaskInputs(
                    meta["query_ids"],
                    qf,
                    meta["query_families"],
                    meta["gallery_families"],
                    gf,
                    hq,
                    hg,
                )
            )
            allowed &= valid
            checkpoint = ROOT / "models" / (key + ".safetensors")
            result = ROOT / "models" / (key + ".json")
            if result.exists():
                saved = load(result)
                if (
                    sha256_file(checkpoint) != saved["checkpoint_sha256"]
                    or sha256_file(ROOT / "models" / (key + ".npy"))
                    != saved["scores_sha256"]
                ):
                    raise ValueError("Projection checkpoint changed")
                scores = np.load(ROOT / "models" / (key + ".npy"), allow_pickle=False)
            else:
                started = time.perf_counter()
                model, scores, trace = fit(
                    q, g, y, allowed, POLICY["seed"] + hq * 3 + hg
                )
                save_file(model.state_dict(), checkpoint)
                np.save(ROOT / "models" / (key + ".npy"), scores, allow_pickle=False)
                saved = {
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "scores_sha256": sha256_file(ROOT / "models" / (key + ".npy")),
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "steps": POLICY["steps"],
                    "trace": trace,
                    "seconds": time.perf_counter() - started,
                }
                once(result, saved)
            held = valid & (qf[:, None] == hq) & (gf[None, :] == hg)
            oof[held] = scores[held]
            audits.append(
                {
                    "model": key,
                    "family_audit": evidence,
                    "known_training_cells": int((allowed & np.isfinite(y)).sum()),
                    "unknown_excluded": int((allowed & ~np.isfinite(y)).sum()),
                    "held_cells": int(held.sum()),
                    "training_held_overlap": bool((allowed & held).any()),
                }
            )
            print(
                json.dumps(
                    {"metric_fold_completed": key, "full_steps": POLICY["steps"]}
                ),
                flush=True,
            )
    if not np.isfinite(oof[valid]).all() or any(
        x["training_held_overlap"] for x in audits
    ):
        raise ValueError("OOF incomplete/leaking")
    np.savez_compressed(ROOT / "OOF_scores.npz", scores=oof, valid=valid)
    model, _, trace = fit(q, g, y, valid, POLICY["seed"] + 100)
    save_file(model.state_dict(), ROOT / "models/all_development.safetensors")
    once(
        ROOT / "models/all_development.json",
        {
            "training_only_not_quality_evaluation": True,
            "trace": trace,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
    )
    once(ROOT / "training_audits.json", audits)
    paths = sorted((ROOT / "models").glob("*")) + [
        ROOT / "OOF_scores.npz",
        ROOT / "training_audits.json",
    ]
    once(
        ROOT / "training_receipt.json",
        {
            "files": {relative(p): sha256_file(p) for p in paths},
            "fold_models": 9,
            "all_dev_candidates": 1,
            "formal_metrics": None,
        },
    )


def evaluate() -> None:
    """Apply the evaluate workflow step.

    Raises:
        ValueError: Missing complete metric score.

    Notes:
        Writes derived artifacts through once, write_json.
    """
    guard()
    for name, digest in load(ROOT / "training_receipt.json")["files"].items():
        if sha256_file(PROJECT_ROOT / name) != digest:
            raise ValueError("Training evidence changed: " + name)
    meta = load(SOURCE / "feature_metadata.json")
    with np.load(ROOT / "full_embedding_bank.npz", allow_pickle=False) as bank:
        q, g = bank["queries"].copy(), bank["gallery"].copy()
    base = q @ g.T
    with np.load(ROOT / "OOF_scores.npz", allow_pickle=False) as bank:
        scores = bank["scores"].copy()
    qi = {s: i for i, s in enumerate(meta["query_ids"])}
    gi = {s: i for i, s in enumerate(meta["gallery_ids"])}
    comparisons = [
        (m, b) for m in NEW for b in ("global", "cross20", "grounded_whole_zoom_union")
    ]
    for c in COHORTS:
        old = load(PARENT / c / "predictions.json")
        queries = load(DATA / c / "queries.public.json")
        ids = [r["id"] for r in load(DATA / c / "catalog.public.json")]
        out = {}
        for query in queries:
            key = query["id"]
            out[key] = {
                m: old[key][m]
                for m in ("global", "cross20", "grounded_whole_zoom_union")
            }
            for method, values in zip(NEW, (base, scores)):
                if query["image_path"]:
                    row = values[qi[c + "/" + key]]
                    if not np.isfinite([row[gi[cid]] for cid in ids]).all():
                        raise ValueError("Missing complete metric score")
                    out[key][method] = sorted(
                        ids, key=lambda cid: (-float(row[gi[cid]]), cid)
                    )
                else:
                    out[key][method] = old[key]["cross20"]
            assert all(len(v) == len(set(v)) == 315 for v in out[key].values())
        once(ROOT / c / "predictions.json", out)
        modes = {r["id"]: "image" if r["image_path"] else "text" for r in queries}
        labels = rows(DATA / c / "judgments.ai.jsonl")
        once(
            ROOT / c / "internal_metrics.ai.development.json",
            evaluate_intervals(out, labels, modes),
        )
        once(
            ROOT / c / "internal_paired_querywise.json",
            paired_querywise(out, labels, modes, comparisons),
        )
    paths = list(ROOT.glob("*/predictions.json")) + list(ROOT.glob("*/internal_*.json"))
    once(
        ROOT / "component_evaluation_receipt.json",
        {
            "files": {relative(p): sha256_file(p) for p in paths},
            "component_methods_only": True,
            "not_final_system_or_independent_acceptance": True,
            "formal_metrics": None,
        },
    )
    write_json(
        ROOT / "public_status.json",
        {
            "phase": "whole_image_metric_component_completed",
            "completed": True,
            "new_full_backbone_calls": 0,
            "new_projection_fold_models": 9,
            "all_development_candidates": 1,
            "retrieval_quality_metrics": None,
            "formal_metrics": None,
            "default_strategy_changed": False,
            "STIR_completed": False,
            "final_fusion_completed": False,
        },
    )
    print(
        json.dumps(
            {
                "whole_metric_component_evaluation_completed": True,
                "formal_metrics": None,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "train", "evaluate"])
    action = parser.parse_args().action
    {"prepare": prepare, "train": train, "evaluate": evaluate}[action]()
