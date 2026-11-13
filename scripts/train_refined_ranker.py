"""Refit the two frozen ranking designs using the additional pixel weak labels."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import run  # noqa: F401,E402
from multimodal_ecommerce.blind_evaluation import verify_files  # noqa: E402
from multimodal_ecommerce.dataset_release import read_rows  # noqa: E402
from multimodal_ecommerce.safety import (  # noqa: E402
    PROJECT_ROOT,
    sha256_file,
    write_json,
)
from multimodal_ecommerce.semantic_ranker import (  # noqa: E402
    machine_topk_bounds,
    ranked_ids,
)
from multimodal_ecommerce.supervised_pipeline import fit_supervised_bundle  # noqa: E402
from multimodal_ecommerce.supervised_pipeline import (  # noqa: E402 - local path bootstrap must run first.
    SupervisedPipelineFitSupervisedBundleInputs,
)

BASE = PROJECT_ROOT / "artifacts/module2_machine_assist/semantic_ranker"
PIXEL = PROJECT_ROOT / "artifacts/module2_continuation/pixel_refinement"
OUT = PROJECT_ROOT / "artifacts/module2_continuation/refined_ranker"


def freeze() -> None:
    """Freeze the inputs and policy used by this workflow.

    Raises:
        FileExistsError: Preserve refined-ranking protocol.

    Notes:
        Writes derived artifacts through OUT.mkdir, write_json.
    """
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve refined-ranking protocol")
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    paths = [
        "scripts/train_refined_ranker.py",
        "src/multimodal_ecommerce/supervised_pipeline.py",
        "src/multimodal_ecommerce/semantic_ranker.py",
    ]
    write_json(
        OUT / "protocol.json",
        dict(
            policy=old["policy"],
            feature_receipt_sha256=sha256_file(BASE / "feature_receipt.json"),
            pixel_protocol_sha256=sha256_file(PIXEL / "protocol.json"),
            implementation={name: sha256_file(PROJECT_ROOT / name) for name in paths},
            single_changed_factor="Additional direct-pixel machine judgments for previously unknown development relations",
            formal_acceptance=False,
            human_labels=0,
            default_changed=False,
        ),
    )
    print(json.dumps(dict(frozen=True, formal_acceptance=False)))


def fit() -> None:
    """Apply the fit workflow step.

    Raises:
        FileExistsError: Preserve refined candidates.
        ValueError: Refined ranking inputs changed.

    Notes:
        Writes derived artifacts through np.savez_compressed, write_json.
    """
    if (OUT / "receipt.json").exists():
        raise FileExistsError("Preserve refined candidates")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    verify_files(PROJECT_ROOT, protocol["implementation"])
    if (
        sha256_file(BASE / "feature_receipt.json") != protocol["feature_receipt_sha256"]
        or sha256_file(PIXEL / "protocol.json") != protocol["pixel_protocol_sha256"]
    ):
        raise ValueError("Refined ranking inputs changed")
    verify_files(
        BASE,
        json.loads((BASE / "feature_receipt.json").read_text(encoding="utf-8"))[
            "files"
        ],
    )
    verify_files(
        PIXEL, json.loads((PIXEL / "receipt.json").read_text(encoding="utf-8"))["files"]
    )
    metadata = json.loads((BASE / "feature_metadata.json").read_text(encoding="utf-8"))
    rows = read_rows(PIXEL / "refined_weak_labels.jsonl")
    weak = {(r["query_id"], r["candidate_id"]): r["grade"] for r in rows}
    labels = np.array(
        [
            [
                np.nan if weak.get((qid, cid)) is None else weak[qid, cid]
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
    result = fit_supervised_bundle(
        SupervisedPipelineFitSupervisedBundleInputs(
            x,
            labels,
            valid,
            qfolds,
            gfolds,
            base,
            metadata,
            protocol["policy"],
            "machine_pixel_refined_development",
        )
    )
    if not result["trained"]:
        raise ValueError(result)
    rankings = json.loads((BASE / "rankings.json").read_text(encoding="utf-8"))
    for i, qid in enumerate(metadata["query_ids"]):
        positions = np.flatnonzero(valid[i])
        ids = [metadata["gallery_ids"][j] for j in positions]
        for name, scores in result["predictions"].items():
            rankings[qid]["refined_weak_oof_" + name] = ranked_ids(
                ids, scores[i, positions]
            )
    bounds = {
        name: machine_topk_bounds({qid: r[name] for qid, r in rankings.items()}, weak)
        for name in next(iter(rankings.values()))
    }
    write_json(OUT / "candidate_models.json", result["candidates"])
    write_json(OUT / "cross_fit_models.json", result["models"])
    write_json(OUT / "cross_fit_audit.json", result["audits"])
    write_json(OUT / "rankings.json", rankings)
    np.savez_compressed(OUT / "oof_scores.npz", **result["predictions"])
    write_json(
        OUT / "machine_agreement_bounds.json",
        dict(
            methods=bounds,
            source="machine_only_not_human_quality",
            formal_metrics=None,
            goal_met=False,
        ),
    )
    write_json(
        OUT / "status.json",
        dict(
            known_machine_labels=result["known_labels"],
            cross_fit_models=len(result["models"]),
            candidate_models=len(result["candidates"]),
            complete_rankings=sum(len(v) for v in rankings.values()),
            human_labels=0,
            formal_acceptance=False,
            formal_metrics=None,
            goal_met=False,
            default_changed=False,
        ),
    )
    write_json(
        OUT / "receipt.json",
        dict(
            files={
                p.relative_to(OUT).as_posix(): sha256_file(p)
                for p in OUT.rglob("*")
                if p.is_file()
            }
        ),
    )
    print((OUT / "status.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "fit"])
    args = parser.parse_args()
    {"freeze": freeze, "fit": fit}[args.action]()
