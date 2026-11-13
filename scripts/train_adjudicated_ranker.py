"""Refit the unchanged group-masked recipe after explicit machine-label re-review."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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

ROOT = PROJECT_ROOT / "artifacts/module2_precision_recall"
OUT = ROOT / "adjudicated_ranker"
BANK = PROJECT_ROOT / "artifacts/module2_machine_assist/semantic_ranker"


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Refit the unchanged group-masked recipe after explicit machine-label
    re-review.
    """
    if (OUT / "receipt.json").exists():
        raise FileExistsError("Preserve completed adjudicated models")
    for folder, name in [
        (ROOT, "analysis_receipt.json"),
        (ROOT / "machine_adjudication", "receipt.json"),
        (BANK, "feature_receipt.json"),
    ]:
        verify_files(folder, load(folder / name)["files"])
    OUT.mkdir(parents=True, exist_ok=True)
    policy = load(BANK / "protocol.json")["policy"]
    write_json(
        OUT / "protocol.json",
        dict(
            policy=policy,
            label_sha256=sha256_file(
                ROOT / "machine_adjudication/revised_weak_labels.jsonl"
            ),
            feature_receipt_sha256=sha256_file(BANK / "feature_receipt.json"),
            implementation={
                n: sha256_file(PROJECT_ROOT / n)
                for n in [
                    "scripts/train_adjudicated_ranker.py",
                    "src/multimodal_ecommerce/supervised_pipeline.py",
                    "src/multimodal_ecommerce/semantic_ranker.py",
                ]
            },
            changed_factor="39 disputed machine positives revised to 3 negatives and 36 unknowns after direct-pixel machine review; no recipe tuning",
            formal_metrics=None,
        ),
    )
    metadata = load(BANK / "feature_metadata.json")
    weak = {
        (r["query_id"], r["candidate_id"]): r["grade"]
        for r in read_rows(ROOT / "machine_adjudication/revised_weak_labels.jsonl")
    }
    labels = np.array(
        [
            [
                np.nan if weak.get((q, c)) is None else weak[q, c]
                for c in metadata["gallery_ids"]
            ]
            for q in metadata["query_ids"]
        ],
        dtype=float,
    )
    with np.load(BANK / "features.npz", allow_pickle=False) as bank:
        valid = bank["valid"]
        fitted = fit_supervised_bundle(
            SupervisedPipelineFitSupervisedBundleInputs(
                bank["features"],
                labels,
                valid,
                bank["query_folds"],
                bank["gallery_folds"],
                bank["base_scores"],
                metadata,
                policy,
                "machine_pixel_refined_development",
            )
        )
    if not fitted["trained"]:
        raise ValueError(fitted)
    rankings = load(ROOT / "rankings.json")
    for i, qid in enumerate(metadata["query_ids"]):
        positions = np.flatnonzero(valid[i])
        ids = [metadata["gallery_ids"][j] for j in positions]
        for name, scores in fitted["predictions"].items():
            rankings[qid]["adjudicated_weak_oof_" + name] = ranked_ids(
                ids, scores[i, positions]
            )
    write_json(OUT / "candidate_models.json", fitted["candidates"])
    write_json(OUT / "cross_fit_models.json", fitted["models"])
    write_json(OUT / "cross_fit_audit.json", fitted["audits"])
    np.savez_compressed(OUT / "oof_scores.npz", **fitted["predictions"])
    write_json(OUT / "rankings.json", rankings)
    write_json(
        OUT / "machine_agreement_bounds.json",
        dict(
            methods={
                name: machine_topk_bounds(
                    {q: r[name] for q, r in rankings.items()}, weak
                )
                for name in next(iter(rankings.values()))
            },
            source="Recomputed for all methods on the same revised machine labels; not comparable as quality gain against earlier label versions",
            formal_metrics=None,
            goal_met=False,
        ),
    )
    write_json(
        OUT / "status.json",
        dict(
            machine_positive=120,
            machine_negative=1729,
            machine_unknown=3294,
            cross_fit_models=18,
            candidate_models=2,
            complete_rankings=sum(len(r) for r in rankings.values()),
            human_labels=0,
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
    print(json.dumps(load(OUT / "status.json")))


if __name__ == "__main__":
    main()
