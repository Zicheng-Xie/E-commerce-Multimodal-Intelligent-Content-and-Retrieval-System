"""Freeze and compare the explicit intent-constraint candidate on old development."""

from __future__ import annotations

import json

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_constraints import ENTITY_RULES, constrain_scores
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.semantic_ranker import machine_topk_bounds, ranked_ids

BASE = PROJECT_ROOT / "artifacts/module2_machine_assist"
REFINED = PROJECT_ROOT / "artifacts/module2_continuation/refined_ranker"
OUT = PROJECT_ROOT / "artifacts/module2_continuation/constraints"


def main() -> None:
    """Freeze and compare the explicit intent-constraint candidate on old
    development.
    """
    if (OUT / "receipt.json").exists():
        raise FileExistsError("Preserve the constraint candidate evaluation")
    OUT.mkdir(parents=True, exist_ok=True)
    verify_files(
        REFINED,
        json.loads((REFINED / "receipt.json").read_text(encoding="utf-8"))["files"],
    )
    verify_files(
        BASE / "semantic_ranker",
        json.loads(
            (BASE / "semantic_ranker/feature_receipt.json").read_text(encoding="utf-8")
        )["files"],
    )
    code = [
        "scripts/evaluate_business_constraints.py",
        "src/multimodal_ecommerce/business_constraints.py",
    ]
    write_json(
        OUT / "protocol.json",
        dict(
            implementation={name: sha256_file(PROJECT_ROOT / name) for name in code},
            entity_rules=ENTITY_RULES,
            source="Query text, machine-generated product main type, original product title; no source taxonomy or relevance answers used as rank features",
            query_rules_or_labels_as_features=False,
            all_candidates_retained=True,
            default_changed=False,
            formal_acceptance=False,
        ),
    )
    context = json.loads((BASE / "development_inputs.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (BASE / "semantic_ranker/feature_metadata.json").read_text(encoding="utf-8")
    )
    titles = {r["id"]: r["title"] for r in context["catalog"]}
    rankings = json.loads((REFINED / "rankings.json").read_text(encoding="utf-8"))
    with np.load(BASE / "semantic_ranker/features.npz", allow_pickle=False) as bank:
        x = bank["features"]
        valid = bank["valid"]
    queries = {q["id"]: q["text"] for q in context["queries"]}
    evidence = {}
    for i, qid in enumerate(metadata["query_ids"]):
        base = 0.5 * x[i, :, 1] + 0.2 * x[i, :, 3] + 0.3 * x[i, :, 4]
        scores, why = constrain_scores(
            queries[qid],
            [titles[k] for k in metadata["gallery_ids"]],
            metadata["profile_texts"],
            base,
        )
        positions = np.flatnonzero(valid[i])
        ids = [metadata["gallery_ids"][j] for j in positions]
        rankings[qid]["constraint_aware_fusion"] = ranked_ids(ids, scores[positions])
        evidence[qid] = {metadata["gallery_ids"][j]: why[j] for j in positions}
    weak = {
        (r["query_id"], r["candidate_id"]): r["grade"]
        for r in read_rows(
            PROJECT_ROOT
            / "artifacts/module2_continuation/pixel_refinement/refined_weak_labels.jsonl"
        )
    }
    bounds = {
        name: machine_topk_bounds(
            {qid: current[name] for qid, current in rankings.items()}, weak
        )
        for name in next(iter(rankings.values()))
    }
    write_json(OUT / "rankings.json", rankings)
    write_json(OUT / "evidence.json", evidence)
    write_json(
        OUT / "machine_agreement_bounds.json",
        dict(
            methods=bounds,
            formal_metrics=None,
            goal_met=False,
            source="machine_agreement_not_business_precision",
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
    print(
        json.dumps(
            dict(
                queries=len(rankings),
                complete_rankings=sum(len(v) for v in rankings.values()),
                formal_acceptance=False,
                goal_met=False,
            )
        )
    )


if __name__ == "__main__":
    main()
