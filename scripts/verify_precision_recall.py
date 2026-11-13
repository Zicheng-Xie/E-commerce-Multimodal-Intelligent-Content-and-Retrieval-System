"""Replay candidate pools, all ranks, label changes and group-masked models."""

from __future__ import annotations

import ast
import hashlib
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
from multimodal_ecommerce.intent_retrieval import (  # noqa: E402
    balanced_pool,
    normalized_ranks,
    public_evidence,
    recall_channels,
    rerank_complete,
)
from multimodal_ecommerce.machine_teacher import strict_grades  # noqa: E402
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
BANK = PROJECT_ROOT / "artifacts/module2_machine_assist/semantic_ranker"


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def verify_receipt(folder: Path, name: str = "receipt.json") -> None:
    """Verify receipt.

    Args:
        folder: Directory containing the operation artifacts.
        name: Name used by this operation.
    """
    verify_files(folder, load(folder / name)["files"])


class StripAnnotation(ast.NodeTransformer):
    """Remove annotation syntax for source-equivalence checks of frozen algorithms."""

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.Assign | None:
        """Return visit AnnAssign for this workflow.

        Args:
            node: Node used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        if node.value is None:
            return None
        return ast.Assign(targets=[node.target], value=node.value)


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through write_json.
    """
    protocol = load(ROOT / "protocol.json")
    amendment = load(ROOT / "typing_amendment.json")
    for name, digest in protocol["implementation"].items():
        path = PROJECT_ROOT / name
        if name == amendment["current_source"]:
            executed = ROOT / amendment["executed_source"]
            assert sha256_file(executed) == digest == amendment["executed_sha256"]
            assert sha256_file(path) == amendment["current_sha256"]
            before = StripAnnotation().visit(
                ast.parse(executed.read_text(encoding="utf-8"))
            )
            after = StripAnnotation().visit(ast.parse(path.read_text(encoding="utf-8")))
            assert ast.dump(before, include_attributes=False) == ast.dump(
                after, include_attributes=False
            )
        else:
            assert sha256_file(path) == digest
    verify_files(PROJECT_ROOT, protocol["parents"])
    verify_files(PROJECT_ROOT, protocol["image_files"])
    for name in ["inputs", "pools", "public_evidence"]:
        assert sha256_file(ROOT / (name + ".json")) == protocol[name + "_sha256"]
    verify_receipt(ROOT, "inference_receipt.json")
    verify_receipt(ROOT, "analysis_receipt.json")
    inputs = load(ROOT / "inputs.json")
    queries = {q["id"]: q for q in inputs["queries"]}
    catalog = {r["id"]: r for r in inputs["catalog"]}
    metadata = load(BANK / "feature_metadata.json")
    pools = load(ROOT / "pools.json")
    evidence = load(ROOT / "public_evidence.json")
    pair_scores = {}
    prior = {}
    for row in read_rows(
        PROJECT_ROOT / "artifacts/module2_precision60/business/pair_scores.jsonl"
    ):
        prior.update(
            {
                (row["query_id"], key): score
                for key, score in zip(row["candidate_ids"], row["logits"])
            }
        )
    origins: dict = {}
    for path in (ROOT / "pair_scores").glob("*.json"):
        row = load(path)
        key = row["query_id"], row["candidate_id"]
        assert key not in pair_scores and np.isfinite(row["logit"])
        doc = catalog[key[1]]
        public = dict(
            query=queries[key[0]],
            document=dict(id=key[1], image_path=doc["image_path"], text=doc["title"]),
        )
        assert (
            hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
            == row["model_input_sha256"]
        )
        if row["origin"] == "frozen_prior_identical_input":
            assert row["logit"] == prior[key]
        else:
            assert row["origin"] == "new_cross_encoder_inference" and row["seconds"] > 0
        origins[row["origin"]] = origins.get(row["origin"], 0) + 1
        pair_scores[key] = row["logit"]
    assert set(pair_scores) == {
        (q, c) for q, candidates in pools.items() for c in candidates
    }
    assert origins == dict(
        frozen_prior_identical_input=357, new_cross_encoder_inference=723
    )
    rankings = load(
        PROJECT_ROOT / "artifacts/module2_continuation/constraints/rankings.json"
    )
    with np.load(BANK / "features.npz", allow_pickle=False) as bank:
        x, valid, qfolds, gfolds, base = (
            bank[k]
            for k in [
                "features",
                "valid",
                "query_folds",
                "gallery_folds",
                "base_scores",
            ]
        )
    for i, qid in enumerate(metadata["query_ids"]):
        positions = np.flatnonzero(valid[i])
        ids = [metadata["gallery_ids"][j] for j in positions]
        rrf, channels = recall_channels(ids, x[i, positions])
        why = [
            public_evidence(
                queries[qid]["text"], catalog[c]["title"], metadata["profile_texts"][j]
            )
            for c, j in zip(ids, positions)
        ]
        assert dict(zip(ids, why)) == evidence[qid]
        offsets = np.array([v["offset"] for v in why])
        adjusted = normalized_ranks(ids, rrf) + offsets
        channels.append(ranked_ids(ids, adjusted))
        assert balanced_pool(channels, 60) == pools[qid]
        rankings[qid]["multi_channel_rrf"] = ranked_ids(ids, rrf)
        rankings[qid]["public_evidence_soft"] = ranked_ids(ids, adjusted)
        for size in [20, 60]:
            pool = pools[qid][:size]
            logits = np.array([pair_scores[qid, c] for c in pool])
            rankings[qid][f"broad_cross_pool{size}"] = rerank_complete(
                ids, rrf, pool, logits
            )
            rankings[qid][f"broad_cross_evidence_pool{size}"] = rerank_complete(
                ids, rrf, pool, logits, offsets
            )
    assert rankings == load(ROOT / "rankings.json")
    adjudication = ROOT / "machine_adjudication"
    verify_receipt(adjudication)
    policy = load(adjudication / "protocol.json")
    assert (
        sha256_file(PROJECT_ROOT / "scripts/adjudicate_machine_conflicts.py")
        == policy["source_sha256"]
    )
    rows = read_rows(
        PROJECT_ROOT
        / "artifacts/module2_continuation/pixel_refinement/refined_weak_labels.jsonl"
    )
    assert (
        sha256_file(
            PROJECT_ROOT
            / "artifacts/module2_continuation/pixel_refinement/refined_weak_labels.jsonl"
        )
        == policy["prior_labels_sha256"]
    )
    updates = {}
    for path in (adjudication / "reviews").glob("*.json"):
        row = load(path)
        assert row["review_origin"] == "model" and row["human_label"] is False
        raw = json.loads(row["raw"]["message"]["content"])
        assert row["grade"] == strict_grades([raw["grade"]], 1)[0]
        assert row["reason"] == raw["reason"]
        verify_files(PROJECT_ROOT, row["image_files"])
        updates[row["query_id"], row["candidate_id"]] = row
    assert set(updates) == {
        (r["query_id"], r["candidate_id"]) for r in policy["selected_pairs"]
    }
    revised = read_rows(adjudication / "revised_weak_labels.jsonl")
    for before, after in zip(rows, revised):
        key = before["query_id"], before["candidate_id"]
        if key in updates:
            assert before["grade"] == after["previous_machine_grade"] == 1
            assert after["grade"] == updates[key]["grade"]
        else:
            assert before == after
    weak = {(r["query_id"], r["candidate_id"]): r["grade"] for r in revised}
    folder = ROOT / "adjudicated_ranker"
    verify_receipt(folder)
    training = load(folder / "protocol.json")
    implementations = dict(training["implementation"])
    formatting = load(ROOT / "import_format_amendment.json")
    expected = implementations.pop(formatting["current_source"])
    old_source = ROOT / formatting["executed_source"]
    new_source = PROJECT_ROOT / formatting["current_source"]
    assert sha256_file(old_source) == expected == formatting["executed_sha256"]
    assert sha256_file(new_source) == formatting["current_sha256"]
    assert ast.dump(
        ast.parse(old_source.read_text(encoding="utf-8")), include_attributes=False
    ) == ast.dump(
        ast.parse(new_source.read_text(encoding="utf-8")), include_attributes=False
    )
    verify_files(PROJECT_ROOT, implementations)
    assert training["label_sha256"] == sha256_file(
        adjudication / "revised_weak_labels.jsonl"
    )
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
    fitted = fit_supervised_bundle(
        SupervisedPipelineFitSupervisedBundleInputs(
            x,
            labels,
            valid,
            qfolds,
            gfolds,
            base,
            metadata,
            training["policy"],
            "machine_pixel_refined_development",
        )
    )
    assert fitted["trained"]
    for name, model_key in [
        ("cross_fit_models", "models"),
        ("candidate_models", "candidates"),
    ]:
        actual = load(folder / (name + ".json"))
        assert fitted[model_key] == actual
    assert fitted["audits"] == load(folder / "cross_fit_audit.json")
    for i, qid in enumerate(metadata["query_ids"]):
        positions = np.flatnonzero(valid[i])
        ids = [metadata["gallery_ids"][j] for j in positions]
        for name, scores in fitted["predictions"].items():
            rankings[qid]["adjudicated_weak_oof_" + name] = ranked_ids(
                ids, scores[i, positions]
            )
        assert all(
            len(order) == len(set(order)) == len(ids) and set(order) == set(ids)
            for order in rankings[qid].values()
        )
    assert rankings == load(folder / "rankings.json")
    bounds = load(folder / "machine_agreement_bounds.json")
    for name, values in bounds["methods"].items():
        assert values == machine_topk_bounds(
            {q: r[name] for q, r in rankings.items()}, weak
        )
    original_roots = [
        "module2_blind",
        "module2_precision",
        "module2_training",
        "module2_precision60",
        "module2_business_review",
        "module2_business_acceptance",
        "module2_machine_assist",
        "module2_continuation",
        "module2_target_p60_r85",
    ]
    for name in original_roots:
        verify_receipt(
            PROJECT_ROOT / "artifacts" / name,
            (
                "preparation_receipt.json"
                if name in ["module2_business_review", "module2_business_acceptance"]
                else "completion_receipt.json"
            ),
        )
    result = dict(
        pair_scores_verified=1080,
        new_inferences_verified=723,
        cached_pairs_verified=357,
        original_rankings_replayed=306,
        latest_rankings_replayed=342,
        machine_adjudications_replayed=59,
        cross_fit_models_retrained=18,
        candidate_models_retrained=2,
        original_sealed_roots_unchanged=9,
        human_labels=0,
        acceptance_model_predictions=0,
        formal_metrics=None,
        goal_met=False,
        default_changed=False,
    )
    if (ROOT / "verification.json").exists():
        assert load(ROOT / "verification.json") == result
    else:
        write_json(ROOT / "verification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
