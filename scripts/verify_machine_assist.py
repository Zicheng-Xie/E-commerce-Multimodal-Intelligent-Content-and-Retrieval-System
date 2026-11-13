"""Replay machine-origin labels, fold exclusions, models and complete rankings."""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import run  # noqa: F401,E402
from multimodal_ecommerce.blind_evaluation import verify_files  # noqa: E402
from multimodal_ecommerce.dataset_release import read_rows  # noqa: E402
from multimodal_ecommerce.machine_teacher import strict_grades  # noqa: E402
from multimodal_ecommerce.safety import (  # noqa: E402
    PROJECT_ROOT,
    sha256_file,
    write_json,
)
from multimodal_ecommerce.semantic_ranker import (  # noqa: E402 - local path bootstrap must run first.; noqa: E402
    SemanticRankerFitRankerInputs,
    SemanticRankerTrainingMaskInputs,
    fit_ranker,
    machine_topk_bounds,
    predict_ranker,
    ranked_ids,
    training_mask,
)

ROOT = PROJECT_ROOT / "artifacts/module2_machine_assist"
OUT = ROOT / "semantic_ranker"
SOURCE = PROJECT_ROOT / "artifacts/module2_precision60/business"


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through write_json.
    """
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    verify_files(PROJECT_ROOT, protocol["implementation"])
    verify_files(PROJECT_ROOT, protocol["input_files"])
    assert (
        sha256_file(ROOT / "development_inputs.json")
        == protocol["development_inputs_sha256"]
    )
    receipt = json.loads((ROOT / "annotation_receipt.json").read_text(encoding="utf-8"))
    verify_files(ROOT, receipt["files"])
    assert sha256_file(ROOT / "weak_labels.jsonl") == receipt["weak_labels_sha256"]
    allowed = json.loads(
        (SOURCE / "allowed_candidates.json").read_text(encoding="utf-8")
    )
    expected = {(qid, cid) for qid, ids in allowed.items() for cid in ids}
    reconstructed = {}
    profile_seconds = 0
    for path in (ROOT / "profiles").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        assert row["origin"] == "machine" and row["human_label"] is False
        assert row["teacher_digest"] == protocol["teacher"]["digest"]
        assert (
            strict_grades(row["raw"]["result"]["grades"], len(row["text_query_ids"]))
            == row["grades"]
        )
        profile_seconds += row["raw"]["response"]["wall_seconds"]
        for qid, grade in zip(row["text_query_ids"], row["grades"]):
            if row["id"] in allowed[qid]:
                reconstructed[qid, row["id"]] = grade
    image_seconds = 0
    for path in (ROOT / "image_batches").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        assert row["origin"] == "machine_from_descriptions_not_direct_image_pair"
        assert (
            strict_grades(row["raw"]["result"]["grades"], len(row["candidate_ids"]))
            == row["grades"]
        )
        image_seconds += row["raw"]["response"]["wall_seconds"]
        for cid, grade in zip(row["candidate_ids"], row["grades"]):
            assert (row["query_id"], cid) not in reconstructed
            reconstructed[row["query_id"], cid] = grade
    assert set(reconstructed) == expected
    weak = read_rows(ROOT / "weak_labels.jsonl")
    assert len(weak) == len(expected)
    for row in weak:
        assert (
            row["review_origin"] == "model"
            and row["human_label"] is False
            and row["role"] == "machine_development_only"
        )
        assert row["grade"] is None or (
            type(row["grade"]) is int and row["grade"] in (0, 1)
        )
        assert row["grade"] == reconstructed[row["query_id"], row["candidate_id"]]
    policy = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    verify_files(PROJECT_ROOT, policy["implementation"])
    verify_files(PROJECT_ROOT, policy["parents"])
    for name in ["feature_receipt.json", "training_receipt.json"]:
        verify_files(OUT, json.loads((OUT / name).read_text(encoding="utf-8"))["files"])
    metadata = json.loads((OUT / "feature_metadata.json").read_text(encoding="utf-8"))
    models = json.loads((OUT / "cross_fit_models.json").read_text(encoding="utf-8"))
    audit = json.loads((OUT / "cross_fit_audit.json").read_text(encoding="utf-8"))
    with np.load(OUT / "features.npz", allow_pickle=False) as bank:
        x = bank["features"]
        y = bank["labels"]
        valid = bank["valid"]
        qfolds = bank["query_folds"]
        gfolds = bank["gallery_folds"]
        base = bank["base_scores"]
    with np.load(OUT / "oof_scores.npz", allow_pickle=False) as bank:
        oof = {name: bank[name].copy() for name in bank.files}
    count = 0
    for item in audit:
        mask, evidence = training_mask(
            SemanticRankerTrainingMaskInputs(
                metadata["query_ids"],
                qfolds,
                metadata["query_families"],
                metadata["gallery_families"],
                gfolds,
                item["held_query_fold"],
                item["held_gallery_fold"],
            )
        )
        for key, value in evidence.items():
            assert item[key] == value
        mask &= valid
        original = models[item["model"]]
        replay = fit_ranker(
            SemanticRankerFitRankerInputs(
                x, y, mask, base, policy["policy"], pairwise=original["pairwise"]
            )
        )
        for key in ["mean", "scale", "weights"]:
            assert np.allclose(replay[key], original[key], rtol=0, atol=1e-12)
        validation = (
            (qfolds[:, None] == item["held_query_fold"])
            & (gfolds[None, :] == item["held_gallery_fold"])
            & valid
        )
        name = "hard_negative_ranknet" if original["pairwise"] else "balanced_logistic"
        assert np.allclose(
            oof[name][validation],
            predict_ranker(x, original)[validation],
            rtol=0,
            atol=1e-12,
        )
        count += 1
    baseline = json.loads((SOURCE / "rankings.json").read_text(encoding="utf-8"))
    rankings = json.loads((OUT / "rankings.json").read_text(encoding="utf-8"))
    assert set(rankings) == set(allowed)
    ranking_count = 0
    for i, qid in enumerate(metadata["query_ids"]):
        positions = np.flatnonzero(valid[i])
        ids = [metadata["gallery_ids"][j] for j in positions]
        assert set(ids) == set(allowed[qid])
        for method, ranked in rankings[qid].items():
            assert (
                len(ranked) == len(ids)
                and len(set(ranked)) == len(ids)
                and set(ranked) == set(ids)
            )
            if method in baseline[qid]:
                assert ranked == baseline[qid][method]
            elif method == "semantic_profile_blend":
                assert ranked == ranked_ids(
                    ids,
                    (0.5 * x[i, :, 1] + 0.2 * x[i, :, 3] + 0.3 * x[i, :, 4])[positions],
                )
            else:
                assert ranked == ranked_ids(
                    ids, oof[method.removeprefix("weak_oof_")][i, positions]
                )
            ranking_count += 1
    bounds = json.loads(
        (OUT / "machine_agreement_bounds.json").read_text(encoding="utf-8")
    )
    for method, values in bounds["methods"].items():
        assert values == machine_topk_bounds(
            {qid: rows[method] for qid, rows in rankings.items()}, reconstructed
        )
    assert (
        bounds["formal_metrics"] is None
        and bounds["goal_met"] is False
        and bounds["default_changed"] is False
    )
    reserved = read_rows(
        PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl"
    )
    assert (
        sha256_file(PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl")
        == protocol["reserved_registry_sha256"]
    )
    assert not set(metadata["gallery_ids"]) & {r["id"] for r in reserved}
    assert not set(protocol["input_files"].values()) & {
        r["image_sha256"] for r in reserved
    }
    verified_roots = 0
    for name, receipt_name in [
        ("module2_blind", "completion_receipt.json"),
        ("module2_precision", "completion_receipt.json"),
        ("module2_training", "completion_receipt.json"),
        ("module2_precision60", "completion_receipt.json"),
        ("module2_business_review", "preparation_receipt.json"),
        ("module2_business_acceptance", "preparation_receipt.json"),
    ]:
        root = PROJECT_ROOT / "artifacts" / name
        verify_files(
            root, json.loads((root / receipt_name).read_text(encoding="utf-8"))["files"]
        )
        verified_roots += 1
    result = dict(
        machine_relation_rows=len(weak),
        machine_known=sum(r["grade"] is not None for r in weak),
        machine_unknown=sum(r["grade"] is None for r in weak),
        models_retrained_and_replayed=count,
        complete_rankings_replayed=ranking_count,
        original_sealed_roots_verified=verified_roots,
        reserved_acceptance_used=False,
        profile_inference_seconds=profile_seconds,
        image_relation_inference_seconds=image_seconds,
        human_labels=0,
        formal_acceptance=False,
        formal_metrics=None,
        goal_met=False,
        default_changed=False,
    )
    if (ROOT / "verification.json").exists():
        assert (
            json.loads((ROOT / "verification.json").read_text(encoding="utf-8"))
            == result
        )
    else:
        write_json(ROOT / "verification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
