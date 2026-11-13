"""Replay this development stage without new inference or acceptance access."""

from __future__ import annotations

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
from multimodal_ecommerce.business_constraints import constrain_scores  # noqa: E402
from multimodal_ecommerce.dataset_release import read_rows  # noqa: E402
from multimodal_ecommerce.human_business_pipeline import (  # noqa: E402
    REVIEW,
    validate_consensus,
)
from multimodal_ecommerce.machine_teacher import strict_grades  # noqa: E402
from multimodal_ecommerce.pixel_refinement import (  # noqa: E402
    PIXEL_PROMPT,
    merge_machine_refinement,
    select_uncertain,
)
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

ROOT = PROJECT_ROOT / "artifacts/module2_continuation"
OLD = PROJECT_ROOT / "artifacts/module2_machine_assist"
BASE = OLD / "semantic_ranker"


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def receipt(root: Path, name: str = "receipt.json") -> None:
    """Record file hashes and provenance for the completed stage.

    Args:
        root: Root directory used to resolve the operation artifacts.
        name: Name used by this operation.
    """
    verify_files(root, load(root / name)["files"])


def compare_models(actual: dict, expected: dict) -> None:
    """Apply the compare models workflow step.

    Args:
        actual: Actual used by this operation.
        expected: Expected used by this operation.
    """
    assert set(actual) == set(expected)
    for name, model in actual.items():
        other = expected[name]
        assert set(model) == set(other)
        for key, value in model.items():
            if key in {"mean", "scale", "weights", "bias", "loss"}:
                assert np.allclose(value, other[key], rtol=0, atol=1e-12)
            else:
                assert value == other[key]


def verify_pixels() -> list[dict]:
    """Verify pixels.

    Returns:
        Values produced by this operation, in the stated return order.

    Notes:
        Writes derived artifacts through receipt.
    """
    pixel = ROOT / "pixel_refinement"
    receipt(pixel)
    policy = load(pixel / "protocol.json")
    verify_files(PROJECT_ROOT, policy["implementation"])
    verify_files(PROJECT_ROOT, policy["input_files"])
    assert sha256_file(pixel / "tasks.json") == policy["tasks_sha256"]
    assert sha256_file(OLD / "weak_labels.jsonl") == policy["original_labels_sha256"]
    original = read_rows(OLD / "weak_labels.jsonl")
    tasks = load(pixel / "tasks.json")
    selected = select_uncertain(original, load(BASE / "rankings.json"), 12)
    assert [(r["query_id"], r["candidate_id"]) for r in tasks] == [
        (r["query_id"], r["candidate_id"]) for r in selected
    ]
    records = []
    for task in tasks:
        key = hashlib.sha256(
            (task["query_id"] + "|" + task["candidate_id"]).encode()
        ).hexdigest()[:24]
        row = load(pixel / "reviews" / (key + ".json"))
        public = {k: task[k] for k in ["query_text", "query_rule", "title"]}
        public["image_order"] = (
            "query then candidate" if len(task["images"]) == 2 else "candidate only"
        )
        prompt = PIXEL_PROMPT + json.dumps(public, ensure_ascii=False)
        raw = json.loads(row["raw"]["message"]["content"])
        assert row["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
        assert row["teacher_digest"] == policy["teacher"]["digest"]
        assert row["grade"] == strict_grades([raw["grade"]], 1)[0]
        assert row["reason"] == raw["reason"] and row["reason"].strip()
        assert row["raw"]["done"] and row["raw"]["done_reason"] == "stop"
        assert row["review_origin"] == "model" and row["human_label"] is False
        records.append(row)
    merged = merge_machine_refinement(original, records)
    assert merged == read_rows(pixel / "refined_weak_labels.jsonl")
    assert len(merged) == 5143 and len(records) == 216
    assert sum(r["grade"] is not None for r in records) == 57
    assert sum(r["grade"] is not None for r in merged) == 1885
    assert sum(r["grade"] is None for r in merged) == 3258
    assert all(a == b for a, b in zip(original, merged) if a["grade"] is not None)
    return merged


def verify_rankers(rows: list[dict]) -> dict:
    """Verify rankers.

    Args:
        rows: Records processed by this operation.

    Returns:
        Mapping with fields: models, candidates, refined_rankings, latest_rankings,
        image_queries_unchanged.

    Notes:
        Writes derived artifacts through receipt.
    """
    refined = ROOT / "refined_ranker"
    receipt(refined)
    receipt(BASE, "feature_receipt.json")
    protocol = load(refined / "protocol.json")
    verify_files(PROJECT_ROOT, protocol["implementation"])
    assert (
        sha256_file(BASE / "feature_receipt.json") == protocol["feature_receipt_sha256"]
    )
    assert (
        sha256_file(ROOT / "pixel_refinement/protocol.json")
        == protocol["pixel_protocol_sha256"]
    )
    metadata = load(BASE / "feature_metadata.json")
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
        x, valid = bank["features"], bank["valid"]
        replay = fit_supervised_bundle(
            SupervisedPipelineFitSupervisedBundleInputs(
                x,
                labels,
                valid,
                bank["query_folds"],
                bank["gallery_folds"],
                bank["base_scores"],
                metadata,
                protocol["policy"],
                "machine_pixel_refined_development",
            )
        )
    assert replay["trained"] and replay["known_labels"] == 1885
    compare_models(replay["models"], load(refined / "cross_fit_models.json"))
    compare_models(replay["candidates"], load(refined / "candidate_models.json"))
    assert replay["audits"] == load(refined / "cross_fit_audit.json")
    with np.load(refined / "oof_scores.npz", allow_pickle=False) as bank:
        for name, values in replay["predictions"].items():
            assert np.allclose(values, bank[name], rtol=0, atol=1e-12, equal_nan=True)
    rankings = load(BASE / "rankings.json")
    for i, qid in enumerate(metadata["query_ids"]):
        positions = np.flatnonzero(valid[i])
        ids = [metadata["gallery_ids"][j] for j in positions]
        for name, scores in replay["predictions"].items():
            rankings[qid]["refined_weak_oof_" + name] = ranked_ids(
                ids, scores[i, positions]
            )
    assert rankings == load(refined / "rankings.json")
    assert sum(len(v) for v in rankings.values()) == 180
    constraints = ROOT / "constraints"
    receipt(constraints)
    verify_files(PROJECT_ROOT, load(constraints / "protocol.json")["implementation"])
    context = load(OLD / "development_inputs.json")
    titles = {r["id"]: r["title"] for r in context["catalog"]}
    queries = {r["id"]: r for r in context["queries"]}
    evidence = {}
    image_unchanged = 0
    for i, qid in enumerate(metadata["query_ids"]):
        scores, why = constrain_scores(
            queries[qid]["text"],
            [titles[k] for k in metadata["gallery_ids"]],
            metadata["profile_texts"],
            0.5 * x[i, :, 1] + 0.2 * x[i, :, 3] + 0.3 * x[i, :, 4],
        )
        positions = np.flatnonzero(valid[i])
        ids = [metadata["gallery_ids"][j] for j in positions]
        rankings[qid]["constraint_aware_fusion"] = ranked_ids(ids, scores[positions])
        evidence[qid] = {metadata["gallery_ids"][j]: why[j] for j in positions}
        if queries[qid]["image_path"]:
            assert (
                rankings[qid]["constraint_aware_fusion"]
                == rankings[qid]["semantic_profile_blend"]
            )
            image_unchanged += 1
    assert image_unchanged == 6
    assert rankings == load(constraints / "rankings.json")
    assert evidence == load(constraints / "evidence.json")
    assert sum(len(v) for v in rankings.values()) == 198
    for folder in [refined, constraints]:
        current = load(folder / "rankings.json")
        bounds = load(folder / "machine_agreement_bounds.json")
        assert bounds["formal_metrics"] is None and bounds["goal_met"] is False
        for name, actual in bounds["methods"].items():
            assert actual == machine_topk_bounds(
                {q: r[name] for q, r in current.items()}, weak
            )
    return dict(
        models=18,
        candidates=2,
        refined_rankings=180,
        latest_rankings=198,
        image_queries_unchanged=image_unchanged,
    )


def verify_human() -> None:
    """Verify human.

    Notes:
        Writes derived artifacts through receipt.
    """
    status = load(ROOT / "human_pipeline/current_status.json")
    target = PROJECT_ROOT / status["revision"]
    receipt(target)
    record = load(target / "receipt.json")
    verify_files(PROJECT_ROOT, record["implementation"])
    merge = PROJECT_ROOT / status["source_merge"]
    assert record["source_label_sha256"] == sha256_file(merge / "judgments.jsonl")
    assert record["source_provenance_sha256"] == sha256_file(
        merge / "input_provenance.json"
    )
    result = validate_consensus(
        load(REVIEW / "manifest.json"),
        read_rows(REVIEW / "judgments.blank.jsonl"),
        load(merge / "input_provenance.json"),
        read_rows(target / "human_consensus.jsonl"),
    )
    assert result["status"]["completed"] == status["human_completed"] == 0
    assert status["human_pending"] == 5143 and status["training_executed"] is False
    assert status["quality_metrics"] is None and status["formal_metrics"] is None
    assert not (target / "candidate_models.json").exists()


def verify_benchmark() -> None:
    """Verify benchmark.

    Notes:
        Writes derived artifacts through receipt.
    """
    root = ROOT / "service"
    receipt(root, "benchmark_receipt.json")
    protocol = load(root / "benchmark_protocol.json")
    verify_files(PROJECT_ROOT, protocol["implementation"])
    summary = load(root / "benchmark.json")
    records = load(root / "benchmark_records.json")
    assert len(protocol["cases"]) == 24 and protocol["repeats"] == 2
    assert len(records) == 48 and len({(r["repeat"], r["case"]) for r in records}) == 48
    for row in records:
        assert len(row["top_ids"]) == len(set(row["top_ids"])) == 10
        assert row["method"] == protocol["cases"][row["case"]]["method"]
        assert row["http_seconds"] > 0 and row["server_seconds"] > 0
    for mode in [None, "text", "image", "image_roi"]:
        values = [
            r["http_seconds"] for r in records if mode is None or r["modality"] == mode
        ]
        actual = summary["http"] if mode is None else summary["by_modality"][mode]
        assert actual == dict(
            count=len(values),
            p50_seconds=float(np.percentile(values, 50)),
            p95_seconds=float(np.percentile(values, 95)),
            maximum_seconds=max(values),
            fraction_below_five_seconds=sum(v < 5 for v in values) / len(values),
        )
    assert summary["guard_checks_passed"] == 5
    assert summary["guards_triggered_no_retrieval"] and summary["image_asset_served"]
    assert summary["embedding_recomputed_every_request"]
    assert summary["formal_performance_acceptance"] is False
    assert (
        summary["formal_quality_metrics"] is None
        and summary["acceptance_model_predictions"] == 0
    )


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through receipt, write_json.
    """
    rows = verify_pixels()
    counts = verify_rankers(rows)
    verify_human()
    verify_benchmark()
    sealed = []
    for name, kind in [
        ("module2_blind", "completion_receipt.json"),
        ("module2_precision", "completion_receipt.json"),
        ("module2_training", "completion_receipt.json"),
        ("module2_precision60", "completion_receipt.json"),
        ("module2_business_review", "preparation_receipt.json"),
        ("module2_business_acceptance", "preparation_receipt.json"),
        ("module2_machine_assist", "completion_receipt.json"),
    ]:
        receipt(PROJECT_ROOT / "artifacts" / name, kind)
        sealed.append(name)
    assert (
        sha256_file(PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl")
        == load(OLD / "protocol.json")["reserved_registry_sha256"]
    )
    result = dict(
        stage="module2_continuation",
        pixel_reviews_replayed=216,
        newly_resolved_machine_relations=57,
        known_machine_relations=1885,
        unknown_machine_relations=3258,
        cross_fit_replay=counts,
        benchmark_records_replayed=48,
        original_sealed_roots_verified=sealed,
        human_labels=0,
        human_training_executed=False,
        acceptance_model_predictions=0,
        reserved_acceptance_registry_unchanged=True,
        formal_acceptance=False,
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
