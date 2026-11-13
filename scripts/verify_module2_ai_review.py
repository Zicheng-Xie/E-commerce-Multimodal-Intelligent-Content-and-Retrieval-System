"""Verify complete AI provenance, metric replay and all historical sealed evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import run  # noqa: F401
from multimodal_ecommerce.ai_review_metrics import metric_ranges
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

ROOT = PROJECT_ROOT / "artifacts/module2_ai_review"


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through write_json.
    """
    protocol = load(ROOT / "protocol.json")
    implementations = dict(protocol["implementation"])
    amendment = load(ROOT / "automatic_runner_amendment.json")
    executed = ROOT / amendment["executed_source"]
    current = PROJECT_ROOT / amendment["current_source"]
    assert (
        sha256_file(executed)
        == implementations.pop(amendment["current_source"])
        == amendment["executed_sha256"]
    )
    assert sha256_file(current) == amendment["current_sha256"]
    needle = 'if __name__ == "__main__":\n'
    assert executed.read_text(encoding="utf-8").replace(
        needle, needle + amendment["inserted_guard"]
    ) == current.read_text(encoding="utf-8")
    verify_files(PROJECT_ROOT, implementations)
    verify_files(PROJECT_ROOT, protocol["guards"])
    verify_files(PROJECT_ROOT, protocol["image_files"])
    native = ROOT / "native_audit"
    verify_files(native, load(native / "protocol.json")["image_files"])
    verify_files(native / "original", load(native / "original/manifest.json")["files"])
    training = load(ROOT / "ranker_protocol.json")
    verify_files(PROJECT_ROOT, training["implementation"])
    verify_files(ROOT, training["native_observation_files"])
    assert training["label_sha256"] == sha256_file(ROOT / "judgments.ai.jsonl")
    inputs = load(ROOT / "inputs.json")
    labels = read_rows(ROOT / "judgments.ai.jsonl")
    assert (
        len(labels) == len({(r["query_id"], r["candidate_id"]) for r in labels}) == 6295
    )
    expected = {
        (q["id"], p["id"])
        for q in inputs["queries"]
        for p in inputs["catalog"]
        if q["exclude_id"] != p["id"]
    }
    assert {(r["query_id"], r["candidate_id"]) for r in labels} == expected
    assert all(
        r["review_origin"] == "model"
        and r["human_label"] is False
        and r["reason"].strip()
        for r in labels
    )
    assert sum(r["grade"] is None for r in labels) == 19
    qa = read_rows(ROOT / "catalog_QA.ai.jsonl")
    assert len(qa) == 64 and sum(r["verdict"] == "usable" for r in qa) == 58
    assert sum(r["verdict"] == "reject" for r in qa) == 3
    assert sum(r["verdict"] == "uncertain" for r in qa) == 3
    assert len(read_rows(ROOT / "catalog.reviewed.jsonl")) == 344
    assert len(read_rows(ROOT / "priority59.ai.jsonl")) == 59
    old_ids = {
        p["id"] for p in inputs["catalog"] if p["catalog_source"] == "original286"
    }
    original = [r for r in labels if r["candidate_id"] in old_ids]
    assert len(original) == 5143
    before = load(
        PROJECT_ROOT
        / "artifacts/module2_precision_recall/adjudicated_ranker/rankings.json"
    )
    after = load(ROOT / "ranker/rankings.json")
    assert all(
        all(after[q][m] == order for m, order in current.items())
        for q, current in before.items()
    )
    modes = {q["id"]: "text" if q["text"] else "image" for q in inputs["queries"]}
    assert metric_ranges(after, original, modes) == load(
        ROOT / "metrics.ai.development.json"
    )
    assert len(load(ROOT / "ranker/cross_fit_models.json")) == 18
    assert len(load(ROOT / "ranker/candidate_models.json")) == 2
    for audit in load(ROOT / "ranker/cross_fit_audit.json"):
        assert not set(audit["training_query_ids"]) & set(audit["excluded_query_ids"])
        assert not set(audit["training_gallery_families"]) & set(
            audit["excluded_gallery_families"]
        )
    history = [
        "module2_blind",
        "module2_precision",
        "module2_training",
        "module2_precision60",
        "module2_business_review",
        "module2_business_acceptance",
        "module2_machine_assist",
        "module2_continuation",
        "module2_target_p60_r85",
        "module2_precision_recall",
        "module2_catalog_review",
    ]
    for name in history:
        folder = PROJECT_ROOT / "artifacts" / name
        receipt = (
            "preparation_receipt.json"
            if name
            in {
                "module2_business_review",
                "module2_business_acceptance",
                "module2_catalog_review",
            }
            else "completion_receipt.json"
        )
        verify_files(folder, load(folder / receipt)["files"])
    human = read_rows(
        PROJECT_ROOT
        / "artifacts/module2_business_review/development/judgments.blank.jsonl"
    )
    assert all(r["grade"] is None and r["review_origin"] is None for r in human)
    result = dict(
        native_visual_products=350,
        full_AI_relations=6295,
        unknown_retained=19,
        original_AI_relations=5143,
        native_QA_usable=58,
        priority_pairs=59,
        methods_replayed=21,
        full_rankings_replayed=378,
        original_methods_unchanged=19,
        old_sealed_roots_unchanged=11,
        human_stream_unchanged=True,
        reserved_registry_unchanged=True,
        acceptance_predictions=0,
        formal_metrics=None,
        goal_met=False,
    )
    if (ROOT / "completion_receipt.json").exists():
        verify_files(ROOT, load(ROOT / "completion_receipt.json")["files"])
    if (ROOT / "verification.json").exists():
        assert load(ROOT / "verification.json") == result
    else:
        write_json(ROOT / "verification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
