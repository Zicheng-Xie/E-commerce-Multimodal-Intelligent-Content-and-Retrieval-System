"""Protect review provenance, completeness and stable precomputed rankings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from multimodal_ecommerce import safety
from multimodal_ecommerce.dataset_release import write_rows
from multimodal_ecommerce.frozen_retrieval import score_predictions
from multimodal_ecommerce.review_workflow import merge_reviews, validate_batch
from multimodal_ecommerce.safety import sha256_file, write_json


def template(
    task: str = "retrieval", status: str = "draft_requires_human_review"
) -> dict:
    """Return template for this workflow.

    Args:
        task: Task used by this operation.
        status: Status used by this operation.

    Returns:
        Mapping with fields: query_id, product_id, grade, id, status,
        relevance_pass, fluency_pass, factual_error.
    """
    row: dict = {
        "task": task,
        "batch_id": "batch",
        "reviewer": "",
        "note": "",
        "review_origin": None,
    }
    if task == "retrieval":
        return {**row, "query_id": "q", "product_id": "p", "grade": None}
    return {
        **row,
        "id": "p",
        "status": status,
        "relevance_pass": None,
        "fluency_pass": None,
        "factual_error": None,
    }


def test_unreviewed_is_never_a_negative() -> None:
    """Verify that unreviewed is never a negative."""
    blank = template()
    assert validate_batch([blank], [blank])["completed"] == 0
    reviewed: dict = {
        **blank,
        "grade": 0,
        "reviewer": "synthetic-test",
        "review_origin": "human",
    }
    assert validate_batch([blank], [reviewed])["completed"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"grade": False},
        {"grade": "0"},
        {"grade": 1, "reviewer": ""},
        {"review_origin": "model_assisted"},
        {"batch_id": "another"},
        {"product_id": "unknown"},
        {"extra": "not_in_schema"},
    ],
)
def test_bad_imports_cannot_become_formal_judgments(change: dict) -> None:
    """Verify that bad imports cannot become formal judgments.

    Args:
        change: Change used by this operation.
    """
    blank = template()
    row: dict = {
        **blank,
        "grade": 1,
        "reviewer": "synthetic-test",
        "review_origin": "human",
        **change,
    }
    with pytest.raises(ValueError):
        validate_batch([blank], [row])


def test_missing_negative_duplicate_and_missing_id_rejected() -> None:
    """Verify that missing negative duplicate and missing id rejected."""
    blank = template()
    other: dict = {**blank, "product_id": "other"}
    with pytest.raises(ValueError, match="cardinality"):
        validate_batch([blank, other], [blank])
    with pytest.raises(ValueError, match="duplicate"):
        validate_batch([blank, other], [blank, blank])
    with pytest.raises(ValueError):
        validate_batch([blank], [{"task": "retrieval"}])


def test_failed_copy_requires_evidence_and_no_output_cannot_be_scored() -> None:
    """Verify that failed copy requires evidence and no output cannot be scored."""
    blank = template("copy")
    row: dict = {
        **blank,
        "reviewer": "synthetic-test",
        "review_origin": "human",
        "relevance_pass": False,
        "fluency_pass": True,
        "factual_error": True,
    }
    assert not validate_batch([blank], [row])["ready"]
    row["note"] = "Synthetic fixture: source 10 cm but output 20 cm."
    assert validate_batch([blank], [row])["ready"]
    with pytest.raises(ValueError, match="Factual errors"):
        validate_batch([blank], [{**row, "relevance_pass": True}])
    absent = template("copy", "blocked_source_conflict")
    acknowledged: dict = {
        **absent,
        "reviewer": "synthetic-test",
        "review_origin": "human",
        "note": "Synthetic fixture: conflicting source.",
    }
    assert validate_batch([absent], [acknowledged])["ready"]
    with pytest.raises(ValueError, match="Absent output"):
        validate_batch([absent], [{**acknowledged, "fluency_pass": False}])


def freeze_files(directory: Path) -> None:
    """Freeze files.

    Args:
        directory: Directory used by this operation.

    Notes:
        Writes derived artifacts through write_json.
    """
    write_json(
        directory / "manifest.json",
        {
            "files": {
                path.name: sha256_file(path)
                for path in directory.iterdir()
                if path.is_file() and path.name != "manifest.json"
            }
        },
    )


def test_scores_reuse_frozen_predictions_and_refuse_blank_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that scores reuse frozen predictions and refuse blank labels.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.

    Notes:
        Writes derived artifacts through kit.mkdir, predictions.mkdir, write_json,
        write_rows.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    kit, predictions = tmp_path / "kit", tmp_path / "predictions"
    kit.mkdir()
    predictions.mkdir()
    write_rows(
        kit / "queries.jsonl", [{"id": "q", "kind": "image", "exclude_id": "self"}]
    )
    write_json(kit / "catalog.json", [{"id": key} for key in ("self", "p", "n")])
    freeze_files(kit)
    write_json(
        predictions / "run.json",
        {
            "kit": str(kit),
            "kit_manifest_sha256": sha256_file(kit / "manifest.json"),
            "target": "image",
        },
    )
    write_json(
        predictions / "rankings.json",
        {"q": [{"id": "p", "rank": 1}, {"id": "n", "rank": 2}]},
    )
    freeze_files(predictions)
    judgments = tmp_path / "labels.jsonl"
    labels = [
        {
            "query_id": "q",
            "product_id": key,
            "grade": grade,
            "reviewer": "synthetic-test",
        }
        for key, grade in (("p", 1), ("n", None))
    ]
    write_rows(judgments, labels)
    with pytest.raises(ValueError, match="Unjudged"):
        score_predictions(predictions, judgments)
    labels[1]["grade"] = 0
    write_rows(judgments, labels)
    report = score_predictions(predictions, judgments)
    assert report["model_inference_performed"] is False
    assert report["metrics"]["macro_over_positive_queries"]["all"]["recall@10"] == 1
    assert (
        report["metrics"]["macro_over_positive_queries"]["all"]["precision@10"] == 0.1
    )
    write_json(
        predictions / "rankings.json",
        {"q": [{"id": "self", "rank": 1}, {"id": "p", "rank": 2}]},
    )
    freeze_files(predictions)
    with pytest.raises(ValueError, match="non-self"):
        score_predictions(predictions, judgments)


def test_merge_keeps_missing_batches_blank_and_rejects_duplicate_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that merge keeps missing batches blank and rejects duplicate versions.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.

    Notes:
        Writes derived artifacts through packet.mkdir, root.mkdir, write_json,
        write_rows.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    packet = tmp_path / "packet"
    packet.mkdir()
    sources: dict = {}
    for name in ("kit", "copy_run"):
        root = tmp_path / name
        root.mkdir()
        write_json(root / "manifest.json", {"files": {}})
        sources[name] = str(root)
        sources[name + "_manifest_sha256"] = sha256_file(root / "manifest.json")
    retrieval = template()
    copy: dict = {**template("copy"), "batch_id": "copy"}
    write_rows(packet / "r.jsonl", [retrieval])
    write_rows(packet / "c.jsonl", [copy])
    write_json(
        packet / "packet.json",
        {
            **sources,
            "batches": [
                {"id": "batch", "task": "retrieval", "template": "r.jsonl"},
                {"id": "copy", "task": "copy", "template": "c.jsonl"},
            ],
            "files": {
                name: sha256_file(packet / name) for name in ("r.jsonl", "c.jsonl")
            },
        },
    )
    output = tmp_path / "merged"
    result = merge_reviews(packet, [], output)
    assert result["completed"] == {"retrieval": 0, "copy": 0}
    assert not list(output.glob("*.reviewed.jsonl"))
    assert json.loads((output / "qrels.partial.jsonl").read_text())["grade"] is None
    exported = tmp_path / "r.reviewed.jsonl"
    write_rows(exported, [retrieval])
    with pytest.raises(ValueError, match="duplicate batch"):
        merge_reviews(packet, [exported, exported], tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
