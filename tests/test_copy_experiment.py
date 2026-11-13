"""Test split isolation and honest scoring denominators for copy experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from multimodal_ecommerce import safety
from multimodal_ecommerce.copy_experiment import score_copy_run, select_records
from multimodal_ecommerce.dataset_release import write_rows
from multimodal_ecommerce.safety import sha256_file, write_json


def test_selection_keeps_heldout_out_of_development(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deterministic development run cannot accidentally select heldout rows.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "rows.jsonl"
    rows = [
        {"id": f"{role}/{number}", "project_role": role, "category": "家居日用"}
        for role in ("development", "heldout")
        for number in range(10)
    ]
    write_rows(path, rows)
    chosen = select_records(path, "development", 3)
    write_rows(path, list(reversed(rows)))
    assert select_records(path, "development", 3) == chosen
    assert len(chosen) == 3
    assert all(row["project_role"] == "development" for row in chosen)


def test_scoring_refuses_blank_and_counts_abstentions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coverage loss cannot disappear from an apparently perfect pass rate.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    records: list[dict] = [
        {"id": "a", "status": "draft_requires_human_review", "copy": {"title": "商品"}},
        {"id": "b", "status": "blocked_source_conflict"},
    ]
    write_rows(root / "results.jsonl", records)
    write_json(
        root / "manifest.json",
        {"files": {"results.jsonl": sha256_file(root / "results.jsonl")}},
    )
    judgments = tmp_path / "judgments.jsonl"
    labels = [
        {
            "id": item["id"],
            "status": item["status"],
            "relevance_pass": None,
            "fluency_pass": None,
            "factual_error": None,
            "reviewer": "",
        }
        for item in records
    ]
    write_rows(judgments, labels)
    with pytest.raises(ValueError, match="reviewer"):
        score_copy_run(root, judgments)
    labels[0].update(
        {
            "relevance_pass": True,
            "fluency_pass": True,
            "factual_error": False,
            "reviewer": "r",
        }
    )
    labels[1]["reviewer"] = "r"
    write_rows(judgments, labels)
    result = score_copy_run(root, judgments)
    assert result["generated_only_pass_rates"]["relevance_pass"] == 1
    assert (
        result["all_selected_pass_rates_counting_abstentions_as_not_passed"][
            "relevance_pass"
        ]
        == 0.5
    )
    labels[1]["fluency_pass"] = True
    write_rows(judgments, labels)
    with pytest.raises(ValueError, match="Absent output"):
        score_copy_run(root, judgments)
    assert json.loads((root / "manifest.json").read_text())["files"]
