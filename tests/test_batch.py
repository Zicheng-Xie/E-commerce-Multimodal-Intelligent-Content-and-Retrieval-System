"""Batch runs must preserve split-level failures and existing evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from multimodal_ecommerce import safety
from multimodal_ecommerce.batch import generate_batch
from multimodal_ecommerce.ollama_client import OllamaError


class StubClient:
    """Return one successful response and one deliberate inference failure."""

    model = "test-model"

    def generate_copy(self, product: dict) -> dict:
        """Fail deterministically for the test record.

        Args:
            product: Product used by this operation.

        Returns:
            Mapping with fields: _meta.

        Raises:
            OllamaError: intentional failure.
        """
        if product["name"] == "fail":
            raise OllamaError("intentional failure")
        return {
            "_meta": {
                "fact_check": {"status": "no_rule_flags"},
                "response": {"wall_seconds": 1.0},
            }
        }


def test_batch_failure_and_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed test request stays visible; reruns cannot overwrite the prior run.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    dataset = tmp_path / "products.jsonl"
    rows = [
        {
            "id": f"product/{split}/1",
            "split": split,
            "category": "服饰穿搭",
            "text": name,
            "attributes": {},
            "image_path": "image.png",
            "evaluation_eligible": True,
        }
        for split, name in [("train", "ok"), ("test", "fail")]
    ]
    dataset.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    output = tmp_path / "run"
    result = generate_batch(StubClient(), dataset, output, per_category=1)
    assert result["all_generated"] is False
    assert result["splits"]["test"]["attempted"] == 1
    assert result["splits"]["test"]["generated"] == 0
    assert result["splits"]["train"]["no_rule_flags"] == 1
    with pytest.raises(FileExistsError):
        generate_batch(StubClient(), dataset, output)
