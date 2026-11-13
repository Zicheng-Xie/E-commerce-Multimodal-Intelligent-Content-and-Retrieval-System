"""Validate ranking, filtering, integrity and metric definitions without a GPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from multimodal_ecommerce import safety
from multimodal_ecommerce.retrieval import VectorIndex, unit_rows
from multimodal_ecommerce.retrieval_eval import ranking_metrics, write_gallery


def test_cosine_ranking_filters_and_excludes_self() -> None:
    """Raw magnitude cannot bias cosine rank; self and category filters apply."""
    records = [
        {"id": key, "category": category}
        for key, category in [("a", "服饰"), ("b", "家居"), ("c", "服饰")]
    ]
    matrix = np.array([[100, 0], [0, 2], [1, 1]])
    index = VectorIndex(records, matrix, matrix)
    result = index.search(np.array([1, 0]), top_k=3)
    assert [r["id"] for r in result] == ["a", "c", "b"]
    assert result[0]["score"] == pytest.approx(1)
    assert (
        index.search(np.array([1, 0]), category="服饰", exclude_id="a")[0]["id"] == "c"
    )
    assert index.search(np.array([1, 0]), category="不存在") == []


@pytest.mark.parametrize(
    "vectors", [np.zeros((1, 2)), np.array([[np.nan, 1]]), np.ones(3)]
)
def test_bad_vectors_rejected(vectors: np.ndarray) -> None:
    """Invalid numerical output must not silently contaminate the index.

    Args:
        vectors: Vectors used by this operation.
    """
    with pytest.raises(ValueError):
        unit_rows(vectors)


def test_index_rejects_changed_vectors(tmp_path: Path) -> None:
    """Persisted metadata cannot be paired with modified vectors unnoticed.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
    """
    (tmp_path / "vectors.npz").write_bytes(b"changed")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"files": {"vectors.npz": "badchecksum", "records.json": "badchecksum"}}
        )
    )
    with pytest.raises(ValueError, match="checksum"):
        VectorIndex.load(tmp_path)


def test_precision_denominator_is_requested_k() -> None:
    """Precision@10 must not become precision@2 on a short result list."""
    metrics = ranking_metrics(["a", "b"], {"a", "c"}, k=10)
    assert metrics == {"precision@10": 0.1, "recall@10": 0.5}
    with pytest.raises(ValueError):
        ranking_metrics(["a", "a"], {"a"})


def test_html_escapes_dataset_titles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Product text is data, not executable markup.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "view.html"
    write_gallery(
        output,
        [
            {
                "query": "<query>",
                "results": [
                    {
                        "rank": 1,
                        "score": 0.7,
                        "title": "<script>alert(1)</script>",
                        "id": "x",
                        "image_path": str(tmp_path / "x.jpg"),
                    }
                ],
            }
        ],
    )
    content = output.read_text(encoding="utf-8")
    assert "<script>" not in content
    assert "&lt;script&gt;" in content


def test_image_and_text_fusion_explicit() -> None:
    """Fusion is the documented mean of two cosine scores."""
    index = VectorIndex(
        [{"id": "a", "category": "x"}], np.array([[1, 0]]), np.array([[0, 1]])
    )
    result = index.search(np.array([1, 0]), target="fused")[0]
    assert result["score"] == pytest.approx(0.5)
