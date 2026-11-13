"""The new context budget is explicit; no old response is relabelled or edited."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import balanced_pair_review as review
from balanced_pair_review import batches, parsed
from test_balanced_supplement_review import sample


def test_complete_queries_and_single_image_query_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve every query while limiting each image batch to one query.

    Args:
        tmp_path: Independent public-query and caption fixture directory.
        monkeypatch: Restore the query source after exercising the planner.
    """
    monkeypatch.setattr(review.base, "DATA", tmp_path)
    for cohort, text_count in (("tune", 8), ("regression", 4), ("fresh", 4)):
        queries: list[dict] = [
            dict(id=f"text_{i}", text=f"商品 {i}", image_path=None)
            for i in range(text_count)
        ]
        captions = tmp_path / "query_captions" / cohort
        captions.mkdir(parents=True)
        for i in range(8):
            query_id = f"image_{i}"
            queries.append(dict(id=query_id, text="", image_path=f"{query_id}.png"))
            (captions / f"{query_id}.json").write_text(
                json.dumps(dict(text="商品主体"))
            )
        public = tmp_path / cohort
        public.mkdir()
        (public / "queries.public.json").write_text(json.dumps(queries))
    planned = batches()
    keys = [q["key"] for _, qs in planned for q in qs]
    assert len(keys) == len(set(keys)) == 40
    assert all(len(qs) == 1 for mode, qs in planned if mode == "image")


def test_context_budget_and_actual_truncation_are_distinct() -> None:
    """Verify that context budget and actual truncation are distinct."""
    raw = sample()
    raw["prompt_eval_count"] = 7370
    assert parsed(raw, 3)["grades"] == [1, None, 0]
    with pytest.raises(AssertionError):
        parsed(dict(raw, prompt_eval_count=16000), 3)
    with pytest.raises(AssertionError):
        parsed(dict(raw, done_reason="length"), 3)
