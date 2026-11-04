"""Verify interval retrieval metrics contracts with isolated regression scenarios."""

from __future__ import annotations

import itertools
import random

import pytest

from multimodal_ecommerce.interval_retrieval_metrics import query_bounds
from multimodal_ecommerce.retrieval_strategies import extended_metrics


def test_dynamic_bounds_match_exhaustive_shared_label_completions() -> None:
    """Verify that dynamic bounds match exhaustive shared label completions."""
    randomizer = random.Random(20261023)
    order = [str(i) for i in range(15)]
    for _ in range(18):
        randomizer.shuffle(order)
        grades: dict = {c: randomizer.choice([0, 0, 1, None]) for c in order}
        unknown = [c for c, v in grades.items() if v is None]
        fixed = {c for c, v in grades.items() if v == 1}
        rows = []
        for bits in itertools.product([False, True], repeat=len(unknown)):
            relevant = fixed | {c for c, keep in zip(unknown, bits) if keep}
            rows.append(
                extended_metrics(order, relevant)
                if relevant
                else dict(
                    ap=0.0,
                    mrr=0.0,
                    **{
                        "precision@10": 0.0,
                        "recall@10": None,
                        "ndcg@10": 0.0,
                        "hit@1": 0.0,
                    },
                )
            )
        actual = query_bounds(order, grades)["ranges"]
        for m in actual:
            values = [value for r in rows if (value := r[m]) is not None]
            assert (
                actual[m] == pytest.approx([min(values), max(values)])
                if values
                else actual[m] == [None, None]
            )


def test_many_unknowns_stay_unknown_without_exponential_filling() -> None:
    """Verify that many unknowns stay unknown without exponential filling."""
    order = [str(i) for i in range(300)]
    result = query_bounds(order, dict.fromkeys(order))
    assert result["unknown"] == 300
    assert result["ranges"]["precision@10"] == [0.0, 1.0]
    assert result["ranges"]["ap"] == [0.0, 1.0]
    assert result["ranges"]["recall@10"] == [0.0, 1.0]


def test_zero_positive_recall_is_not_invented() -> None:
    """Verify that zero positive recall is not invented."""
    result = query_bounds(["a", "b"], {"a": 0, "b": 0})
    assert result["ranges"]["recall@10"] == [None, None]
    assert result["ranges"]["precision@10"] == [0.0, 0.0]
