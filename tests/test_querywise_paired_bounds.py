"""Verify querywise paired bounds contracts with isolated regression scenarios."""

from __future__ import annotations

import itertools
from typing import NoReturn

import pytest

import querywise_paired_bounds as paired_module
from multimodal_ecommerce.business_evaluation import METRICS
from multimodal_ecommerce.interval_retrieval_metrics import query_bounds
from multimodal_ecommerce.retrieval_strategies import extended_metrics
from querywise_paired_bounds import (
    CONSERVATIVE,
    EXACT,
    IDENTICAL,
    MIXED,
    UNDEFINED,
    paired_querywise,
)


def labels_for(grades: dict) -> list:
    """Return labels for for this workflow.

    Args:
        grades: Grades used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return [
        dict(
            query_id=qid,
            candidate_id=cid,
            grade=grade,
            review_origin="model",
            human_label=False,
        )
        for qid, current in grades.items()
        for cid, grade in current.items()
    ]


def brute_shared(
    rankings: dict, grades: dict, modes: dict, comparisons: list[tuple[str, str]]
) -> dict:
    """Return brute shared for this workflow.

    Args:
        rankings: Candidate rankings keyed by query and retrieval method.
        grades: Grades used by this operation.
        modes: Modes used by this operation.
        comparisons: Comparisons used by this operation.

    Returns:
        Result mapping produced by the workflow.
    """
    unknown = [
        (q, c) for q, current in grades.items() for c, g in current.items() if g is None
    ]
    values: dict = {
        (scope, candidate, baseline, metric): []
        for scope in ["all", *sorted(set(modes.values()))]
        for candidate, baseline in comparisons
        for metric in METRICS
    }
    for bits in itertools.product([0, 1], repeat=len(unknown)):
        completion: dict = dict(zip(unknown, bits))
        rows: dict = {}
        for qid, methods in rankings.items():
            positive = {
                c
                for c, g in grades[qid].items()
                if g == 1 or g is None and completion[qid, c]
            }
            rows[qid] = {
                method: (
                    extended_metrics(order, positive)
                    if positive
                    else {**dict.fromkeys(METRICS, 0.0), "recall@10": None}
                )
                for method, order in methods.items()
            }
        for key in values:
            scope, candidate, baseline, metric = key
            differences = [
                methods[candidate][metric] - methods[baseline][metric]
                for qid, methods in rows.items()
                if (scope == "all" or modes[qid] == scope)
                and methods[candidate][metric] is not None
            ]
            if differences:
                values[key].append(sum(differences) / len(differences))
    return {key: [min(v), max(v)] if v else [None, None] for key, v in values.items()}


def test_querywise_ranges_match_joint_exhaustion_with_optional_recall_denominator() -> (
    None
):
    """Verify that querywise ranges match joint exhaustion with optional recall
    denominator.
    """
    ids = [str(i) for i in range(12)]
    orders: dict = {"base": ids, "candidate": ids[::-1], "other": ids[5:] + ids[:5]}
    rankings: dict = {qid: orders for qid in ["mandatory", "optional", "zero"]}
    grades: dict = {qid: dict.fromkeys(ids, 0) for qid in rankings}
    grades["mandatory"].update({"0": 1, "1": None, "11": None})
    grades["optional"].update({"2": None, "10": None})
    modes: dict = dict(mandatory="image", optional="text", zero="image")
    comparisons = [("candidate", "base"), ("other", "base")]
    result = paired_querywise(rankings, labels_for(grades), modes, comparisons)
    expected = brute_shared(rankings, grades, modes, comparisons)
    for (scope, candidate, baseline, metric), bounds in expected.items():
        actual = result["scopes"][scope][candidate][baseline][metric]
        assert actual["delta_range"] == pytest.approx(bounds)
        assert actual["bound_type"] == EXACT
    zero = [row for row in result["query_pair_bounds"] if row["query_id"] == "zero"]
    assert all(row["recall_presence"] == "undefined" for row in zero)
    assert all(
        row["metrics"]["recall@10"]["delta_range"] == [None, None] for row in zero
    )
    assert all(
        row["metrics"]["precision@10"]["delta_range"] == [0.0, 0.0] for row in zero
    )
    assert any(
        row["recall_presence"] == "optional" for row in result["query_pair_bounds"]
    )


def test_identical_text_with_seventeen_unknowns_does_not_weaken_exact_image_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that identical text with seventeen unknowns does not weaken exact image
    pairs.

    Args:
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    ids = [str(i) for i in range(20)]
    rankings: dict = {
        "text_many": {"base": ids, "candidate": list(ids)},
        "text_zero": {"base": ids, "candidate": list(ids)},
        "image": {"base": ids, "candidate": ids[::-1]},
    }
    grades: dict = {q: dict.fromkeys(ids, 0) for q in rankings}
    grades["text_many"].update({str(i): None for i in range(17)})
    grades["image"].update({"0": 1, "1": None, "19": None})
    modes: dict = dict(text_many="text", text_zero="text", image="image")

    def forbidden_fallback(*args: object, **kwargs: object) -> NoReturn:
        """Apply the forbidden fallback workflow step.

        Args:
            *args: Extra arguments accepted to match the patched dependency.
            **kwargs: Extra arguments accepted to match the patched dependency.

        Raises:
            AssertionError: No nonidentical pair exceeds the exact budget.
        """
        raise AssertionError("No nonidentical pair exceeds the exact budget")

    monkeypatch.setattr(paired_module, "query_bounds", forbidden_fallback)
    result = paired_querywise(
        rankings, labels_for(grades), modes, [("candidate", "base")]
    )
    for metric in METRICS:
        text = result["scopes"]["text"]["candidate"]["base"][metric]
        image = result["scopes"]["image"]["candidate"]["base"][metric]
        assert text["delta_range"] == [0.0, 0.0]
        assert text["bound_type"] == IDENTICAL
        assert image["bound_type"] == EXACT
        assert image["conservative_query_ids"] == []
    assert result["enumeration_work"] == [
        dict(query_id="image", completions=4, blocks=1, unique_orders=2)
    ]
    zero = next(
        row for row in result["query_pair_bounds"] if row["query_id"] == "text_zero"
    )
    assert zero["metrics"]["recall@10"]["bound_type"] == UNDEFINED
    assert zero["metrics"]["recall@10"]["delta_range"] == [None, None]


def test_only_nonidentical_many_unknown_query_uses_disclosed_conservative_bounds() -> (
    None
):
    """Verify that only nonidentical many unknown query uses disclosed conservative
    bounds.
    """
    ids = [str(i) for i in range(24)]
    rankings: dict = {
        "many": {"base": ids, "candidate": ids[::-1]},
        "few": {"base": ids, "candidate": ids[1:] + ids[:1]},
    }
    grades: dict = {q: dict.fromkeys(ids, 0) for q in rankings}
    grades["many"].update({str(i): None for i in range(17)})
    grades["many"]["20"] = 1
    grades["few"].update({"0": 1, "4": None})
    result = paired_querywise(
        rankings,
        labels_for(grades),
        dict(many="text", few="image"),
        [("candidate", "base"), ("base", "base")],
    )
    a = query_bounds(rankings["many"]["candidate"], grades["many"])["ranges"]
    b = query_bounds(rankings["many"]["base"], grades["many"])["ranges"]
    for metric in METRICS:
        text = result["scopes"]["text"]["candidate"]["base"][metric]
        assert text["bound_type"] == CONSERVATIVE
        assert text["conservative_query_ids"] == ["many"]
        assert text["delta_range"] == pytest.approx(
            [a[metric][0] - b[metric][1], a[metric][1] - b[metric][0]]
        )
        assert (
            result["scopes"]["image"]["candidate"]["base"][metric]["bound_type"]
            == EXACT
        )
        assert (
            result["scopes"]["all"]["candidate"]["base"][metric]["bound_type"] == MIXED
        )
        assert result["scopes"]["all"]["base"]["base"][metric]["delta_range"] == [
            0.0,
            0.0,
        ]
        assert (
            result["scopes"]["all"]["base"]["base"][metric]["bound_type"] == IDENTICAL
        )
