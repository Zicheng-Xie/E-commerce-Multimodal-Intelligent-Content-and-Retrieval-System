"""Human provenance, exact precision denominators, and mathematically sound recall bounds."""

from __future__ import annotations

import itertools
import random

import pytest

from multimodal_ecommerce.human_retrieval_evaluation import (
    evaluate_human_retrieval,
    retrieval_error_queue,
)

TARGET = dict(k=10, precision_at_10_required=0.8, recall_at_10_preferred=0.9)


def _query(grades: list, mode: str = "image") -> tuple[list[str], dict]:
    order = [str(i) for i in range(len(grades))]
    return order, dict(
        mode=mode,
        expected_pairs=len(order),
        exclusions=[],
        judgments={cid: dict(grade=g, human=True) for cid, g in zip(order, grades)},
    )


def test_sixteen_confirmed_positives_make_ninety_percent_top10_impossible() -> None:
    """Even a perfect first ten recover only ten of sixteen relevant products."""
    order, query = _query([1] * 16 + [0] * 4)
    result = evaluate_human_retrieval({"q": order}, {"q": query}, TARGET)["scopes"][
        "all"
    ]
    assert result["precision_at_10"]["value"] == 1
    assert result["recall_at_10"]["value"] == pytest.approx(0.625)
    assert result["recall_capacity"]["optimistic_ideal_macro_upper"] == pytest.approx(
        0.625
    )
    assert result["recall_capacity"]["incompatible"]
    assert (
        result["recall_at_10"]["status"]
        == "incompatible_with_confirmed_human_labels_and_top10"
    )
    assert result["documented_positive_coverage"]["20"] == 1


def test_ai_labels_never_complete_the_human_view() -> None:
    """Machine positives cannot turn one human positive into a human gold set."""
    order, query = _query([1, 0] + [1] * 18)
    for cid in order[2:]:
        query["judgments"][cid]["human"] = False
    result = evaluate_human_retrieval({"q": order}, {"q": query}, TARGET)["scopes"][
        "all"
    ]
    p, r = result["precision_at_10"], result["recall_at_10"]
    assert p["value"] is None and p["judged"] == 2 and p["missing"] == 8
    assert p["bounds"] == pytest.approx([0.1, 0.9])
    assert r["value"] is None and r["bounds"] == pytest.approx([1 / 11, 1])
    assert result["coverage"]["mixed_binary"] == 20
    assert result["coverage"]["human_positive"] == 1


def test_uncertain_inside_top10_stays_unresolved_even_if_recall_bounds_collapse() -> (
    None
):
    """A tight conditional bound does not imply a complete human relevance set."""
    order, query = _query([1] * 9 + [None, 0, 0])
    result = evaluate_human_retrieval({"q": order}, {"q": query}, TARGET)["scopes"][
        "all"
    ]
    assert result["precision_at_10"]["value"] is None
    assert result["precision_at_10"]["bounds"] == [0.9, 1]
    assert result["recall_at_10"]["bounds"] == [1, 1]
    assert result["recall_at_10"]["value"] is None
    assert result["coverage"]["human_uncertain"] == 1


def test_complete_precision_uses_all_550_slots_and_keeps_formal_acceptance_false() -> (
    None
):
    """Do not drop zero-hit queries or change overall/modal precision denominators."""
    order, query = _query([1] * 8 + [0] * 4)
    ranks = {f"q{i}": order for i in range(55)}
    queries = {
        key: {**query, "mode": "image" if i < 36 else "text"}
        for i, key in enumerate(ranks)
    }
    result = evaluate_human_retrieval(ranks, queries, TARGET)
    for mode, expected in [("all", 550), ("image", 360), ("text", 190)]:
        p = result["scopes"][mode]["precision_at_10"]
        assert p["expected"] == p["judged"] == expected and p["missing"] == 0
        assert p["value"] == pytest.approx(0.8) and p["status"] == "met"
    assert not result["formal_acceptance"]


def test_zero_positive_queries_are_undefined_for_recall_but_remain_in_precision() -> (
    None
):
    """Preserve all queries in precision while naming undefined recall cohorts."""
    order, a = _query([1] * 5 + [0] * 7)
    _, b = _query([0] * 12, mode="text")
    result = evaluate_human_retrieval(
        {"a": order, "b": order}, {"a": a, "b": b}, TARGET
    )
    assert result["scopes"]["all"]["precision_at_10"]["value"] == 0.25
    assert result["scopes"]["all"]["recall_at_10"]["value"] == 1
    assert (
        result["scopes"]["all"]["recall_at_10"]["zero_positive_undefined_queries"] == 1
    )
    assert result["scopes"]["text"]["recall_at_10"]["value"] is None


def test_human_recall_bounds_match_every_possible_unresolved_completion() -> None:
    """Check both inside/outside unknowns against exhaustive positive-set completions."""
    rng = random.Random(20261109)
    for _ in range(25):
        grades = [rng.choice([0, 1, None]) for _ in range(12)]
        order, query = _query(grades)
        unknown = [i for i, v in enumerate(grades) if v is None]
        fixed = {i for i, v in enumerate(grades) if v == 1}
        values = []
        for bits in itertools.product((False, True), repeat=len(unknown)):
            positive = fixed | {i for i, on in zip(unknown, bits) if on}
            if positive:
                values.append(len(positive.intersection(range(10))) / len(positive))
        actual = evaluate_human_retrieval({"q": order}, {"q": query}, TARGET)["scopes"][
            "all"
        ]["recall_at_10"]["bounds"]
        assert (
            actual == pytest.approx([min(values), max(values)])
            if values
            else actual == [None, None]
        )


def test_missing_queries_and_boolean_relevance_are_rejected() -> None:
    """Fail visibly rather than silently shrinking cohorts or accepting false labels."""
    order, q = _query([1] * 12)
    with pytest.raises(ValueError, match="All original queries"):
        evaluate_human_retrieval({}, {"q": q}, TARGET)
    q["judgments"]["0"]["grade"] = True
    with pytest.raises(ValueError, match="Binary integer"):
        evaluate_human_retrieval({"q": order}, {"q": q}, TARGET)


def test_error_queue_uses_human_errors_and_retains_unresolved_and_pool_evidence() -> (
    None
):
    """AI negatives stay unresolved and human reasons support only explicit tags."""
    order, q = _query([0, 0] + [1] * 19)
    q["judgments"]["1"]["human"] = False
    manifest = dict(
        catalog=[dict(id=c, title="product " + c) for c in order],
        query_manifest={"q": q},
    )
    annotations = {
        ("q", "0"): [
            dict(
                grade=0,
                reason="商品主体不同，候选是手机保护套",
                source_csv="answers.csv",
                task_id="t",
            )
        ]
    }
    row = retrieval_error_queue({"q": order}, manifest, annotations)[0]
    assert row["confirmed_top10_error_count"] == 1 and row["unresolved_top10"] == 1
    assert row["errors"][0]["error_types"] == ["subject_or_accessory"]
    assert len(row["known_relevant_in_top20_not_top10"]) == 10
    assert len(row["known_relevant_outside_top20"]) == 1
