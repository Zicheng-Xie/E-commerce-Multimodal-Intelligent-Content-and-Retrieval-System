"""Critical AI provenance and unknown-denominator checks."""

from __future__ import annotations

import itertools

import pytest

from multimodal_ecommerce.ai_review import (
    diagnostic_intervals,
    recall_interval,
    validate_profile,
)


def test_recall_intervals_match_binary_completions() -> None:
    """Verify that recall intervals match binary completions."""
    for positives in range(4):
        for a in range(positives + 1):
            for ut, uo in itertools.product(range(4), repeat=2):
                actual = []
                for top, outside in itertools.product(range(ut + 1), range(uo + 1)):
                    if positives + top + outside:
                        actual.append((a + top) / (positives + top + outside))
                expected = (min(actual), max(actual)) if actual else (None, None)
                assert (
                    recall_interval(a, positives, ut, uo) == pytest.approx(expected)
                    if actual
                    else recall_interval(a, positives, ut, uo) == expected
                )


def test_ai_diagnostic_retains_unknown_and_fixed_ten() -> None:
    """Verify that ai diagnostic retains unknown and fixed ten."""
    order = [str(i) for i in range(12)]
    labels = [
        dict(
            query_id="q",
            candidate_id=c,
            grade=1 if i == 0 else None if i == 1 else 0,
            review_origin="model",
            human_label=False,
        )
        for i, c in enumerate(order)
    ]
    result = diagnostic_intervals({"q": {"global": order}}, labels, {"q": "text"})
    values = result["scopes"]["all"]["global"]
    assert values["precision_lower"] == 0.1
    assert values["precision_upper"] == 0.2
    assert result["unknown"] == 1 and result["formal_metrics"] is None
    labels[0]["human_label"] = True
    with pytest.raises(ValueError, match="AI provenance"):
        diagnostic_intervals({"q": {"global": order}}, labels, {"q": "text"})


def test_profile_rejects_boolean_grade_and_missing_evidence() -> None:
    """Verify that profile rejects boolean grade and missing evidence."""
    value: dict = dict(
        observed_type="cup",
        readable=True,
        evidence="杯子主体",
        uncertainty="",
        text_grades=[0] * 12,
    )
    assert validate_profile(value)["observed_type"] == "cup"
    value["text_grades"][0] = True
    with pytest.raises(ValueError):
        validate_profile(value)
