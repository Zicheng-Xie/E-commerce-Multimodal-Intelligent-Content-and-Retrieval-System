"""Native label parsing never converts uncertain or malformed output to negatives."""

from __future__ import annotations

import json

import pytest

from balanced_supplement import MODEL, parsed


def sample() -> dict:
    """Return sample for this workflow.

    Returns:
        Mapping with fields: model, done, done_reason, prompt_eval_count,
        eval_count, message.
    """
    return dict(
        model=MODEL,
        done=True,
        done_reason="stop",
        prompt_eval_count=2000,
        eval_count=180,
        message=dict(
            role="assistant",
            content=json.dumps(
                dict(
                    grades=[1, None, 0],
                    evidence=["可见同类主体", "属性无法确定", "可见主体不同"],
                    candidate_product_type="耳机",
                    candidate_appearance="带耳罩及头梁",
                )
            ),
        ),
    )


def test_native_unknown_and_order_remain_explicit() -> None:
    """Verify that native unknown and order remain explicit."""
    result = parsed(sample(), 3)
    assert result["grades"] == [1, None, 0]


def test_engineering_invalidity_does_not_produce_labels() -> None:
    """Verify that engineering invalidity does not produce labels."""
    truncated: dict = dict(sample(), done_reason="length")
    boolean_grade = sample()
    value = json.loads(boolean_grade["message"]["content"])
    value["grades"][0] = True
    boolean_grade["message"]["content"] = json.dumps(value)
    for raw, count in [(truncated, 3), (boolean_grade, 3), (sample(), 2)]:
        with pytest.raises((AssertionError, ValueError)):
            parsed(raw, count)
