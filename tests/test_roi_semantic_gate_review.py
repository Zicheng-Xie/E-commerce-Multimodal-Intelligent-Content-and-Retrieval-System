"""Native review validity is independent of the semantic accept/reject decision."""

from __future__ import annotations

import json

from roi_semantic_gate_review import CHECKS, MODEL, parse_response


def response(decision: str = "accept", **changes: object) -> dict:
    """Return response for this workflow.

    Args:
        decision: Decision used by this operation.
        **changes: Named options forwarded to the underlying operation.

    Returns:
        Mapping with fields: model, done, done_reason, prompt_eval_count,
        eval_count, message.
    """
    value: dict = dict(
        decision=decision,
        **{key: True for key in CHECKS},
        evidence_cn="两张图均保留主商品及可见关键结构。",
    )
    value.update(changes)
    return dict(
        model=MODEL,
        done=True,
        done_reason="stop",
        prompt_eval_count=2000,
        eval_count=100,
        message=dict(role="assistant", content=json.dumps(value, ensure_ascii=False)),
    )


def test_valid_uncertainty_is_a_semantic_fallback_not_an_engineering_failure() -> None:
    """Verify that valid uncertainty is a semantic fallback not an engineering
    failure.
    """
    accepted = parse_response(response())
    uncertain = parse_response(response("uncertain", primary_product_retained=None))
    rejected = parse_response(response("reject", same_primary_product=False))
    assert accepted["valid"] and accepted["accepted"]
    assert uncertain["valid"] and uncertain["accepted"] is False
    assert rejected["valid"] and rejected["accepted"] is False


def test_truncated_schema_invalid_or_failed_requests_never_create_false_verdicts() -> (
    None
):
    """Verify that truncated schema invalid or failed requests never create false
    verdicts.
    """
    truncated: dict = dict(response(), done_reason="length")
    invalid = response(primary_product_retained=False)
    missing_counts = response()
    missing_counts.pop("prompt_eval_count")
    values = [
        parse_response(truncated),
        parse_response(invalid),
        parse_response(missing_counts),
        parse_response(None, {"type": "TimeoutError"}),
    ]
    assert all(value["valid"] is False for value in values)
    assert all("accepted" not in value for value in values)
    assert all("engineering_error" in value for value in values)
