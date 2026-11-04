"""Critical-path checks for role extraction and four-arm pool attribution."""

from __future__ import annotations

import pytest

from grounded_subject_runtime import validate
from role_pool_profiles import POLICY, SCHEMA, digest, request_identity


def test_role_schema_keeps_accessory_distinct_from_host_and_props() -> None:
    """Verify that role schema keeps accessory distinct from host and props."""
    value: dict = dict(
        sale_item="包内胆",
        role="accessory",
        host_object="手提包",
        display_props=["手提包"],
        function="收纳小件",
        appearance="多口袋布袋",
        primary_unclear=False,
        evidence="图中文字和独立袋体",
    )
    validate(value, SCHEMA)
    with pytest.raises(ValueError):
        validate(dict(value, relevance=1), SCHEMA)
    assert value["sale_item"] != value["host_object"]


def test_query_role_request_never_includes_candidate_title() -> None:
    """Verify that query role request never includes candidate title."""
    row: dict = dict(
        kind="query", source_sha256="a" * 64, public_title="secret candidate"
    )
    query = request_identity(row)
    assert "secret candidate" not in query["content"]
    assert digest(query) == digest(dict(reversed(list(query.items()))))
    candidate = request_identity(dict(row, kind="gallery"))
    assert "secret candidate" in candidate["content"]
    assert digest(query) != digest(candidate)


def test_fixed_comparison_policy() -> None:
    """Verify that fixed comparison policy."""
    assert len(POLICY["four_arms"]) == 4
    assert POLICY["fixed_k"] == 10
    assert POLICY["new_channel_budget"] == 20
    assert POLICY["new_masks_enabled"] is False
    assert POLICY["labels_or_metrics_used_to_select_pool"] is False
    assert POLICY["post_result_search"] is False
    assert POLICY["formal_metrics"] is None
