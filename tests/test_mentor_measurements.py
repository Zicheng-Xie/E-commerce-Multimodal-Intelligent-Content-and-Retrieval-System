"""Check task definitions, score statistics and factual/marketing separation."""

from __future__ import annotations

import pytest

from multimodal_ecommerce.content_contract import (
    MARKETING_LINES,
    boundary_cases,
    generate_content,
)
from multimodal_ecommerce.measurement import (
    cluster_interval,
    distribution,
    wilson_interval,
)
from multimodal_ecommerce.retrieval_benchmark import same_item_protocol


def test_source_pair_ids_are_split_scoped_and_queries_not_self_matches() -> None:
    """A shared numeric pair ID in another split cannot become a positive."""
    rows = []
    for name, source, split, digest in [
        ("q", "user", "train", "a"),
        ("s", "shop", "train", "b"),
        ("wrong_split", "shop", "validation", "c"),
        ("same_bytes", "shop", "train", "a"),
    ]:
        rows.append(
            {
                "id": name,
                "source": source,
                "split": split,
                "category": "shorts",
                "instance_count": 1,
                "item_keys": [f"{split}/1/1/7"],
                "image_sha256": digest,
            }
        )
    protocol = same_item_protocol(rows)
    assert protocol["queries"][0]["relevant_ids"] == ["s"]
    assert "q" not in protocol["gallery_ids"]


def test_confidence_interval_narrows_without_changing_accuracy() -> None:
    """More independent judgments improve precision of an estimate, not p."""
    a = wilson_interval(24, 30)
    b = wilson_interval(72, 90)
    assert a["estimate"] == b["estimate"] == 0.8
    assert b["upper"] - b["lower"] < a["upper"] - a["lower"]
    assert wilson_interval(0, 10)["lower"] == pytest.approx(0)
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        distribution([float("nan")])
    assert cluster_interval([0.1, 0.9], ["same", "same"])["interval"] is None


class StubClient:
    """Record whether inference happens for an intentionally incomplete input."""

    def __init__(self) -> None:
        """Initialize the component and validate its configured dependencies."""
        self.calls = 0

    def generate_copy(self, product: dict) -> dict:
        """Return generate copy for this workflow.

        Args:
            product: Product used by this operation.

        Returns:
            Mapping with fields: title, description, selling_points, _meta.
        """
        self.calls += 1
        return {
            "title": "商品",
            "description": "明确事实",
            "selling_points": ["一", "二", "三"],
            "_meta": {"release_status": "draft_requires_human_review"},
        }


def test_missing_required_fields_abstain_without_inference() -> None:
    """Verify that missing required fields abstain without inference."""
    client = StubClient()
    result = generate_content(
        client, {"name": "音箱", "category": "3C数码", "attributes": {}}, "friendly"
    )
    assert result["status"] == "abstained_missing_required_fields"
    assert result["marketing_line"] is None
    assert client.calls == 0


def test_blank_attribute_values_are_missing_not_facts() -> None:
    """Verify that blank attribute values are missing not facts."""
    client = StubClient()
    result = generate_content(
        client,
        {"name": "盒子", "category": "家居", "attributes": {"材质": ["", "  "]}},
    )
    assert result["missing_required_fields"] == ["attributes"]
    assert client.calls == 0


def test_marketing_is_separate_and_closed_vocabulary() -> None:
    """Verify that marketing is separate and closed vocabulary."""
    client = StubClient()
    result = generate_content(
        client,
        {"name": "盒子", "category": "家居日用", "attributes": {"颜色": "蓝色"}},
        "friendly",
    )
    assert result["marketing_line"] == MARKETING_LINES["friendly"]
    assert result["factual_copy"]["description"] == "明确事实"
    with pytest.raises(ValueError):
        generate_content(client, {}, "防水认证百年耐用")
    cases = boundary_cases()
    assert len(cases) == 30
    assert len({row["id"] for row in cases}) == 30
    assert {row["category"] for row in cases} == {"3C数码", "服饰穿搭", "家居日用"}


def test_attribute_name_cleanup_preserves_conflicting_values() -> None:
    """Verify that attribute name cleanup preserves conflicting values."""
    result = generate_content(
        StubClient(),
        {
            "name": "盒子",
            "category": "家居",
            "attributes": {"材质": "棉", " 材质 ": "塑料"},
        },
    )
    assert result["source"]["attributes"]["材质"] == ["棉", "塑料"]


def test_flagged_draft_does_not_receive_marketing_line() -> None:
    """Verify that flagged draft does not receive marketing line."""

    class FlaggedClient(StubClient):
        """Return a controlled rule-flagged response to exercise release blocking."""

        def generate_copy(self, product: dict) -> dict:
            """Return generate copy for this workflow.

            Args:
                product: Product used by this operation.

            Returns:
                Mapping with fields: _meta.
            """
            return {"_meta": {"release_status": "blocked_by_rules"}}

    result = generate_content(
        FlaggedClient(),
        {"name": "盒子", "category": "家居", "attributes": {"颜色": "白色"}},
        "friendly",
    )
    assert result["status"] == "blocked_by_rules"
    assert result["marketing_line"] is None
