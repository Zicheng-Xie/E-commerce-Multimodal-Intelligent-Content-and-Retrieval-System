"""Regression cases for factual errors observed in real model output."""

from __future__ import annotations

from multimodal_ecommerce.fact_check import check_copy, normalize_product


def _copy(text: str) -> dict:
    return {
        "title": "儿童书架",
        "description": text,
        "selling_points": ["书架", "收纳", "家用"],
    }


def test_age_typo_and_unsupported_durability() -> None:
    """20 months cannot authorize 2 months or an invented durability claim."""
    product: dict = {
        "name": "儿童书架",
        "attributes": {"年龄": ["20个月", "2岁", "9岁"]},
    }
    result = check_copy(product, _copy("适用于2个月至9岁，稳固耐用"))
    rules = {finding["rule"] for finding in result["findings"]}
    assert {
        "unsupported_number_or_unit",
        "new_numeric_range",
        "unsupported_claim_term",
    } <= rules
    assert any(finding["claim"] == "2个月" for finding in result["findings"])


def test_literal_facts_do_not_imply_verified_quality() -> None:
    """Rule-clean answers still require human review."""
    product: dict = {"name": "儿童书架", "attributes": {"年龄": ["20个月", "9岁"]}}
    result = check_copy(product, _copy("适用年龄包括20个月、9岁"))
    assert result["status"] == "no_rule_flags"
    assert result["human_review_required"] is True


def test_deduplicate_without_range_conversion() -> None:
    """Deduplication preserves original units and the input object."""
    source: dict = {"attributes": {"年龄": ["20个月", "9岁", "20个月"]}}
    normalized = normalize_product(source)
    assert normalized["attributes"]["年龄"] == ["20个月", "9岁"]
    assert source["attributes"]["年龄"] == ["20个月", "9岁", "20个月"]
