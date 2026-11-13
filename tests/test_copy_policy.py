"""Regression checks for source ambiguity, negative facts and SKU claims."""

from __future__ import annotations

import httpx
import pytest

from multimodal_ecommerce.copy_policy import SourceConflictError, source_policy
from multimodal_ecommerce.fact_check import check_copy, normalize_product
from multimodal_ecommerce.ollama_client import OllamaClient


def copy_text(text: str) -> dict:
    """Build a minimal three-field response for checking a single assertion.

    Args:
        text: Text input used by the operation.

    Returns:
        Mapping with fields: title, description, selling_points.
    """
    return {
        "title": "商品",
        "description": text,
        "selling_points": ["商品", "商品", "商品"],
    }


@pytest.mark.parametrize("bad", ["配备童锁功能（无）", "具备童锁", "童锁消毒柜"])
def test_absent_child_lock_not_affirmed(bad: str) -> None:
    """Parenthesized absence does not excuse an affirmative claim.

    Args:
        bad: Bad used by this operation.
    """
    source: dict = {"attributes": {"是否有童锁": ["否"]}}
    result = check_copy(source, copy_text(bad))
    assert any(
        item["rule"] == "negative_attribute_not_preserved"
        for item in result["findings"]
    )
    assert check_copy(source, copy_text("无童锁"))["status"] == "no_rule_flags"


def test_nonremovable_hat_and_negative_foldability() -> None:
    """Keep the observed hat error and a generic boolean regression covered."""
    source: dict = {"attributes": {"是否带帽子": ["有帽不可拆"]}}
    assert check_copy(source, copy_text("可拆帽设计"))["status"] == "flagged"
    assert check_copy(source, copy_text("有帽不可拆"))["status"] == "no_rule_flags"
    source = {"attributes": {"是否可折叠": ["否"]}}
    assert check_copy(source, copy_text("不可折叠"))["status"] == "no_rule_flags"
    assert check_copy(source, copy_text("可折叠"))["status"] == "flagged"


def test_ram_is_an_option_not_universal() -> None:
    """Qualify option values in the same clause; do not match 8GB inside 128GB."""
    source: dict = {
        "attributes": {"运行内存RAM": ["8GB", "12GB"], "机身内存ROM": ["128GB"]}
    }
    assert any(
        item["rule"] == "option_value_presented_as_fixed"
        for item in check_copy(source, copy_text("配备12GB运行内存"))["findings"]
    )
    assert (
        check_copy(source, copy_text("运行内存可选8GB、12GB"))["status"]
        == "no_rule_flags"
    )
    assert check_copy(source, copy_text("128GB机身内存"))["status"] == "no_rule_flags"


@pytest.mark.parametrize(
    "source,code",
    [
        (
            {
                "name": "车载屏",
                "attributes": {
                    "屏幕尺寸": ["12.2英寸"],
                    "颜色分类": ["黑色15.6寸", "白色15.6寸"],
                },
            },
            "screen_size_option_conflict",
        ),
        (
            {"name": "Tory Burch手提包", "attributes": {"品牌": ["other/其他"]}},
            "unknown_brand_with_title_name_candidate",
        ),
        (
            {"name": "消毒柜", "attributes": {"是否有童锁": ["是", "否"]}},
            "contradictory_boolean",
        ),
    ],
)
def test_conflict_stops_before_model_request(source: dict, code: str) -> None:
    """Never ask the model to choose between contradictory source statements.

    Args:
        source: Source used by this operation.
        code: Code used by this operation.
    """
    calls = []

    def _record_transport(request: httpx.Request) -> httpx.Response:
        """Record a forbidden request without violating the transport interface."""
        calls.append(request)
        return httpx.Response(500)

    client = OllamaClient(
        client=httpx.Client(transport=httpx.MockTransport(_record_transport))
    )
    try:
        with pytest.raises(SourceConflictError) as caught:
            client.generate_copy(source)
        assert code in {item["code"] for item in caught.value.policy["findings"]}
        assert not calls
    finally:
        client.close()


def test_ambiguous_multivalue_not_assumed_to_be_skus() -> None:
    """Multiple source models are not silently made into validated choices."""
    source = normalize_product(
        {"attributes": {"型号": ["骁龙865", "红魔5S"], "颜色分类": ["红色", "蓝色"]}}
    )
    policy = source_policy(source)
    assert "型号" in policy["generation_input"]["omitted_attributes"]
    assert "颜色分类" in policy["generation_input"]["variant_attributes"]
    assert source["attributes"]["型号"] == ["骁龙865", "红魔5S"]


def test_different_numeric_attribute_and_supported_scalar_are_not_false_flags() -> None:
    """Do not confuse fabric percentages with height or discard corroboration."""
    source: dict = {
        "attributes": {
            "参考身高": ["100", "110"],
            "材质": ["聚酰胺纤维100%"],
            "型号": ["骁龙865", "红魔5S"],
            "CPU型号": ["骁龙865"],
        }
    }
    assert (
        check_copy(source, copy_text("聚酰胺纤维100%，CPU型号骁龙865"))["status"]
        == "no_rule_flags"
    )


def test_source_range_spaces_and_correct_negation() -> None:
    """Do not flag spacing around a supported interval or an explicit '非'."""
    source: dict = {"attributes": {"价格区间": ["101-200元"], "是否商场同款": ["否"]}}
    assert (
        check_copy(source, copy_text("来源价格区间101-200 元，非商场同款"))["status"]
        == "no_rule_flags"
    )


def test_option_qualifier_does_not_authorize_cross_field_sku() -> None:
    """Two marginal option lists are not evidence of their Cartesian product."""
    source: dict = {
        "attributes": {
            "运行内存RAM": ["8GB", "12GB"],
            "机身内存ROM": ["128GB", "256GB"],
        }
    }
    result = check_copy(source, copy_text("可选8GB运行内存加256GB机身存储的组合"))
    assert any(
        item["rule"] == "unverified_option_combination" for item in result["findings"]
    )


def test_single_sku_can_also_conflict_with_scalar() -> None:
    """Screen conflict checks must not depend on the number of color options."""
    source = normalize_product(
        {"attributes": {"屏幕尺寸": ["12.2英寸"], "颜色分类": ["黑色15.6寸"]}}
    )
    assert source_policy(source)["status"] == "source_review_required"
