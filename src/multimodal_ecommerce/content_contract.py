"""Separate factual copy from a small approved subjective marketing channel."""

from __future__ import annotations

import logging
from typing import Any

from multimodal_ecommerce.content_guard import (
    check_claim_support,
    check_source_consistency,
)
from multimodal_ecommerce.copy_policy import SourceConflictError
from multimodal_ecommerce.fact_check import normalize_product
from multimodal_ecommerce.model_interfaces import CopyGenerator
from multimodal_ecommerce.ollama_client import OllamaError

MARKETING_LINES: dict[str, str | None] = {
    "neutral": None,
    "concise": "查看商品信息，按需选购。",
    "friendly": "为日常挑选一份新意。",
}
CONTRACT_VERSION = "facts-and-marketing-v2"


def generate_content(
    client: CopyGenerator, product: dict, style: str = "neutral"
) -> dict[str, Any]:
    """Preserve the frozen factual generator; never mix marketing into parameters.

    Missing optional attributes are omitted from facts, never defaulted. Missing
    name/category/all attributes causes abstention before an inference request.
    Approved marketing lines have no material, numerical or functional claims.

    Args:
        client: Configured local model client used for inference.
        product: Product used by this operation.
        style: Style used by this operation.

    Returns:
        Mapping with fields: status, source_policy, error.

    Raises:
        ValueError: Unknown controlled marketing style; Attribute names must be strings.
    """
    if style not in MARKETING_LINES:
        raise ValueError("Unknown controlled marketing style")
    normalized = normalize_product(product)
    normalized["name"] = normalized["name"].strip()
    normalized["category"] = normalized["category"].strip()
    attributes: dict[str, list[str]] = {}
    for raw_key, raw in normalized["attributes"].items():
        if not isinstance(raw_key, str):
            raise ValueError("Attribute names must be strings")
        key = raw_key.strip()
        values = raw if isinstance(raw, list) else [raw]
        values = [value.strip() for value in values if value.strip()]
        if key and values:
            # Whitespace-normalized names can collide: retain all facts, never
            # silently overwrite one conflicting source with the other.
            attributes[key] = list(dict.fromkeys(attributes.get(key, []) + values))
    normalized["attributes"] = attributes
    missing = [key for key in ("name", "category", "attributes") if not normalized[key]]
    result: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "source": normalized,
        "style": style,
        "factual_copy": None,
        "marketing_line": None,
        "missing_required_fields": missing,
        "omission_policy": "Unknown optional attributes are omitted; never imputed or completed from world knowledge.",
    }
    if missing:
        return {**result, "status": "abstained_missing_required_fields"}
    result["source_consistency"] = check_source_consistency(normalized)
    if result["source_consistency"]["issues"]:
        return {**result, "status": "blocked_source_conflict"}
    try:
        copy = client.generate_copy(normalized)
    except SourceConflictError as error:
        logging.getLogger(__name__).debug(
            "Handled exception in generate_content", exc_info=True
        )
        return {
            **result,
            "status": "blocked_source_conflict",
            "source_policy": error.policy,
        }
    except (OllamaError, ValueError) as error:
        logging.getLogger(__name__).debug(
            "Handled exception in generate_content", exc_info=True
        )
        return {**result, "status": "generation_failed", "error": str(error)}
    result.update({"factual_copy": copy, "status": copy["_meta"]["release_status"]})
    result["claim_support"] = check_claim_support(normalized, copy)
    if result["claim_support"]["issues"]:
        result["status"] = "blocked_by_contract_rules"
    if result["status"] == "draft_requires_human_review":
        result["marketing_line"] = MARKETING_LINES[style]
    return result


def boundary_cases() -> list[dict]:
    """Author 30 explicit synthetic cases, distinct from the 90 real test items.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    sources: list[dict[str, Any]] = [
        {
            "name": "便携音箱",
            "category": "3C数码",
            "attributes": {
                "品牌": ["示例甲"],
                "材质": ["塑料"],
                "尺寸": ["10厘米"],
                "颜色": ["黑色"],
                "连接方式": ["有线连接"],
                "是否支持蓝牙": ["否"],
            },
        },
        {
            "name": "短袖上衣",
            "category": "服饰穿搭",
            "attributes": {
                "品牌": ["示例乙"],
                "材质": ["棉"],
                "尺寸": ["66厘米"],
                "颜色": ["蓝色"],
                "是否可拆卸": ["否"],
            },
        },
        {
            "name": "收纳盒",
            "category": "家居日用",
            "attributes": {
                "品牌": ["示例丙"],
                "材质": ["塑料"],
                "尺寸": ["20厘米"],
                "颜色": ["白色"],
                "是否可折叠": ["否"],
            },
        },
    ]
    scenarios = [
        "complete",
        "missing_brand",
        "missing_material",
        "missing_dimensions",
        "missing_functions",
        "all_attributes_missing",
        "missing_name",
        "missing_category",
        "conflicting_boolean",
        "multi_sku",
    ]
    cases = []
    for source_index, source in enumerate(sources):
        for number, scenario in enumerate(scenarios):
            product: dict[str, Any] = {
                **source,
                "attributes": {
                    key: list(value) for key, value in source["attributes"].items()
                },
            }
            removed = []
            for code, fields in {
                "missing_brand": ["品牌"],
                "missing_material": ["材质"],
                "missing_dimensions": ["尺寸"],
                "missing_functions": [
                    key
                    for key in product["attributes"]
                    if "是否" in key or key == "连接方式"
                ],
            }.items():
                if scenario == code:
                    for field in fields:
                        removed.append(
                            {"field": field, "value": product["attributes"].pop(field)}
                        )
            if scenario == "all_attributes_missing":
                product["attributes"] = {}
            elif scenario == "missing_name":
                product["name"] = ""
            elif scenario == "missing_category":
                product["category"] = ""
            elif scenario == "conflicting_boolean":
                key = next(key for key in product["attributes"] if "是否" in key)
                product["attributes"][key] = ["是", "否"]
            elif scenario == "multi_sku":
                product["attributes"]["颜色"] = ["黑色", "白色"]
                product["attributes"]["尺寸"] = ["10厘米", "20厘米"]
            expected = (
                "abstained_missing_required_fields"
                if scenario
                in {"all_attributes_missing", "missing_name", "missing_category"}
                else (
                    "blocked_source_conflict"
                    if scenario == "conflicting_boolean"
                    else "generation_attempt"
                )
            )
            cases.append(
                {
                    "id": f"boundary/{source_index + 1}/{scenario}",
                    "category": source["category"],
                    "scenario": scenario,
                    "product": product,
                    "style": list(MARKETING_LINES)[number % 3],
                    "removed_facts": removed,
                    "expected_behavior": expected,
                    "data_origin": "hand-authored synthetic input, not an observed product or human rating",
                }
            )
    return cases
