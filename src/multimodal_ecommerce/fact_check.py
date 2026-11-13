"""Conservative copywriting checks; findings are not a semantic truth oracle."""

from __future__ import annotations

import re
from typing import Any

from multimodal_ecommerce.copy_policy import policy_findings, source_policy

NUMBER = re.compile(
    r"(?<![\d.])\d+(?:\.\d+)?\s*"
    + r"(?:个月|毫米|厘米|公斤|千克|毫升|小时|分钟|英寸|年|月|岁|天|米|寸|克|斤|升|"
    + r"mah|ghz|mhz|gb|tb|mb|hz|kg|mm|cm|ml|kw|w|v|g|%|％)?",
    re.IGNORECASE,
)
RANGE = re.compile(
    NUMBER.pattern + r"\s*(?:至|到|~|～|-)\s*" + NUMBER.pattern, re.IGNORECASE
)
CLAIMS = (
    "稳固",
    "耐用",
    "透气",
    "舒适",
    "防水",
    "防晒",
    "抗菌",
    "杀菌",
    "美白",
    "祛痘",
    "无毒",
    "安全",
    "环保",
    "纯棉",
    "真皮",
    "正品",
    "官方",
    "认证",
    "包邮",
    "保修",
    "最低",
    "第一",
    "顶级",
    "永久",
    "性能强劲",
    "快速",
    "高效",
    "节能",
)


def normalize_product(product: dict[str, Any]) -> dict[str, Any]:
    """Deduplicate repeated property values without merging ranges or units.

    Args:
        product: Product used by this operation.

    Returns:
        Mapping with fields: name, category, attributes.

    Raises:
        ValueError: Product and attributes must be JSON objects; Attribute values must
            be strings; Attribute values must be strings or lists of strings.
    """
    if not isinstance(product, dict) or not isinstance(
        product.get("attributes", {}), dict
    ):
        raise ValueError("Product and attributes must be JSON objects")
    attributes: dict[str, Any] = {}
    for key, value in product.get("attributes", {}).items():
        if isinstance(value, list):
            unique = []
            for item in value:
                if not isinstance(item, str):
                    raise ValueError("Attribute values must be strings")
                if item not in unique:
                    unique.append(item)
            attributes[key] = unique
        elif isinstance(value, str):
            attributes[key] = value
        else:
            raise ValueError("Attribute values must be strings or lists of strings")
    return {
        "name": str(product.get("name") or product.get("text") or ""),
        "category": str(product.get("category") or ""),
        "attributes": attributes,
    }


def _text(value: Any) -> str:
    if isinstance(value, dict):
        return "\n".join(_text(item) for item in value.values())
    if isinstance(value, list):
        return "\n".join(_text(item) for item in value)
    return str(value)


def numeric_tokens(text: str) -> set[str]:
    """Extract Arabic numeric literals with units, not substring matches.

    Args:
        text: Text input used by the operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return {
        re.sub(r"\s+", "", match.group()).casefold().replace("％", "%")
        for match in NUMBER.finditer(text)
    }


def check_copy(product: dict[str, Any], copy: dict[str, Any]) -> dict[str, Any]:
    """Flag unsupported numeric/unit claims, new ranges and selected risky terms.

    This lexical checker is deliberately conservative. It cannot establish
    entailment, verify supplier claims, or detect every hallucinated assertion.

    Args:
        product: Product used by this operation.
        copy: Copy used by this operation.

    Returns:
        Mapping with fields: status, human_review_required, findings,
        source_numeric_tokens, rule_version, source_policy_status, limitation.
    """
    normalized = normalize_product(product)
    source = _text(normalized)
    supported = numeric_tokens(source)
    findings = []
    fields = {"title": copy["title"], "description": copy["description"]}
    fields.update(
        {f"selling_points[{i}]": item for i, item in enumerate(copy["selling_points"])}
    )

    def canonical_range(value: str) -> str:
        """Return canonical range.

        Args:
            value: Input value to validate or transform.

        Returns:
            Text produced by this operation.
        """
        return re.sub(r"\s+", "", value).casefold().replace("％", "%")

    source_ranges = {canonical_range(match.group()) for match in RANGE.finditer(source)}
    if len(copy["description"]) > 100:
        findings.append(
            {
                "field": "description",
                "rule": "description_too_long",
                "claim": len(copy["description"]),
            }
        )
    for field, value in fields.items():
        for token in sorted(numeric_tokens(value) - supported):
            findings.append(
                {"field": field, "rule": "unsupported_number_or_unit", "claim": token}
            )
        for term in CLAIMS:
            if term in value and term not in source:
                findings.append(
                    {"field": field, "rule": "unsupported_claim_term", "claim": term}
                )
            elif term in value and any(
                prefix + term in source for prefix in ("不", "非", "未", "无")
            ):
                findings.append(
                    {
                        "field": field,
                        "rule": "source_negation_needs_review",
                        "claim": term,
                    }
                )
        for match in RANGE.finditer(value):
            if canonical_range(match.group()) not in source_ranges:
                findings.append(
                    {
                        "field": field,
                        "rule": "new_numeric_range",
                        "claim": match.group(),
                    }
                )
    policy = source_policy(normalized)
    findings.extend(policy_findings(policy, fields))
    if policy["status"] == "source_review_required":
        findings.append(
            {
                "field": "source",
                "rule": "source_conflict_requires_review",
                "claim": policy["findings"],
            }
        )
    return {
        "status": "flagged" if findings else "no_rule_flags",
        "human_review_required": True,
        "findings": findings,
        "source_numeric_tokens": sorted(supported),
        "rule_version": "lexical-v2-source-constraints",
        "source_policy_status": policy["status"],
        "limitation": "No flags does not establish factual correctness; lexical negation/option checks do not prove entailment or valid SKU combinations.",
    }
