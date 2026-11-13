"""Conservative source/claim guards layered over the immutable v3 generator.

These targeted rules flag evidence gaps for review; absence of flags is not a
semantic factuality certificate. SKU conflicts never create inferred SKU tuples.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

GUARD_VERSION = "source-and-claims-v2"
VARIANT_KEYS = ("颜色分类", "颜色", "套餐类型", "套餐")


def values(attributes: dict, key: str) -> list[str]:
    """Accept both normalized list attributes and scalar string inputs.

    Args:
        attributes: Attributes used by this operation.
        key: Key used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    raw = attributes.get(key, [])
    return [raw] if isinstance(raw, str) else raw


def _text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).lower()


def _affirmed(text: str, pattern: str) -> bool:
    """Negated material/function words must not unlock a positive claim."""
    return any(
        not re.search(r"非|不|无|未", text[max(0, match.start() - 4) : match.start()])
        for match in re.finditer(pattern, text)
    )


def _numbers(text: str, kind: str) -> list[float]:
    text = _text(text)
    if kind == "capacity_ml":
        return [
            float(number) * (1000 if unit in {"l", "升"} else 1)
            for number, unit in re.findall(
                r"(\d+(?:\.\d+)?)\s*(ml|毫升|l|升)(?![a-z])", text
            )
        ]
    if kind == "quantity":
        return [float(n) for n in re.findall(r"(\d+)\s*只", text)]
    if kind == "ram_gb":
        return [float(n) for n in re.findall(r"(\d+)\s*g(?:b)?(?:\s*内存)?", text)]
    return []


def check_source_consistency(product: dict) -> dict[str, Any]:
    """Check explicit fixed parameters against recognizable SKU descriptions.

    Multi-valued fixed parameters are outside this narrow check. A conflicting
    option means the global value cannot safely describe every option; it does
    not prove either source value false. Missing units are never guessed.

    Args:
        product: Product used by this operation.

    Returns:
        Mapping with fields: version, status, issues.
    """
    attrs = product.get("attributes", {})
    issues = []
    for field, kind in (
        ("容量", "capacity_ml"),
        ("只数", "quantity"),
        ("内存容量", "ram_gb"),
    ):
        fixed_values = values(attrs, field)
        if len(fixed_values) != 1:
            continue
        fixed = fixed_values[0]
        numbers = _numbers(fixed, kind)
        is_range = bool(re.search(r"[-~～至到]", _text(fixed)))
        if len(numbers) == 1:
            lower = upper = numbers[0]
        elif len(numbers) == 2 and is_range:
            lower, upper = numbers
            if lower > upper:
                continue
        else:
            continue
        for variant_field in VARIANT_KEYS:
            for option in values(attrs, variant_field):
                # Storage and video-memory capacities are not system RAM.
                option_text = _text(option)
                if kind == "ram_gb":
                    ram_matches = re.findall(r"(\d+)\s*g(?:b)?\s*内存", option_text)
                    option_numbers = [float(n) for n in ram_matches]
                else:
                    option_numbers = _numbers(option, kind)
                contradictory = [n for n in option_numbers if not lower <= n <= upper]
                if contradictory:
                    issues.append(
                        {
                            "code": "fixed_parameter_vs_sku_option",
                            "field": field,
                            "fixed_value": fixed,
                            "normalized_interval": [lower, upper],
                            "unit": kind,
                            "variant_field": variant_field,
                            "variant_value": option,
                            "outside_values": contradictory,
                            "action": "require_source_or_selected_sku_confirmation",
                        }
                    )
    # Only explicit RTX/GTX family codes are compared. Bare model digits such
    # as 3060Z8 are ambiguous and are deliberately not treated as GPU evidence.
    for field in ("显卡类型", "显卡芯片"):
        fixed_values = values(attrs, field)
        if len(fixed_values) != 1:
            continue
        fixed_gpu = re.findall(
            r"(?:rtx|gtx)\s*\d{3,4}(?:\s*ti)?", _text(fixed_values[0])
        )
        if len(fixed_gpu) != 1:
            continue
        for variant_field in VARIANT_KEYS:
            for option in values(attrs, variant_field):
                explicit = re.findall(r"(?:rtx|gtx)\s*\d{3,4}(?:\s*ti)?", _text(option))
                if any(
                    re.sub(r"\s", "", x) != re.sub(r"\s", "", fixed_gpu[0])
                    for x in explicit
                ):
                    issues.append(
                        {
                            "code": "fixed_gpu_vs_sku_option",
                            "field": field,
                            "fixed_value": fixed_values[0],
                            "variant_field": variant_field,
                            "variant_value": option,
                            "action": "require_source_or_selected_sku_confirmation",
                        }
                    )
    return {
        "version": GUARD_VERSION,
        "status": "needs_source_review" if issues else "no_targeted_conflict_found",
        "issues": issues,
    }


def check_claim_support(product: dict, copy: dict) -> dict[str, Any]:
    """Flag selected high-risk assertions lacking structured-attribute support.

    A merchant title is not authoritative evidence for material/purity/function
    claims under the frozen v3 source policy. This is a targeted lexical guard,
    not comprehensive entailment or a human relevance/fluency assessment.

    Args:
        product: Product used by this operation.
        copy: Copy used by this operation.

    Returns:
        Mapping with fields: version, status, issues.
    """
    attrs = product.get("attributes", {})
    material = " ".join(
        v
        for k in attrs
        if any(t in k for t in ("材质", "材料", "成分", "面料"))
        for v in values(attrs, k)
    )
    seasons = " ".join(v for k in attrs if "季节" in k for v in values(attrs, k))
    power_control = " ".join(
        v for k in attrs if "调节" in k or "功能" in k for v in values(attrs, k)
    )
    rules = [
        (
            "cotton_purity",
            r"纯棉|全棉|100\s*%.{0,4}棉",
            _affirmed(material, r"纯棉|全棉|100\s*%.{0,4}棉|棉.{0,4}100\s*%"),
            "structured material/purity",
        ),
        (
            "aluminium_alloy",
            r"铝合金",
            _affirmed(material, r"铝合金"),
            "structured material",
        ),
        (
            "african_rosewood",
            r"非洲酸枝木?",
            _affirmed(material, r"非洲酸枝"),
            "structured material provenance",
        ),
        (
            "solid_wood",
            r"全实木",
            _affirmed(material, r"全实木|100\s*%实木"),
            "structured material",
        ),
        (
            "adjustable_power",
            r"功率调节|调节功率",
            bool(re.search(r"功率.{0,4}(?:可调|调节)|调功", power_control)),
            "explicit power adjustment function, not a rated power interval",
        ),
        (
            "all_seasons",
            r"四季可用|四季通用",
            "四季" in seasons or all(s in seasons for s in "春夏秋冬"),
            "structured applicable seasons",
        ),
        (
            "comparative_softness",
            r"比[^，。；、]{1,10}?更?柔软",
            False,
            "independently confirmed comparative performance evidence",
        ),
    ]
    output_fields = {k: copy.get(k, "") for k in ("title", "description")}
    output_fields.update(
        {f"selling_points/{i}": v for i, v in enumerate(copy.get("selling_points", []))}
    )
    issues = []
    for field, text in output_fields.items():
        for code, pattern, supported, required in rules:
            if not supported:
                for match in re.finditer(pattern, text):
                    if code == "cotton_purity" and re.search(
                        r"比$|较$", text[max(0, match.start() - 2) : match.start()]
                    ):
                        # A comparison to cotton does not assert this product
                        # is cotton. Its performance claim has a separate rule.
                        continue
                    issues.append(
                        {
                            "code": code,
                            "output_field": field,
                            "quote": match.group(),
                            "required_evidence": required,
                            "action": "require_review_or_remove_unsupported_claim",
                        }
                    )
    return {
        "version": GUARD_VERSION,
        "status": "needs_claim_review" if issues else "no_targeted_gap_found",
        "issues": issues,
    }
