"""Stable Top20 conflict handling using public product words, never relevance labels."""

from __future__ import annotations

from multimodal_ecommerce.business_constraints import canonical
from multimodal_ecommerce.intent_retrieval import attributes, public_entity

COMPATIBLE = (
    frozenset({"bag", "phone_bag", "computer_bag", "camera_bag", "storage_bag"}),
    frozenset({"chair", "stool"}),
    frozenset({"outerwear", "knitwear"}),
    frozenset({"sports_shoes", "other_shoes"}),
)
EXPLICIT_CONFLICTS = {
    "phone_case": frozenset({"phone_mount", "power_bank", "charger", "cable"}),
    "t_shirt": frozenset({"trousers", "shorts", "skirt"}),
    "trousers": frozenset({"shorts", "skirt"}),
    "chair": frozenset({"other_shoes", "sports_shoes", "pillow", "chair_accessory"}),
    "bag": frozenset({"pillow", "headphone_case", "other_shoes", "sports_shoes"}),
    "valve_cap": frozenset({"phone_mount", "phone_case", "headphone"}),
}


def _entity(text: str) -> str | None:
    value = canonical(text)
    # A battery case is sold for its power function, not automatically a plain case.
    if any(term in value for term in ("背夹电池", "背夹式电池", "背甲电池")):
        return "power_bank"
    if any(term in value for term in ("气嘴帽", "气门帽", "气嘴盖")):
        return "valve_cap"
    if any(term in value for term in ("枕头包", "圆桶包", "波士顿包")):
        return "bag"
    return public_entity(value)


def _intent(text: str | None, caption: dict | None) -> tuple[str | None, str]:
    if text:
        # Multi-product disjunctions keep the baseline rather than forcing one noun.
        if any(word in text for word in ("或", "以及", "、")):
            return None, text
        return _entity(text), text
    if not caption or caption.get("unclear"):
        return None, ""
    product = caption["product"]
    if product.endswith("包"):
        kind = _entity(product)
        return (kind if kind in COMPATIBLE[0] else "bag"), product
    return _entity(product), product


def public_conflict(query_type: str | None, query_text: str, title: str) -> dict:
    """Describe explicit title conflicts without making a relevance judgment.

    Args:
        query_type: Public product noun inferred from query input alone.
        query_text: Original query words or query-only product caption.
        title: Unmodified candidate title, without annotations or source categories.

    Returns:
        Public noun and attribute evidence; uncertainty abstains from a type penalty.
    """
    kind = _entity(title)
    compatible = query_type == kind or any(
        {query_type, kind} <= family for family in COMPATIBLE
    )
    attr = attributes(query_text, title)
    conflict = bool(query_type and kind and not compatible)
    explicit = kind in EXPLICIT_CONFLICTS.get(query_type or "", frozenset())
    actionable_attributes = [
        a for a in attr["conflicts"] if a in {"long_trousers", "short_sleeve"}
    ]
    return dict(
        query_type=query_type,
        title_type=kind,
        type_conflict=conflict,
        attribute_conflicts=attr["conflicts"],
        conflict=conflict or bool(attr["conflicts"]),
        explicit_conflict=explicit or bool(actionable_attributes),
        source="query input and public title only; not human relevance truth",
    )


def reorder_public_conflicts(
    order: list[str],
    titles: dict[str, str],
    text: str | None = None,
    caption: dict | None = None,
    options: dict | None = None,
) -> tuple[list[str], dict]:
    """Demote explicit conflicts within the existing Top20 and preserve all candidates.

    Args:
        order: Complete existing eligible product order, independent of labels.
        titles: Public titles keyed by exactly the eligible product IDs.
        text: Original text query, when this is the text branch.
        caption: Existing query-only image description, when this is the image branch.
        options: Protected prefix (0 or 2), broad control flag, and matched-only flag.

    Returns:
        Full stable-partition order and public evidence for the original Top20.

    Raises:
        ValueError: Candidate coverage or protected-prefix configuration is invalid.
    """
    policy = dict(protect=2, broad=False, matched_only=True)
    if options is not None:
        if set(options) - set(policy):
            raise ValueError("Unknown conflict policy option")
        policy.update(options)
    protect, broad, matched_only = (
        policy[k] for k in ("protect", "broad", "matched_only")
    )
    if (
        len(order) < 20
        or len(order) != len(set(order))
        or set(order) != set(titles)
        or type(protect) is not int
        or protect not in (0, 2)
        or type(broad) is not bool
        or type(matched_only) is not bool
    ):
        raise ValueError("Unique full candidate coverage and protect=0 or 2 required")
    kind, words = _intent(text, caption)
    evidence = {cid: public_conflict(kind, words, titles[cid]) for cid in order[:20]}
    if kind is None:
        return list(order), dict(
            applied=False, reason="ambiguous_query", evidence=evidence
        )
    field = "conflict" if broad else "explicit_conflict"
    pool = order[protect:20]
    if matched_only:
        replacements = [
            cid
            for cid in order[10:20]
            if not evidence[cid][field]
            and (
                evidence[cid]["title_type"] == kind
                or any(
                    {kind, evidence[cid]["title_type"]} <= family
                    for family in COMPATIBLE
                )
            )
        ]
        revised = list(order)
        positions = {cid: i for i, cid in enumerate(order)}
        swaps = []
        for index in range(protect, 10):
            if evidence[order[index]][field] and replacements:
                incoming = replacements.pop(0)
                swaps.append(
                    dict(outgoing=order[index], incoming=incoming, position=index + 1)
                )
                revised[index], revised[positions[incoming]] = incoming, order[index]
        return revised, dict(
            applied=revised != order, protect=protect, swaps=swaps, evidence=evidence
        )
    revised = (
        order[:protect]
        + [cid for cid in pool if not evidence[cid][field]]
        + [cid for cid in pool if evidence[cid][field]]
        + order[20:]
    )
    return revised, dict(applied=revised != order, protect=protect, evidence=evidence)
