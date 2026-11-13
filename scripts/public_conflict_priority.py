"""Stable public primary-item conflict priority; unknowns preserve neural order."""

from __future__ import annotations

from primary_subject_ranker import penalty, signature
from subject_attribute_ranker import public_descriptor


def descriptor(public: dict, *, candidate: bool = False) -> dict:
    """Return descriptor for this workflow.

    Args:
        public: Public used by this operation.
        candidate: Candidate used by this operation.

    Returns:
        Result mapping produced by the workflow.
    """
    value = public_descriptor(public)
    if (
        value["family"] == "craft_material"
        and value["subtype"] == "phone_case_material"
    ):
        # Preserve the existing broad business-relevance rubric; materials and
        # phone-model disagreements are diagnostics, not new relevance labels.
        value.update(family="phone_case", subtype=None)
    if candidate and not public["primary_unclear"]:
        title = signature(public["visible_text"])
        if title["family"] == "cabinet" and not any(
            word in public["visible_text"]
            for word in (
                "衣柜",
                "鞋柜",
                "储物柜",
                "收纳柜",
                "斗柜",
                "餐边柜",
                "展示柜",
                "抽屉柜",
            )
        ):
            title = dict(family=None, subtype=None, structures=[])
        if value["family"] is None and title["family"] is not None:
            value.update(**title)
        elif value["family"] is not None and title["family"] is not None:
            if value["family"] != title["family"]:
                value.update(family=None, subtype=None, structures=[])
            elif value["subtype"] is None:
                value["subtype"] = title["subtype"]
            elif title["subtype"] is not None and value["subtype"] != title["subtype"]:
                value["subtype"] = None
    return {k: value[k] for k in ("family", "subtype", "structures")}


def conflict(query: dict, candidate: dict) -> tuple[bool, dict]:
    """Return conflict for this workflow.

    Args:
        query: Query record or representation used by this operation.
        candidate: Candidate used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    q, d = descriptor(query), descriptor(candidate, candidate=True)
    evidence = penalty(q, d)
    return bool(evidence["value"] > 0), dict(
        query=q, candidate=d, reason=evidence["reason"]
    )


def reorder(
    original: list[str], query: dict, profiles: dict, *, pool: list[str] | None = None
) -> tuple[list, dict]:
    """Return reorder for this workflow.

    Args:
        original: Original used by this operation.
        query: Query record or representation used by this operation.
        profiles: Profiles used by this operation.
        pool: Pool used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Unique complete public ranking required; Aligned original candidate
            pool required.
    """
    if len(original) != len(set(original)) or not set(original) <= set(profiles):
        raise ValueError("Unique complete public ranking required")
    selected = (
        list(original)
        if pool is None
        else [cid for cid in original if cid in set(pool)]
    )
    if pool is not None and (len(pool) != len(set(pool)) or set(selected) != set(pool)):
        raise ValueError("Aligned original candidate pool required")
    flags = {}
    evidence = {}
    for cid in selected:
        flags[cid], evidence[cid] = conflict(query, profiles[cid])
    head = sorted(selected, key=lambda cid: flags[cid])
    tail = [] if pool is None else [cid for cid in original if cid not in set(pool)]
    return head + tail, dict(
        public_conflicts=flags,
        evidence=evidence,
        all_items_retained=True,
        neural_relative_order_preserved_within_tiers=True,
        model_compatibility_hard_filter=False,
    )
