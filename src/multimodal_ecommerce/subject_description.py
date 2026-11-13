"""Separate explicit scene props from the primary product in image-query expansion."""

from __future__ import annotations

import re

from multimodal_ecommerce.intent_retrieval import attributes, public_entity
from multimodal_ecommerce.public_conflict_guard import COMPATIBLE, public_conflict
from multimodal_ecommerce.retrieval_safeguards import compound_caption_conflict

PROP_MARKERS = ("脚穿", "脚上", "手持", "内搭", "背景", "旁边", "身后", "身旁", "后方")
FOOTWEAR = frozenset({"other_shoes", "sports_shoes"})


def compatible(kind: str | None, other: str | None) -> bool:
    """Compare recognized public entity families, abstaining on missing types.

    Args:
        kind: Primary product entity.
        other: Candidate or clause entity.

    Returns:
        Whether both nouns identify the same compatible product family.
    """
    return bool(
        kind
        and other
        and (kind == other or any({kind, other} <= family for family in COMPATIBLE))
    )


def clean_caption(caption: dict) -> dict:
    """Remove explicitly marked foreign props without rewriting the primary product.

    Args:
        caption: Query-only product, appearance and unclear fields.

    Returns:
        Original/filtered descriptions and public reasons for every removed clause.
    """
    if (
        set(caption) != {"product", "appearance", "unclear"}
        or type(caption["unclear"]) is not bool
    ):
        raise ValueError("A complete original query-only caption is required")
    kind = public_entity(caption["product"])
    abstain = caption["unclear"] or compound_caption_conflict(caption)
    kept, removed = [], []
    for clause in re.split(r"[，,。；;]", caption["appearance"]):
        clause = clause.strip()
        if not clause:
            continue
        other = public_entity(clause)
        marked = any(m in clause for m in PROP_MARKERS)
        footwear_prop = (
            any(m in clause for m in ("脚穿", "脚上")) and kind not in FOOTWEAR
        )
        drop = (
            not abstain
            and kind
            and marked
            and not compatible(kind, other)
            and (other or footwear_prop)
        )
        if drop:
            removed.append(
                dict(clause=clause, type=other, reason="explicit_foreign_display_prop")
            )
        else:
            kept.append(clause)
    cleaned = dict(caption, appearance="，".join(kept))
    return dict(
        original=caption,
        cleaned=cleaned,
        query_type=kind,
        removed=removed,
        removed_types=sorted({r["type"] for r in removed if r["type"]}),
        changed=bool(removed),
    )


def prop_conflict_ids(
    order: list[str], titles: dict[str, str], observation: dict
) -> list[str]:
    """Identify explicit returned prop types using public titles, never annotations.

    Args:
        order: Complete selected baseline order.
        titles: Original public titles.
        observation: Query-only cleaning observation.

    Returns:
        Unprotected Top10 products explicitly matching removed foreign prop types.
    """
    kind = observation["query_type"]
    prop_types = set(observation["removed_types"])
    return [
        cid
        for cid in order[2:10]
        if (other := public_conflict(kind, "", titles[cid])["title_type"]) in prop_types
        and not compatible(kind, other)
    ]


def subject_swap(inputs: dict) -> tuple[list[str], dict]:
    """Replace at most one explicit display prop under a fixed soft or hard constraint.

    Args:
        inputs: Baseline, scored pool, titles, query observation and optional type_veto.

    Returns:
        Full preserved candidate universe and the actual optional swap trace.
    """
    order, pool, titles, scores, observation = (
        inputs[k] for k in ("order", "pool", "titles", "scores", "observation")
    )
    if (
        len(order) != len(set(order))
        or len(pool) != 20
        or len(set(pool)) != 20
        or pool[:10] != order[:10]
        or not set(pool) <= set(order)
        or set(scores) != set(pool)
    ):
        raise ValueError(
            "Original Top10 anchors and a unique, completely scored Top20 required"
        )
    import math

    if not all(math.isfinite(v) for v in scores.values()):
        raise ValueError("Actual finite logits are required")
    kind = observation["query_type"]
    veto = inputs.get("type_veto", False)
    if type(veto) is not bool:
        raise ValueError("Explicit Boolean type-veto configuration required")
    outgoing = prop_conflict_ids(order, titles, observation)
    incoming = [
        c
        for c in pool[10:]
        if compatible(kind, public_conflict(kind, "", titles[c])["title_type"])
    ]
    if veto:
        words = (
            observation["cleaned"]["product"]
            + " "
            + observation["cleaned"]["appearance"]
        )
        incoming = [
            cid for cid in incoming if not attributes(words, titles[cid])["conflicts"]
        ]
    result = list(order)
    trace: dict = dict(changed=False, incoming=None, outgoing=None)
    if not outgoing or not incoming:
        return result, trace
    worst = min(outgoing, key=lambda c: (scores[c], -order.index(c)))
    best = max(incoming, key=lambda c: (scores[c], -pool.index(c)))
    if veto or scores[best] > scores[worst]:
        i, j = order.index(worst), order.index(best)
        result[i], result[j] = result[j], result[i]
        trace = dict(
            changed=True,
            incoming=best,
            outgoing=worst,
            incoming_old_rank=j + 1,
            target_rank=i + 1,
            model_logit_gap=scores[best] - scores[worst],
            type_veto=veto,
        )
    return result, trace
