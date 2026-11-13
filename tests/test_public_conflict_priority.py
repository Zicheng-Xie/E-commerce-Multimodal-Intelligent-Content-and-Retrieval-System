"""Public-only conflict abstention and stable complete candidate ordering."""

from __future__ import annotations

import pytest

from public_conflict_priority import conflict, descriptor, reorder


def public(product: str, text: str = "", unclear: bool = False) -> dict:
    """Return public for this workflow.

    Args:
        product: Product used by this operation.
        text: Text input used by the operation.
        unclear: Unclear used by this operation.

    Returns:
        Mapping with fields: product, appearance, visible_text, primary_unclear.
    """
    return dict(
        product=product, appearance="", visible_text=text, primary_unclear=unclear
    )


def test_models_and_materials_preserve_broad_business_rubric_and_unknowns_abstain() -> (
    None
):
    """Verify that models and materials preserve broad business rubric and unknowns
    abstain.
    """
    query = public("手机壳", "华为Mate10")
    assert not conflict(query, public("手机壳", "华为Mate20Pro"))[0]
    assert not conflict(query, public("手机壳", "手机壳水晶滴胶材料"))[0]
    assert conflict(query, public("头戴式耳机"))[0]
    assert not conflict(query, public("头戴式耳机", unclear=True))[0]
    assert not conflict(query, public("头戴式耳机", "手机壳"))[0]
    assert descriptor(public("化妆品", "专柜正品"), candidate=True)["family"] is None
    with pytest.raises(ValueError):
        conflict(query, dict(public("手机壳"), grade=1))


def test_stable_pool_priority_preserves_unknown_order_tail_and_complete_gallery() -> (
    None
):
    """Verify that stable pool priority preserves unknown order tail and complete
    gallery.
    """
    query = public("内胆包")
    profiles: dict = dict(
        wrong=public("手提包"),
        unknown=public("无法识别"),
        right=public("内胆包"),
        laptop=public("电脑内胆包"),
        tail=public("内胆包"),
    )
    original = ["wrong", "unknown", "right", "laptop", "tail"]
    ranked, evidence = reorder(original, query, profiles, pool=original[:4])
    assert ranked == ["unknown", "right", "wrong", "laptop", "tail"]
    assert set(ranked) == set(original) and evidence["all_items_retained"]
    assert not evidence["model_compatibility_hard_filter"]
    with pytest.raises(ValueError):
        reorder(original, query, profiles, pool=["right", "right"])
