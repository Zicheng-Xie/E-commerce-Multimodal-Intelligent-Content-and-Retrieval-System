"""Public compound nouns, props, abstention and full-tail scoring contracts."""

from __future__ import annotations

import numpy as np
import pytest

from primary_subject_ranker import (
    PrimarySubjectRankerGuardedOrderInputs,
    guarded_order,
    penalty,
    signature,
)


def test_compound_product_nouns_and_props() -> None:
    """Verify that compound product nouns and props."""
    assert signature("耳机盒保护套", "盒内白色耳机展示")["family"] == "audio_accessory"
    assert signature("笔记本电脑内胆包")["subtype"] == "laptop"
    assert signature("手袋内胆包")["subtype"] == "organizer"
    assert signature("枕头包")["family"] == "bag"
    assert signature("鞋收纳盒", "图中运动鞋")["family"] == "shoe_accessory"
    assert signature("鞋柜")["family"] == "cabinet"
    assert signature("耳机", "没有头梁")["subtype"] is None
    assert signature("外套")["subtype"] is None
    assert signature("学饮杯")["subtype"] is None


def test_public_conflicts_do_not_reward_or_use_age() -> None:
    """Verify that public conflicts do not reward or use age."""
    assert penalty(signature("头戴耳机"), signature("耳机盒保护套"))["value"] == 0.35
    assert penalty(signature("头戴耳机"), signature("单耳蓝牙耳机"))["value"] == 0.25
    assert penalty(signature("成人羽绒服"), signature("儿童羽绒服"))["value"] == 0
    assert (
        penalty(signature("短靴", "尖头细跟"), signature("短靴", "黑色粗跟"))["value"]
        == 0.12
    )
    assert penalty(signature("不明确"), signature("羽绒服"))["value"] == 0
    assert penalty(signature("有线耳机"), signature("头戴耳机"))["value"] == 0
    assert penalty(signature("外套"), signature("羽绒马甲"))["value"] == 0
    assert (
        penalty(signature("短靴", "厚底细跟"), signature("短靴", "细高跟"))["value"]
        == 0
    )


def test_guard_keeps_twenty_candidates_and_full_gallery() -> None:
    """Verify that guard keeps twenty candidates and full gallery."""
    ids = [f"doc{i:02d}" for i in range(25)]
    pool = ids[:20]
    docs: dict = {
        cid: dict(id=cid, product_type="头戴耳机" if i else "耳机保护套", appearance="")
        for i, cid in enumerate(ids)
    }
    base, raw = np.linspace(1, 0, 25), np.linspace(1, 0, 20)
    query: dict = dict(
        product="头戴耳机", appearance="大耳罩", visible_text="", primary_unclear=False
    )
    order, detail = guarded_order(
        PrimarySubjectRankerGuardedOrderInputs(ids, base, pool, raw, None, query, docs)
    )
    assert set(order) == set(ids) and set(order[:20]) == set(pool)
    assert order[20:] == ids[20:] and order[0] != ids[0]
    assert detail["conflicts"][ids[0]]["value"] == 0.35
    docs[ids[0]]["grade"] = 0
    with pytest.raises(ValueError, match="evaluator"):
        guarded_order(
            PrimarySubjectRankerGuardedOrderInputs(
                ids, base, pool, raw, None, query, docs
            )
        )
