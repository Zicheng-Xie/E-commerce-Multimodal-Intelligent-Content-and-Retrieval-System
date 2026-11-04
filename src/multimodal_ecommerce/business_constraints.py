"""Query-only product intent and conservative public-evidence ranking constraints."""

from __future__ import annotations

import re
import unicodedata

import numpy as np

ENTITY_RULES = [
    (
        "headphone_case",
        [
            "耳机保护套",
            "耳机保护壳",
            "耳机套",
            "耳机壳",
            "耳机收纳盒",
            "耳机收纳包",
            "耳机充电仓",
            "耳机充电盒",
            "耳塞套",
        ],
    ),
    ("phone_case", ["手机保护套", "手机保护壳", "手机壳", "手机套"]),
    ("keyboard_accessory", ["键帽", "键盘膜", "键盘保护", "键盘支架"]),
    ("mouse_pad", ["鼠标垫", "键盘垫", "桌垫"]),
    ("cup_accessory", ["杯垫", "杯盖", "杯架"]),
    ("pillow_case", ["枕套", "抱枕套"]),
    ("shorts", ["短裤", "热裤", "五分裤"]),
    ("skirt", ["半身裙"]),
    ("generic_accessory", ["配件", "保护套", "保护壳", "挂绳", "挂件", "耳罩", "鞋垫"]),
    ("headphone", ["耳机", "耳麦", "耳塞"]),
    ("keyboard", ["键盘"]),
    ("mouse", ["鼠标"]),
    ("power_bank", ["充电宝", "移动电源"]),
    ("charger", ["充电器"]),
    ("cable", ["数据线", "充电线"]),
    ("laptop", ["笔记本电脑", "手提电脑", "笔记本计算机"]),
    ("notebook", ["记事本", "文具本", "日记本"]),
    ("phone", ["手机"]),
    ("t_shirt", ["t恤", "t-shirt", "tshirt"]),
    ("dress", ["连衣裙"]),
    ("shirt", ["衬衫"]),
    ("trousers", ["长裤", "牛仔裤", "休闲裤", "运动裤", "西裤"]),
    ("sports_shoes", ["运动鞋", "跑步鞋", "篮球鞋", "网球鞋", "球鞋"]),
    ("other_shoes", ["皮鞋", "凉鞋", "拖鞋", "高跟鞋", "靴"]),
    ("cup", ["水杯", "保温杯", "玻璃杯", "吸管杯", "马克杯", "杯子"]),
    ("pillow", ["枕头", "枕芯"]),
    ("cabinet", ["柜"]),
    ("shelf", ["置物架", "鞋架", "书架"]),
    ("table_lamp", ["台灯"]),
    ("other_lamp", ["吊灯", "壁灯"]),
    ("table", ["桌", "茶几"]),
    ("chair", ["椅"]),
    ("quilt", ["被子", "被芯", "蚕丝被", "棉花被", "羽绒被"]),
    ("bag", ["背包", "双肩包", "书包", "手提包"]),
    ("umbrella", ["雨伞", "遮阳伞"]),
]
ACCESSORY_INTENTS = {
    "headphone_case",
    "phone_case",
    "keyboard_accessory",
    "mouse_pad",
    "cup_accessory",
    "pillow_case",
    "generic_accessory",
}


def canonical(text: str) -> str:
    """Return canonical for this workflow.

    Args:
        text: Text input used by the operation.

    Returns:
        Text produced by this operation.
    """
    return unicodedata.normalize("NFKC", text).casefold()


def entity(text: str) -> str | None:
    """Return entity for this workflow.

    Args:
        text: Text input used by the operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    text = canonical(text)
    for name, aliases in ENTITY_RULES:
        if any(word in text for word in aliases):
            return name
    return None


def query_intent(text: str) -> tuple[str | None, str]:
    """Return query intent for this workflow.

    Args:
        text: Text input used by the operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    positive = re.split(r"不要|不需要|排除|别给我|不是", canonical(text), maxsplit=1)[0]
    parts = re.split(r"和|与|以及|、", positive)
    types = {entity(part) for part in parts} - {None}
    return (None if len(types) > 1 else entity(positive)), positive


def candidate_entity(profile: str) -> str | None:
    """Return candidate entity for this workflow.

    Args:
        profile: Profile used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    head = profile.split("商品主体：", 1)[-1].split("。可见外观：", 1)[0]
    head = re.split(r"[（(，,；;]|附赠|附送|搭配|配有|带有", head, maxsplit=1)[0]
    return entity(head)


def attribute_evidence(query: str, title: str) -> dict:
    """Return attribute evidence for this workflow.

    Args:
        query: Query record or representation used by this operation.
        title: Public product title used as input.

    Returns:
        Mapping with fields: supported, conflicts, unknown.
    """
    q = canonical(query)
    t = canonical(title)
    supported = []
    conflicts = []
    unknown = []
    checks = [
        ("wireless", ("无线", "蓝牙"), ("有线",)),
        ("short_sleeve", ("短袖",), ("长袖",)),
        ("long_sleeve", ("长袖",), ("短袖",)),
        ("long_trousers", ("长裤",), ("短裤", "五分裤")),
        ("female", ("女士", "女式", "女款"), ("男士", "男式", "男款")),
    ]
    for name, yes, no in checks:
        if not any(word in q for word in yes):
            continue
        has_yes = any(word in t for word in yes)
        has_no = any(word in t for word in no)
        if has_yes and not has_no:
            supported.append(name)
        elif has_no and not has_yes:
            conflicts.append(name)
        else:
            unknown.append(name)
    return dict(supported=supported, conflicts=conflicts, unknown=unknown)


def constrain_scores(
    text: str | None, titles: list[str], profiles: list[str], scores: np.ndarray
) -> tuple[np.ndarray, list[dict]]:
    """Return constrain scores for this workflow.

    Args:
        text: Text input used by the operation.
        titles: Titles used by this operation.
        profiles: Profiles used by this operation.
        scores: Scores used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Constraint feature alignment mismatch.
    """
    values = np.asarray(scores, dtype=float)
    if (
        len(titles) != len(profiles)
        or values.shape != (len(titles),)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Constraint feature alignment mismatch")
    if text is None:
        return values.copy(), [dict(state="not_applied_image_query") for _ in titles]
    intent, positive = query_intent(text)
    output = values.copy()
    evidence = []
    for i, (title, profile) in enumerate(zip(titles, profiles)):
        found = candidate_entity(profile)
        attributes = attribute_evidence(positive, title)
        matched = intent is not None and found == intent
        different = intent is not None and found is not None and found != intent
        if found == "generic_accessory" and intent in ACCESSORY_INTENTS:
            different = False
        conflict = different or bool(attributes["conflicts"])
        if conflict:
            tier = 0
            state = "explicit_product_or_attribute_conflict"
        elif matched:
            tier = 2 if attributes["unknown"] else 3
            state = "matching_product_type"
        else:
            tier = 1
            state = "unknown_product_type"
        if (
            intent is not None
            or attributes["supported"]
            or attributes["conflicts"]
            or attributes["unknown"]
        ):
            output[i] += 3 * tier
        evidence.append(
            dict(
                state=state,
                query_intent=intent,
                candidate_intent=found,
                tier=tier,
                attributes=attributes,
                source="machine_product_profile_and_original_title_not_ground_truth",
            )
        )
    return output, evidence
