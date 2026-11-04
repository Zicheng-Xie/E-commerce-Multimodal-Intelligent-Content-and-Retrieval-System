"""Multi-channel candidate recall and bounded public-evidence ranking adjustments."""

from __future__ import annotations

import re

import numpy as np
import numpy.typing as npt

from multimodal_ecommerce.business_constraints import (
    ENTITY_RULES,
    canonical,
    query_intent,
)
from multimodal_ecommerce.semantic_ranker import ranked_ids

# Public product nouns, independent of query IDs, source categories and judgments.
EXTRA_TYPES = [
    ("phone_mount", ["手机支架", "车载支架"]),
    ("computer_mount", ["电脑支架", "显示器支架", "硬盘托架"]),
    ("phone_bag", ["手机包"]),
    ("computer_bag", ["电脑包", "笔记本电脑包"]),
    ("camera_bag", ["相机包", "相机内胆包", "相机电池收纳盒"]),
    ("storage_bag", ["收纳包", "收纳袋", "收纳盒", "鞋盒"]),
    ("watch_strap", ["表带", "腕带", "手表替换", "手表扣环"]),
    ("chair_accessory", ["椅套", "椅垫", "椅子配件", "办公椅底座", "脚轮组件"]),
    ("furniture_accessory", ["家具连接件", "餐桌五金", "预埋螺母"]),
    ("shoe_accessory", ["鞋带", "鞋套", "鞋垫"]),
    ("lens_accessory", ["手机镜头", "镜头保护", "手机望远镜"]),
    ("router", ["路由器", "交换机"]),
    ("storage_device", ["硬盘", "u盘", "存储卡"]),
    ("camera", ["相机", "摄像头", "倒车影像"]),
    ("camera_accessory", ["相机配件", "相机电池", "闪光灯"]),
    ("speaker", ["音箱", "音响"]),
    ("display", ["显示器", "电视机"]),
    ("smart_watch", ["智能手表", "电话手表"]),
    ("screen_accessory", ["屏幕防窥膜", "屏幕保护膜", "插座面板"]),
    ("bed", ["儿童床", "双人床", "双层床", "婴儿床", "储物床", "午睡床", "情趣床"]),
    ("mattress", ["床垫"]),
    ("bedding", ["被套", "床单", "被褥", "床上用品", "婴儿被", "婴儿包被"]),
    ("sofa", ["沙发"]),
    ("stool", ["凳"]),
    ("towel", ["浴巾", "毛巾", "擦手纸"]),
    ("carpet", ["地毯", "防尘罩", "垫巾"]),
    ("dishware", ["餐具", "碗", "筷子", "洗菜盆", "沥水篮"]),
    ("measuring_cup", ["量杯"]),
    ("cleaning_tool", ["拖把", "清洁桶", "洗鞋机", "烘鞋器"]),
    ("hat", ["帽"]),
    ("outerwear", ["外套", "羽绒服", "西服", "西装", "棒球衫", "开衫", "卫衣"]),
    ("knitwear", ["毛衣", "针织衫", "针织套装"]),
    ("underwear", ["内裤", "纸尿裤", "拉拉裤"]),
    ("sleepwear", ["睡衣", "家居服"]),
    ("phone_charm", ["手机挂", "手机数据线配件", "防尘塞", "钥匙扣"]),
    ("light_strip", ["灯带"]),
    ("power_adapter", ["电源适配器", "电脑插座"]),
]

POLICY: dict = dict(
    pool_sizes=[20, 60],
    rrf_constant=60,
    channel_indices=[0, 1, 2, 3, 4, 5, 6],
    channel_weights=[0.15, 0.20, 0.10, 0.15, 0.25, 0.10, 0.05],
    cross_encoder_weight=0.65,
    public_match_bonus=0.12,
    explicit_type_conflict_penalty=0.35,
    explicit_attribute_conflict_penalty=0.25,
    conflicting_sources_penalty=0.08,
    source="Query text, original title and machine product main type only",
    labels_or_source_categories_as_rank_features=False,
    default_promotion_allowed=False,
)


def public_entity(text: str) -> str | None:
    """Return public entity for this workflow.

    Args:
        text: Text input used by the operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    text = canonical(text)
    matches = [
        (len(alias), order, kind)
        for order, (kind, aliases) in enumerate([*EXTRA_TYPES, *ENTITY_RULES])
        for alias in aliases
        if alias in text
    ]
    # Longest explicit product noun avoids e.g. "phone" overriding "phone mount".
    return min(matches, key=lambda r: (-r[0], r[1]))[2] if matches else None


def main_type(profile: str) -> str | None:
    """Return main type for this workflow.

    Args:
        profile: Profile used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    head = profile.split("商品主体：", 1)[-1].split("。可见外观：", 1)[0]
    head = re.split(r"[（(，,；;]|附赠|附送|搭配|配有|带有", head, maxsplit=1)[0]
    return public_entity(head)


def attributes(query: str, title: str) -> dict:
    """Return attributes for this workflow.

    Args:
        query: Query record or representation used by this operation.
        title: Public product title used as input.

    Returns:
        Result mapping produced by the workflow.
    """
    q, t = canonical(query), canonical(title)
    checks = [
        ("wireless", ["无线", "蓝牙"], ["有线"]),
        ("short_sleeve", ["短袖"], ["长袖", "无袖", "背心"]),
        ("long_sleeve", ["长袖"], ["短袖", "无袖"]),
        ("long_trousers", ["长裤"], ["短裤", "五分裤", "七分裤"]),
        (
            "female",
            ["女士", "女式", "女款", "女装", "女童", "女生"],
            ["男士", "男式", "男款", "男童"],
        ),
    ]
    result: dict[str, list[str]] = dict(supported=[], conflicts=[], unknown=[])
    for name, yes, no in checks:
        if not any(s in q for s in yes):
            continue
        positive, negative = any(s in t for s in yes), any(s in t for s in no)
        key = (
            "supported"
            if positive and not negative
            else "conflicts" if negative and not positive else "unknown"
        )
        result[key].append(name)
    return result


def public_evidence(text: str | None, title: str, profile: str) -> dict:
    """Return public evidence for this workflow.

    Args:
        text: Text input used by the operation.
        title: Public product title used as input.
        profile: Profile used by this operation.

    Returns:
        Mapping with fields: state, offset, query_type, profile_type, title_type,
        attributes, source.
    """
    if text is None:
        return dict(
            state="image_query_abstain",
            offset=0.0,
            query_type=None,
            profile_type=main_type(profile),
        )
    intent, positive = query_intent(text)
    profile_type, title_type = main_type(profile), public_entity(title)
    found = profile_type or title_type
    attr = attributes(positive, title)
    mismatch = intent is not None and found is not None and found != intent
    disagreement = (
        profile_type is not None
        and title_type is not None
        and profile_type != title_type
    )
    offset, state = 0.0, "unknown"
    if mismatch:
        offset, state = (
            -POLICY["explicit_type_conflict_penalty"],
            "explicit_type_conflict",
        )
    elif intent is not None and found == intent and not disagreement:
        offset, state = POLICY["public_match_bonus"], "public_product_match"
    elif disagreement:
        offset, state = (
            -POLICY["conflicting_sources_penalty"],
            "conflicting_public_sources",
        )
    if attr["conflicts"]:
        offset -= POLICY["explicit_attribute_conflict_penalty"]
        state = "explicit_attribute_conflict"
    # Unsupported explicit attributes cannot receive the positive match bonus.
    if attr["unknown"] and offset > 0:
        offset = 0.0
    return dict(
        state=state,
        offset=offset,
        query_type=intent,
        profile_type=profile_type,
        title_type=title_type,
        attributes=attr,
        source="public_evidence_not_relevance_truth",
    )


def normalized_ranks(ids: list[str], scores: npt.ArrayLike) -> np.ndarray:
    """Return normalized ranks.

    Args:
        ids: Identifiers in the order expected by the associated arrays.
        scores: Scores used by this operation.

    Returns:
        Computed array in the order of the supplied records.
    """
    order = ranked_ids(ids, scores)
    ranks = {key: i for i, key in enumerate(order)}
    return np.array([1.0 - ranks[k] / max(1, len(ids) - 1) for k in ids])


def recall_channels(
    ids: list[str], features: np.ndarray
) -> tuple[np.ndarray, list[list[str]]]:
    """Return recall channels for this workflow.

    Args:
        ids: Identifiers in the order expected by the associated arrays.
        features: Feature array aligned with the candidate or query records.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Expected aligned finite public features.
    """
    if features.shape != (len(ids), 13) or not np.isfinite(features).all():
        raise ValueError("Expected aligned finite public features")
    output = np.zeros(len(ids))
    channels = []
    for column, weight in zip(POLICY["channel_indices"], POLICY["channel_weights"]):
        if column in (5, 6) and not np.any(features[:, column] > 0):
            continue
        order = ranked_ids(ids, features[:, column])
        ranks = {key: i + 1 for i, key in enumerate(order)}
        output += np.array([weight / (POLICY["rrf_constant"] + ranks[k]) for k in ids])
        channels.append(order)
    return output, channels


def balanced_pool(channels: list[list[str]], budget: int) -> list[str]:
    """Return balanced pool for this workflow.

    Args:
        channels: Channels used by this operation.
        budget: Budget used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Need a positive budget and recall channels.
    """
    if budget <= 0 or not channels:
        raise ValueError("Need a positive budget and recall channels")
    output, seen = [], set()
    for position in range(max(map(len, channels))):
        for channel in channels:
            if position < len(channel) and channel[position] not in seen:
                key = channel[position]
                output.append(key)
                seen.add(key)
                if len(output) == budget:
                    return output
    return output


def rerank_complete(
    ids: list[str],
    base: np.ndarray,
    pool: list[str],
    logits: np.ndarray,
    offsets: np.ndarray | None = None,
) -> list[str]:
    """Return rerank complete for this workflow.

    Args:
        ids: Identifiers in the order expected by the associated arrays.
        base: Base used by this operation.
        pool: Pool used by this operation.
        logits: Logits used by this operation.
        offsets: Offsets used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Cross-encoder pool/score alignment mismatch; Public evidence
            alignment mismatch.
    """
    if (
        len(pool) != len(set(pool))
        or not set(pool) <= set(ids)
        or logits.shape != (len(pool),)
        or not np.isfinite(logits).all()
    ):
        raise ValueError("Cross-encoder pool/score alignment mismatch")
    positions = {key: i for i, key in enumerate(ids)}
    selected = np.array([positions[k] for k in pool])
    score = POLICY["cross_encoder_weight"] * normalized_ranks(pool, logits)
    score += (1 - POLICY["cross_encoder_weight"]) * normalized_ranks(ids, base)[
        selected
    ]
    if offsets is not None:
        if offsets.shape != (len(ids),) or not np.isfinite(offsets).all():
            raise ValueError("Public evidence alignment mismatch")
        score += offsets[selected]
    first = ranked_ids(pool, score)
    return first + [k for k in ranked_ids(ids, base) if k not in set(pool)]
