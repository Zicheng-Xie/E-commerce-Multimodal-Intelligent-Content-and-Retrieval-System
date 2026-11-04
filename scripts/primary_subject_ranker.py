"""Public subject consistency ablation; no IDs, labels, grades or rankings as
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from multimodal_ecommerce.intent_retrieval import normalized_ranks
from multimodal_ecommerce.semantic_ranker import ranked_ids


class _PolicySpec(TypedDict):
    """Exact fields of the POLICY configuration and provenance record."""

    main_type_conflict: float
    subtype_conflict: float
    structure_conflict: float
    stack_penalties: bool
    positive_bonus: int
    unknown_penalty: int
    image_only: bool
    candidate_budget: int
    qrels_used: bool
    baseline_cross_weight: float
    unchanged_full_tail: bool


POLICY: _PolicySpec = dict(
    main_type_conflict=0.35,
    subtype_conflict=0.25,
    structure_conflict=0.12,
    stack_penalties=False,
    positive_bonus=0,
    unknown_penalty=0,
    image_only=True,
    candidate_budget=20,
    qrels_used=False,
    baseline_cross_weight=0.65,
    unchanged_full_tail=True,
)
# Longest complete product noun wins, so object names inside compound nouns
# cannot turn cases into devices or a named handbag into bedding.
NOUNS = [
    (
        "bag",
        "laptop",
        ["笔记本电脑内胆包", "电脑内胆包", "电脑包", "笔记本包", "平板内胆包"],
    ),
    ("bag", "camera", ["相机内胆包", "摄影内胆包", "相机包", "摄影包"]),
    (
        "bag",
        "organizer",
        [
            "手提包内胆",
            "手袋内胆",
            "包内收纳",
            "收纳内胆",
            "内胆包",
            "包内胆",
            "包中包",
        ],
    ),
    ("bag", "handbag", ["枕头包", "手提包", "手袋", "单肩包", "挎包"]),
    (
        "audio_accessory",
        "case",
        ["耳机盒保护套", "耳机保护套", "耳机盒", "耳机收纳盒", "耳机套"],
    ),
    ("audio_accessory", "pad", ["耳机耳罩", "耳垫", "耳机线", "耳机支架"]),
    ("audio", "overear", ["头戴式耳机", "头戴耳机", "耳麦", "头戴式耳麦"]),
    ("audio", "mono", ["单耳通话耳机", "单耳蓝牙耳机", "单耳耳机"]),
    ("audio", "earbud", ["有线入耳耳机", "入耳式耳机"]),
    ("audio", "earbud", ["真无线耳机", "无线耳塞", "双耳耳机", "蓝牙耳塞"]),
    ("audio", None, ["有线耳机", "蓝牙耳机", "无线耳机", "耳机", "耳塞"]),
    ("phone_accessory", "strap", ["手机挂绳", "手机挂链"]),
    ("phone_accessory", "pouch", ["手机防水袋", "手机收纳袋"]),
    ("phone_case", None, ["手机保护壳", "手机背壳", "手机壳", "手机保护套"]),
    ("shoe_accessory", None, ["鞋收纳盒", "鞋盒", "鞋垫", "鞋带"]),
    ("shoe", "work", ["劳保鞋", "安全鞋"]),
    ("shoe", "ballet", ["芭蕾舞鞋", "舞蹈鞋", "舞鞋"]),
    ("shoe", "boot", ["中筒靴", "马丁靴", "短靴", "踝靴", "袜靴", "时装靴", "靴"]),
    ("shoe", "sport", ["运动鞋", "跑步鞋", "老爹鞋", "登山鞋", "户外鞋", "休闲鞋"]),
    ("coat", "vest", ["羽绒马甲", "牛仔马甲", "马甲", "背心"]),
    ("coat", "sleeved", ["羽绒服", "棉服", "夹克", "开衫"]),
    ("coat", None, ["外套"]),
    ("cabinet", "wardrobe", ["推拉门衣柜", "衣柜"]),
    ("cabinet", "shoe_storage", ["鞋柜"]),
    ("cabinet", "sideboard", ["餐边收纳柜", "餐边柜", "茶水柜", "边柜"]),
    ("cabinet", "dresser", ["抽屉斗柜", "斗柜"]),
    ("cabinet", None, ["储物柜", "展示柜", "柜"]),
    ("rack", None, ["台面置物架", "置物架", "货架"]),
    ("pillow_accessory", None, ["枕头套", "枕套"]),
    ("pillow", "arch", ["拱形颈枕", "木颈枕", "竹颈枕", "拱形枕"]),
    ("pillow", None, ["定型枕", "颈枕", "枕头", "枕芯", "枕"]),
    ("bed", None, ["床头靠背板", "床头板", "床架"]),
    ("cup_accessory", None, ["杯盖", "杯套", "杯刷"]),
    ("cup", "straw", ["吸管学饮杯", "吸管杯"]),
    ("cup", "nipple", ["奶瓶"]),
    ("cup", None, ["学饮杯", "水杯", "保温杯", "杯"]),
    ("fruit", None, ["榴莲果肉", "榴莲", "水果"]),
    ("intercom", None, ["窗口对讲机", "对讲机"]),
]


def affirmed(text: str, word: str) -> bool:
    """Return affirmed for this workflow.

    Args:
        text: Text input used by the operation.
        word: Word used by this operation.

    Returns:
        Whether the condition checked by this operation holds.
    """
    start = 0
    negatives = ("无", "非", "没有", "不带", "不含", "无需")
    while (position := text.find(word, start)) >= 0:
        if not text[:position].endswith(negatives):
            return True
        start = position + len(word)
    return False


def signature(product: str, appearance: str = "", *, uncertain: bool = False) -> dict:
    """Return signature for this workflow.

    Args:
        product: Product used by this operation.
        appearance: Appearance used by this operation.
        uncertain: Uncertain used by this operation.

    Returns:
        Mapping with fields: family, subtype, structures.
    """
    if (
        uncertain
        or not product.strip()
        or any(s in product for s in ["无法识别", "不明确", "不明", "展示道具", "组合"])
    ):
        return dict(family=None, subtype=None, structures=[])
    matches = [
        (len(alias), -i, family, subtype)
        for i, (family, subtype, aliases) in enumerate(NOUNS)
        for alias in aliases
        if affirmed(product, alias)
    ]
    if not matches:
        return dict(family=None, subtype=None, structures=[])
    _, _, family, subtype = max(matches)
    detail = product + "。" + appearance
    if family == "audio" and subtype is None:
        if any(affirmed(detail, w) for w in ["头梁", "头戴", "大耳罩", "包耳"]):
            subtype = "overear"
        elif affirmed(detail, "单耳"):
            subtype = "mono"
        elif any(affirmed(detail, w) for w in ["真无线", "双耳", "充电盒", "入耳"]):
            subtype = "earbud"
    if family == "coat" and subtype is None:
        if any(affirmed(detail, w) for w in ["无袖", "背心", "马甲"]):
            subtype = "vest"
        elif any(affirmed(detail, w) for w in ["有袖", "长袖", "袖口", "袖子"]):
            subtype = "sleeved"
    if family == "cup" and subtype is None:
        if affirmed(detail, "吸管"):
            subtype = "straw"
        elif affirmed(detail, "奶嘴"):
            subtype = "nipple"
    flags = []
    if family == "shoe" and subtype == "boot":
        if any(affirmed(detail, w) for w in ["细跟", "细高跟", "尖细跟"]):
            flags.append("stiletto")
        if any(affirmed(detail, w) for w in ["粗底", "粗跟", "锯齿底", "齿纹底"]):
            flags.append("chunky")
    if family == "pillow":
        if any(affirmed(detail, w) for w in ["拱形", "木质", "竹木", "木制", "弧形硬"]):
            flags.append("hard_arch")
        elif any(affirmed(detail, w) for w in ["软枕", "软垫", "填充枕", "蓬松"]):
            flags.append("soft")
    if family == "cup":
        if any(affirmed(detail, w) for w in ["双柄", "双耳", "双把手", "两侧手柄"]):
            flags.append("double_handle")
        if any(w in detail for w in ["无把手", "无柄"]):
            flags.append("handleless")
    return dict(family=family, subtype=subtype, structures=sorted(flags))


def penalty(query: dict, document: dict) -> dict:
    """Only clear public conflicts receive a single penalty; unknowns abstain.

    Args:
        query: Query record or representation used by this operation.
        document: Document used by this operation.

    Returns:
        Mapping with fields: value, reason.
    """
    qf, df = query["family"], document["family"]
    if qf is None or df is None:
        return dict(value=0.0, reason="unresolved public primary type; abstain")
    if qf != df:
        return dict(
            value=POLICY["main_type_conflict"],
            reason="different explicit primary types or primary/accessory roles",
        )
    qs, ds = query["subtype"], document["subtype"]
    incompatible = (
        qs is not None
        and ds is not None
        and qs != ds
        and (
            qf in {"audio", "bag", "coat", "shoe"}
            or qf == "cup"
            and {qs, ds} == {"straw", "nipple"}
            or qf == "cabinet"
            and "wardrobe" in {qs, ds}
        )
    )
    if incompatible:
        return dict(
            value=POLICY["subtype_conflict"],
            reason="explicit mutually incompatible sale-item subtypes",
        )
    qflags, dflags = set(query["structures"]), set(document["structures"])
    for left, right in [
        ("stiletto", "chunky"),
        ("hard_arch", "soft"),
        ("double_handle", "handleless"),
    ]:
        if (
            left in qflags
            and right not in qflags
            and right in dflags
            and left not in dflags
            or right in qflags
            and left not in qflags
            and left in dflags
            and right not in dflags
        ):
            return dict(
                value=POLICY["structure_conflict"],
                reason="explicit incompatible visible structures",
            )
    return dict(value=0.0, reason="no explicit conflict; no positive reward")


def query_signature(value: dict) -> dict:
    """Return query signature for this workflow.

    Args:
        value: Input value to validate or transform.

    Returns:
        Result mapping produced by the workflow.
    """
    return signature(
        value["product"], value["appearance"], uncertain=value["primary_unclear"]
    )


@dataclass(frozen=True)
class PrimarySubjectRankerGuardedOrderInputs:
    """Input values for guarded order.

    Attributes:
        ids: Identifiers in the order expected by the associated arrays.
        base: Base used by this operation.
        pool: Pool used by this operation.
        logits: Logits used by this operation.
        offsets: Offsets used by this operation.
        query: Query record or representation used by this operation.
        profiles: Profiles used by this operation.
    """

    ids: list[str]
    base: object
    pool: list[str]
    logits: npt.ArrayLike
    offsets: npt.ArrayLike | None
    query: dict
    profiles: dict


def guarded_order(
    options: PrimarySubjectRankerGuardedOrderInputs,
) -> tuple[list[str], dict]:
    """Return guarded order for this workflow.

    Args:
        options: Validated input fields described by
            PrimarySubjectRankerGuardedOrderInputs.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Full gallery public profiles and twenty unique candidates required;
            Runtime profiles cannot carry evaluator fields.
    """
    ids = options.ids
    base = options.base
    pool = options.pool
    logits = options.logits
    offsets = options.offsets
    query = options.query
    profiles = options.profiles
    if len(pool) != 20 or len(set(pool)) != 20 or set(profiles) != set(ids):
        raise ValueError(
            "Full gallery public profiles and twenty unique candidates required"
        )
    if any(
        set(row) != {"id", "product_type", "appearance"} for row in profiles.values()
    ):
        raise ValueError("Runtime profiles cannot carry evaluator fields")
    position = {cid: i for i, cid in enumerate(ids)}
    selected = np.array([position[cid] for cid in pool])
    score = (
        0.65 * normalized_ranks(pool, np.asarray(logits))
        + 0.35 * normalized_ranks(ids, np.asarray(base))[selected]
    )
    if offsets is not None:
        score += np.asarray(offsets)[selected]
    q = query_signature(query)
    conflicts = {
        cid: penalty(
            q, signature(profiles[cid]["product_type"], profiles[cid]["appearance"])
        )
        for cid in pool
    }
    score -= np.array([conflicts[cid]["value"] for cid in pool])
    order = ranked_ids(pool, score) + [
        cid for cid in ranked_ids(ids, np.asarray(base)) if cid not in set(pool)
    ]
    return order, dict(
        query_signature=q,
        conflicts=conflicts,
        pool_scores=dict(zip(pool, score.tolist())),
    )
