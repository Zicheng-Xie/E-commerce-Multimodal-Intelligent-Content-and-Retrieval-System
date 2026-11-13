"""Metadata-only development coverage sampling and declared-human catalog QA."""

from __future__ import annotations

import re

from multimodal_ecommerce.business_constraints import canonical
from multimodal_ecommerce.business_review_jobs import identity

BUCKETS: dict = {
    "wireless_headphones": dict(
        query_id="text_01",
        name="无线蓝牙耳机",
        labels=[
            "蓝牙耳机",
            "耳机/耳麦",
            "电脑耳机/耳麦",
            "真无线降噪耳机",
            "普通真无线耳机",
        ],
        required=r"无线|蓝牙",
        forbidden=r"保护|配件|耳罩|单只|左耳|右耳|充电盒|充电仓|收纳|有线",
    ),
    "computer_keyboards": dict(
        query_id="text_02",
        name="电脑键盘",
        labels=["键盘"],
        required=r"键盘",
        forbidden=r"保护|膜|键帽|支架|midi|电子琴|编曲",
    ),
    "phone_cases": dict(
        query_id="text_03",
        name="手机保护壳",
        labels=["手机保护套/壳"],
        required=r"手机壳|保护套|保护壳|手机套",
        forbidden=r"镜头膜|屏幕膜",
    ),
    "power_banks": dict(
        query_id="text_04",
        name="便携充电宝",
        labels=["移动电源"],
        required=r"充电宝|移动电源",
        forbidden=r"保护套|收纳|外壳",
    ),
    "short_tshirts": dict(
        query_id="text_05",
        name="短袖T恤",
        labels=["T恤", "速干T恤", "瑜伽t恤"],
        required=r"短袖|半袖|半截袖",
        forbidden=r"长袖|无袖|衬衫",
    ),
    "women_dresses": dict(
        query_id="text_06",
        name="女士连衣裙",
        labels=["连衣裙", "运动连衣裙"],
        required=r"女",
        forbidden=r"裙撑|半身裙|男士|男装",
    ),
    "sleep_pillows": dict(
        query_id="text_11",
        name="睡眠枕头",
        labels=["枕头/枕芯", "乳胶枕", "枕头"],
        required=r"枕",
        forbidden=r"枕套|抱枕|靠垫|车载|汽车|u型|u形|颈枕",
    ),
    "home_chairs": dict(
        query_id="text_12",
        name="家用椅子",
        labels=["餐椅", "普通餐椅", "休闲椅", "椅子", "折叠椅/躺椅"],
        required=r"椅",
        forbidden=r"椅套|椅垫|配件|脚轮|椅脚|理发|美发|轮椅|医院|诊所|公园|商场",
    ),
}
POLICY: dict = dict(
    seed=2026102208,
    source_split="train",
    scan_limit=200000,
    buckets=BUCKETS,
    per_bucket=8,
    metadata_budget_per_bucket=120,
    download_workers=8,
    dhash_threshold=4,
    role="exposed_development_catalog_candidates_pending_human_QA",
    model_inference_allowed=False,
    labels_are_relevance_truth=False,
    change_existing_queries_or_qrels=False,
)
QA_EDITABLE = {"verdict", "product_type", "note", "reviewer", "review_origin"}


def metadata_bucket(label: str, title: str) -> str | None:
    """Return metadata bucket for this workflow.

    Args:
        label: Label used by this operation.
        title: Public product title used as input.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    title = canonical(title)
    for key, rule in BUCKETS.items():
        if (
            canonical(label) in {canonical(v) for v in rule["labels"]}
            and re.search(rule["required"], title)
            and not re.search(rule["forbidden"], title)
        ):
            return key
    return None


def priority_order(rows: list[dict], preferred: set[str]) -> list[dict]:
    """Only reorder complete unchanged original batches; do not prefill judgments.

    Args:
        rows: Records processed by this operation.
        preferred: Preferred used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Duplicate original batch row IDs.
    """
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate original batch row IDs")
    return [r for r in rows if r["id"] in preferred] + [
        r for r in rows if r["id"] not in preferred
    ]


def validate_catalog_submission(
    manifest: dict, templates: list[dict], value: dict
) -> list[dict]:
    """Validate catalog submission.

    Args:
        manifest: Manifest used by this operation.
        templates: Templates used by this operation.
        value: Input value to validate or transform.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Catalog QA release or slot mismatch; A named actual human
            declaration is required; Full catalog QA row coverage required; Unknown or
            duplicate catalog item; Immutable catalog fields changed; Only catalog QA
            decisions are supported; Invalid catalog QA origin or text fields; Each
            judgment needs the same actual reviewer; Usable items require an observed
            product type; Rejected or uncertain items require a reason.
    """
    keys = {"release_id", "review_slot", "reviewer", "declared_human", "rows"}
    if (
        set(value) != keys
        or value["release_id"] != manifest["release_id"]
        or value["review_slot"] not in ["A", "B"]
    ):
        raise ValueError("Catalog QA release or slot mismatch")
    if (
        value["declared_human"] is not True
        or not isinstance(value["reviewer"], str)
        or not identity(value["reviewer"])
    ):
        raise ValueError("A named actual human declaration is required")
    expected = {r["id"]: r for r in templates}
    if not isinstance(value["rows"], list) or len(value["rows"]) != len(templates):
        raise ValueError("Full catalog QA row coverage required")
    seen = set()
    for row in value["rows"]:
        key = row.get("id")
        if key not in expected or key in seen:
            raise ValueError("Unknown or duplicate catalog item")
        seen.add(key)
        template = expected[key]
        if set(row) != set(template) or any(
            row[k] != template[k] for k in template if k not in QA_EDITABLE
        ):
            raise ValueError("Immutable catalog fields changed")
        if row["role"] != "development_catalog_QA_only" or row["verdict"] not in [
            None,
            "usable",
            "reject",
            "uncertain",
        ]:
            raise ValueError("Only catalog QA decisions are supported")
        if not all(
            isinstance(row[k], str) for k in ["product_type", "note", "reviewer"]
        ) or row["review_origin"] not in [None, "human"]:
            raise ValueError("Invalid catalog QA origin or text fields")
        if (
            row["verdict"] is not None
            or row["product_type"].strip()
            or row["note"].strip()
        ):
            if row["review_origin"] != "human" or identity(row["reviewer"]) != identity(
                value["reviewer"]
            ):
                raise ValueError("Each judgment needs the same actual reviewer")
        if row["verdict"] == "usable" and not row["product_type"].strip():
            raise ValueError("Usable items require an observed product type")
        if row["verdict"] in ["reject", "uncertain"] and not row["note"].strip():
            raise ValueError("Rejected or uncertain items require a reason")
    return value["rows"]


def merge_catalog_qa(
    manifest: dict, templates: list[dict], submissions: list[dict]
) -> dict:
    """Merge catalog qa.

    Args:
        manifest: Manifest used by this operation.
        templates: Templates used by this operation.
        submissions: Submissions used by this operation.

    Returns:
        Mapping with fields: accepted_catalog_ids, rejected_catalog_ids,
        conflict_ids, pending, declared_reviewers, human_relevance_labels,
        relevance_labels_created, formal_metrics, goal_met.

    Raises:
        ValueError: Keep only one latest export per slot; A and B must be different
            actual reviewers.
    """
    slots = {}
    for value in submissions:
        rows = validate_catalog_submission(manifest, templates, value)
        slot = value["review_slot"]
        if slot in slots:
            raise ValueError("Keep only one latest export per slot")
        slots[slot] = (identity(value["reviewer"]), {r["id"]: r for r in rows})
    if len(slots) == 2 and slots["A"][0] == slots["B"][0]:
        raise ValueError("A and B must be different actual reviewers")
    accepted, rejected, conflicts = [], [], []
    for row in templates:
        a = slots.get("A", ("", {}))[1].get(row["id"])
        b = slots.get("B", ("", {}))[1].get(row["id"])
        if (
            a is None
            or b is None
            or a["verdict"] in [None, "uncertain"]
            or b["verdict"] in [None, "uncertain"]
        ):
            continue
        if a["verdict"] == b["verdict"] == "reject":
            rejected.append(row["id"])
        elif a["verdict"] == b["verdict"] == "usable" and identity(
            a["product_type"]
        ) == identity(b["product_type"]):
            accepted.append(row["id"])
        else:
            conflicts.append(row["id"])
    return dict(
        accepted_catalog_ids=accepted,
        rejected_catalog_ids=rejected,
        conflict_ids=conflicts,
        pending=len(templates) - len(accepted) - len(rejected) - len(conflicts),
        declared_reviewers=len(slots),
        human_relevance_labels=0,
        relevance_labels_created=False,
        formal_metrics=None,
        goal_met=False,
    )
