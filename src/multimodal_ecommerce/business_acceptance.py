"""Metadata-defined, model-unseen business acceptance candidate preparation."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import ijson

from multimodal_ecommerce.data import parse_pv
from multimodal_ecommerce.dataset_release import (
    normalized_title,
    order_key,
    read_rows,
    write_rows,
)
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    PROTECTED_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)
from multimodal_ecommerce.taxonomy import normalize_label

FINE_TYPES: dict[str, list[str]] = {
    "3C数码": [
        "手机",
        "笔记本电脑",
        "智能手表",
        "耳机/耳麦",
        "有线鼠标",
        "移动电源",
        "手机保护套/壳",
        "平板电脑保护套/壳",
        "车载充电器",
        "硬盘盒",
    ],
    "服饰穿搭": [
        "T恤",
        "牛仔裤",
        "羽绒服",
        "卫衣/绒衫",
        "西装",
        "汉服",
        "时装靴",
        "跑步鞋",
        "帆布鞋",
        "双肩包",
    ],
    "家居日用": [
        "茶几",
        "餐边柜",
        "单个书柜",
        "换鞋凳",
        "吸管杯",
        "蚕丝被",
        "棉花被",
        "装饰台灯",
        "懒人沙发",
        "婴童枕头/枕芯",
    ],
}
ACCEPTANCE_POLICY: dict[str, Any] = {
    "seed": 2026102204,
    "source_split": "test",
    "scan_limit": 200000,
    "catalog_per_fine_type": 10,
    "image_queries_per_fine_type": 1,
    "text_queries_per_fine_type": 1,
    "fine_types": FINE_TYPES,
    "minimum_available_metadata": 20,
    "max_metadata_candidates_per_fine_type": 200,
    "dhash_threshold": 4,
    "download_batch_per_fine_type": 20,
    "download_workers": 8,
    "scope": "Balanced closed catalog: 30 predefined fine types, 300 distinct-family candidate products, 30 image and 30 independently authored text queries; not a traffic-representative sample",
    "query_or_catalog_selection_uses_model_scores": False,
    "drop_queries_for_low_relevant_counts": False,
    "human_query_approval_required": True,
    "complete_human_relevance_labels_required": True,
    "quality_targets": {"precision@10": 0.6, "recall@10": 0.7},
}


def product_family_keys(row: dict) -> set[str]:
    """Conservative source ID/title/image/merchant-SKU and branded-model links.

    Args:
        row: Row used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    keys = set()
    source = str(row.get("source_id") or row.get("id", "").rsplit("/", 1)[-1])
    if source:
        keys.add("id:" + source)
    title = normalized_title(str(row.get("text") or row.get("title") or ""))
    if title:
        keys.add("title:" + title)
    url = str(row.get("image_url") or row.get("url") or "")
    parsed = urlsplit(url)
    if parsed.hostname and parsed.path:
        keys.add("url:" + parsed.hostname.casefold() + parsed.path)
    attrs = row.get("attributes") or parse_pv(row.get("pv") or "")

    def single(names: list[str]) -> str | None:
        """Return single for this workflow.

        Args:
            names: Names to inspect or resolve.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        values = {
            normalize_label(str(v)).casefold()
            for name in names
            for v in attrs.get(name, [])
            if str(v).strip()
        }
        return next(iter(values)) if len(values) == 1 else None

    fine = normalize_label(str(row.get("label") or ""))
    brand = single(["品牌"])
    model = single(["型号", "产品型号"])
    if (
        brand
        and model
        and brand not in {"other/其他", "其他", "其它", "无品牌", "无", "other"}
        and model not in {"其他", "其它", "通用", "无", "001", "1"}
    ):
        keys.add("brand_model:" + fine + "|" + brand + "|" + model)
    merchant = re.search(r"/i\d+/(\d+)/", parsed.path) or re.search(
        r"!!(\d+)", parsed.path
    )
    sku = single(["货号", "款号"])
    if merchant and sku:
        keys.add("merchant_sku:" + fine + "|" + merchant.group(1) + "|" + sku)
    return keys


def prior_product_keys() -> set[str]:
    """Return prior product keys for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    keys = set()
    for path in sorted((PROJECT_ROOT / "data").rglob("*.jsonl")):
        for row in read_rows(path):
            if (
                str(row.get("id", "")).startswith("product/")
                or row.get("dataset") == "product"
            ):
                keys.update(product_family_keys(row))
    return keys


def prepare_acceptance_metadata(output: Path) -> dict:
    """Freeze the candidate policy and source prefix before any media selection.

    Args:
        output: Output used by this operation.

    Returns:
        Mapping with fields: scanned, source_id_prefix_sha256,
        available_by_fine_type, queue_records, excluded, candidate_file_sha256,
        model_inference_performed, formal_acceptance.

    Raises:
        FileExistsError: Preserve the acceptance preparation protocol.
        ValueError: Frozen metadata prefix is shorter than expected.
    """
    output = ensure_output_path(output)
    if (output / "preparation_protocol.json").exists():
        raise FileExistsError("Preserve the acceptance preparation protocol")
    exposure_path = (
        PROJECT_ROOT / "artifacts/module2_training/post_training_exposure_snapshot.json"
    )
    exposure = json.loads(exposure_path.read_text(encoding="utf-8"))
    from multimodal_ecommerce.blind_evaluation import verify_files

    verify_files(PROJECT_ROOT, exposure["files"])
    source = PROTECTED_ROOT / "多模态电商数据/product1m_product5m_test_id_label.json"
    old = prior_product_keys()
    protocol = {
        "policy": ACCEPTANCE_POLICY,
        "role": "reserved_acceptance_candidates_not_yet_ready",
        "confirmed_business_relevance_by_user": True,
        "source_sha256": sha256_file(source),
        "prior_exposure_sha256": sha256_file(exposure_path),
        "old_product_family_keys": sorted(old),
        "implementation": {
            name: sha256_file(PROJECT_ROOT / name)
            for name in [
                "src/multimodal_ecommerce/business_acceptance.py",
                "scripts/prepare_business_acceptance.py",
            ]
        },
        "taxonomy_policy_sha256": sha256_file(
            PROJECT_ROOT / "configs/taxonomy_v2.json"
        ),
        "human_review_status": "Query authors and independent relevance assessors not yet assigned; no human judgments",
        "model_inference_allowed": False,
        "formal_acceptance": False,
        "formal_metrics": None,
    }
    selected = {label: cat for cat, labels in FINE_TYPES.items() for label in labels}
    candidates: dict[str, list[dict]] = {label: [] for label in selected}
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    digest = hashlib.sha256()
    scanned = 0
    with source.open("rb") as handle:
        for sid, raw in ijson.kvitems(handle, ""):
            scanned += 1
            label = normalize_label(str(raw.get("label") or ""))
            if label in selected and raw.get("title") and raw.get("url"):
                record = {
                    "id": f"product/test/{sid}",
                    "source_id": str(sid),
                    "dataset": "product",
                    "split": "test",
                    "label": label,
                    "category": selected[label],
                    "text": raw["title"],
                    "image_url": raw["url"],
                    "attributes": parse_pv(raw.get("pv") or ""),
                    "source_position": scanned,
                    "source_record_sha256": hashlib.sha256(
                        json.dumps(raw, sort_keys=True, ensure_ascii=False).encode()
                    ).hexdigest(),
                }
                keys = product_family_keys(record)
                if keys & old:
                    skipped["previous_source_family_key"] += 1
                else:
                    record["family_keys"] = sorted(keys)
                    record["priority"] = order_key(
                        record["id"], ACCEPTANCE_POLICY["seed"]
                    )
                    candidates[label].append(record)
                    counts[label] += 1
            digest.update(str(sid).encode() + b"\n")
            if scanned >= ACCEPTANCE_POLICY["scan_limit"]:
                break
    if scanned != ACCEPTANCE_POLICY["scan_limit"]:
        raise ValueError("Frozen metadata prefix is shorter than expected")
    rows = []
    for label, records in candidates.items():
        if len(records) < ACCEPTANCE_POLICY["minimum_available_metadata"]:
            raise ValueError(
                f"Insufficient source metadata for declared fine type: {label}"
            )
        rows.extend(
            sorted(records, key=lambda r: r["priority"])[
                : ACCEPTANCE_POLICY["max_metadata_candidates_per_fine_type"]
            ]
        )
    write_json(output / "preparation_protocol.json", protocol)
    write_json(output / "prior_exposure.json", exposure)
    write_rows(output / "source_candidates.jsonl", rows)
    summary = {
        "scanned": scanned,
        "source_id_prefix_sha256": digest.hexdigest(),
        "available_by_fine_type": dict(counts),
        "queue_records": len(rows),
        "excluded": dict(skipped),
        "candidate_file_sha256": sha256_file(output / "source_candidates.jsonl"),
        "model_inference_performed": False,
        "formal_acceptance": False,
    }
    write_json(output / "metadata_receipt.json", summary)
    return summary
