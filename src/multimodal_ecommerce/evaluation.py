"""Prepare explicit relevance judgments and refuse incomplete quality scores."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from multimodal_ecommerce.dataset_release import order_key, read_rows, write_rows
from multimodal_ecommerce.retrieval_eval import ranking_metrics
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)

# Generic needs written before inspecting the expanded candidate rankings.
TEXT_QUERIES = [
    (
        "3C数码",
        "无线蓝牙耳机",
        "耳机本体且明确支持无线蓝牙；排除保护套、耳塞配件和有线耳机。",
    ),
    ("3C数码", "电脑键盘", "用于电脑输入的键盘本体；排除键帽、保护膜和支架。"),
    ("3C数码", "手机保护壳", "手机保护壳或保护套；排除手机本体、贴膜和其他配件。"),
    (
        "3C数码",
        "便携充电宝",
        "可为移动设备充电的便携移动电源；排除单独充电器和数据线。",
    ),
    ("服饰穿搭", "短袖T恤", "T恤且明确为短袖；排除衬衫、长袖、无袖和其他衣物。"),
    ("服饰穿搭", "女士连衣裙", "女式连衣裙；排除半身裙和上衣，年龄不限。"),
    ("服饰穿搭", "长裤", "衣着长裤；排除短裤、裙子和其他配件。"),
    ("服饰穿搭", "运动鞋", "供穿着的运动鞋本体；排除鞋垫、鞋架和非运动鞋。"),
    ("家居日用", "喝水用的杯子", "饮水杯本体；排除杯垫、杯盖、杯架和装饰品。"),
    ("家居日用", "家用收纳柜", "用于家庭收纳物品的柜体；排除盒、袋和无柜体的架子。"),
    ("家居日用", "枕头", "睡眠枕芯或含枕芯的枕头；排除单独枕套、抱枕套和靠垫。"),
    ("家居日用", "家用椅子", "供坐用的家用椅子本体；排除椅套、垫子、桌子和配件。"),
]


def verify_files(root: Path, manifest: dict) -> None:
    """Verify every frozen path before deriving another artifact.

    Args:
        root: Root directory used to resolve the operation artifacts.
        manifest: Manifest used by this operation.
    """
    for relative, digest in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or sha256_file(path) != digest:
            raise ValueError(f"Frozen artifact changed: {relative}")


def prepare_evaluation(dataset: Path, index: Path, output: Path) -> dict:
    """Freeze queries and blank exhaustive qrels without inventing judgments.

    Args:
        dataset: Dataset used by this operation.
        index: Index used by this operation.
        output: Output used by this operation.

    Returns:
        Mapping with fields: schema_version, status, dataset, dataset_sha256, index,
        index_manifest_sha256, query_count, judgments_required, candidate_count, files,
        scoring_policy.

    Raises:
        FileExistsError: Use a new evaluation directory.
        ValueError: Index belongs to a different dataset.
    """
    output = ensure_output_path(output)
    if output.exists():
        raise FileExistsError("Use a new evaluation directory")
    verify_files(
        dataset.parent,
        json.loads((dataset.parent / "manifest.json").read_text(encoding="utf-8")),
    )
    records = read_rows(dataset)
    index_manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
    if index_manifest["dataset_sha256"] != sha256_file(dataset):
        raise ValueError("Index belongs to a different dataset")
    queries = [
        {
            "id": f"text_{number:02d}",
            "kind": "text",
            "category_for_reporting": category,
            "text": text,
            "image_path": None,
            "exclude_id": None,
            "relevance_rule": rule,
        }
        for number, (category, text, rule) in enumerate(TEXT_QUERIES, 1)
    ]
    counts: Counter = Counter()
    for row in sorted(records, key=lambda item: order_key(item["id"], 17)):
        if row["project_role"] != "heldout" or counts[row["category"]] >= 2:
            continue
        counts[row["category"]] += 1
        queries.append(
            {
                "id": f"image_{len(queries) - len(TEXT_QUERIES) + 1:02d}",
                "kind": "image",
                "category_for_reporting": row["category"],
                "text": None,
                "image_path": str((dataset.parent / row["image_path"]).resolve()),
                "image_sha256": row["image_sha256"],
                "exclude_id": row["id"],
                "relevance_rule": "主要商品类型与用途相同，且视觉外形相似。仅品类相同不足以相关；排除仅场景或背景相似。看不清主体时标记无法判断，不猜测不可见材质或参数。",
            }
        )
    labels = [
        {
            "query_id": query["id"],
            "product_id": row["id"],
            "grade": None,
            "reviewer": "",
            "note": "",
        }
        for query in queries
        for row in records
        if row["id"] != query["exclude_id"]
    ]
    catalog = [
        {
            "id": row["id"],
            "title": row["text"],
            "category": row["category"],
            "image_path": str((dataset.parent / row["image_path"]).resolve()),
            "attributes": row["attributes"],
        }
        for row in records
    ]
    write_rows(output / "queries.jsonl", queries)
    write_rows(output / "qrels.template.jsonl", labels)
    write_json(output / "catalog.json", catalog)
    manifest = {
        "schema_version": 1,
        "status": "awaiting_human_judgments",
        "dataset": str(dataset.resolve()),
        "dataset_sha256": sha256_file(dataset),
        "index": str(index.resolve()),
        "index_manifest_sha256": sha256_file(index / "manifest.json"),
        "query_count": len(queries),
        "judgments_required": len(labels),
        "candidate_count": len(records),
        "files": {
            name: sha256_file(output / name)
            for name in ("queries.jsonl", "qrels.template.jsonl", "catalog.json")
        },
        "scoring_policy": "Grade 1=relevant, 0=irrelevant, null=unjudged. All non-self candidates must be judged for Recall. Macro mean over positive-qrel queries; no-positive queries reported separately, never silently removed. Primary image target, no category filtering. Fusion may be reported separately, not tuned on heldout labels.",
    }
    write_json(output / "manifest.json", manifest)
    display_queries = [dict(row) for row in queries]
    display_catalog = [dict(row) for row in catalog]
    for row in display_queries + display_catalog:
        if row.get("image_path"):
            row["image_path"] = os.path.relpath(row["image_path"], output).replace(
                "\\", "/"
            )
    display = {"queries": display_queries, "catalog": display_catalog, "labels": labels}
    template = (PROJECT_ROOT / "assets/relevance_review.html").read_text(
        encoding="utf-8"
    )
    page = template.replace(
        "__REVIEW_DATA__",
        json.dumps(display, ensure_ascii=False).replace("<", "\\u003c"),
    )
    ensure_output_path(output / "review.html").write_text(page, encoding="utf-8")
    return manifest


def validate_qrels(
    queries: list[dict], product_ids: set[str], labels: list[dict]
) -> dict[str, set[str]]:
    """Require exhaustive, reviewed binary judgments for every valid pair.

    Args:
        queries: Queries used by this operation.
        product_ids: Product ids used by this operation.
        labels: Judgments used for the stated development or evaluation protocol.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Duplicate query IDs; Unexpected or duplicate qrels pair; Unjudged
            pair or missing reviewer; formal metrics unavailable; Incomplete candidate
            judgments; Recall denominator unknown.
    """
    if len({query["id"] for query in queries}) != len(queries):
        raise ValueError("Duplicate query IDs")
    expected = {
        (query["id"], key)
        for query in queries
        for key in product_ids
        if key != query.get("exclude_id")
    }
    seen: set[tuple[str, str]] = set()
    relevant: dict[str, set[str]] = {query["id"]: set() for query in queries}
    for label in labels:
        pair = (label["query_id"], label["product_id"])
        if pair not in expected or pair in seen:
            raise ValueError("Unexpected or duplicate qrels pair")
        if (
            type(label.get("grade")) is not int
            or label["grade"] not in (0, 1)
            or not str(label.get("reviewer") or "").strip()
        ):
            raise ValueError(
                "Unjudged pair or missing reviewer; formal metrics unavailable"
            )
        seen.add(pair)
        if label["grade"] == 1:
            relevant[pair[0]].add(pair[1])
    if seen != expected:
        raise ValueError("Incomplete candidate judgments; Recall denominator unknown")
    return relevant


def score_rankings(
    queries: list[dict],
    relevant: dict[str, set[str]],
    rankings: dict[str, list[str]],
    k: int = 10,
) -> dict[str, Any]:
    """Aggregate explicit precision and recall, reporting no-positive queries.

    Args:
        queries: Queries used by this operation.
        relevant: Relevant used by this operation.
        rankings: Candidate rankings keyed by query and retrieval method.
        k: K used by this operation.

    Returns:
        Mapping with fields: positive_queries, no_positive_queries,
        macro_over_positive_queries, scope.
    """
    positive, no_positive = [], []
    for query in queries:
        qid = query["id"]
        if not relevant[qid]:
            no_positive.append(qid)
            continue
        positive.append(
            {
                "query_id": qid,
                "kind": query["kind"],
                "relevant_count": len(relevant[qid]),
                **ranking_metrics(rankings[qid], relevant[qid], k),
            }
        )
    groups = {}
    for kind in ("all", "text", "image"):
        subset = [row for row in positive if kind == "all" or row["kind"] == kind]
        groups[kind] = {
            "query_count": len(subset),
            **{
                name: float(np.mean([row[name] for row in subset])) if subset else None
                for name in (f"precision@{k}", f"recall@{k}")
            },
        }
    return {
        "positive_queries": positive,
        "no_positive_queries": no_positive,
        "macro_over_positive_queries": groups,
        "scope": "Exhaustive judgments in this frozen candidate pool; no-positive queries excluded from macro Recall and listed explicitly.",
    }
