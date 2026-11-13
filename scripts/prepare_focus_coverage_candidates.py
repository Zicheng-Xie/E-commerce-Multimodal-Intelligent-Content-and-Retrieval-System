"""Freeze more specific public metadata for future development coverage only."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import TypedDict

import ijson

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_acceptance import product_family_keys
from multimodal_ecommerce.business_review_jobs import canonical_hash
from multimodal_ecommerce.data import parse_pv
from multimodal_ecommerce.dataset_release import write_rows
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    PROTECTED_ROOT,
    sha256_file,
    write_json,
)
from prepare_visual_coverage import ROOT

DESTINATION = ROOT / "coverage_next"
REGISTRY = PROJECT_ROOT / "data/module2_focus_coverage_candidates/records.jsonl"


class _PolicySpec(TypedDict):
    """Exact fields of the POLICY configuration and provenance record."""

    seed: int
    source_split: str
    scan_limit: int
    metadata_per_bucket: int
    future_images_per_bucket: int
    acquisition_uses_known_development_queries: bool
    acquisition_uses_reserved_data_or_scores: bool
    role: str
    labels_created: int
    model_inferences: int
    independent_test_use_allowed: bool


POLICY: _PolicySpec = dict(
    seed=2026102310,
    source_split="train",
    scan_limit=200000,
    metadata_per_bucket=32,
    future_images_per_bucket=12,
    acquisition_uses_known_development_queries=True,
    acquisition_uses_reserved_data_or_scores=False,
    role="query_aware_exposed_development_metadata_only",
    labels_created=0,
    model_inferences=0,
    independent_test_use_allowed=False,
)


def bucket(title: str) -> str | None:
    """Return bucket for this workflow.

    Args:
        title: Public product title used as input.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    text = title.casefold()
    if (
        any(t in text for t in ["调料", "调味"])
        and "架" in text
        and any(t in text for t in ["壁挂", "挂墙", "墙", "免打孔", "挂式"])
        and not any(
            t in text for t in ["纸巾", "卷纸", "毛巾", "洞洞", "三角", "角码", "支撑"]
        )
    ):
        return "wall_bottle_racks"
    if (
        any(t in text for t in ["餐边柜", "餐柜", "餐厅柜", "碗柜", "酒柜"])
        and any(t in text for t in ["红木", "中式", "仿古", "花梨", "紫檀", "明清"])
        and not any(
            t in text
            for t in ["博古", "多宝", "鞋柜", "换鞋", "凳", "鱼缸", "盆景", "展示架"]
        )
    ):
        return "classical_storage_cabinets"
    return None


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve prepared future metadata.

    Notes:
        Writes derived artifacts through DESTINATION.mkdir, REGISTRY.parent.mkdir,
        write_json, write_rows.
    """
    if DESTINATION.exists() or REGISTRY.exists():
        raise FileExistsError("Preserve prepared future metadata")
    source = PROTECTED_ROOT / "多模态电商数据/product1m_product5m_train_id_label.json"
    snapshot = ROOT / "post_development_exclusion_snapshot.json"
    prior = json.loads(snapshot.read_text(encoding="utf-8"))
    verify_files(PROJECT_ROOT, prior["files"])
    excluded = set(prior["family_keys"])
    DESTINATION.mkdir()
    write_json(
        DESTINATION / "protocol.json",
        dict(
            policy=POLICY,
            source=source.relative_to(PROJECT_ROOT).as_posix(),
            source_sha256=sha256_file(source),
            prior_snapshot_sha256=sha256_file(snapshot),
            implementation_sha256=sha256_file(
                PROJECT_ROOT / "scripts/prepare_focus_coverage_candidates.py"
            ),
            selection="Public original titles only, deterministic hash order and family dedup; no relevance filtering, no new image pixels opened",
            limitation="Development query-aware acquisition and biased coverage; never independent validation",
        ),
    )
    groups: dict = {
        name: [] for name in ["wall_bottle_racks", "classical_storage_cabinets"]
    }
    with source.open("rb") as handle:
        for number, (sid, raw) in enumerate(ijson.kvitems(handle, ""), 1):
            name = bucket(str(raw.get("title") or ""))
            if name and raw.get("url"):
                row = dict(
                    id=f"product/train/{sid}",
                    source_id=str(sid),
                    dataset="product",
                    split="train",
                    label=raw.get("label"),
                    text=raw["title"],
                    image_url=raw["url"],
                    attributes=parse_pv(raw.get("pv") or ""),
                    bucket=name,
                    source_position=number,
                    source_record_sha256=canonical_hash(raw),
                )
                keys = product_family_keys(row)
                if not keys & excluded:
                    row.update(
                        family_keys=sorted(keys),
                        priority=hashlib.sha256(
                            f"{POLICY['seed']}:{row['id']}".encode()
                        ).hexdigest(),
                        project_role="exposed_development_metadata_not_relevance_truth",
                        image_path=None,
                    )
                    groups[name].append(row)
            if number == POLICY["scan_limit"]:
                break
    assert number == POLICY["scan_limit"]
    rows, seen = [], set(excluded)
    for name, values in groups.items():
        count = 0
        for row in sorted(values, key=lambda r: r["priority"]):
            if set(row["family_keys"]) & seen:
                continue
            rows.append(row)
            seen.update(row["family_keys"])
            count += 1
            if count == POLICY["metadata_per_bucket"]:
                break
    write_rows(DESTINATION / "source_candidates.jsonl", rows)
    REGISTRY.parent.mkdir()
    write_rows(REGISTRY, rows)
    write_json(
        DESTINATION / "receipt.json",
        dict(
            scanned=number,
            selected_metadata=len(rows),
            per_bucket=dict(Counter(r["bucket"] for r in rows)),
            eligible_before_family_dedup={b: len(v) for b, v in groups.items()},
            future_target_images=24,
            images_downloaded=0,
            labels_created=0,
            model_inferences=0,
            acceptance_predictions=0,
            exposure_registry=REGISTRY.relative_to(PROJECT_ROOT).as_posix(),
            exposure_registry_sha256=sha256_file(REGISTRY),
            files={
                p.name: sha256_file(p) for p in DESTINATION.iterdir() if p.is_file()
            },
        ),
    )
    snapshot2 = dict(prior)
    snapshot2["ids"] = sorted(set(prior["ids"]) | {r["id"] for r in rows})
    snapshot2["product_source_ids"] = sorted(
        set(prior["product_source_ids"]) | {r["source_id"] for r in rows}
    )
    snapshot2["family_keys"] = sorted(seen)
    snapshot2["files"] = dict(
        prior["files"],
        **{REGISTRY.relative_to(PROJECT_ROOT).as_posix(): sha256_file(REGISTRY)},
    )
    snapshot2["parent_snapshot_sha256"] = sha256_file(snapshot)
    write_json(ROOT / "post_preparation_exclusion_snapshot.json", snapshot2)
    print(
        json.dumps(
            dict(
                metadata=len(rows),
                per_bucket=dict(Counter(r["bucket"] for r in rows)),
                future_target_images=24,
                labels_created=0,
                acceptance_predictions=0,
            )
        )
    )


if __name__ == "__main__":
    main()
