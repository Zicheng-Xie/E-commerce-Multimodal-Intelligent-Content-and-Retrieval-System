"""Prepare train-source development metadata; never use heldout quality outcomes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

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

ROOT = PROJECT_ROOT / "artifacts/module2_expanded_development/coverage_next"
REGISTRY = PROJECT_ROOT / "data/module2_visual_coverage_candidates/records.jsonl"
POLICY: dict = dict(
    seed=2026102209,
    source_split="train",
    scan_limit=200000,
    metadata_per_bucket=16,
    future_images_per_bucket=8,
    acquisition_uses_query_coverage=True,
    role="exposed_development_candidates_only",
    acceptance_selection_or_predictions=False,
    labels_created=0,
    model_inferences=0,
)


def bucket(title: str) -> str | None:
    """Return bucket for this workflow.

    Args:
        title: Public product title used as input.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    text = title.casefold()
    if any(
        t in text
        for t in ["气门帽", "气嘴帽", "气门嘴帽", "气嘴盖", "气门嘴盖", "轮胎帽"]
    ):
        return "tire_valve_caps"
    if "手机支架" in text and any(
        t in text for t in ["车载", "自行车", "摩托", "电动"]
    ):
        return "vehicle_phone_holders"
    if (
        "置物架" in text
        and any(t in text for t in ["调料", "调味", "厨房"])
        and any(t in text for t in ["壁挂", "墙", "免打孔"])
    ):
        return "wall_spice_racks"
    if (
        any(t in text for t in ["西装", "西服"])
        and "女" in text
        and "裤" in text
        and any(t in text for t in ["套装", "两件套"])
    ):
        return "women_pantsuits"
    if any(t in text for t in ["餐边柜", "边柜", "玄关柜"]) and any(
        t in text for t in ["红木", "中式"]
    ):
        return "wooden_sideboards"
    return None


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve prepared coverage candidates.

    Notes:
        Writes derived artifacts through REGISTRY.parent.mkdir, ROOT.mkdir,
        write_json, write_rows.
    """
    if ROOT.exists() or REGISTRY.exists():
        raise FileExistsError("Preserve prepared coverage candidates")
    ROOT.mkdir(parents=True)
    source = PROTECTED_ROOT / "多模态电商数据/product1m_product5m_train_id_label.json"
    snapshot = (
        PROJECT_ROOT
        / "artifacts/module2_catalog_review/post_preparation_exclusion_snapshot.json"
    )
    prior = json.loads(snapshot.read_text(encoding="utf-8"))
    verify_files(PROJECT_ROOT, prior["files"])
    excluded = set(prior["family_keys"])
    protocol = dict(
        policy=POLICY,
        source=str(source.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        source_sha256=sha256_file(source),
        prior_snapshot_sha256=sha256_file(snapshot),
        reserved_registry_sha256=sha256_file(
            PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl"
        ),
        implementation_sha256=sha256_file(
            PROJECT_ROOT / "scripts/prepare_visual_coverage_candidates.py"
        ),
        no_relevance_judgments=True,
        independent_test_use_allowed=False,
        limitations="Query-aware development coverage acquisition; not representative traffic or validation evidence",
    )
    write_json(ROOT / "protocol.json", protocol)
    groups: dict = {
        n: []
        for n in [
            "wooden_sideboards",
            "vehicle_phone_holders",
            "wall_spice_racks",
            "women_pantsuits",
            "tire_valve_caps",
        ]
    }
    scanned = 0
    with source.open("rb") as handle:
        for sid, raw in ijson.kvitems(handle, ""):
            scanned += 1
            name = bucket(str(raw.get("title") or ""))
            if name and raw.get("url"):
                row = dict(
                    id=f"product/train/{sid}",
                    source_id=str(sid),
                    dataset="product",
                    split="train",
                    label=raw.get("label"),
                    text=raw.get("title"),
                    image_url=raw["url"],
                    attributes=parse_pv(raw.get("pv") or ""),
                    bucket=name,
                    source_position=scanned,
                    source_record_sha256=canonical_hash(raw),
                )
                keys = product_family_keys(row)
                if not keys & excluded:
                    row.update(
                        family_keys=sorted(keys),
                        priority=hashlib.sha256(
                            f"{POLICY['seed']}:{row['id']}".encode()
                        ).hexdigest(),
                    )
                    groups[name].append(row)
            if scanned == POLICY["scan_limit"]:
                break
    assert scanned == POLICY["scan_limit"]
    rows, seen = [], set(excluded)
    for name, values in groups.items():
        count = 0
        for row in sorted(values, key=lambda r: r["priority"]):
            if set(row["family_keys"]) & seen:
                continue
            rows.append(
                dict(
                    row,
                    project_role="exposed_development_metadata_not_relevance_truth",
                    image_path=None,
                )
            )
            seen.update(row["family_keys"])
            count += 1
            if count == POLICY["metadata_per_bucket"]:
                break
    write_rows(ROOT / "source_candidates.jsonl", rows)
    REGISTRY.parent.mkdir(parents=True)
    write_rows(REGISTRY, rows)
    write_json(
        ROOT / "receipt.json",
        dict(
            scanned=scanned,
            metadata_candidates=len(rows),
            per_bucket=dict(Counter(r["bucket"] for r in rows)),
            target_future_images=40,
            candidate_images_downloaded=0,
            inference_count=0,
            relevance_labels=0,
            files={
                "source_candidates.jsonl": sha256_file(
                    ROOT / "source_candidates.jsonl"
                ),
                "protocol.json": sha256_file(ROOT / "protocol.json"),
            },
            exposure_registry=str(REGISTRY.relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ),
            exposure_registry_sha256=sha256_file(REGISTRY),
            next_steps="Download public source media, deduplicate pixels and product families, AI visual QA, then complete all development relations before index extension",
        ),
    )
    print(
        json.dumps(
            dict(
                metadata_candidates=len(rows),
                buckets=dict(Counter(r["bucket"] for r in rows)),
                future_target_images=40,
                acceptance_predictions=0,
            )
        )
    )


if __name__ == "__main__":
    main()
