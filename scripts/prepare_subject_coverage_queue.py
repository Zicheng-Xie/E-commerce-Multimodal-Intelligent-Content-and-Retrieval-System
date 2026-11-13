"""Freeze a broad, non-metric-filtered next development metadata queue."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import run  # noqa: F401
from multimodal_ecommerce.dataset_release import write_rows
from multimodal_ecommerce.safety import PROJECT_ROOT, PROTECTED_ROOT, sha256_file
from primary_form_coverage import PrimaryFormCoverageScanInputs, scan
from subject_roi_experiment import ROOT, load, once

QUEUE = PROJECT_ROOT / "data/module2_subject_coverage_candidates/records.jsonl"
BUCKETS = [
    "audio_main",
    "audio_accessory",
    "phone_case",
    "footwear",
    "outerwear",
    "cup",
    "cabinet",
    "pillow",
    "bag_organizer",
    "fruit",
]


class _PolicySpec(TypedDict):
    """Exact fields of the POLICY configuration and provenance record."""

    scan_after: int
    scan_limit: int
    seed: int
    metadata_per_bucket: int
    proposed_technical_images_per_bucket: int
    filter_by_relevance_or_scores: bool
    role: str


POLICY: _PolicySpec = dict(
    scan_after=2100000,
    scan_limit=2500000,
    seed=2026102311,
    metadata_per_bucket=8,
    proposed_technical_images_per_bucket=4,
    filter_by_relevance_or_scores=False,
    role="next_exposed_development_metadata_not_acceptance",
)


def classify(title: str) -> str | None:
    """Return classify for this workflow.

    Args:
        title: Public product title used as input.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    if any(x in title for x in ["内胆包", "包内胆", "包中包", "包内收纳"]):
        return "bag_organizer"
    if "耳机" in title or "耳罩" in title:
        return (
            "audio_accessory"
            if any(
                x in title
                for x in ["耳罩", "耳垫", "升级线", "耳机线", "保护套", "收纳包"]
            )
            else "audio_main"
        )
    rules = [
        ("phone_case", ["手机壳", "手机背壳", "手机保护壳"], []),
        ("footwear", ["鞋", "靴"], ["鞋垫", "鞋带", "鞋套", "鞋刷"]),
        (
            "outerwear",
            ["羽绒", "马甲", "外套", "夹克", "棉服"],
            ["宠物", "布料", "修补", "面料"],
        ),
        ("cup", ["水杯", "吸管杯", "学饮杯", "奶瓶"], ["杯盖", "杯刷", "配件"]),
        ("cabinet", ["柜"], ["柜门", "拉手", "铰链", "板材", "贴纸"]),
        ("pillow", ["枕"], ["榴莲", "枕套", "枕头包", "抱枕套", "面包"]),
        (
            "fruit",
            ["榴莲"],
            ["干", "饼", "糖", "糕", "酥", "泥", "壳", "模型", "冰淇淋"],
        ),
    ]
    return next(
        (
            name
            for name, yes, no in rules
            if any(x in title for x in yes) and not any(x in title for x in no)
        ),
        None,
    )


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve frozen coverage queue.

    Notes:
        Writes derived artifacts through once, write_rows.
    """
    folder = ROOT / "next_coverage"
    if QUEUE.exists() or (folder / "protocol.json").exists():
        raise FileExistsError("Preserve frozen coverage queue")
    prior_path = ROOT / "post_experiment_exclusion_snapshot.json"
    prior = load(prior_path)
    source = PROTECTED_ROOT / "多模态电商数据/product1m_product5m_train_id_label.json"
    once(
        folder / "protocol.json",
        dict(
            policy=POLICY,
            buckets=BUCKETS,
            implementation_sha256=sha256_file(Path(__file__)),
            source=source.relative_to(PROJECT_ROOT).as_posix(),
            source_sha256=sha256_file(source),
            prior_sha256=sha256_file(prior_path),
        ),
    )
    rows, meta = scan(
        PrimaryFormCoverageScanInputs(
            source,
            prior,
            POLICY["scan_after"],
            POLICY["scan_limit"],
            POLICY["seed"],
            BUCKETS,
            classify,
            POLICY["metadata_per_bucket"],
        )
    )
    for row in rows:
        row["project_role"] = (
            "next_subject_coverage_development_metadata_not_acceptance"
        )
    write_rows(QUEUE, rows)
    snapshot = dict(prior)
    snapshot.update(
        scope="All prior exposures plus prepared next development metadata; not reusable as unseen",
        files=dict(
            prior["files"],
            **{QUEUE.relative_to(PROJECT_ROOT).as_posix(): sha256_file(QUEUE)},
        ),
        ids=sorted(set(prior["ids"]) | {r["id"] for r in rows}),
        product_source_ids=sorted(
            set(prior["product_source_ids"]) | {r["source_id"] for r in rows}
        ),
        family_keys=sorted(
            set(prior["family_keys"]) | {k for r in rows for k in r["family_keys"]}
        ),
        prepared_subject_coverage_metadata=len(rows),
        parent_snapshot_sha256=sha256_file(prior_path),
    )
    once(folder / "post_preparation_exclusion_snapshot.json", snapshot)
    receipt = dict(
        metadata_candidates=len(rows),
        by_bucket={b: sum(r["bucket"] == b for r in rows) for b in BUCKETS},
        scan=meta,
        queue_sha256=sha256_file(QUEUE),
        new_images_downloaded=0,
        new_AI_judgments=0,
        added_to_current_comparison_gallery=False,
    )
    once(folder / "receipt.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
