"""Freeze and acquire precise primary-subject coverage on exposed development data."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

import httpx
from PIL import Image, ImageDraw, ImageFont, ImageOps

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.holdout_data import is_exposed
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    PROTECTED_ROOT,
    sha256_file,
    write_json,
)
from primary_form_coverage import (
    PrimaryFormCoverageScanInputs,
    fetch_one,
    receipt,
    scan,
)
from subject_roi_experiment import load, once

ROOT = PROJECT_ROOT / "artifacts/module2_primary_subjects"
PARENT = PROJECT_ROOT / "artifacts/module2_subject_coverage_expansion"
QUEUE = PROJECT_ROOT / "data/module2_primary_subject_candidates/records.jsonl"
MEDIA = PROJECT_ROOT / "data/module2_primary_subject_media/records.jsonl"
SOURCE = PROTECTED_ROOT / "多模态电商数据/product1m_product5m_train_id_label.json"
BUCKETS = [
    "overear_headset",
    "mono_call_earphone",
    "handbag_organizer",
    "stiletto_ankle_boot",
    "black_chunky_ankle_boot",
    "adult_male_fur_puffer",
    "down_vest",
    "glass_sideboard",
    "door_drawer_sideboard",
    "painted_sideboard",
    "handled_straw_cup",
    "chunky_sneaker",
    "outdoor_mesh_shoe",
    "arched_neck_pillow",
]


class _PolicySpec(TypedDict):
    """Exact fields of the POLICY configuration and provenance record."""

    role: str
    scan_after: int
    scan_limit: int
    seed: int
    metadata_per_bucket: int
    download_every_frozen_candidate: bool
    network_workers: int
    timeout_seconds: int
    max_attempts_per_original_url: int
    respect_environment_proxy: bool
    bypass_access_rejection: bool
    keep_all_technical_success_nonduplicates: bool
    technical_dhash_distance: int
    relevance_or_score_filter: bool
    labels_available_during_acquisition: bool
    formal_acceptance: bool
    formal_metrics: None


POLICY: _PolicySpec = dict(
    role="query_aware_exposed_primary_subject_development_not_acceptance",
    scan_after=2500000,
    scan_limit=3500000,
    seed=2026102403,
    metadata_per_bucket=24,
    download_every_frozen_candidate=True,
    network_workers=4,
    timeout_seconds=25,
    max_attempts_per_original_url=1,
    respect_environment_proxy=True,
    bypass_access_rejection=False,
    keep_all_technical_success_nonduplicates=True,
    technical_dhash_distance=4,
    relevance_or_score_filter=False,
    labels_available_during_acquisition=False,
    formal_acceptance=False,
    formal_metrics=None,
)


def any_of(text: str, words: Sequence[str]) -> bool:
    """Return any of for this workflow.

    Args:
        text: Text input used by the operation.
        words: Words used by this operation.

    Returns:
        Whether the condition checked by this operation holds.
    """
    return any(word in text for word in words)


def classify(title: str) -> str | None:
    """Source-title coverage strata, never relevance judgments or rank labels.

    Args:
        title: Public product title used as input.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    t = title.casefold()
    audio_accessories = [
        "耳机套",
        "耳罩套",
        "耳机罩",
        "保护套",
        "耳机支架",
        "收纳",
        "耳机包",
        "耳机线",
        "升级线",
        "替换线",
        "延长线",
        "耳垫",
        "配件",
        "耳机架",
    ]
    if "耳机" in t and not any_of(t, audio_accessories):
        if "头戴" in t:
            return "overear_headset"
        if (
            "单耳" in t
            and "蓝牙" in t
            and any_of(t, ["通话", "商务", "开车", "挂耳"])
            and not any_of(t, ["tws", "双耳", "一对"])
        ):
            return "mono_call_earphone"
    if any_of(t, ["内胆包", "包内胆", "包中包", "包内收纳"]) and any_of(
        t, ["手袋", "手提", "整理", "收纳", "女包", "托特", "化妆", "水桶包"]
    ):
        if not any_of(
            t,
            [
                "电脑",
                "相机",
                "单反",
                "镜头",
                "平板",
                "ipad",
                "macbook",
                "笔记本",
                "摄影",
                "无人机",
            ],
        ):
            return "handbag_organizer"
    if not any_of(t, ["鞋垫", "鞋套", "鞋带", "鞋架", "鞋盒", "靴撑"]):
        if "短靴" in t and "尖头" in t and "细跟" in t:
            return "stiletto_ankle_boot"
        if (
            "马丁" in t
            and "靴" in t
            and "厚底" in t
            and "黑" in t
            and any_of(t, ["短", "踝", "低帮", "系带"])
        ):
            return "black_chunky_ankle_boot"
        if "老爹鞋" in t and "厚底" in t:
            return "chunky_sneaker"
        if (
            "鞋" in t
            and any_of(t, ["户外", "徒步", "登山", "涉水", "越野"])
            and any_of(t, ["网面", "透气", "网布"])
        ):
            return "outdoor_mesh_shoe"
    if "羽绒" in t and not any_of(t, ["布料", "面料", "修补", "宠物", "狗狗", "猫咪"]):
        if any_of(t, ["马甲", "背心"]):
            return "down_vest"
        if (
            "男" in t
            and any_of(t, ["中长", "长款"])
            and any_of(t, ["毛领", "狐狸毛", "貉子毛", "獭兔毛", "毛帽"])
            and not any_of(t, ["儿童", "童装", "男童", "女童", "宝宝", "小童", "中童"])
        ):
            return "adult_male_fur_puffer"
    if "餐边柜" in t and not any_of(t, ["拉手", "铰链", "柜门板", "贴纸", "模型"]):
        if any_of(t, ["彩绘", "彩色", "多色", "拼色"]) and not any_of(
            t, ["高柜", "立柜", "顶柜"]
        ):
            return "painted_sideboard"
        if "玻璃" in t:
            return "glass_sideboard"
        if "抽屉" in t and "门" in t:
            return "door_drawer_sideboard"
    if (
        "吸管杯" in t
        and any_of(t, ["双耳", "双柄", "手柄", "把手"])
        and any_of(t, ["透明", "塑料", "ppsu", "tritan", "儿童", "宝宝", "婴儿"])
    ):
        if not any_of(t, ["杯盖", "杯刷", "替换", "配件", "保温", "不锈钢"]):
            return "handled_straw_cup"
    if (
        "枕" in t
        and any_of(t, ["木", "竹"])
        and any_of(t, ["颈", "颈椎"])
        and any_of(t, ["拱", "弧形", "弯曲"])
    ):
        if not any_of(t, ["枕套", "榴莲", "面包", "内胆包"]):
            return "arched_neck_pillow"
    return None


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        FileExistsError: Preserve frozen precise-subject acquisition.

    Notes:
        Writes derived artifacts through once, write_rows.
    """
    if QUEUE.exists() or (ROOT / "acquisition_protocol.json").exists():
        raise FileExistsError("Preserve frozen precise-subject acquisition")
    prior_path = PARENT / "post_experiment_exclusion_snapshot.json"
    prior = load(prior_path)
    verify_files(PROJECT_ROOT, prior["files"])
    parent = load(PARENT / "completion_receipt.json")
    verify_files(PARENT, parent["files"])
    protocol = dict(
        policy=POLICY,
        buckets=BUCKETS,
        source=SOURCE.relative_to(PROJECT_ROOT).as_posix(),
        source_sha256=sha256_file(SOURCE),
        source_bytes=SOURCE.stat().st_size,
        source_record_count="not separately counted; streaming scan reports its real stop position",
        prior_exclusion=prior_path.relative_to(PROJECT_ROOT).as_posix(),
        prior_sha256=sha256_file(prior_path),
        files={
            p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
            for p in [
                Path(__file__),
                PROJECT_ROOT / "scripts/primary_form_coverage.py",
                PROJECT_ROOT / "src/multimodal_ecommerce/media.py",
                prior_path,
                PARENT / "completion_receipt.json",
            ]
        },
    )
    once(ROOT / "acquisition_protocol.json", protocol)
    rows, meta = scan(
        PrimaryFormCoverageScanInputs(
            SOURCE,
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
        row["project_role"] = POLICY["role"]
    write_rows(QUEUE, rows)
    counts = Counter(r["bucket"] for r in rows)
    snapshot = dict(prior)
    snapshot.update(
        scope="All prior exposures plus all precise-subject development candidate metadata; never independent acceptance",
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
        parent_snapshot_sha256=sha256_file(prior_path),
        primary_subject_metadata=len(rows),
    )
    once(ROOT / "post_preparation_exclusion_snapshot.json", snapshot)
    once(
        ROOT / "metadata_receipt.json",
        dict(
            candidates=len(rows),
            by_bucket={b: counts[b] for b in BUCKETS},
            scan=meta,
            shortages={
                b: POLICY["metadata_per_bucket"] - counts[b]
                for b in BUCKETS
                if counts[b] < POLICY["metadata_per_bucket"]
            },
            queue_sha256=sha256_file(QUEUE),
            no_new_images=True,
            no_relevance_labels=True,
        ),
    )
    print(load(ROOT / "metadata_receipt.json"), flush=True)


def check() -> None:
    """Verify the prerequisites and integrity of the current workflow."""
    protocol = load(ROOT / "acquisition_protocol.json")
    assert protocol["policy"] == POLICY
    verify_files(PROJECT_ROOT, protocol["files"])
    assert sha256_file(QUEUE) == load(ROOT / "metadata_receipt.json")["queue_sha256"]


def acquire() -> None:
    """Apply the acquire workflow step.

    Raises:
        FileExistsError: Preserve completed downloads.

    Notes:
        Writes derived artifacts through once, receipt, write_json, write_rows.
    """
    check()
    if (ROOT / "acquisition_receipt.json").exists():
        raise FileExistsError("Preserve completed downloads")
    rows = read_rows(QUEUE)
    cache_path = ROOT / "download_cache.json"
    cache = load(cache_path) if cache_path.exists() else {}
    pending = [r for r in rows if r["id"] not in cache]
    with httpx.Client(timeout=POLICY["timeout_seconds"], trust_env=True) as client:
        with ThreadPoolExecutor(max_workers=POLICY["network_workers"]) as executor:
            for row, result in zip(
                pending, executor.map(lambda r: fetch_one(client, r, ROOT), pending)
            ):
                cache[row["id"]] = result
                write_json(cache_path, cache)
                if len(cache) % 12 == 0 or len(cache) == len(rows):
                    print(
                        dict(
                            attempted=len(cache),
                            total=len(rows),
                            downloaded=sum(
                                v["status"] == "downloaded" for v in cache.values()
                            ),
                        ),
                        flush=True,
                    )
    prior = load(PARENT / "post_experiment_exclusion_snapshot.json")
    signatures = list(prior["image_fingerprints"])
    attempts, chosen = [], []
    for row in rows:
        result = cache[row["id"]]
        duplicate = result["status"] == "downloaded" and is_exposed(
            result["fingerprints"], signatures, POLICY["technical_dhash_distance"]
        )
        selected = result["status"] == "downloaded" and not duplicate
        attempts.append(
            dict(
                **row,
                **result,
                technical_duplicate=bool(duplicate),
                selected_for_gallery=selected,
                attempt_count=1,
            )
        )
        if selected:
            chosen.append(dict(**row, **result))
            signatures.append(result["fingerprints"])
    write_rows(MEDIA, attempts)
    write_rows(ROOT / "selected.evaluator.jsonl", chosen)
    once(
        ROOT / "additions/catalog.public.json",
        [dict(id=r["id"], title=r["text"], image_path=r["image_path"]) for r in chosen],
    )
    errors = Counter(
        "HTTP420" if "420" in r.get("error", "") else r.get("error_type", "unknown")
        for r in attempts
        if r["status"] != "downloaded"
    )
    receipt(
        ROOT,
        "acquisition_receipt.json",
        [
            ROOT / "download_cache.json",
            ROOT / "selected.evaluator.jsonl",
            ROOT / "additions/catalog.public.json",
        ],
        candidates=len(rows),
        attempted=len(cache),
        downloaded=sum(r["status"] == "downloaded" for r in attempts),
        failed=sum(r["status"] != "downloaded" for r in attempts),
        failure_reasons=dict(errors),
        technical_duplicates=sum(r["technical_duplicate"] for r in attempts),
        selected=len(chosen),
        selected_by_bucket=dict(Counter(r["bucket"] for r in chosen)),
        media_registry_sha256=sha256_file(MEDIA),
        all_technical_success_nonduplicates_retained=True,
        AI_relevance_labels=0,
        formal_metrics=None,
    )
    snapshot = load(ROOT / "post_preparation_exclusion_snapshot.json")
    snapshot.update(
        files=dict(
            snapshot["files"],
            **{MEDIA.relative_to(PROJECT_ROOT).as_posix(): sha256_file(MEDIA)},
        ),
        image_fingerprints=snapshot["image_fingerprints"]
        + [r["fingerprints"] for r in attempts if r["status"] == "downloaded"],
        primary_subject_downloaded=sum(r["status"] == "downloaded" for r in attempts),
        scope="All prior exposures plus every precise-subject metadata and captured image, including technical duplicates; not acceptance",
    )
    once(ROOT / "post_acquisition_exclusion_snapshot.json", snapshot)
    print(load(ROOT / "acquisition_receipt.json"), flush=True)


def sheets() -> None:
    """Apply the sheets workflow step.

    Raises:
        FileExistsError: Preserve native contact sheets.

    Notes:
        Writes derived artifacts through canvas.save, path.parent.mkdir, receipt.
    """
    check()
    verify_files(ROOT, load(ROOT / "acquisition_receipt.json")["files"])
    docs = load(ROOT / "additions/catalog.public.json")
    folder = ROOT / "native"
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
    for start in range(0, len(docs), 8):
        canvas = Image.new("RGB", (1600, 820), "white")
        draw = ImageDraw.Draw(canvas)
        for off, doc in enumerate(docs[start : start + 8]):
            x, y = off % 4 * 400, off // 4 * 410
            with Image.open(doc["image_path"]) as im:
                pic = ImageOps.contain(
                    ImageOps.exif_transpose(im).convert("RGB"), (390, 300)
                )
            canvas.paste(pic, (x + (400 - pic.width) // 2, y))
            draw.text(
                (x + 5, y + 300),
                f"gallery {start + off + 1} | {doc['id']}",
                font=font,
                fill="black",
            )
            for line in range(4):
                draw.text(
                    (x + 5, y + 325 + 19 * line),
                    doc["title"][line * 22 : (line + 1) * 22],
                    font=font,
                    fill="black",
                )
        path = folder / f"gallery_{start // 8 + 1:02d}.jpg"
        if path.exists():
            raise FileExistsError("Preserve native contact sheets")
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=97)
    receipt(
        ROOT,
        "native_image_receipt.json",
        sorted(folder.glob("gallery_*.jpg")),
        images=len(docs),
        pages=(len(docs) + 7) // 8,
        native_relevance_review_completed=False,
    )
    print(dict(contact_sheets=(len(docs) + 7) // 8, images=len(docs)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "acquire", "sheets"])
    globals()[parser.parse_args().action]()
