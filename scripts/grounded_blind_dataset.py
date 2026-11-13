"""One new independent closed-catalog AI blind dataset; never tune on its pixels."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw, ImageFont, ImageOps

import run  # noqa: F401
from grounded_subject_experiment import ROOT as DEVELOPMENT
from learned_fusion_experiment import load, once, rel
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.holdout_data import is_exposed
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    PROTECTED_ROOT,
    sha256_file,
    write_json,
)
from primary_form_coverage import PrimaryFormCoverageScanInputs, fetch_one, scan

ROOT = PROJECT_ROOT / "artifacts/module2_grounded_independent_blind"
QUEUE = ROOT / "evaluator/candidates.jsonl"
CONSUMED = PROJECT_ROOT / "data/module2_grounded_blind_consumed/records.jsonl"
BUCKETS = [
    "overear",
    "earbuds",
    "phone_case",
    "organizer",
    "laptop_sleeve",
    "down_vest",
    "puffer",
    "ankle_boot",
    "sneaker",
    "sideboard",
    "straw_cup",
    "pillow",
]
TEXTS = [
    "头戴式耳机",
    "入耳式双耳耳机",
    "手机保护壳",
    "手袋内胆收纳包",
    "笔记本电脑内胆包",
    "羽绒马甲",
    "有袖羽绒服或棉服",
    "短靴",
    "运动鞋或跑步鞋",
    "餐边收纳柜",
    "吸管杯",
    "枕头或枕芯",
]
POLICY: dict = dict(
    seed=2026102502,
    scan_after=4000000,
    scan_limit=5000000,
    metadata_per_bucket=48,
    buckets=BUCKETS,
    gallery_per_bucket=8,
    image_queries_per_bucket=1,
    text_queries=TEXTS,
    download_workers=4,
    timeout_seconds=25,
    attempts_per_original_url=1,
    respect_proxy=True,
    bypass_access_rejection=False,
    technical_dhash_distance=4,
    select="first nine readable globally nonduplicate images in frozen seed order per stratum: first eight gallery, ninth image query; no relevance or ranking selection",
    role="new one-shot independently sourced AI blind, balanced 12 predefined types; not traffic-representative or human validation",
    gallery=96,
    queries=24,
    relations=2304,
    K=10,
    human_labels=0,
    relevance="Same PDF business relevance: text matches requested main type/use/explicit attributes; image matches primary product type/use and similar key visible form, not necessarily SKU or brand. Background or coarse category alone is insufficient. Unknown material/compatibility/unclear primary stays null. No label or relevance-definition changes for high scores.",
    annotation="Codex native image review, all gallery profiles before query opening, complete gallery-by-query judgments before predictions; reviewer knows project background, no independent human or annotator-model claim",
    AI_labels_frozen_before_predictions=True,
    one_predeclared_candidate=True,
    development_gate_failure="If run as user-authorized diagnostic, disclose development failure; no promotion or attainment claim. Never select another method on blind results.",
    retain_all_selected=True,
    unknowns_retained=True,
    score_driven_label_edits=False,
    formal_metrics=None,
)
SOURCE = PROTECTED_ROOT / "多模态电商数据/product1m_product5m_train_id_label.json"
SOURCES = [
    "scripts/grounded_blind_dataset.py",
    "scripts/grounded_blind_runtime.py",
    "scripts/grounded_blind_report.py",
    "scripts/grounded_subject_runtime.py",
    "scripts/grounded_subject_recovery.py",
    "scripts/primary_form_coverage.py",
    "scripts/primary_subject_captions_safe.py",
    "scripts/primary_subject_experiment.py",
    "scripts/learned_fusion_experiment.py",
    "scripts/balanced_supplement.py",
    "scripts/roi_semantic_gate_review.py",
    "scripts/caption_roi_assets_safe.py",
    "tests/test_grounded_blind_runtime.py",
]


def classify(title: str) -> str | None:
    """Return classify for this workflow.

    Args:
        title: Public product title used as input.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    t = title.casefold()
    if "手机壳" in t and not any(w in t for w in ["材料", "配件", "滴胶", "挂绳"]):
        return "phone_case"
    if any(w in t for w in ["电脑内胆包", "笔记本包", "笔记本电脑包"]):
        return "laptop_sleeve"
    if any(w in t for w in ["内胆包", "包中包", "包内胆"]) and not any(
        w in t for w in ["电脑", "相机", "平板"]
    ):
        return "organizer"
    if "耳机" in t and not any(
        w in t
        for w in ["保护", "耳垫", "耳罩套", "耳机套", "收纳", "支架", "耳机线", "配件"]
    ):
        if "头戴" in t:
            return "overear"
        if any(w in t for w in ["入耳", "双耳", "真无线"]) and "单耳" not in t:
            return "earbuds"
    if any(w in t for w in ["羽绒", "棉服"]) and not any(
        w in t for w in ["布料", "修补", "面料", "宠物"]
    ):
        if any(w in t for w in ["马甲", "背心"]):
            return "down_vest"
        if any(w in t for w in ["羽绒服", "棉服"]):
            return "puffer"
    if not any(w in t for w in ["鞋带", "鞋垫", "鞋盒", "鞋套", "靴撑"]):
        if "短靴" in t:
            return "ankle_boot"
        if any(w in t for w in ["运动鞋", "跑步鞋"]):
            return "sneaker"
    if "餐边柜" in t and not any(w in t for w in ["拉手", "柜门配件"]):
        return "sideboard"
    if "吸管杯" in t and not any(w in t for w in ["配件", "杯盖", "替换", "杯刷"]):
        return "straw_cup"
    if any(w in t for w in ["枕头", "枕芯"]) and not any(
        w in t for w in ["枕套", "填充棉", "面料"]
    ):
        return "pillow"
    return None


def receipt(name: str, paths: list[Path], **extra: object) -> None:
    """Record file hashes and provenance for the completed stage.

    Args:
        name: Name used by this operation.
        paths: Paths used by this operation.
        **extra: Named options forwarded to the underlying operation.

    Notes:
        Writes derived artifacts through once.
    """
    once(ROOT / name, dict(files={rel(p): sha256_file(p) for p in paths}, **extra))


def checked(name: str) -> Any:
    """Return checked for this workflow.

    Args:
        name: Name used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    r = load(ROOT / name)
    verify_files(PROJECT_ROOT, r["files"])
    return r


def status(phase: str, **extra: object) -> None:
    """Apply the status workflow step.

    Args:
        phase: Phase used by this operation.
        **extra: Named options forwarded to the underlying operation.

    Notes:
        Writes derived artifacts through write_json.
    """
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=False,
            phase=phase,
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            **extra,
        ),
    )


def freeze() -> None:
    """Freeze code/method and sampling before scanning or opening any new image."""
    if (ROOT / "release_protocol.json").exists():
        guard()
        return
    assert load(ROOT / "engineering_checks.pre.json")["passed"]
    paths = [PROJECT_ROOT / s for s in SOURCES] + sorted(
        (PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")
    )
    paths += [
        ROOT / "engineering_checks.pre.json",
        DEVELOPMENT / "protocol.json",
        DEVELOPMENT / "semantic_abstention_execution_protocol.json",
        DEVELOPMENT / "grounder_model_manifest.json",
        PROJECT_ROOT / "configs/module2_current_target.json",
        PROJECT_ROOT / "configs/embedding_model.json",
        PROJECT_ROOT / "configs/reranker_model.json",
        PROJECT_ROOT / "configs/retrieval_regions_v1.json",
        PROJECT_ROOT / "configs/local.json",
        PROJECT_ROOT / "vendor/qwen3_vl_reranker/qwen3_vl_reranker.py",
        PROJECT_ROOT / "vendor/qwen3_vl_embedding/qwen3_vl_embedding.py",
    ]
    once(
        ROOT / "release_protocol.json",
        dict(
            files={rel(p): sha256_file(p) for p in paths},
            policy=POLICY,
            candidate="grounded_whole_zoom_union",
            candidate_source_protocol_sha256=sha256_file(DEVELOPMENT / "protocol.json"),
            sampling_source=rel(SOURCE),
            source_sha256=sha256_file(SOURCE),
            heldout_pixels_opened=0,
            method_and_data_rules_frozen_before_new_metadata_scan=True,
            formal_metrics=None,
        ),
    )
    status("candidate_and_blind_rules_frozen_no_pixels_opened")


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    p = load(ROOT / "release_protocol.json")
    verify_files(PROJECT_ROOT, p["files"])
    assert p["policy"] == POLICY


def reserve() -> None:
    """Return reserve for this workflow.

    Notes:
        Writes derived artifacts through QUEUE.parent.mkdir, receipt, write_rows.
    """
    guard()
    if (ROOT / "reservation_receipt.json").exists():
        checked("reservation_receipt.json")
        return
    # Wait for the final crop exposure snapshot before constructing the heldout.
    from grounded_subject_experiment import verify

    verify()
    prior_path = DEVELOPMENT / "post_experiment_exclusion_snapshot.json"
    prior = load(prior_path)
    verify_files(PROJECT_ROOT, prior["files"])
    rows, counts = scan(
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
    assert {r["bucket"] for r in rows} == set(BUCKETS)
    for row in rows:
        row["project_role"] = "reserved_new_grounded_AI_blind_never_development"
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    if QUEUE.exists():
        assert read_rows(QUEUE) == rows
    else:
        write_rows(QUEUE, rows)
    receipt(
        "reservation_receipt.json",
        [QUEUE, prior_path],
        prior_snapshot_sha256=sha256_file(prior_path),
        candidates=len(rows),
        by_bucket=dict(Counter(r["bucket"] for r in rows)),
        scan=counts,
        heldout_pixels_opened=0,
    )
    status("new_blind_metadata_reserved_no_pixels_opened")


def acquire() -> None:
    """Return acquire for this workflow.

    Notes:
        Writes derived artifacts through once, receipt, write_rows.
    """
    guard()
    checked("reservation_receipt.json")
    if (ROOT / "acquisition_receipt.json").exists():
        checked("acquisition_receipt.json")
        return
    rows = read_rows(QUEUE)

    def one(row: dict) -> Any:
        """Return one for this workflow.

        Args:
            row: Row used by this operation.

        Returns:
            Decoded JSON value; the caller must validate its expected schema.

        Notes:
            Writes derived artifacts through once.
        """
        p = (
            ROOT
            / "downloads"
            / (hashlib.sha256(row["id"].encode()).hexdigest() + ".json")
        )
        if not p.exists():
            with httpx.Client(
                timeout=POLICY["timeout_seconds"], trust_env=True
            ) as client:
                once(p, dict(id=row["id"], **fetch_one(client, row, ROOT)))
        return load(p)

    with ThreadPoolExecutor(max_workers=POLICY["download_workers"]) as workers:
        downloads = list(workers.map(one, rows))
    prior = load(DEVELOPMENT / "post_experiment_exclusion_snapshot.json")
    seen = list(prior["image_fingerprints"])
    selected = []
    technical = []
    for bucket in BUCKETS:
        candidates = [(r, d) for r, d in zip(rows, downloads) if r["bucket"] == bucket]
        usable = []
        for row, d in candidates:
            if d["status"] != "downloaded":
                continue
            fp = d["fingerprints"]
            if is_exposed(fp, seen, 4):
                technical.append(
                    dict(id=row["id"], reason="prior_or_within_release_image_duplicate")
                )
                continue
            seen.append(fp)
            technical.append(dict(id=row["id"], reason="readable_nonduplicate"))
            usable.append(dict(row, **{k: v for k, v in d.items() if k != "id"}))
        if len(usable) < 9:
            raise RuntimeError(
                f"Frozen blind stratum {bucket} has only {len(usable)} readable nonduplicates; no semantic replacement or score-based resampling"
            )
        selected += [
            dict(
                r,
                role="gallery" if i < 8 else "image_query",
                opaque_id=hashlib.sha256(
                    ("grounded-blind:" + r["id"]).encode()
                ).hexdigest()[:20],
            )
            for i, r in enumerate(usable[:9])
        ]
    p = ROOT / "evaluator/selected.jsonl"
    prior_keys = set(prior["family_keys"])
    own_keys: set
    own_images: list
    own_keys, own_images = set(), []
    for row in selected:
        assert row["id"] not in prior["ids"]
        assert not set(row["family_keys"]) & (prior_keys | own_keys)
        assert not is_exposed(
            row["fingerprints"], prior["image_fingerprints"] + own_images, 4
        )
        own_keys.update(row["family_keys"])
        own_images.append(row["fingerprints"])
    once(
        ROOT / "isolation_before_opening.json",
        dict(
            passed=True,
            selected_records=len(selected),
            recorded_family_and_image_checks=True,
            sha_pixels_dhash_distance=4,
            prior_snapshot_sha256=sha256_file(
                DEVELOPMENT / "post_experiment_exclusion_snapshot.json"
            ),
            assessor_pixels_opened=0,
            limitation="Recorded keys and byte/pixel/dHash isolation do not prove absence of all semantic duplicates",
        ),
    )
    if p.exists():
        assert read_rows(p) == selected
    else:
        write_rows(p, selected)
    once(ROOT / "technical_selection.json", technical)
    receipt(
        "acquisition_receipt.json",
        [
            p,
            ROOT / "technical_selection.json",
            ROOT / "isolation_before_opening.json",
            *sorted((ROOT / "downloads").glob("*.json")),
            *[Path(r["image_path"]) for r in selected],
        ],
        attempted=len(rows),
        downloaded=sum(d["status"] == "downloaded" for d in downloads),
        selected_records=len(selected),
        gallery=96,
        image_queries=12,
        selection_uses_relevance=False,
        heldout_pixels_opened_to_assessor=0,
    )
    status("blind_media_acquired_not_opened_to_assessor")


def font(size: int) -> ImageFont.FreeTypeFont:
    """Return font for this workflow.

    Args:
        size: Size used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)


def sheet(rows: list[dict], path: Path, *, titles: object) -> None:
    """Apply the sheet workflow step.

    Args:
        rows: Records processed by this operation.
        path: Path of the input or output artifact used by this operation.
        titles: Titles used by this operation.

    Notes:
        Writes derived artifacts through canvas.save, path.parent.mkdir.
    """
    canvas = Image.new("RGB", (2400, 1350), "white")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        x, y = (i % 4) * 600, (i // 4) * 450
        with Image.open(row["image_path"]) as raw:
            thumb = ImageOps.contain(
                ImageOps.exif_transpose(raw).convert("RGB"), (585, 345)
            )
        canvas.paste(thumb, (x + (600 - thumb.width) // 2, y + 32))
        draw.text((x + 8, y + 5), row["display_id"], font=font(22), fill="black")
        if titles:
            for j in range(2):
                draw.text(
                    (x + 8, y + 380 + j * 27),
                    row["title"][j * 27 : (j + 1) * 27],
                    font=font(18),
                    fill="black",
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def open_gallery() -> None:
    """Return open gallery for this workflow.

    Notes:
        Writes derived artifacts through CONSUMED.parent.mkdir, once, receipt,
        write_rows.
    """
    guard()
    checked("acquisition_receipt.json")
    if (ROOT / "gallery_open_receipt.json").exists():
        checked("gallery_open_receipt.json")
        return
    selected = read_rows(ROOT / "evaluator/selected.jsonl")
    # Consumption is registered before the first assessor-visible sheet is made.
    consumed = [
        dict(r, project_role="consumed_grounded_AI_blind_not_reusable_as_unseen")
        for r in selected
    ]
    CONSUMED.parent.mkdir(parents=True, exist_ok=True)
    if CONSUMED.exists():
        assert read_rows(CONSUMED) == consumed
    else:
        write_rows(CONSUMED, consumed)
    docs = [
        dict(id=r["opaque_id"], title=r["text"], image_path=r["image_path"])
        for r in selected
        if r["role"] == "gallery"
    ]
    once(ROOT / "catalog.public.json", docs)
    mapping = [dict(d, display_id=f"G{i+1:03d}") for i, d in enumerate(docs)]
    once(ROOT / "evaluator/gallery_mapping.json", mapping)
    for start in range(0, 96, 12):
        sheet(
            mapping[start : start + 12],
            ROOT / "review_gallery" / f"{start//12+1:02d}.png",
            titles=True,
        )
    receipt(
        "gallery_open_receipt.json",
        [
            CONSUMED,
            ROOT / "catalog.public.json",
            ROOT / "evaluator/gallery_mapping.json",
            *sorted((ROOT / "review_gallery").glob("*.png")),
        ],
        queries_opened=False,
        profiles_must_precede_query_opening=True,
    )
    status("gallery_opened_AI_profiles_required_before_queries")


def freeze_profiles() -> None:
    """Freeze profiles.

    Notes:
        Writes derived artifacts through receipt, write_rows.
    """
    guard()
    checked("gallery_open_receipt.json")
    rows = load(ROOT / "evaluator/profiles.native_AI.json")
    docs = load(ROOT / "evaluator/gallery_mapping.json")
    assert set(rows) == {d["display_id"] for d in docs}
    profiles = []
    for d in docs:
        v = rows[d["display_id"]]
        assert (
            set(v) == {"product_type", "appearance", "primary_unclear"}
            and type(v["primary_unclear"]) is bool
        )
        assert all(
            isinstance(v[k], str) and 0 < len(v[k]) <= 200
            for k in ("product_type", "appearance")
        )
        profiles.append(dict(id=d["id"], **v))
    p = ROOT / "profiles.public.jsonl"
    if p.exists():
        assert read_rows(p) == profiles
    else:
        write_rows(p, profiles)
    receipt(
        "profile_receipt.json",
        [p, ROOT / "evaluator/profiles.native_AI.json"],
        review_origin="Codex native image review",
        human_labels=0,
        query_images_or_rankings_seen=False,
    )


def open_queries() -> None:
    """Return open queries for this workflow.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    checked("profile_receipt.json")
    if (ROOT / "query_open_receipt.json").exists():
        checked("query_open_receipt.json")
        return
    selected = read_rows(ROOT / "evaluator/selected.jsonl")
    queries = []
    mapping = []
    for i, bucket in enumerate(BUCKETS):
        row = next(
            r for r in selected if r["bucket"] == bucket and r["role"] == "image_query"
        )
        queries.extend(
            [
                dict(id=f"text_{i+1:02d}", text=TEXTS[i], image_path=None),
                dict(id=f"image_{i+1:02d}", text=None, image_path=row["image_path"]),
            ]
        )
        mapping.append(
            dict(display_id=f"I{i+1:02d}", title="", image_path=row["image_path"])
        )
    once(ROOT / "queries.public.json", queries)
    sheet(mapping, ROOT / "review_queries/all.png", titles=False)
    receipt(
        "query_open_receipt.json",
        [ROOT / "queries.public.json", ROOT / "review_queries/all.png"],
        queries=24,
        candidate_rankings_seen=False,
    )
    status("AI_full_gallery_labels_required_before_inference")


def freeze_labels() -> None:
    """Freeze labels.

    Notes:
        Writes derived artifacts through receipt, write_rows.
    """
    guard()
    checked("query_open_receipt.json")
    assert not (ROOT / "prediction_receipt.json").exists()
    authored = load(ROOT / "evaluator/judgments.native_AI.json")
    mapping = load(ROOT / "evaluator/gallery_mapping.json")
    qs = load(ROOT / "queries.public.json")
    assert set(authored) == {d["display_id"] for d in mapping}
    out = []
    for d in mapping:
        v = authored[d["display_id"]]
        assert set(v) == {
            "text_positive",
            "image_positive",
            "text_unknown",
            "image_unknown",
            "evidence",
        }
        for mode in ("text", "image"):
            positive = set(v[mode + "_positive"])
            unknown = {int(k): reason for k, reason in v[mode + "_unknown"].items()}
            assert (
                len(positive) == len(v[mode + "_positive"])
                and not positive & set(unknown)
                and positive | set(unknown) <= set(range(1, 13))
            )
            for i in range(1, 13):
                grade = 1 if i in positive else None if i in unknown else 0
                out.append(
                    dict(
                        query_id=f"{mode}_{i:02d}",
                        candidate_id=d["id"],
                        grade=grade,
                        reason=unknown[i] if i in unknown else v["evidence"],
                        review_origin="Codex native full-gallery AI image review",
                        human_label=False,
                        evidence="evaluator/judgments.native_AI.json",
                    )
                )
    assert len(out) == len(mapping) * len(qs) == 2304
    p = ROOT / "judgments.ai.jsonl"
    if p.exists():
        assert read_rows(p) == out
    else:
        write_rows(p, out)
    receipt(
        "label_receipt.json",
        [
            p,
            ROOT / "evaluator/judgments.native_AI.json",
            ROOT / "queries.public.json",
            ROOT / "catalog.public.json",
            ROOT / "profiles.public.jsonl",
        ],
        AI_relations=2304,
        human_labels=0,
        unknown=sum(r["grade"] is None for r in out),
        labels_frozen_before_predictions=True,
        all_selected_records_retained=True,
        independent_human_or_annotator_claim=False,
    )
    status("independent_AI_labels_frozen_ready_for_one_candidate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "freeze",
            "reserve",
            "acquire",
            "open_gallery",
            "freeze_profiles",
            "open_queries",
            "freeze_labels",
        ],
    )
    a = parser.parse_args().action
    globals()[a]()
