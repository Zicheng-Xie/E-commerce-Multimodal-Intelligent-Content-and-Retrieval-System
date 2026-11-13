"""Family-disjoint development coverage, reranker ablations and fixed validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import ijson
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_acceptance import product_family_keys
from multimodal_ecommerce.business_retrieval import BUSINESS_INSTRUCTION
from multimodal_ecommerce.business_review_jobs import canonical_hash
from multimodal_ecommerce.data import parse_pv
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.intent_retrieval import (
    balanced_pool,
    normalized_ranks,
    public_evidence,
    recall_channels,
    rerank_complete,
)
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.machine_teacher import MODEL
from multimodal_ecommerce.media import fetch_image, inspect_image
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.precision_reranking import color_descriptor
from multimodal_ecommerce.query_fusion import (
    QueryFusionVisualChannelsInputs,
    visual_channels,
)
from multimodal_ecommerce.regional_retrieval import (
    RegionalIndex,
    RegionalRetrievalEncodeViewsInputs,
    encode_views,
    region_configuration,
)
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    PROTECTED_ROOT,
    sha256_file,
    write_json,
)
from multimodal_ecommerce.semantic_ranker import (
    SemanticRankerFeatureMatrixInputs,
    feature_matrix,
    ranked_ids,
)
from multimodal_ecommerce.visual_query import DescribedImageReranker, describe_query

ROOT = PROJECT_ROOT / "artifacts/module2_structure_development"
PARENT = PROJECT_ROOT / "artifacts/module2_independent_business_blind"
QUEUE = PROJECT_ROOT / "data/module2_structure_candidates/records.jsonl"
MEDIA = PROJECT_ROOT / "data/module2_structure_media/records.jsonl"
BUCKETS = [
    ("earphones", "耳机", ["耳机"], ["耳机套", "耳机壳", "充电仓", "耳机线"]),
    ("phone_case", "手机保护壳", ["手机壳", "手机套"], ["支架", "贴膜"]),
    ("pillow", "枕头或枕芯", ["枕头", "枕芯"], ["枕套", "抱枕", "靠枕"]),
    ("boots", "时装靴", ["短靴", "长靴", "女靴"], ["鞋垫", "鞋套"]),
    ("outerwear", "羽绒服", ["羽绒服"], ["布料", "面料", "清洗"]),
    ("cabinet", "餐边收纳柜", ["餐边柜", "茶水柜"], ["柜门", "拉手"]),
    ("cup", "吸管杯", ["吸管杯"], ["杯盖", "杯套", "配件", "杯刷"]),
    ("shoes", "跑步鞋", ["跑步鞋"], ["鞋垫", "鞋套", "鞋带"]),
]
STRUCTURE_INSTRUCTION = BUSINESS_INSTRUCTION + (
    " First identify the primary sale item in each image. Compare that item's "
    + "visible silhouette, opening/closure, component arrangement and construction. "
    + "Examples of meaningful conflicts include in-ear versus over-ear headphones, "
    + "a flip cover versus a back shell, a pillow versus a pillowcase, high heels "
    + "versus flat soles, and different cabinet door/drawer layouts. "
    + "These are examples, not a restricted category list. A matching title or an "
    + "incidental prop cannot override a clearly incompatible visible primary item. "
    + "Treat any query description as unverified auxiliary evidence: when it "
    + "conflicts with the query pixels, use the pixels; do not infer hidden details."
)
METHODS = {
    "original": dict(instruction=BUSINESS_INSTRUCTION, description=True),
    "structure": dict(instruction=STRUCTURE_INSTRUCTION, description=True),
    "structure_pixels": dict(instruction=STRUCTURE_INSTRUCTION, description=False),
}
POLICY: dict = dict(
    role="AI_reviewed_development_and_fixed_family_disjoint_validation",
    scan_after=1100000,
    scan_limit=1400000,
    seed=2026102307,
    buckets=BUCKETS,
    gallery_per_bucket=12,
    queries_per_bucket=1,
    metadata_per_bucket_per_cohort=16,
    cohorts=["tune", "validation"],
    technical_spares_only=True,
    replace_for_relevance_or_scores=False,
    duplicate_dhash_distance=4,
    relevance="Original PDF business relevance: primary type/use and similar visible form for images; requested type/use/explicit attributes for text. No identical SKU requirement. Keep uncertain labels unresolved.",
    fixed_k=10,
    precision_required=0.6,
    recall_preferred=0.85,
    selection="Only tune: eligible if paired lower bounds of all six metrics versus original are nonnegative in all/text/image; among eligible choose maximum image P@10 lower bound, then Recall lower bound; ties retain original. Validation cannot select or alter a method.",
    catalog_scope="balanced closed development catalog; not traffic representative; no claim of same-query before/after coverage gain",
    validation_inspection_requires_selection_freeze=True,
    human_labels=0,
    formal_acceptance=False,
    independent_AI_blind_predictions=0,
    default_promotion_allowed=False,
)


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def receipt(folder: Path, name: str, paths: list[Path], **extra: object) -> None:
    """Record file hashes and provenance for the completed stage.

    Args:
        folder: Directory containing the operation artifacts.
        name: Name used by this operation.
        paths: Paths used by this operation.
        **extra: Named options forwarded to the underlying operation.

    Notes:
        Writes derived artifacts through write_json.
    """
    write_json(
        folder / name,
        dict(
            files={p.relative_to(folder).as_posix(): sha256_file(p) for p in paths},
            **extra,
        ),
    )


def check() -> Any:
    """Verify the prerequisites and integrity of the current workflow.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    protocol = load(ROOT / "protocol.json")
    assert protocol["policy"] == json.loads(json.dumps(POLICY))
    verify_files(PROJECT_ROOT, protocol["files"])
    return protocol


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        FileExistsError: Preserve stage and exposure registries.

    Notes:
        Writes derived artifacts through ROOT.mkdir, receipt, write_json, write_rows.
    """
    if ROOT.exists() or QUEUE.exists() or MEDIA.exists():
        raise FileExistsError("Preserve stage and exposure registries")
    prior = load(PARENT / "post_blind_exclusion_snapshot.json")
    verify_files(PROJECT_ROOT, prior["files"])
    source = PROTECTED_ROOT / "多模态电商数据/product1m_product5m_train_id_label.json"
    paths = [
        Path(__file__),
        *sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")),
    ]
    paths += [
        PROJECT_ROOT / ("configs/" + n)
        for n in ["embedding_model.json", "reranker_model.json"]
    ]
    paths += [
        PARENT / "completion_receipt.json",
        PARENT / "post_blind_exclusion_snapshot.json",
    ]
    ROOT.mkdir(parents=True)
    write_json(
        ROOT / "protocol.json",
        dict(
            policy=POLICY,
            methods=METHODS,
            regions=region_configuration(),
            files={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in paths
            },
            source=source.relative_to(PROJECT_ROOT).as_posix(),
            source_sha256=sha256_file(source),
            teacher=load(PARENT / "release.json")["teacher"],
            formal_metrics=None,
        ),
    )
    groups: dict = {b[0]: [] for b in BUCKETS}
    excluded = set(prior["family_keys"])
    with source.open("rb") as handle:
        for number, (sid, raw) in enumerate(ijson.kvitems(handle, ""), 1):
            if number > POLICY["scan_after"] and raw.get("url"):
                title = str(raw.get("title") or "")
                matches = [
                    b[0]
                    for b in BUCKETS
                    if any(t in title for t in b[2])
                    and not any(t in title for t in b[3])
                ]
                if matches:
                    row = dict(
                        id=f"product/train/{sid}",
                        dataset="product",
                        split="train",
                        source_id=str(sid),
                        text=title,
                        image_url=raw["url"],
                        attributes=parse_pv(raw.get("pv") or ""),
                        bucket=matches[0],
                        source_position=number,
                        source_record_sha256=canonical_hash(raw),
                        project_role="exposed_structure_development_metadata_not_acceptance",
                    )
                    keys = product_family_keys(row)
                    if not keys & excluded:
                        row["family_keys"] = sorted(keys)
                        row["priority"] = hashlib.sha256(
                            f"{POLICY['seed']}:{row['id']}".encode()
                        ).hexdigest()
                        groups[matches[0]].append(row)
            if number == POLICY["scan_limit"]:
                break
    selected, seen = [], set(excluded)
    for bucket in groups:
        eligible = sorted(groups[bucket], key=lambda r: r["priority"])
        picked = []
        for row in eligible:
            if not seen & set(row["family_keys"]):
                picked.append(row)
                seen.update(row["family_keys"])
            if len(picked) == 32:
                break
        if len(picked) != 32:
            raise ValueError(f"Insufficient metadata: {bucket}")
        for i, row in enumerate(picked):
            selected.append(
                dict(
                    **row,
                    cohort="tune" if i % 2 == 0 else "validation",
                    cohort_priority=i // 2,
                )
            )
    write_rows(QUEUE, selected)
    receipt(
        ROOT,
        "metadata_receipt.json",
        [ROOT / "protocol.json"],
        candidates=len(selected),
        source_scanned=number,
        queue_sha256=sha256_file(QUEUE),
        eligible={b: len(g) for b, g in groups.items()},
        model_inferences=0,
    )
    print(json.dumps(dict(metadata=len(selected), stage=str(ROOT))), flush=True)


def acquire() -> None:
    """Apply the acquire workflow step.

    Raises:
        FileExistsError: Preserve acquisition.

    Notes:
        Writes derived artifacts through folder.mkdir, receipt, target.parent.mkdir,
        target.write_bytes, write_json, write_rows.
    """
    check()
    if (ROOT / "acquisition_receipt.json").exists():
        raise FileExistsError("Preserve acquisition")
    assert sha256_file(QUEUE) == load(ROOT / "metadata_receipt.json")["queue_sha256"]
    prior = load(PARENT / "post_blind_exclusion_snapshot.json")
    cache_path = ROOT / "download_cache.json"
    cache = load(cache_path) if cache_path.exists() else {}
    candidates = read_rows(QUEUE)
    signatures = list(prior["image_fingerprints"])
    chosen = []
    with httpx.Client(timeout=20, trust_env=False) as client:
        for cohort in POLICY["cohorts"]:
            for bucket, *_ in BUCKETS:
                accepted = 0
                group = sorted(
                    [
                        r
                        for r in candidates
                        if r["cohort"] == cohort and r["bucket"] == bucket
                    ],
                    key=lambda r: r["cohort_priority"],
                )
                for row in group:
                    key = row["id"]
                    if key not in cache:
                        try:
                            content, url = fetch_image(client, row["image_url"])
                            info = inspect_image(content)
                            target = ROOT / "media" / (info["sha256"] + info["suffix"])
                            target.parent.mkdir(exist_ok=True)
                            if not target.exists():
                                target.write_bytes(content)
                            cache[key] = dict(
                                status="downloaded",
                                image_path=str(target),
                                fingerprints=fingerprints(target),
                                final_url=url,
                            )
                        except (httpx.HTTPError, OSError, ValueError) as e:
                            logging.getLogger(__name__).debug(
                                "Handled exception in acquire", exc_info=True
                            )
                            cache[key] = dict(
                                status="failed",
                                error_type=type(e).__name__,
                                error=str(e),
                            )
                        write_json(cache_path, cache)
                    entry = cache[key]
                    if entry["status"] != "downloaded":
                        continue
                    if is_exposed(entry["fingerprints"], signatures, 4):
                        continue
                    signatures.append(entry["fingerprints"])
                    role = "query" if accepted == 0 else "gallery"
                    chosen.append(dict(**row, **entry, sample_role=role))
                    accepted += 1
                    if accepted == 13:
                        break
                if accepted != 13:
                    raise ValueError(
                        f"Technical spares exhausted: {cohort}/{bucket}; no relevance replacement"
                    )
                print(
                    json.dumps(dict(cohort=cohort, bucket=bucket, acquired=accepted)),
                    flush=True,
                )
    write_rows(ROOT / "selected.evaluator.jsonl", chosen)
    write_rows(
        MEDIA,
        [
            dict(
                id=k,
                **v,
                project_role="exposed_structure_development_download_not_acceptance",
            )
            for k, v in cache.items()
        ],
    )
    for cohort in POLICY["cohorts"]:
        folder = ROOT / cohort
        folder.mkdir(exist_ok=True)
        rows = [r for r in chosen if r["cohort"] == cohort]
        docs = [
            dict(id=r["id"], title=r["text"], image_path=r["image_path"])
            for r in rows
            if r["sample_role"] == "gallery"
        ]
        queries = []
        for number, (bucket, text, *_) in enumerate(BUCKETS, 1):
            image = next(
                r for r in rows if r["bucket"] == bucket and r["sample_role"] == "query"
            )
            queries += [
                dict(id=f"text_{number:02d}", text=text, image_path=None),
                dict(
                    id=f"image_{number:02d}", text=None, image_path=image["image_path"]
                ),
            ]
        write_json(folder / "catalog.public.json", docs)
        write_json(folder / "queries.public.json", queries)
    receipt(
        ROOT,
        "acquisition_receipt.json",
        [
            ROOT / "selected.evaluator.jsonl",
            cache_path,
            *[
                ROOT / c / n
                for c in POLICY["cohorts"]
                for n in ["catalog.public.json", "queries.public.json"]
            ],
        ],
        metadata_sha256=sha256_file(QUEUE),
        media_registry_sha256=sha256_file(MEDIA),
        selected=len(chosen),
        downloaded=len(cache),
        isolation_checked_against_previous_exposures_and_between_cohorts=True,
        sample_relevance_inspected=False,
    )


def allowed(cohort: str) -> None:
    """Apply the allowed workflow step.

    Args:
        cohort: Cohort used by this operation.
    """
    check()
    verify_files(ROOT, load(ROOT / "acquisition_receipt.json")["files"])
    if cohort == "validation":
        verify_files(ROOT, load(ROOT / "selection_receipt.json")["files"])


def sheets(cohort: str) -> None:
    """Apply the sheets workflow step.

    Args:
        cohort: Cohort used by this operation.

    Raises:
        FileExistsError: Preserve native review inputs.

    Notes:
        Writes derived artifacts through canvas.save, folder.mkdir.
    """
    allowed(cohort)
    folder = ROOT / cohort / "native"
    if folder.exists():
        raise FileExistsError("Preserve native review inputs")
    folder.mkdir()
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17)
    docs = load(ROOT / cohort / "catalog.public.json")
    for start in range(0, len(docs), 12):
        canvas = Image.new("RGB", (1600, 1140), "white")
        draw = ImageDraw.Draw(canvas)
        for off, d in enumerate(docs[start : start + 12]):
            x, y = off % 4 * 400, off // 4 * 380
            with Image.open(d["image_path"]) as im:
                pic = ImageOps.contain(
                    ImageOps.exif_transpose(im).convert("RGB"), (388, 290)
                )
            canvas.paste(pic, (x + (400 - pic.width) // 2, y))
            draw.text((x + 5, y + 294), str(start + off + 1), fill="black", font=font)
            for line in range(3):
                draw.text(
                    (x + 5, y + 317 + line * 20),
                    d["title"][line * 20 : (line + 1) * 20],
                    fill="black",
                    font=font,
                )
        canvas.save(folder / f"gallery_{start // 12 + 1:02d}.jpg", quality=96)


def profiles(cohort: str) -> None:
    """Apply the profiles workflow step.

    Args:
        cohort: Cohort used by this operation.

    Raises:
        FileExistsError: Preserve frozen public profiles.

    Notes:
        Writes derived artifacts through canvas.save, receipt, write_rows.
    """
    allowed(cohort)
    folder = ROOT / cohort
    if (folder / "profile_receipt.json").exists():
        raise FileExistsError("Preserve frozen public profiles")
    native = load(folder / "native/profiles.json")
    docs = load(folder / "catalog.public.json")
    assert set(native) == {str(i) for i in range(1, len(docs) + 1)}
    rows = [
        dict(
            id=d["id"],
            product_type=native[str(i)]["product_type"],
            appearance=native[str(i)]["appearance"],
        )
        for i, d in enumerate(docs, 1)
    ]
    write_rows(folder / "profiles.public.jsonl", rows)
    receipt(
        folder,
        "profile_receipt.json",
        [folder / "native/profiles.json", folder / "profiles.public.jsonl"],
        profiles=len(rows),
        image_queries_inspected_before_profiles=False,
        annotator_knew_predefined_text_queries=True,
        review_origin="model",
        human_label=False,
    )
    canvas = Image.new("RGB", (1600, 780), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
    for i, q in enumerate(
        q for q in load(folder / "queries.public.json") if q["image_path"]
    ):
        x, y = i % 4 * 400, i // 4 * 390
        with Image.open(q["image_path"]) as im:
            pic = ImageOps.contain(
                ImageOps.exif_transpose(im).convert("RGB"), (390, 345)
            )
        canvas.paste(pic, (x + (400 - pic.width) // 2, y))
        draw.text((x + 8, y + 350), q["id"], fill="black", font=font)
    canvas.save(folder / "native/queries.jpg", quality=97)


def labels(cohort: str) -> None:
    """Apply the labels workflow step.

    Args:
        cohort: Cohort used by this operation.

    Raises:
        FileExistsError: Preserve frozen judgments.

    Notes:
        Writes derived artifacts through receipt, write_rows.
    """
    allowed(cohort)
    folder = ROOT / cohort
    verify_files(folder, load(folder / "profile_receipt.json")["files"])
    if (folder / "label_receipt.json").exists():
        raise FileExistsError("Preserve frozen judgments")
    native = load(folder / "native/judgments.json")
    docs = load(folder / "catalog.public.json")
    queries = load(folder / "queries.public.json")
    assert set(native["queries"]) == {q["id"] for q in queries}
    assert native["all_gallery_sheets_viewed"] == list(range(1, 9))
    assert native["query_sheet_viewed"] is True
    output = []
    for q in queries:
        v = native["queries"][q["id"]]
        positive, unknown = set(v["positive"]), {
            int(k): val for k, val in v["unknown"].items()
        }
        assert not positive & set(unknown) and positive | set(unknown) <= set(
            range(1, len(docs) + 1)
        )
        for i, doc in enumerate(docs, 1):
            grade = 1 if i in positive else None if i in unknown else 0
            output.append(
                dict(
                    query_id=q["id"],
                    candidate_id=doc["id"],
                    grade=grade,
                    reason=(
                        v["reason"]
                        if grade == 1
                        else (
                            unknown[i]
                            if grade is None
                            else "主要商品类型、用途或可见结构不匹配该查询需求。"
                        )
                    ),
                    review_origin="model",
                    human_label=False,
                    evidence="native/judgments.json",
                    native_position=i,
                )
            )
    write_rows(folder / "judgments.ai.jsonl", output)
    receipt(
        folder,
        "label_receipt.json",
        [
            folder / "native/judgments.json",
            folder / "judgments.ai.jsonl",
            folder / "queries.public.json",
        ],
        relations=len(output),
        unknown=sum(r["grade"] is None for r in output),
        labels_frozen_before_predictions=True,
        human_labels=0,
    )


def encode(cohort: str) -> None:
    """Encode the configured inputs into model feature representations.

    Args:
        cohort: Cohort used by this operation.

    Raises:
        FileExistsError: Preserve completed full embedding run.

    Notes:
        Writes derived artifacts through cache.mkdir, np.save, np.savez_compressed,
        receipt, target.mkdir, write_json.
    """
    allowed(cohort)
    folder = ROOT / cohort
    verify_files(folder, load(folder / "label_receipt.json")["files"])
    if (folder / "encoding_receipt.json").exists():
        raise FileExistsError("Preserve completed full embedding run")
    config = check()
    docs = load(folder / "catalog.public.json")
    queries = load(folder / "queries.public.json")
    images = [q for q in queries if q["image_path"]]
    client = OllamaClient(model=MODEL)
    try:
        health = client.health()
        m = next(m for m in health["models"] if m["name"] == MODEL)
        assert m["digest"].removeprefix("sha256:") == config["teacher"]["digest"]
        assert health["version"] == config["teacher"]["server_version"]
        for i, q in enumerate(images):
            path = folder / "queries" / q["id"] / "description.json"
            if not path.exists():
                write_json(
                    path,
                    describe_query(
                        client, Path(q["image_path"]), last=i == len(images) - 1
                    ),
                )
    finally:
        client.close()
    profiles = read_rows(folder / "profiles.public.jsonl")
    texts = [
        "商品主体：" + p["product_type"] + "。可见结构：" + p["appearance"] + "。"
        for p in profiles
    ]
    encoder = QwenEmbedder()
    records, regions, titles, semantic = [], [], [], []
    try:
        for i, (doc, text) in enumerate(zip(docs, texts), 1):
            cache = folder / "index" / "products" / f"{i:03d}"
            cache.mkdir(parents=True, exist_ok=True)
            if not (cache / "vectors.npz").exists():
                views, metadata = encode_views(
                    RegionalRetrievalEncodeViewsInputs(
                        encoder,
                        Path(doc["image_path"]),
                        cache / "views",
                        config["regions"],
                    )
                )
                title, profile = encoder.encode(text=doc["title"]), encoder.encode(
                    text=text
                )
                np.savez_compressed(
                    cache / "vectors.npz", views=views, title=title, profile=profile
                )
                write_json(cache / "views.json", metadata)
            with np.load(cache / "vectors.npz", allow_pickle=False) as bank:
                padded = np.zeros((7, 2048), dtype=np.float32)
                padded[: len(bank["views"])] = bank["views"]
                regions.append(padded)
                titles.append(bank["title"].copy())
                semantic.append(bank["profile"].copy())
            records.append(dict(**doc, views=load(cache / "views.json")))
            if i % 12 == 0:
                print(
                    json.dumps(dict(cohort=cohort, gallery_encoded=i, total=len(docs))),
                    flush=True,
                )
        for q in queries:
            target = folder / "queries" / q["id"]
            target.mkdir(parents=True, exist_ok=True)
            if (target / "vectors.npz").exists():
                continue
            if q["image_path"]:
                views, metadata = encode_views(
                    RegionalRetrievalEncodeViewsInputs(
                        encoder,
                        Path(q["image_path"]),
                        target / "views",
                        config["regions"],
                        query=True,
                    )
                )
                desc = load(target / "description.json")["text"]
                caption = (
                    encoder.encode(text=desc, query=True)
                    if desc
                    else np.zeros(2048, dtype=np.float32)
                )
                write_json(target / "views.json", metadata)
            else:
                views, caption = np.array(
                    [encoder.encode(text=q["text"], query=True)]
                ), np.zeros(2048, dtype=np.float32)
            np.savez_compressed(target / "vectors.npz", views=views, caption=caption)
        runtime = encoder.diagnostics()
    finally:
        encoder.close()
    np.savez_compressed(
        folder / "index/vectors.npz",
        regions=np.array(regions),
        texts=np.array(titles),
        profiles=np.array(semantic),
    )
    np.save(
        folder / "index/colors.npy",
        np.array([color_descriptor(Path(d["image_path"])) for d in docs]),
    )
    write_json(folder / "index/records.json", records)
    write_json(folder / "index/profile_texts.json", texts)
    write_json(folder / "index/embedding_runtime.json", runtime)
    receipt(
        folder / "index",
        "manifest.json",
        [p for p in sorted((folder / "index").rglob("*")) if p.is_file()],
        model=load(PROJECT_ROOT / "configs/embedding_model.json"),
        regions=config["regions"],
    )
    receipt(
        folder,
        "encoding_receipt.json",
        [
            p
            for base in [folder / "queries", folder / "index"]
            for p in sorted(base.rglob("*"))
            if p.is_file()
        ],
        full_embedding_calls=len(runtime["calls"]),
    )


def inputs(
    cohort: str,
) -> tuple[RegionalIndex, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Return inputs for this workflow.

    Args:
        cohort: Cohort used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    folder = ROOT / cohort
    index, titles, _ = RegionalIndex.load(folder / "index")
    if titles is None:
        raise ValueError("Published-title features are required for this workflow")
    with np.load(folder / "index/vectors.npz", allow_pickle=False) as bank:
        profiles = bank["profiles"].copy()
    return (
        index,
        titles,
        profiles,
        load(folder / "index/profile_texts.json"),
        np.load(folder / "index/colors.npy", allow_pickle=False),
    )


def infer(cohort: str) -> None:
    """Run model inference using the configured inputs.

    Args:
        cohort: Cohort used by this operation.

    Raises:
        FileExistsError: Preserve fixed completed predictions.

    Notes:
        Writes derived artifacts through receipt, target.mkdir, write_json.
    """
    allowed(cohort)
    folder = ROOT / cohort
    verify_files(folder, load(folder / "encoding_receipt.json")["files"])
    if (folder / "predictions.json").exists():
        raise FileExistsError("Preserve fixed completed predictions")
    methods = (
        list(METHODS)
        if cohort == "tune"
        else list(
            dict.fromkeys(["original", load(ROOT / "selection.json")["selected"]])
        )
    )
    index, titles, profiles, profile_texts, colors = inputs(cohort)
    ids = [d["id"] for d in index.records]
    docs = {d["id"]: d for d in index.records}
    ranks = {}
    reranker = DescribedImageReranker()
    try:
        for q in load(folder / "queries.public.json"):
            target = folder / "runs" / q["id"]
            target.mkdir(parents=True, exist_ok=True)
            with np.load(
                folder / "queries" / q["id"] / "vectors.npz", allow_pickle=False
            ) as bank:
                views, caption = bank["views"].copy(), bank["caption"].copy()
            image = Path(q["image_path"]) if q["image_path"] else None
            features = feature_matrix(
                SemanticRankerFeatureMatrixInputs(
                    index,
                    titles,
                    profiles,
                    profile_texts,
                    views[0],
                    text=q["text"],
                    image=image,
                    color_bank=colors,
                )
            )
            description, offsets = "", None
            if image:
                description = load(folder / "queries" / q["id"] / "description.json")[
                    "text"
                ]
                fusion = visual_channels(
                    QueryFusionVisualChannelsInputs(
                        index,
                        titles,
                        profiles,
                        features,
                        views,
                        caption,
                        description,
                        np.arange(len(ids), dtype=int),
                    )
                )
                base, pool = np.array(fusion["fused_scores"]), fusion["pool"][:20]
            else:
                base, channels = recall_channels(ids, features)
                offsets = np.array(
                    [
                        public_evidence(q["text"], d["title"], p)["offset"]
                        for d, p in zip(index.records, profile_texts)
                    ]
                )
                channels.append(ranked_ids(ids, normalized_ranks(ids, base) + offsets))
                pool = balanced_pool(channels, 60)[:20]
            write_json(
                target / "features.json",
                dict(
                    ids=ids,
                    base=base.tolist(),
                    pool=pool,
                    offsets=None if offsets is None else offsets.tolist(),
                    description=description,
                    features=features.tolist(),
                ),
            )
            order = {
                "global": ranked_ids(ids, features[:, 0]),
                "region_title": ranked_ids(
                    ids, 0.8 * features[:, 1] + 0.2 * features[:, 3]
                ),
            }
            pairs = [
                dict(
                    id=cid, image_path=docs[cid]["image_path"], text=docs[cid]["title"]
                )
                for cid in pool
            ]
            for method in methods:
                if not image and method != "original":
                    order[method] = order["original"]
                    continue
                policy = METHODS[method] if image else METHODS["original"]
                reranker.adapter.config = dict(
                    reranker.adapter.config, instruction=policy["instruction"]
                )
                path = target / (method + ".json")
                if path.exists():
                    raw = load(path)
                    assert raw["pool"] == pool and raw["policy"] == policy
                    scores = np.array(raw["logits"])
                else:
                    before = len(reranker.adapter.timings)
                    scores = (
                        reranker.score_described(
                            pairs, query_image=image, description=description
                        )
                        if image and policy["description"] and description
                        else reranker.score(
                            pairs, query_image=image, query_text=q["text"]
                        )
                    )
                    write_json(
                        path,
                        dict(
                            pool=pool,
                            policy=policy,
                            logits=scores.tolist(),
                            timings=reranker.adapter.timings[before:],
                        ),
                    )
                order[method] = rerank_complete(ids, base, pool, scores, offsets)
            ranks[q["id"]] = order
            print(
                json.dumps(dict(cohort=cohort, query=q["id"], methods=methods)),
                flush=True,
            )
        runtime = reranker.diagnostics()
    finally:
        reranker.close()
    write_json(folder / "cross_runtime.json", runtime)
    write_json(folder / "predictions.json", ranks)
    receipt(
        folder,
        "prediction_receipt.json",
        [
            folder / "predictions.json",
            folder / "cross_runtime.json",
            *sorted((folder / "runs").rglob("*.json")),
        ],
        actual_full_pairs=len(runtime["timings"]),
        fixed_methods=methods,
    )


def report(cohort: str) -> None:
    """Summarize the recorded results of this development workflow.

    Args:
        cohort: Cohort used by this operation.

    Notes:
        Writes derived artifacts through receipt, write_json.
    """
    from report_focus_coverage import paired_differences

    allowed(cohort)
    folder = ROOT / cohort
    verify_files(folder, load(folder / "prediction_receipt.json")["files"])
    verify_files(folder, load(folder / "label_receipt.json")["files"])
    rankings = load(folder / "predictions.json")
    labels = read_rows(folder / "judgments.ai.jsonl")
    modes = {q: q.split("_")[0] for q in rankings}
    metrics = evaluate_intervals(rankings, labels, modes)
    paired = paired_differences(rankings, labels, modes)
    # Include original as an anchor without modifying the frozen historical helper.
    renamed = {
        q: {
            ("cross_evidence_pool60" if m == "original" else m): v
            for m, v in methods.items()
        }
        for q, methods in rankings.items()
    }
    original_deltas = paired_differences(renamed, labels, modes)
    write_json(folder / "internal_metrics.ai.development.json", metrics)
    write_json(folder / "internal_paired.json", paired)
    write_json(folder / "internal_vs_original.json", original_deltas)
    receipt(
        folder,
        "evaluation_receipt.json",
        [
            folder / n
            for n in [
                "internal_metrics.ai.development.json",
                "internal_paired.json",
                "internal_vs_original.json",
            ]
        ],
        formal_metrics=None,
        human_labels=0,
    )
    print(json.dumps(dict(evaluated=cohort, formal_acceptance=False)))


def select() -> None:
    """Apply the select workflow step.

    Raises:
        FileExistsError: Selection is frozen before validation inspection.

    Notes:
        Writes derived artifacts through receipt, write_json.
    """
    check()
    if (ROOT / "selection.json").exists():
        raise FileExistsError("Selection is frozen before validation inspection")
    folder = ROOT / "tune"
    verify_files(folder, load(folder / "evaluation_receipt.json")["files"])
    metrics = load(folder / "internal_metrics.ai.development.json")["scopes"]
    paired = load(folder / "internal_vs_original.json")["scopes"]
    eligible = ["original"]
    gates = {}
    for method in ["structure", "structure_pixels"]:
        gate = all(
            value["nonregression_proven_for_shared_unknown_labels"]
            for scope in ["all", "image", "text"]
            for value in paired[scope][method]["cross_evidence_pool60"].values()
        )
        gates[method] = gate
        if gate:
            eligible.append(method)
    selected = max(
        eligible,
        key=lambda m: (
            metrics["image"][m]["ranges"]["precision@10"][0],
            metrics["image"][m]["ranges"]["recall@10"][0] or 0,
            m == "original",
        ),
    )
    write_json(
        ROOT / "selection.json",
        dict(
            selected=selected,
            eligible=eligible,
            nonregression_gates=gates,
            source="tune only; validation media not inspected and no validation inference",
            policy=POLICY["selection"],
            validation_cannot_change_selection=True,
            default_promotion_allowed=False,
            formal_acceptance=False,
        ),
    )
    receipt(
        ROOT,
        "selection_receipt.json",
        [ROOT / "selection.json", folder / "evaluation_receipt.json"],
        validation_profiles=0,
        validation_image_queries_inspected=0,
        validation_predictions=0,
    )
    print(json.dumps(dict(selected=selected, gates=gates, formal_acceptance=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "acquire",
            "sheets",
            "profiles",
            "labels",
            "encode",
            "infer",
            "report",
            "select",
        ],
    )
    parser.add_argument("--cohort", choices=POLICY["cohorts"], default="tune")
    args = parser.parse_args()
    if args.action in {"prepare", "acquire", "select"}:
        globals()[args.action]()
    else:
        globals()[args.action](args.cohort)
