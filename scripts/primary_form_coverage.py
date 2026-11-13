"""Adaptive development coverage; fixed galleries and honest fresh-query validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import ijson
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_acceptance import product_family_keys
from multimodal_ecommerce.business_review_jobs import canonical_hash
from multimodal_ecommerce.data import parse_pv
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.development_expansion import load_expanded_index
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.intent_retrieval import (
    balanced_pool,
    normalized_ranks,
    public_evidence,
    recall_channels,
    rerank_complete,
)
from multimodal_ecommerce.machine_teacher import MODEL
from multimodal_ecommerce.media import fetch_image, inspect_image
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.precision_reranking import color_descriptor
from multimodal_ecommerce.query_fusion import (
    QueryFusionVisualChannelsInputs,
    visual_channels,
)
from multimodal_ecommerce.regional_retrieval import (
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
from structure_development import BUCKETS

ROOT = PROJECT_ROOT / "artifacts/module2_primary_form_coverage"
PARENT = PROJECT_ROOT / "artifacts/module2_structure_development"
QUEUE = PROJECT_ROOT / "data/module2_primary_form_candidates/records.jsonl"
MEDIA = PROJECT_ROOT / "data/module2_primary_form_media/records.jsonl"
FRESH_QUEUE = PROJECT_ROOT / "data/module2_primary_form_fresh_candidates/records.jsonl"
FRESH_MEDIA = PROJECT_ROOT / "data/module2_primary_form_fresh_media/records.jsonl"
FACETS = [
    "headset",
    "stiletto_boot",
    "arch_pillow",
    "down_vest",
    "glass_sideboard",
    "drawer_sideboard",
    "handled_cup",
    "chunky_shoe",
    "durian",
]
POLICY: dict = dict(
    role="adaptive_query_aware_development_coverage_not_acceptance",
    priorities_from="Both previously exposed cohorts; neither remains unseen validation",
    scan_after=1400000,
    scan_limit=2000000,
    seed=2026102401,
    metadata_per_facet=12,
    images_per_facet=8,
    facets=FACETS,
    technical_spares_only=True,
    keep_all_readable_nonduplicate_selected_products=True,
    discard_for_relevance_or_scores=False,
    duplicate_dhash_distance=4,
    unchanged_queries_per_exposed_cohort=16,
    unchanged_original_gallery_per_cohort=96,
    unchanged_original_judgments=True,
    fixed_k=10,
    methods=["cross20", "cross60"],
    method_difference="rerank candidate budget only; identical full weights, public features and instruction",
    selection="Tune cohort only: candidate must not regress on all six paired metrics in all/text/image versus cross20; then maximize image Precision lower bound and Recall lower bound; ties retain cross20",
    precision_required=0.6,
    recall_preferred=0.85,
    fresh_validation="Eight new train-source image queries sampled after selection freeze; query families disjoint from all earlier exposures, gallery shared with development and explicitly not independent-gallery acceptance",
    fresh_scan_after=2000000,
    fresh_scan_limit=2100000,
    fresh_seed=2026102402,
    fresh_metadata_per_bucket=3,
    fresh_queries_per_bucket=1,
    relevance="Same frozen business relevance: primary type/use and similar visible form for image; requested type/use/explicit attributes for text; unresolved remains null",
    review_origin="model",
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


def facet(title: str) -> str | None:
    """Return facet for this workflow.

    Args:
        title: Public product title used as input.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    t = title.casefold()
    if (
        "头戴" in t
        and "耳机" in t
        and not any(
            x in t for x in ["耳罩套", "耳机套", "支架", "收纳", "耳机包", "保护套"]
        )
    ):
        return "headset"
    if "短靴" in t and "尖头" in t and "细跟" in t:
        return "stiletto_boot"
    if (
        "枕" in t
        and any(x in t for x in ["竹枕", "木枕", "拱形", "弧形"])
        and any(x in t for x in ["颈椎", "颈枕", "颈部"])
    ):
        return "arch_pillow"
    if (
        "羽绒" in t
        and any(x in t for x in ["马甲", "背心"])
        and not any(x in t for x in ["宠物", "面料", "布料", "修补"])
    ):
        return "down_vest"
    if "餐边柜" in t and not any(
        x in t for x in ["柜门", "拉手", "高柜", "立柜", "顶柜"]
    ):
        if "玻璃" in t:
            return "glass_sideboard"
        if "抽屉" in t:
            return "drawer_sideboard"
    if (
        "吸管杯" in t
        and any(x in t for x in ["双耳", "双柄", "手柄", "把手"])
        and not any(x in t for x in ["杯盖", "配件", "杯刷"])
    ):
        return "handled_cup"
    if "老爹鞋" in t and not any(x in t for x in ["鞋垫", "鞋带", "鞋套"]):
        return "chunky_shoe"
    if (
        "榴莲" in t
        and any(x in t for x in ["果肉", "榴莲肉", "金枕"])
        and not any(
            x in t
            for x in [
                "干",
                "饼",
                "糖",
                "糕",
                "冰淇淋",
                "酥",
                "泥",
                "花束",
                "壳",
                "模型",
            ]
        )
    ):
        return "durian"
    return None


def query_bucket(title: str) -> str | None:
    """Return query bucket for this workflow.

    Args:
        title: Public product title used as input.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return next(
        (
            b[0]
            for b in BUCKETS
            if any(t in title for t in b[2]) and not any(t in title for t in b[3])
        ),
        None,
    )


def check() -> Any:
    """Verify the prerequisites and integrity of the current workflow.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    protocol = load(ROOT / "protocol.json")
    assert protocol["policy"] == POLICY
    verify_files(PROJECT_ROOT, protocol["files"])
    return protocol


@dataclass(frozen=True)
class PrimaryFormCoverageScanInputs:
    """Input values for scan.

    Attributes:
        source: Source used by this operation.
        prior: Prior used by this operation.
        after: After used by this operation.
        limit: Limit used by this operation.
        seed: Seed used to reproduce randomized ordering.
        buckets: Buckets used by this operation.
        classifier: Classifier used by this operation.
        count: Count used by this operation.
    """

    source: Path
    prior: dict
    after: int
    limit: int
    seed: int
    buckets: Sequence[str]
    classifier: Callable[[str], str | None]
    count: int


def scan(options: PrimaryFormCoverageScanInputs) -> tuple[list, dict]:
    """Return scan for this workflow.

    Args:
        options: Validated input fields described by PrimaryFormCoverageScanInputs.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    source = options.source
    prior = options.prior
    after = options.after
    limit = options.limit
    seed = options.seed
    buckets = options.buckets
    classifier = options.classifier
    count = options.count
    excluded = set(prior["family_keys"])
    groups: dict = {b: [] for b in buckets}
    with source.open("rb") as handle:
        for number, (sid, raw) in enumerate(ijson.kvitems(handle, ""), 1):
            if number > after and raw.get("url"):
                name = classifier(str(raw.get("title") or ""))
                if name:
                    row = dict(
                        id=f"product/train/{sid}",
                        source_id=str(sid),
                        dataset="product",
                        split="train",
                        text=raw["title"],
                        image_url=raw["url"],
                        attributes=parse_pv(raw.get("pv") or ""),
                        bucket=name,
                        source_position=number,
                        source_record_sha256=canonical_hash(raw),
                        project_role="exposed_primary_form_development_not_acceptance",
                    )
                    keys = product_family_keys(row)
                    if not keys & excluded:
                        row.update(
                            family_keys=sorted(keys),
                            priority=hashlib.sha256(
                                f"{seed}:{row['id']}".encode()
                            ).hexdigest(),
                        )
                        groups[name].append(row)
            if number == limit:
                break
    selected, seen = [], set(excluded)
    for name, rows in groups.items():
        picked = 0
        for row in sorted(rows, key=lambda r: r["priority"]):
            if not seen & set(row["family_keys"]):
                selected.append(row)
                seen.update(row["family_keys"])
                picked += 1
            if picked == count:
                break
    return selected, dict(
        scanned=number, eligible={b: len(g) for b, g in groups.items()}
    )


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        FileExistsError: Preserve frozen acquisition.

    Notes:
        Writes derived artifacts through ROOT.mkdir, receipt, write_json, write_rows.
    """
    if ROOT.exists() or QUEUE.exists() or MEDIA.exists():
        raise FileExistsError("Preserve frozen acquisition")
    prior = load(PARENT / "post_development_exclusion_snapshot.json")
    verify_files(PARENT, load(PARENT / "completion_receipt.json")["files"])
    verify_files(PROJECT_ROOT, prior["files"])
    source = PROTECTED_ROOT / "多模态电商数据/product1m_product5m_train_id_label.json"
    paths = [
        Path(__file__),
        PROJECT_ROOT / "scripts/structure_development.py",
        *sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")),
        PARENT / "completion_receipt.json",
        PARENT / "post_development_exclusion_snapshot.json",
        PARENT / "next_development_coverage_queue.json",
    ]
    paths += [
        PROJECT_ROOT / "configs" / f
        for f in ["embedding_model.json", "reranker_model.json"]
    ]
    ROOT.mkdir(parents=True)
    write_json(
        ROOT / "protocol.json",
        dict(
            policy=POLICY,
            source=source.relative_to(PROJECT_ROOT).as_posix(),
            source_sha256=sha256_file(source),
            regions=region_configuration(),
            teacher=load(PARENT / "protocol.json")["teacher"],
            files={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in paths
            },
            formal_metrics=None,
        ),
    )
    rows, meta = scan(
        PrimaryFormCoverageScanInputs(
            source,
            prior,
            POLICY["scan_after"],
            POLICY["scan_limit"],
            POLICY["seed"],
            FACETS,
            facet,
            POLICY["metadata_per_facet"],
        )
    )
    write_rows(QUEUE, rows)
    receipt(
        ROOT,
        "metadata_receipt.json",
        [ROOT / "protocol.json"],
        queue_sha256=sha256_file(QUEUE),
        candidates=len(rows),
        **meta,
    )
    print(json.dumps(dict(prepared=len(rows), **meta)), flush=True)


def fetch_one(client: httpx.Client, row: dict, base: Path) -> dict:
    """Return fetch one for this workflow.

    Args:
        client: Configured local model client used for inference.
        row: Row used by this operation.
        base: Base used by this operation.

    Returns:
        Mapping with fields: status, image_path, fingerprints, final_url,
        error_type, error.

    Notes:
        Writes derived artifacts through target.parent.mkdir, target.write_bytes.
    """
    try:
        content, url = fetch_image(client, row["image_url"])
        info = inspect_image(content)
        target = base / "media" / (info["sha256"] + info["suffix"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(content)
        return dict(
            status="downloaded",
            image_path=str(target),
            fingerprints=fingerprints(target),
            final_url=url,
        )
    except (httpx.HTTPError, OSError, ValueError) as e:
        logging.getLogger(__name__).debug(
            "Handled exception in fetch_one", exc_info=True
        )
        return dict(status="failed", error_type=type(e).__name__, error=str(e))


@dataclass(frozen=True)
class PrimaryFormCoverageAcquireRowsInputs:
    """Input values for acquire rows.

    Attributes:
        base: Base used by this operation.
        queue: Queue used by this operation.
        registry: Registry used by this operation.
        prior: Prior used by this operation.
        buckets: Buckets used by this operation.
        per_bucket: Per bucket used by this operation.
    """

    base: Path
    queue: Path
    registry: Path
    prior: dict
    buckets: list[str]
    per_bucket: int


def acquire_rows(options: PrimaryFormCoverageAcquireRowsInputs) -> list[dict]:
    """Return acquire rows for this workflow.

    Args:
        options: Validated input fields described by
            PrimaryFormCoverageAcquireRowsInputs.

    Returns:
        Values produced by this operation, in the stated return order.

    Notes:
        Writes derived artifacts through write_json, write_rows.
    """
    base = options.base
    queue = options.queue
    registry = options.registry
    prior = options.prior
    buckets = options.buckets
    per_bucket = options.per_bucket
    cache_path = base / "download_cache.json"
    cache = load(cache_path) if cache_path.exists() else {}
    chosen, signatures = [], list(prior["image_fingerprints"])
    with httpx.Client(timeout=20, trust_env=False) as client:
        for name in buckets:
            accepted = 0
            for row in [r for r in read_rows(queue) if r["bucket"] == name]:
                if row["id"] not in cache:
                    cache[row["id"]] = fetch_one(client, row, base)
                    write_json(cache_path, cache)
                entry = cache[row["id"]]
                if entry["status"] != "downloaded" or is_exposed(
                    entry["fingerprints"], signatures, 4
                ):
                    continue
                signatures.append(entry["fingerprints"])
                chosen.append(dict(**row, **entry))
                accepted += 1
                if accepted == per_bucket:
                    break
            print(
                json.dumps(dict(bucket=name, selected=accepted, requested=per_bucket)),
                flush=True,
            )
    write_rows(base / "selected.evaluator.jsonl", chosen)
    write_rows(
        registry,
        [
            dict(id=k, **v, project_role="exposed_development_download_not_acceptance")
            for k, v in cache.items()
        ],
    )
    return chosen


def acquire() -> None:
    """Apply the acquire workflow step.

    Raises:
        FileExistsError: Preserve acquisition.

    Notes:
        Writes derived artifacts through receipt, write_json.
    """
    check()
    if (ROOT / "acquisition_receipt.json").exists():
        raise FileExistsError("Preserve acquisition")
    assert sha256_file(QUEUE) == load(ROOT / "metadata_receipt.json")["queue_sha256"]
    rows = acquire_rows(
        PrimaryFormCoverageAcquireRowsInputs(
            ROOT,
            QUEUE,
            MEDIA,
            load(PARENT / "post_development_exclusion_snapshot.json"),
            FACETS,
            POLICY["images_per_facet"],
        )
    )
    assert rows, "No usable images; preserve all failures"
    write_json(
        ROOT / "additions/catalog.public.json",
        [dict(id=r["id"], title=r["text"], image_path=r["image_path"]) for r in rows],
    )
    receipt(
        ROOT,
        "acquisition_receipt.json",
        [
            ROOT / "selected.evaluator.jsonl",
            ROOT / "download_cache.json",
            ROOT / "additions/catalog.public.json",
        ],
        metadata_sha256=sha256_file(QUEUE),
        media_registry_sha256=sha256_file(MEDIA),
        selected=len(rows),
        no_relevance_filter=True,
    )


def sheets() -> None:
    """Apply the sheets workflow step.

    Raises:
        FileExistsError: Preserve native gallery input.

    Notes:
        Writes derived artifacts through canvas.save, folder.mkdir.
    """
    check()
    verify_files(ROOT, load(ROOT / "acquisition_receipt.json")["files"])
    folder = ROOT / "native"
    if folder.exists():
        raise FileExistsError("Preserve native gallery input")
    folder.mkdir()
    docs = load(ROOT / "additions/catalog.public.json")
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17)
    for start in range(0, len(docs), 8):
        canvas = Image.new("RGB", (1600, 760), "white")
        draw = ImageDraw.Draw(canvas)
        for off, d in enumerate(docs[start : start + 8]):
            x, y = off % 4 * 400, off // 4 * 380
            with Image.open(d["image_path"]) as im:
                pic = ImageOps.contain(
                    ImageOps.exif_transpose(im).convert("RGB"), (390, 290)
                )
            canvas.paste(pic, (x + (400 - pic.width) // 2, y))
            draw.text((x + 5, y + 293), str(start + off + 1), fill="black", font=font)
            for line in range(3):
                draw.text(
                    (x + 5, y + 317 + 20 * line),
                    d["title"][line * 20 : (line + 1) * 20],
                    fill="black",
                    font=font,
                )
        sheet_number = start // 8 + 1
        canvas.save(folder / f"gallery_{sheet_number:02d}.jpg", quality=96)


def profiles() -> None:
    """Apply the profiles workflow step.

    Raises:
        FileExistsError: Preserve profiles.

    Notes:
        Writes derived artifacts through receipt, write_rows.
    """
    check()
    docs = load(ROOT / "additions/catalog.public.json")
    native = load(ROOT / "native/profiles.json")
    assert set(native) == {str(i) for i in range(1, len(docs) + 1)}
    rows = [dict(id=d["id"], **native[str(i)]) for i, d in enumerate(docs, 1)]
    assert all(set(r) == {"id", "product_type", "appearance"} for r in rows)
    if (ROOT / "profile_receipt.json").exists():
        raise FileExistsError("Preserve profiles")
    write_rows(ROOT / "additions/profiles.public.jsonl", rows)
    receipt(
        ROOT,
        "profile_receipt.json",
        [ROOT / "native/profiles.json", ROOT / "additions/profiles.public.jsonl"],
        profiles=len(rows),
        review_origin="model",
        human_labels=0,
        exposed_queries_known_to_annotator=True,
        rankings_of_expanded_catalog_seen=False,
    )


def labels() -> None:
    """Apply the labels workflow step.

    Raises:
        FileExistsError: Preserve expanded labels.

    Notes:
        Writes derived artifacts through folder.mkdir, receipt, write_json, write_rows.
    """
    check()
    verify_files(ROOT, load(ROOT / "profile_receipt.json")["files"])
    if (ROOT / "label_receipt.json").exists():
        raise FileExistsError("Preserve expanded labels")
    docs = load(ROOT / "additions/catalog.public.json")
    native = load(ROOT / "native/judgments.json")
    paths = []
    for cohort, old in [("tune", "tune"), ("regression", "validation")]:
        parent = PARENT / old
        folder = ROOT / cohort
        folder.mkdir(exist_ok=True)
        queries = load(parent / "queries.public.json")
        assert set(native[cohort]) == {q["id"] for q in queries}
        old_labels = read_rows(parent / "judgments.ai.jsonl")
        added = []
        for q in queries:
            v = native[cohort][q["id"]]
            positive, unknown = set(v["positive"]), {
                int(k): val for k, val in v["unknown"].items()
            }
            assert not positive & set(unknown) and positive | set(unknown) <= set(
                range(1, len(docs) + 1)
            )
            for i, d in enumerate(docs, 1):
                grade = 1 if i in positive else None if i in unknown else 0
                added.append(
                    dict(
                        query_id=q["id"],
                        candidate_id=d["id"],
                        grade=grade,
                        reason=(
                            v["reason"]
                            if grade == 1
                            else (
                                unknown[i]
                                if grade is None
                                else "主要主体、用途或关键可见结构不匹配该查询。"
                            )
                        ),
                        review_origin="model",
                        human_label=False,
                        evidence="native/judgments.json",
                        addition_position=i,
                    )
                )
        write_rows(folder / "judgments.ai.jsonl", old_labels + added)
        write_json(folder / "queries.public.json", queries)
        write_json(
            folder / "catalog.public.json", load(parent / "catalog.public.json") + docs
        )
        write_rows(
            folder / "profiles.public.jsonl",
            read_rows(parent / "profiles.public.jsonl")
            + read_rows(ROOT / "additions/profiles.public.jsonl"),
        )
        paths += [
            folder / n
            for n in [
                "judgments.ai.jsonl",
                "queries.public.json",
                "catalog.public.json",
                "profiles.public.jsonl",
            ]
        ]
    receipt(
        ROOT,
        "label_receipt.json",
        [ROOT / "native/judgments.json", *paths],
        original_labels_unchanged=True,
        additions_relations=32 * len(docs),
        human_labels=0,
        frozen_before_expanded_predictions=True,
    )


def encode() -> None:
    """Encode the configured inputs into model feature representations.

    Raises:
        FileExistsError: Preserve completed expansion encoding.

    Notes:
        Writes derived artifacts through cache.mkdir, np.save, np.savez_compressed,
        receipt, target.mkdir, write_json.
    """
    config = check()
    verify_files(ROOT, load(ROOT / "label_receipt.json")["files"])
    if (ROOT / "encoding_receipt.json").exists():
        raise FileExistsError("Preserve completed expansion encoding")
    folder = ROOT / "additions/index"
    docs = load(ROOT / "additions/catalog.public.json")
    profiles = read_rows(ROOT / "additions/profiles.public.jsonl")
    texts = [
        "商品主体：" + p["product_type"] + "。可见结构：" + p["appearance"] + "。"
        for p in profiles
    ]
    encoder = QwenEmbedder()
    records, regions, titles, semantic = [], [], [], []
    try:
        for i, (d, text) in enumerate(zip(docs, texts), 1):
            cache = folder / "products" / f"{i:03d}"
            cache.mkdir(parents=True, exist_ok=True)
            if not (cache / "vectors.npz").exists():
                views, metadata = encode_views(
                    RegionalRetrievalEncodeViewsInputs(
                        encoder,
                        Path(d["image_path"]),
                        cache / "views",
                        config["regions"],
                    )
                )
                title, profile = encoder.encode(text=d["title"]), encoder.encode(
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
            records.append(dict(**d, views=load(cache / "views.json")))
            if i % 8 == 0:
                print(
                    json.dumps(dict(encoded_additions=i, total=len(docs))), flush=True
                )
        runtime = encoder.diagnostics()
    finally:
        encoder.close()
    np.savez_compressed(
        folder / "vectors.npz",
        regions=np.array(regions),
        texts=np.array(titles),
        profiles=np.array(semantic),
    )
    np.save(
        folder / "colors.npy",
        np.array([color_descriptor(Path(d["image_path"])) for d in docs]),
    )
    write_json(folder / "records.json", records)
    write_json(folder / "profile_texts.json", texts)
    write_json(folder / "runtime.json", runtime)
    for cohort, old in [("tune", "tune"), ("regression", "validation")]:
        parent = PARENT / old / "index"
        target = ROOT / cohort / "index"
        target.mkdir(exist_ok=True)
        with (
            np.load(parent / "vectors.npz", allow_pickle=False) as a,
            np.load(folder / "vectors.npz", allow_pickle=False) as b,
        ):
            np.savez_compressed(
                target / "vectors.npz",
                allow_pickle=False,
                **{
                    k: np.concatenate([a[k], b[k]])
                    for k in ["regions", "texts", "profiles"]
                },
            )
        write_json(target / "records.json", load(parent / "records.json") + records)
        write_json(
            target / "profile_texts.json", load(parent / "profile_texts.json") + texts
        )
        np.save(
            target / "colors.npy",
            np.concatenate(
                [
                    np.load(parent / "colors.npy", allow_pickle=False),
                    np.load(folder / "colors.npy", allow_pickle=False),
                ]
            ),
        )
        receipt(
            target,
            "manifest.json",
            [p for p in sorted(target.iterdir()) if p.is_file()],
            model=load(PROJECT_ROOT / "configs/embedding_model.json"),
            regions=config["regions"],
            upstream_files={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                for p in [
                    parent / "manifest.json",
                    parent / "vectors.npz",
                    folder / "vectors.npz",
                ]
            },
        )
    receipt(
        ROOT,
        "encoding_receipt.json",
        [
            p
            for base in [
                ROOT / "additions/index",
                ROOT / "tune/index",
                ROOT / "regression/index",
            ]
            for p in sorted(base.rglob("*"))
            if p.is_file()
        ],
        actual_embedding_calls=len(runtime["calls"]),
        original_gallery_vectors_reused_without_inference=True,
    )


def infer(cohort: str) -> None:
    """Run model inference using the configured inputs.

    Args:
        cohort: Cohort used by this operation.

    Raises:
        FileExistsError: Preserve fixed completed expanded predictions.

    Notes:
        Writes derived artifacts through receipt, target.mkdir, write_json.
    """
    check()
    verify_files(ROOT, load(ROOT / "encoding_receipt.json")["files"])
    folder = ROOT / cohort
    if (folder / "predictions.json").exists():
        raise FileExistsError("Preserve fixed completed expanded predictions")
    if cohort == "fresh":
        verify_files(ROOT, load(ROOT / "selection_receipt.json")["files"])
        verify_files(folder, load(folder / "query_encoding_receipt.json")["files"])
        bank_root = folder / "queries"
        index_root = ROOT / "tune/index"
        methods = list(
            dict.fromkeys(["cross20", load(ROOT / "selection.json")["selected"]])
        )
    else:
        verify_files(ROOT, load(ROOT / "label_receipt.json")["files"])
        bank_root = PARENT / ("tune" if cohort == "tune" else "validation") / "queries"
        index_root = folder / "index"
        methods = POLICY["methods"]
    index, titles = load_expanded_index(index_root)
    with np.load(index_root / "vectors.npz", allow_pickle=False) as bank:
        profiles = bank["profiles"].copy()
    texts = load(index_root / "profile_texts.json")
    colors = np.load(index_root / "colors.npy", allow_pickle=False)
    ids = [d["id"] for d in index.records]
    by_id = {d["id"]: d for d in index.records}
    ranks = {}
    reranker = DescribedImageReranker()
    try:
        for q in load(folder / "queries.public.json"):
            target = folder / "runs" / q["id"]
            target.mkdir(parents=True, exist_ok=True)
            with np.load(
                bank_root / q["id"] / "vectors.npz", allow_pickle=False
            ) as bank:
                views, caption = bank["views"].copy(), bank["caption"].copy()
            image = Path(q["image_path"]) if q["image_path"] else None
            features = feature_matrix(
                SemanticRankerFeatureMatrixInputs(
                    index,
                    titles,
                    profiles,
                    texts,
                    views[0],
                    text=q["text"],
                    image=image,
                    color_bank=colors,
                )
            )
            description = ""
            offsets = None
            if image:
                description = load(bank_root / q["id"] / "description.json")["text"]
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
                base, pool = np.array(fusion["fused_scores"]), fusion["pool"]
            else:
                base, channels = recall_channels(ids, features)
                offsets = np.array(
                    [
                        public_evidence(q["text"], d["title"], p)["offset"]
                        for d, p in zip(index.records, texts)
                    ]
                )
                channels.append(ranked_ids(ids, normalized_ranks(ids, base) + offsets))
                pool = balanced_pool(channels, 60)
            budget = 60 if "cross60" in methods else 20
            pool = pool[:budget]
            write_json(
                target / "features.json",
                dict(
                    ids=ids,
                    base=base.tolist(),
                    pool=pool,
                    features=features.tolist(),
                    description=description,
                    offsets=None if offsets is None else offsets.tolist(),
                ),
            )
            path = target / "cross.json"
            if path.exists():
                raw = load(path)
                assert raw["pool"] == pool
            else:
                cache = {}
                cache_file = None
                if cohort != "fresh":
                    old = PARENT / ("tune" if cohort == "tune" else "validation")
                    cache_file = old / "runs" / q["id"] / "original.json"
                    prior = load(cache_file)
                    assert (
                        load(old / "queries.public.json")[
                            next(
                                i
                                for i, v in enumerate(load(old / "queries.public.json"))
                                if v["id"] == q["id"]
                            )
                        ]
                        == q
                    )
                    assert description == (
                        load(old / "queries" / q["id"] / "description.json")["text"]
                        if image
                        else ""
                    )
                    cache = dict(zip(prior["pool"], prior["logits"]))
                missing = [cid for cid in pool if cid not in cache]
                pairs = [
                    dict(
                        id=cid,
                        image_path=by_id[cid]["image_path"],
                        text=by_id[cid]["title"],
                    )
                    for cid in missing
                ]
                before = len(reranker.adapter.timings)
                scores = (
                    (
                        reranker.score_described(
                            pairs, query_image=image, description=description
                        )
                        if image and description
                        else reranker.score(
                            pairs, query_image=image, query_text=q["text"]
                        )
                    )
                    if pairs
                    else np.array([])
                )
                computed = dict(zip(missing, scores.tolist()))
                raw = dict(
                    pool=pool,
                    logits=[
                        cache[cid] if cid in cache else computed[cid] for cid in pool
                    ],
                    new_pair_ids=missing,
                    reused_pair_ids=[cid for cid in pool if cid in cache],
                    cache_file=(
                        None
                        if cache_file is None
                        else cache_file.relative_to(PROJECT_ROOT).as_posix()
                    ),
                    cache_sha256=(
                        None if cache_file is None else sha256_file(cache_file)
                    ),
                    timings=reranker.adapter.timings[before:],
                )
                write_json(path, raw)
            order = {
                "global": ranked_ids(ids, features[:, 0]),
                "region_title": ranked_ids(
                    ids, 0.8 * features[:, 1] + 0.2 * features[:, 3]
                ),
            }
            for method in methods:
                count = 20 if method == "cross20" else 60
                order[method] = rerank_complete(
                    ids, base, pool[:count], np.array(raw["logits"][:count]), offsets
                )
            ranks[q["id"]] = order
            print(
                json.dumps(
                    dict(
                        cohort=cohort,
                        query=q["id"],
                        new_pairs=len(raw["new_pair_ids"]),
                        reused=len(raw["reused_pair_ids"]),
                    )
                ),
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
        actual_new_full_pairs=len(runtime["timings"]),
        methods=methods,
        qrels_used_in_runtime=False,
    )


def prepare_fresh() -> None:
    """Prepare fresh.

    Raises:
        FileExistsError: Never resample a completed validation.

    Notes:
        Writes derived artifacts through folder.mkdir, receipt, write_json, write_rows.
    """
    protocol = check()
    verify_files(ROOT, load(ROOT / "selection_receipt.json")["files"])
    folder = ROOT / "fresh"
    if folder.exists() or FRESH_QUEUE.exists() or FRESH_MEDIA.exists():
        raise FileExistsError("Never resample a completed validation")
    prior = load(ROOT / "post_coverage_exclusion_snapshot.json")
    source = PROJECT_ROOT / protocol["source"]
    folder.mkdir()
    write_json(
        folder / "protocol.json",
        dict(
            selection_sha256=sha256_file(ROOT / "selection.json"),
            implementation_sha256=sha256_file(Path(__file__)),
            policy=POLICY,
            sampling_uses_query_scores=False,
            gallery_is_shared_development_not_independent_catalog=True,
        ),
    )
    rows, meta = scan(
        PrimaryFormCoverageScanInputs(
            source,
            prior,
            POLICY["fresh_scan_after"],
            POLICY["fresh_scan_limit"],
            POLICY["fresh_seed"],
            [b[0] for b in BUCKETS],
            query_bucket,
            POLICY["fresh_metadata_per_bucket"],
        )
    )
    assert all(sum(r["bucket"] == b[0] for r in rows) == 3 for b in BUCKETS)
    write_rows(FRESH_QUEUE, rows)
    receipt(
        folder,
        "metadata_receipt.json",
        [folder / "protocol.json"],
        queue_sha256=sha256_file(FRESH_QUEUE),
        **meta,
    )


def acquire_fresh() -> None:
    """Apply the acquire fresh workflow step.

    Raises:
        FileExistsError: Preserve fresh queries.

    Notes:
        Writes derived artifacts through canvas.save, receipt, write_json.
    """
    check()
    verify_files(ROOT, load(ROOT / "selection_receipt.json")["files"])
    folder = ROOT / "fresh"
    if (folder / "acquisition_receipt.json").exists():
        raise FileExistsError("Preserve fresh queries")
    rows = acquire_rows(
        PrimaryFormCoverageAcquireRowsInputs(
            folder,
            FRESH_QUEUE,
            FRESH_MEDIA,
            load(ROOT / "post_coverage_exclusion_snapshot.json"),
            [b[0] for b in BUCKETS],
            1,
        )
    )
    assert len(rows) == 8, "Keep failures; no relevance resampling"
    queries = [
        dict(id=f"image_{i:02d}", text=None, image_path=r["image_path"])
        for i, r in enumerate(rows, 1)
    ]
    write_json(folder / "queries.public.json", queries)
    canvas = Image.new("RGB", (1600, 760), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
    for i, q in enumerate(queries):
        x, y = i % 4 * 400, i // 4 * 380
        with Image.open(q["image_path"]) as im:
            pic = ImageOps.contain(
                ImageOps.exif_transpose(im).convert("RGB"), (390, 335)
            )
        canvas.paste(pic, (x + (400 - pic.width) // 2, y))
        draw.text((x + 5, y + 340), q["id"], fill="black", font=font)
    canvas.save(folder / "queries.jpg", quality=97)
    receipt(
        folder,
        "acquisition_receipt.json",
        [
            folder / n
            for n in [
                "queries.public.json",
                "queries.jpg",
                "selected.evaluator.jsonl",
                "download_cache.json",
            ]
        ],
        selection_frozen_before_query_pixels=True,
        queries=8,
    )


def fresh_labels() -> None:
    """Apply the fresh labels workflow step.

    Raises:
        FileExistsError: Preserve fresh labels.

    Notes:
        Writes derived artifacts through receipt, write_rows.
    """
    check()
    folder = ROOT / "fresh"
    verify_files(folder, load(folder / "acquisition_receipt.json")["files"])
    if (folder / "label_receipt.json").exists():
        raise FileExistsError("Preserve fresh labels")
    native = load(folder / "judgments.native.json")
    docs = load(ROOT / "tune/catalog.public.json")
    rows = []
    for q in load(folder / "queries.public.json"):
        v = native[q["id"]]
        positive = set(v["positive"])
        unknown = {int(k): reason for k, reason in v["unknown"].items()}
        assert not positive & set(unknown) and positive | set(unknown) <= set(
            range(1, len(docs) + 1)
        )
        for i, d in enumerate(docs, 1):
            grade = 1 if i in positive else None if i in unknown else 0
            rows.append(
                dict(
                    query_id=q["id"],
                    candidate_id=d["id"],
                    grade=grade,
                    reason=(
                        v["reason"]
                        if grade == 1
                        else (
                            unknown[i]
                            if grade is None
                            else "主要商品类型、用途或关键外形不匹配。"
                        )
                    ),
                    review_origin="model",
                    human_label=False,
                    native_position=i,
                )
            )
    write_rows(folder / "judgments.ai.jsonl", rows)
    receipt(
        folder,
        "label_receipt.json",
        [folder / "judgments.native.json", folder / "judgments.ai.jsonl"],
        relations=len(rows),
        labels_frozen_before_predictions=True,
        human_labels=0,
    )


def encode_fresh() -> None:
    """Encode fresh.

    Raises:
        FileExistsError: Preserve completed fresh embeddings.

    Notes:
        Writes derived artifacts through np.savez_compressed, receipt, write_json.
    """
    config = check()
    folder = ROOT / "fresh"
    verify_files(folder, load(folder / "label_receipt.json")["files"])
    if (folder / "query_encoding_receipt.json").exists():
        raise FileExistsError("Preserve completed fresh embeddings")
    queries = load(folder / "queries.public.json")
    client = OllamaClient(model=MODEL)
    try:
        health = client.health()
        m = next(m for m in health["models"] if m["name"] == MODEL)
        assert (
            m["digest"].removeprefix("sha256:") == config["teacher"]["digest"]
            and health["version"] == config["teacher"]["server_version"]
        )
        for i, q in enumerate(queries):
            path = folder / "queries" / q["id"] / "description.json"
            if not path.exists():
                write_json(
                    path,
                    describe_query(
                        client, Path(q["image_path"]), last=i == len(queries) - 1
                    ),
                )
    finally:
        client.close()
    encoder = QwenEmbedder()
    try:
        for q in queries:
            target = folder / "queries" / q["id"]
            if (target / "vectors.npz").exists():
                continue
            views, meta = encode_views(
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
            np.savez_compressed(target / "vectors.npz", views=views, caption=caption)
            write_json(target / "views.json", meta)
        runtime = encoder.diagnostics()
    finally:
        encoder.close()
    write_json(folder / "query_runtime.json", runtime)
    receipt(
        folder,
        "query_encoding_receipt.json",
        [
            folder / "query_runtime.json",
            *[p for p in sorted((folder / "queries").rglob("*")) if p.is_file()],
        ],
        actual_full_embedding_calls=len(runtime["calls"]),
    )


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
            "prepare_fresh",
            "acquire_fresh",
            "fresh_labels",
            "encode_fresh",
        ],
    )
    parser.add_argument(
        "--cohort", choices=["tune", "regression", "fresh"], default="tune"
    )
    args = parser.parse_args()
    if args.action == "infer":
        infer(args.cohort)
    else:
        globals()[args.action]()
