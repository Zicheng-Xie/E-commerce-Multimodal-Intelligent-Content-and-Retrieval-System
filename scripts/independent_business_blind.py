"""Frozen one-shot business heldout runner; no development score lookup."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_acceptance import product_family_keys
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.holdout_data import is_exposed
from multimodal_ecommerce.intent_retrieval import (
    balanced_pool,
    normalized_ranks,
    public_evidence,
    recall_channels,
    rerank_complete,
)
from multimodal_ecommerce.machine_teacher import MODEL
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
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.semantic_ranker import (
    SemanticRankerFeatureMatrixInputs,
    feature_matrix,
    ranked_ids,
)
from multimodal_ecommerce.visual_query import DescribedImageReranker, describe_query

ROOT = PROJECT_ROOT / "artifacts/module2_independent_business_blind"
RESERVED = PROJECT_ROOT / "artifacts/module2_business_acceptance"
DATASET = RESERVED / "dataset"
SNAPSHOT = (
    PROJECT_ROOT
    / "artifacts/module2_rank_stability/post_development_exclusion_snapshot.json"
)
CONSUMED = PROJECT_ROOT / "data/module2_business_blind_consumed/records.jsonl"
METHOD = "business_cross20_text_described20_image"
TEXTS = [
    "智能手机",
    "笔记本电脑",
    "智能手表",
    "耳机",
    "有线鼠标",
    "移动电源充电宝",
    "手机保护壳",
    "平板电脑保护壳",
    "车载充电器",
    "移动硬盘盒",
    "T恤",
    "牛仔裤",
    "羽绒服",
    "卫衣",
    "西装外套",
    "汉服",
    "时装靴",
    "跑步鞋",
    "帆布鞋",
    "双肩包",
    "家用茶几",
    "餐边收纳柜",
    "书柜",
    "换鞋凳",
    "吸管杯",
    "蚕丝被",
    "棉花被",
    "装饰台灯",
    "懒人沙发",
    "婴童枕头或枕芯",
]
POLICY: dict = dict(
    version="business_blind_v1",
    candidate=METHOD,
    text_method="cross_evidence_pool20",
    image_method="described_cross_pool20",
    baselines=["global", "region_title"],
    candidate_pool=20,
    image_recall_fusion_pool=60,
    k=10,
    region_prefix_protection=False,
    tail_recovery=False,
    learned_rankers=False,
    manual_query_ROI=False,
    automatic_contrast_ROI_center_and_tiles=True,
    main_models="Full locally pinned Qwen3-VL-Embedding-2B and Qwen3-VL-Reranker-2B",
    query_description="Pinned existing Qwen3.5 prompt/schema/options; one image, no gallery or labels",
    gallery_profile="Native AI author sees only each gallery image and original title, before any heldout query images; type and visible appearance only, no ranking or relevance fields",
    relevance="Text: primary product type/use/explicit attributes. Image: primary product type/use and similar visible form. No identical SKU requirement. Background/person/category alone insufficient. Do not guess invisible material/specifications. Published title claims may support requested attributes; this does not certify physical composition.",
    review_origin="model",
    human_label=False,
    all_gallery_and_queries_retained=True,
    replace_or_remove_for_low_relevance=False,
    labels_frozen_before_query_prediction=True,
    gallery_embedding_before_label_freeze_allowed=True,
    independent_annotator_claim_allowed=False,
    scope="Balanced closed catalog of 30 predefined fine types, not traffic-representative",
    precision_required=0.6,
    recall_preferred=0.85,
    recall_preferred_is_mandatory=False,
    goal_scopes=["all", "text", "image"],
    one_frozen_candidate_only=True,
    tuning_on_this_heldout=False,
    failed_test_reusable_as_unseen=False,
    failed_default_promotion_allowed=False,
)


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def audit_isolation() -> dict:
    """Use precomputed metadata/fingerprints only; never open heldout pixels here.

    Returns:
        Mapping with fields: verified, reserved_records, development_fingerprints,
        development_family_keys, leaks, files, snapshot_sha256, heldout_pixels_opened,
        limitation.
    """
    previous = load(SNAPSHOT)
    verify_files(PROJECT_ROOT, previous["files"])
    lineage = read_rows(DATASET / "evaluator/lineage.jsonl")
    reserved_sha = {r["fingerprints"]["sha256"] for r in lineage}
    # The combined snapshot deliberately includes the reservation. Explicit media
    # registries below also detect any later actual exposure of those same bytes.
    signatures = [
        f for f in previous["image_fingerprints"] if f["sha256"] not in reserved_sha
    ]
    keys = set(load(RESERVED / "preparation_protocol.json")["old_product_family_keys"])
    files = set(previous["files"]) - {"data/module2_reserved_acceptance/records.jsonl"}
    files.add("artifacts/module2_catalog_review/dataset/products.jsonl")
    sources = {}
    for name in sorted(files):
        path = PROJECT_ROOT / name
        if path.suffix != ".jsonl":
            continue
        sources[name] = sha256_file(path)
        for row in read_rows(path):
            if (
                str(row.get("id", "")).startswith("product/")
                or row.get("dataset") == "product"
            ):
                keys.update(row.get("family_keys") or product_family_keys(row))
            for field in ["fingerprints", "image_fingerprints"]:
                fp = row.get(field)
                if isinstance(fp, dict) and {"sha256", "pixels", "dhash"} <= set(fp):
                    signatures.append(fp)
    signatures = list(
        {(f["sha256"], f["pixels"], f["dhash"]): f for f in signatures}.values()
    )
    own_keys: set
    own_images: list
    leaks, own_keys, own_images = [], set(), []
    for row in lineage:
        current = set(row["family_keys"])
        if current & keys:
            leaks.append(dict(id=row["opaque_id"], type="development_family"))
        if current & own_keys:
            leaks.append(dict(id=row["opaque_id"], type="heldout_cross_family"))
        if is_exposed(row["fingerprints"], signatures, 4):
            leaks.append(dict(id=row["opaque_id"], type="development_image"))
        if is_exposed(row["fingerprints"], own_images, 4):
            leaks.append(dict(id=row["opaque_id"], type="heldout_cross_image"))
        own_keys.update(current)
        own_images.append(row["fingerprints"])
    assert len(lineage) == 330 and not leaks, leaks
    return dict(
        verified=True,
        reserved_records=330,
        development_fingerprints=len(signatures),
        development_family_keys=len(keys),
        leaks=leaks,
        files=sources,
        snapshot_sha256=sha256_file(SNAPSHOT),
        heldout_pixels_opened=0,
        limitation="Conservative recorded family keys and exact/pixel/dHash<=4 isolation, not proof of every semantic duplicate",
    )


def freeze() -> None:
    """Freeze the inputs and policy used by this workflow.

    Raises:
        FileExistsError: Preserve this one-shot release.

    Notes:
        Writes derived artifacts through ROOT.mkdir, write_json.
    """
    if ROOT.exists() or CONSUMED.exists():
        raise FileExistsError("Preserve this one-shot release")
    receipt = load(RESERVED / "preparation_receipt.json")
    verify_files(RESERVED, receipt["files"])
    old = load(
        PROJECT_ROOT / "artifacts/module2_rank_stability/completion_receipt.json"
    )
    verify_files(PROJECT_ROOT / "artifacts/module2_rank_stability", old["files"])
    verify_files(PROJECT_ROOT, old["project_files"])
    audit = audit_isolation()
    ROOT.mkdir(parents=True)
    write_json(ROOT / "isolation_before_opening.json", audit)
    files = [
        Path(__file__),
        PROJECT_ROOT / "src/multimodal_ecommerce/interval_retrieval_metrics.py",
        PROJECT_ROOT / "tests/test_interval_retrieval_metrics.py",
        SNAPSHOT,
        PROJECT_ROOT / "configs/module2_current_target.json",
        PROJECT_ROOT / "configs/embedding_model.json",
        PROJECT_ROOT / "configs/reranker_model.json",
        PROJECT_ROOT / "configs/retrieval_regions_v1.json",
        RESERVED / "preparation_receipt.json",
        DATASET / "manifest.json",
        DATASET / "gallery/products.jsonl",
        DATASET / "evaluator/queries.draft.jsonl",
        DATASET / "evaluator/lineage.jsonl",
        PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl",
    ]
    files += [
        PROJECT_ROOT / "src/multimodal_ecommerce" / name
        for name in [
            "embedding.py",
            "pair_reranker.py",
            "regional_retrieval.py",
            "intent_retrieval.py",
            "business_retrieval.py",
            "semantic_ranker.py",
            "query_fusion.py",
            "visual_query.py",
            "machine_teacher.py",
            "ollama_client.py",
            "precision_reranking.py",
            "retrieval_strategies.py",
        ]
    ]
    prior_config = load(PROJECT_ROOT / "artifacts/module2_focus_coverage/protocol.json")
    teacher = load(
        PROJECT_ROOT / "artifacts/module2_focus_coverage/query_enrichment/protocol.json"
    )["teacher"]
    for configuration in [prior_config["embedding"], prior_config["reranker"]]:
        verify_files(Path(configuration["path"]), configuration["checkpoint_files"])
    verify_files(
        Path("E:/LLM_for_Ollama"),
        {
            str(Path(p).relative_to(Path("E:/LLM_for_Ollama"))): h
            for p, h in teacher["files"].items()
        },
    )
    write_json(
        ROOT / "release.json",
        dict(
            frozen_at=datetime.now(timezone.utc).isoformat(),
            policy=POLICY,
            text_queries=TEXTS,
            files={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in files
            },
            embedding=prior_config["embedding"],
            reranker=prior_config["reranker"],
            teacher=teacher,
            regions=region_configuration(),
            isolation_sha256=sha256_file(ROOT / "isolation_before_opening.json"),
            selection_reason="Use existing full-model twenty-pool methods; retain their known development failures. Do not stack failed adaptive prefix/tail variants or search on heldout.",
            human_requirements_superseded_by="User explicitly delegated all review to AI and asked to proceed to independent blind evaluation",
            human_labels=0,
            formal_metrics=None,
            goal_met=False,
        ),
    )
    write_json(
        ROOT / "public_status.json",
        dict(
            state="final_release_and_AI_blind_protocol_frozen_heldout_unopened",
            heldout_unsealed=False,
            final_model_frozen=True,
            AI_blind_protocol_frozen=True,
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            formal_acceptance=False,
            acceptance_predictions=0,
        ),
    )
    print(
        json.dumps(
            dict(release_frozen=True, isolation_verified=True, heldout_unsealed=False)
        )
    )


def check() -> Any:
    """Verify the prerequisites and integrity of the current workflow.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    release = load(ROOT / "release.json")
    assert release["policy"] == POLICY and release["text_queries"] == TEXTS
    verify_files(PROJECT_ROOT, release["files"])
    assert (
        sha256_file(ROOT / "isolation_before_opening.json")
        == release["isolation_sha256"]
    )
    return release


def font(size: int) -> ImageFont.FreeTypeFont:
    """Return font for this workflow.

    Args:
        size: Size used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)


def open_gallery() -> None:
    """Apply the open gallery workflow step.

    Raises:
        FileExistsError: Heldout has already been unsealed; never reset.

    Notes:
        Writes derived artifacts through panel.save, path.parent.mkdir, write_json,
        write_rows.
    """
    check()
    if CONSUMED.exists() or (ROOT / "heldout_unsealed.json").exists():
        raise FileExistsError("Heldout has already been unsealed; never reset")
    marker = dict(
        unsealed_at=datetime.now(timezone.utc).isoformat(),
        release_sha256=sha256_file(ROOT / "release.json"),
        purpose="One-shot fixed-version business blind evaluation with rank-blinded AI judgments",
        reusable_as_unseen=False,
    )
    with (ROOT / "heldout_unsealed.json").open("x", encoding="utf-8") as handle:
        json.dump(marker, handle, ensure_ascii=False, indent=2)
    rows = read_rows(PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl")
    write_rows(
        CONSUMED,
        [
            {
                **r,
                "project_role": "consumed_business_blind_not_reusable_as_unseen",
                "release_sha256": marker["release_sha256"],
            }
            for r in rows
        ],
    )
    catalog = read_rows(DATASET / "gallery/products.jsonl")
    public = [
        dict(
            id=r["id"],
            title=r["title"],
            image_path=str((DATASET / r["image_path"]).resolve()),
        )
        for r in catalog
    ]
    write_json(ROOT / "catalog.public.json", public)
    mapping = [dict(position=i, **r) for i, r in enumerate(public, 1)]
    write_rows(ROOT / "native/gallery_mapping.jsonl", mapping)
    for start in range(0, 300, 10):
        first, last = start + 1, start + 10
        panel = Image.new("RGB", (2000, 800), "white")
        draw = ImageDraw.Draw(panel)
        draw.text(
            (8, 4),
            f"Gallery {first}–{last}; captions before heldout image-query inspection",
            fill="black",
            font=font(20),
        )
        for offset, row in enumerate(public[start : start + 10]):
            x, y = (offset % 5) * 400, 38 + (offset // 5) * 380
            with Image.open(row["image_path"]) as im:
                thumb = ImageOps.contain(
                    ImageOps.exif_transpose(im).convert("RGB"), (388, 290)
                )
            panel.paste(thumb, (x + (400 - thumb.width) // 2, y))
            draw.text(
                (x + 5, y + 291), str(start + offset + 1), fill="black", font=font(23)
            )
            draw.text((x + 5, y + 322), row["title"][:21], fill="black", font=font(17))
            draw.text(
                (x + 5, y + 347), row["title"][21:42], fill="black", font=font(16)
            )
        sheet_number = start // 10 + 1
        path = ROOT / f"native/gallery_{sheet_number:02d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        panel.save(path, quality=96)
    status = load(ROOT / "public_status.json")
    status.update(
        state="heldout_consumed_gallery_profile_authoring_no_predictions",
        heldout_unsealed=True,
    )
    write_json(ROOT / "public_status.json", status)
    print(
        json.dumps(
            dict(
                heldout_unsealed=True, gallery=300, queries=60, acceptance_predictions=0
            )
        )
    )


def finish_profiles() -> None:
    """Apply the finish profiles workflow step.

    Raises:
        FileExistsError: Preserve gallery profile freeze.
        ValueError: Invalid/duplicate public profile.

    Notes:
        Writes derived artifacts through canvas.save, write_json, write_rows.
    """
    check()
    if (ROOT / "profile_receipt.json").exists():
        raise FileExistsError("Preserve gallery profile freeze")
    mapping = read_rows(ROOT / "native/gallery_mapping.jsonl")
    lines = (ROOT / "native/profiles.tsv").read_text(encoding="utf-8").splitlines()
    authored = {}
    for line in lines:
        if not line.strip():
            continue
        position, kind, appearance = line.split("\t")
        number = int(position)
        if (
            number in authored
            or not kind.strip()
            or not appearance.strip()
            or len(appearance) > 200
        ):
            raise ValueError("Invalid/duplicate public profile")
        authored[number] = dict(product_type=kind, appearance=appearance)
    assert set(authored) == set(range(1, 301))
    profiles = [dict(id=r["id"], **authored[r["position"]]) for r in mapping]
    write_rows(ROOT / "profiles.public.jsonl", profiles)
    write_json(
        ROOT / "profile_receipt.json",
        dict(
            files={
                n: sha256_file(ROOT / n)
                for n in [
                    "native/profiles.tsv",
                    "profiles.public.jsonl",
                    "catalog.public.json",
                    "native/gallery_mapping.jsonl",
                ]
            },
            queries_or_judgments_used=False,
            review_origin="model",
            human_label=False,
        ),
    )
    drafts = read_rows(DATASET / "evaluator/queries.draft.jsonl")
    queries = []
    for q in drafts:
        queries.append(
            dict(
                id=q["id"],
                text=(
                    TEXTS[int(q["id"].split("_")[1]) - 1]
                    if q["kind"] == "text"
                    else None
                ),
                image_path=(
                    str((DATASET / q["image_path"]).resolve())
                    if q["kind"] == "image"
                    else None
                ),
            )
        )
    write_json(ROOT / "queries.public.json", queries)
    image_queries = [q for q in queries if q["image_path"]]
    for start in range(0, 30, 10):
        canvas = Image.new("RGB", (2000, 800), "white")
        draw = ImageDraw.Draw(canvas)
        for i, q in enumerate(image_queries[start : start + 10]):
            x, y = (i % 5) * 400, (i // 5) * 400
            with Image.open(q["image_path"]) as im:
                thumb = ImageOps.contain(
                    ImageOps.exif_transpose(im).convert("RGB"), (388, 345)
                )
            canvas.paste(thumb, (x + (400 - thumb.width) // 2, y))
            draw.text((x + 8, y + 350), q["id"], fill="black", font=font(23))
        sheet_number = start // 10 + 1
        canvas.save(ROOT / f"native/queries_{sheet_number:02d}.jpg", quality=96)
    print(json.dumps(dict(gallery_profiles_frozen=300, image_query_sheets=3)))


def finish_labels() -> None:
    """Apply the finish labels workflow step.

    Raises:
        FileExistsError: Preserve frozen AI judgments.

    Notes:
        Writes derived artifacts through write_json, write_rows.
    """
    check()
    verify_files(ROOT, load(ROOT / "profile_receipt.json")["files"])
    if (ROOT / "label_receipt.json").exists():
        raise FileExistsError("Preserve frozen AI judgments")
    mapping = read_rows(ROOT / "native/gallery_mapping.jsonl")
    observations = load(ROOT / "native/judgments.json")
    assert observations["gallery_sheets_viewed"] == list(range(1, 31)) and observations[
        "query_sheets_viewed"
    ] == [1, 2, 3]
    entries = observations["products"]
    assert set(entries) == {str(i) for i in range(1, 301)}
    output = []
    for doc in mapping:
        v = entries[str(doc["position"])]
        for mode in ["text", "image"]:
            positive = set(v[mode + "_positive"])
            unknown = {int(k): reason for k, reason in v[mode + "_unknown"].items()}
            assert not positive & set(unknown) and positive | set(unknown) <= set(
                range(1, 31)
            )
            for number in range(1, 31):
                grade = 1 if number in positive else None if number in unknown else 0
                reason = (
                    "主要主体、用途及该查询明确要求的外形/属性匹配；按冻结规则直接看图与原标题判断。"
                    if grade == 1
                    else (
                        unknown[number]
                        if grade is None
                        else "按冻结业务规则复核，非该需求匹配的主要商品主体/用途/可见外形或明确属性。"
                    )
                )
                output.append(
                    dict(
                        query_id=f"{mode}_{number:02d}",
                        candidate_id=doc["id"],
                        grade=grade,
                        reason=reason,
                        native_position=doc["position"],
                        review_origin="model",
                        human_label=False,
                        evidence="native/judgments.json",
                    )
                )
    write_rows(ROOT / "judgments.ai.jsonl", output)
    write_json(
        ROOT / "label_receipt.json",
        dict(
            files={
                n: sha256_file(ROOT / n)
                for n in [
                    "native/judgments.json",
                    "judgments.ai.jsonl",
                    "queries.public.json",
                    "profiles.public.jsonl",
                ]
            },
            relations=18000,
            unknown=sum(r["grade"] is None for r in output),
            labels_frozen_before_predictions=True,
            review_origin="model",
            human_labels=0,
            query_or_product_removals=0,
        ),
    )
    print(
        json.dumps(
            dict(
                labels_frozen=True,
                relations=len(output),
                unknown=sum(r["grade"] is None for r in output),
            )
        )
    )


def build_index() -> None:
    """Build index.

    Raises:
        FileExistsError: Preserve completed heldout index.

    Notes:
        Writes derived artifacts through index_root.mkdir, np.save,
        np.savez_compressed, path.mkdir, write_json.
    """
    release = check()
    verify_files(ROOT, load(ROOT / "profile_receipt.json")["files"])
    index_root = ROOT / "index"
    if (index_root / "manifest.json").exists():
        raise FileExistsError("Preserve completed heldout index")
    index_root.mkdir(exist_ok=True)
    documents = load(ROOT / "catalog.public.json")
    profiles = {r["id"]: r for r in read_rows(ROOT / "profiles.public.jsonl")}
    records = []
    regions = np.zeros((300, 7, 2048), dtype=np.float32)
    titles = np.zeros((300, 2048), dtype=np.float32)
    profile_bank = np.zeros_like(titles)
    texts = []
    encoder = QwenEmbedder()
    try:
        for i, doc in enumerate(documents):
            product_number = i + 1
            path = index_root / "product_cache" / f"{product_number:03d}"
            path.mkdir(parents=True, exist_ok=True)
            profile = profiles[doc["id"]]
            text = (
                "商品主体："
                + profile["product_type"]
                + "。可见结构："
                + profile["appearance"]
                + "。"
            )
            texts.append(text)
            if (path / "vectors.npz").exists():
                with np.load(path / "vectors.npz", allow_pickle=False) as bank:
                    views = bank["views"].copy()
                    title = bank["title"].copy()
                    semantic = bank["profile"].copy()
                metadata = load(path / "views.json")
            else:
                views, metadata = encode_views(
                    RegionalRetrievalEncodeViewsInputs(
                        encoder,
                        Path(doc["image_path"]),
                        path / "views",
                        release["regions"],
                    )
                )
                title = encoder.encode(text=doc["title"])
                semantic = encoder.encode(text=text)
                np.savez_compressed(
                    path / "vectors.npz", views=views, title=title, profile=semantic
                )
                write_json(path / "views.json", metadata)
            regions[i, : len(views)] = views
            titles[i] = title
            profile_bank[i] = semantic
            records.append(dict(**doc, views=metadata))
            if (i + 1) % 10 == 0:
                write_json(
                    index_root / "embedding_progress.json",
                    dict(completed=i + 1, total=300, diagnostics=encoder.diagnostics()),
                )
                print(json.dumps(dict(gallery_encoded=i + 1, total=300)), flush=True)
        diagnostics = encoder.diagnostics()
    finally:
        encoder.close()
    np.savez_compressed(
        index_root / "vectors.npz", regions=regions, texts=titles, profiles=profile_bank
    )
    np.save(
        index_root / "colors.npy",
        np.array([color_descriptor(Path(d["image_path"])) for d in documents]),
    )
    write_json(index_root / "records.json", records)
    write_json(index_root / "profile_texts.json", texts)
    write_json(index_root / "embedding_runtime.json", diagnostics)
    write_json(
        index_root / "manifest.json",
        dict(
            model=load(PROJECT_ROOT / "configs/embedding_model.json"),
            regions=release["regions"],
            release_sha256=sha256_file(ROOT / "release.json"),
            files={
                p.relative_to(index_root).as_posix(): sha256_file(p)
                for p in sorted(index_root.rglob("*"))
                if p.is_file()
            },
        ),
    )
    print(json.dumps(dict(index_completed=True, gallery=300)))


def prepare_queries() -> None:
    """Prepare queries.

    Raises:
        FileExistsError: Preserve query preprocessing.

    Notes:
        Writes derived artifacts through folder.mkdir, np.savez_compressed, write_json.
    """
    release = check()
    verify_files(ROOT, load(ROOT / "label_receipt.json")["files"])
    if (ROOT / "query_preparation_receipt.json").exists():
        raise FileExistsError("Preserve query preprocessing")
    queries = load(ROOT / "queries.public.json")
    images = [q for q in queries if q["image_path"]]
    client = OllamaClient(model=MODEL)
    try:
        health = client.health()
        models = [m for m in health["models"] if m["name"] == MODEL]
        assert (
            len(models) == 1
            and models[0]["digest"].removeprefix("sha256:")
            == release["teacher"]["digest"]
        )
        assert health["version"] == release["teacher"]["server_version"]
        for i, q in enumerate(images):
            path = ROOT / "query_preparation" / q["id"] / "description.json"
            if not path.exists():
                value = describe_query(
                    client, Path(q["image_path"]), last=i == len(images) - 1
                )
                write_json(path, value)
            print(
                json.dumps(dict(query_description=q["id"], completed=True)), flush=True
            )
    finally:
        client.close()
    encoder = QwenEmbedder()
    try:
        for q in queries:
            folder = ROOT / "query_preparation" / q["id"]
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / "vectors.npz").exists():
                continue
            if q["image_path"]:
                views, metadata = encode_views(
                    RegionalRetrievalEncodeViewsInputs(
                        encoder,
                        Path(q["image_path"]),
                        folder / "views",
                        release["regions"],
                        query=True,
                    )
                )
                description = load(folder / "description.json")["text"]
                caption = (
                    encoder.encode(text=description, query=True)
                    if description
                    else np.zeros(2048, dtype=np.float32)
                )
                write_json(folder / "views.json", metadata)
            else:
                views = np.array([encoder.encode(text=q["text"], query=True)])
                caption = np.zeros(2048, dtype=np.float32)
            np.savez_compressed(folder / "vectors.npz", views=views, caption=caption)
        diagnostics = encoder.diagnostics()
    finally:
        encoder.close()
    write_json(ROOT / "query_preparation/runtime.json", diagnostics)
    write_json(
        ROOT / "query_preparation_receipt.json",
        dict(
            files={
                p.relative_to(ROOT).as_posix(): sha256_file(p)
                for p in sorted((ROOT / "query_preparation").rglob("*"))
                if p.is_file()
            },
            queries=60,
            image_descriptions=30,
            labels_frozen_before_processing=True,
        ),
    )
    print(json.dumps(dict(queries_encoded=60, image_descriptions=30)))


def public_features() -> (
    tuple[RegionalIndex, np.ndarray, np.ndarray, list[str], np.ndarray]
):
    """Return public features for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    index, titles, _ = RegionalIndex.load(ROOT / "index")
    if titles is None:
        raise ValueError("Published-title features are required for this workflow")
    with np.load(ROOT / "index/vectors.npz", allow_pickle=False) as bank:
        profiles = bank["profiles"].copy()
    return (
        index,
        titles,
        profiles,
        load(ROOT / "index/profile_texts.json"),
        np.load(ROOT / "index/colors.npy", allow_pickle=False),
    )


def predict() -> None:
    """Apply the predict workflow step.

    Raises:
        FileExistsError: Never re-run completed blind predictions.

    Notes:
        Writes derived artifacts through target.mkdir, write_json.
    """
    check()
    verify_files(ROOT, load(ROOT / "label_receipt.json")["files"])
    verify_files(ROOT, load(ROOT / "query_preparation_receipt.json")["files"])
    if (ROOT / "predictions.json").exists():
        raise FileExistsError("Never re-run completed blind predictions")
    start_path = ROOT / "predictions_started.json"
    marker = dict(
        release_sha256=sha256_file(ROOT / "release.json"),
        label_receipt_sha256=sha256_file(ROOT / "label_receipt.json"),
        continuation_of_same_frozen_run_only=True,
    )
    if start_path.exists():
        assert load(start_path) == marker
    else:
        write_json(start_path, marker)
    index, titles, profiles, profile_texts, colors = public_features()
    ids = [d["id"] for d in index.records]
    by_id = {d["id"]: d for d in index.records}
    rankings = {}
    reranker = DescribedImageReranker()
    try:
        for number, q in enumerate(load(ROOT / "queries.public.json"), 1):
            folder = ROOT / "query_preparation" / q["id"]
            with np.load(folder / "vectors.npz", allow_pickle=False) as bank:
                views = bank["views"].copy()
                caption = bank["caption"].copy()
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
            methods = dict(
                global_=ranked_ids(ids, features[:, 0]),
                region_title=ranked_ids(
                    ids, 0.8 * features[:, 1] + 0.2 * features[:, 3]
                ),
            )
            methods["global"] = methods.pop("global_")
            if image:
                description = load(folder / "description.json")["text"]
                fusion = visual_channels(
                    QueryFusionVisualChannelsInputs(
                        index,
                        titles,
                        profiles,
                        features,
                        views,
                        caption,
                        description,
                        np.arange(300, dtype=int),
                    )
                )
                pool = fusion["pool"][:20]
                base = np.array(fusion["fused_scores"])
                offsets = None
            else:
                description = ""
                base, channels = recall_channels(ids, features)
                offsets = np.array(
                    [
                        public_evidence(q["text"], d["title"], p)["offset"]
                        for d, p in zip(index.records, profile_texts)
                    ]
                )
                channels.append(ranked_ids(ids, normalized_ranks(ids, base) + offsets))
                pool = balanced_pool(channels, 60)[:20]
            target = ROOT / "raw_predictions" / q["id"]
            target.mkdir(parents=True, exist_ok=True)
            write_json(
                target / "features.json",
                dict(
                    ids=ids,
                    base=base.tolist(),
                    offsets=offsets.tolist() if offsets is not None else None,
                    pool=pool,
                    description=description,
                    features=features.tolist(),
                ),
            )
            documents = [
                dict(
                    id=cid,
                    image_path=by_id[cid]["image_path"],
                    text=by_id[cid]["title"],
                )
                for cid in pool
            ]
            path = target / "cross.json"
            if path.exists():
                raw = load(path)
                assert raw["pool"] == pool and raw["query_sha256"] == sha256_file(
                    folder / "vectors.npz"
                )
                scores = np.array(raw["logits"])
            else:
                previous = len(reranker.adapter.timings)
                scores = (
                    reranker.score_described(
                        documents, query_image=image, description=description
                    )
                    if image and description
                    else reranker.score(
                        documents, query_image=image, query_text=q["text"]
                    )
                )
                write_json(
                    path,
                    dict(
                        pool=pool,
                        logits=scores.tolist(),
                        query_sha256=sha256_file(folder / "vectors.npz"),
                        timings=reranker.adapter.timings[previous:],
                    ),
                )
            methods[METHOD] = rerank_complete(ids, base, pool, scores, offsets)
            write_json(target / "rankings.json", methods)
            rankings[q["id"]] = methods
            write_json(
                ROOT / "prediction_progress.json",
                dict(completed=number, total=60, raw_pair_scores_completed=number * 20),
            )
            print(
                json.dumps(dict(query=q["id"], completed=number, total=60)), flush=True
            )
        diagnostics = reranker.diagnostics()
    finally:
        reranker.close()
    write_json(ROOT / "cross_runtime.json", diagnostics)
    write_json(ROOT / "predictions.json", rankings)
    write_json(
        ROOT / "prediction_receipt.json",
        dict(
            queries=60,
            full_rankings=180,
            full_cross_pairs=1200,
            files={
                p.relative_to(ROOT).as_posix(): sha256_file(p)
                for p in [
                    ROOT / "predictions.json",
                    ROOT / "cross_runtime.json",
                    *sorted((ROOT / "raw_predictions").rglob("*.json")),
                ]
            },
            one_shot_fixed_version=True,
            parameter_search=False,
            review_origin="model",
            human_labels=0,
        ),
    )
    print(
        json.dumps(dict(predictions_complete=True, queries=60, full_cross_pairs=1200))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "freeze",
            "open_gallery",
            "finish_profiles",
            "finish_labels",
            "build_index",
            "prepare_queries",
            "predict",
        ],
    )
    args = parser.parse_args()
    globals()[args.action]()
