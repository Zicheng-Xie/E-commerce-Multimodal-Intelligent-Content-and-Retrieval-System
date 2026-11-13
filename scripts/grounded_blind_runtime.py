"""One immutable candidate on new data; full Qwen, automatic query-only ROI."""

from __future__ import annotations

import argparse
import base64
import io
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import balanced_supplement as admission
import grounded_blind_dataset as dataset
import primary_subject_captions_safe as safe_caption
import run  # noqa: F401
from grounded_subject_recovery import safe_localize
from grounded_subject_runtime import (
    GroundedSubjectRuntimeFuseInputs,
    fuse,
    parse_native,
    proposal,
)
from learned_fusion_experiment import load, np_once, once
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.intent_retrieval import (
    balanced_pool,
    normalized_ranks,
    public_evidence,
    recall_channels,
    rerank_complete,
)
from multimodal_ecommerce.machine_teacher import MODEL, resolve_local_teacher
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
from multimodal_ecommerce.safety import PROJECT_ROOT
from multimodal_ecommerce.semantic_ranker import (
    SemanticRankerFeatureMatrixInputs,
    feature_matrix,
    ranked_ids,
)
from multimodal_ecommerce.visual_query import (
    CAPTION_OPTIONS,
    CAPTION_PROMPT,
    CAPTION_SCHEMA,
    DescribedImageReranker,
    caption_text,
)
from primary_subject_experiment import pair_identity

ROOT = dataset.ROOT
PRIMARY = "grounded_whole_zoom_union"
METHODS = ("global", "region_title", "cross20", PRIMARY)


@dataclass(frozen=True)
class GroundedBlindRuntimeFeatureValuesInputs:
    """Input values for feature values.

    Attributes:
        index: Index used by this operation.
        titles: Titles used by this operation.
        profiles: Profiles used by this operation.
        texts: Texts used by this operation.
        colors: Colors used by this operation.
        query: Query record or representation used by this operation.
        views: Views used by this operation.
        caption: Caption used by this operation.
        description: Description used by this operation.
    """

    index: RegionalIndex
    titles: np.ndarray
    profiles: np.ndarray
    texts: list[str]
    colors: np.ndarray
    query: dict
    views: np.ndarray
    caption: np.ndarray
    description: str


def feature_values(options: GroundedBlindRuntimeFeatureValuesInputs) -> dict:
    """Identical original full-gallery retrieval function, arbitrary new IDs.

    Args:
        options: Validated input fields described by
            GroundedBlindRuntimeFeatureValuesInputs.

    Returns:
        Mapping with fields: ids, base, pool, features, description, offsets.
    """
    index = options.index
    titles = options.titles
    profiles = options.profiles
    texts = options.texts
    colors = options.colors
    query = options.query
    views = options.views
    caption = options.caption
    description = options.description
    ids = [d["id"] for d in index.records]
    image = dataset.Path(query["image_path"]) if query["image_path"] else None
    matrix = feature_matrix(
        SemanticRankerFeatureMatrixInputs(
            index,
            titles,
            profiles,
            texts,
            views[0],
            text=query["text"],
            image=image,
            color_bank=colors,
        )
    )
    offsets = None
    if image:
        fusion = visual_channels(
            QueryFusionVisualChannelsInputs(
                index,
                titles,
                profiles,
                matrix,
                views,
                caption,
                description,
                np.arange(len(ids), dtype=int),
            )
        )
        base = np.asarray(fusion["fused_scores"])
        pool = fusion["pool"][:20]
    else:
        base, channels = recall_channels(ids, matrix)
        offsets = np.asarray(
            [
                public_evidence(query["text"], d["title"], p)["offset"]
                for d, p in zip(index.records, texts)
            ]
        )
        channels.append(ranked_ids(ids, normalized_ranks(ids, base) + offsets))
        pool = balanced_pool(channels, 60)[:20]
    return dict(
        ids=ids,
        base=base.tolist(),
        pool=pool,
        features=matrix.tolist(),
        description=description,
        offsets=None if offsets is None else offsets.tolist(),
    )


def caption(client: OllamaClient, image: Path, *, safe: bool) -> dict:
    """Return caption for this workflow.

    Args:
        client: Configured local model client used for inference.
        image: Image input used by the operation.
        safe: Safe used by this operation.

    Returns:
        Mapping with fields: raw, value, text, seconds.

    Notes:
        Writes derived artifacts through picture.save.
    """
    prompt, schema, options = (
        (safe_caption.PROMPT, safe_caption.SCHEMA, safe_caption.OPTIONS)
        if safe
        else (CAPTION_PROMPT, CAPTION_SCHEMA, CAPTION_OPTIONS)
    )
    with Image.open(image) as source_image:
        picture = ImageOps.exif_transpose(source_image).convert("RGB")
        picture.thumbnail((768, 768))
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=92)
    start = time.perf_counter()
    raw = client._request(
        "/api/chat",
        dict(
            model=MODEL,
            messages=[
                dict(
                    role="user",
                    content=prompt,
                    images=[base64.b64encode(buffer.getvalue()).decode("ascii")],
                )
            ],
            format=schema,
            think=False,
            stream=False,
            options=options,
            keep_alive="1m",
        ),
    )
    value = parse_native(raw, schema, options)
    text = (
        (
            ""
            if value["primary_unclear"]
            else "商品主体："
            + value["product"]
            + "。可见结构："
            + value["appearance"]
            + "。"
        )
        if safe
        else caption_text(value)
    )
    return dict(raw=raw, value=value, text=text, seconds=time.perf_counter() - start)


def native_queries() -> bool:
    """Return native queries for this workflow.

    Returns:
        Whether the condition checked by this operation holds.

    Notes:
        Writes derived artifacts through once.
    """
    dataset.guard()
    dataset.checked("label_receipt.json")
    qs = [q for q in load(ROOT / "queries.public.json") if q["image_path"]]
    if any(
        not (ROOT / "query_preparation" / q["id"] / "new_description.json").exists()
        for q in qs
    ):
        if not admission.gpu_available():
            dataset.status("waiting_natural_GPU_release_for_blind_query_preparation")
            return False
        client = OllamaClient(timeout=360)
        try:
            model = resolve_local_teacher(
                client,
                dataset.Path(
                    load(PROJECT_ROOT / "configs/local.json")["ollama_models_root"]
                ),
            )
            assert (
                model["digest"]
                == load(dataset.DEVELOPMENT / "grounder_model_manifest.json")["digest"]
            )
            once(ROOT / "native_query_model.json", model)
            for q in qs:
                folder = ROOT / "query_preparation" / q["id"]
                safe_localize(
                    dataset.Path(q["image_path"]),
                    folder / "grounding",
                    client,
                    once,
                    load,
                )
                for safe, name in [
                    (False, "old_description.json"),
                    (True, "new_description.json"),
                ]:
                    if not (folder / name).exists():
                        once(folder / name, caption(client, q["image_path"], safe=safe))
                print(dict(blind_query_prepared=q["id"]), flush=True)
        finally:
            client.close()
    dataset.receipt(
        "native_query_receipt.json",
        [
            ROOT / "native_query_model.json",
            *[p for p in (ROOT / "query_preparation").rglob("*") if p.is_file()],
        ],
        query_only=True,
        labels_or_gallery_sent=False,
        new_image_queries=12,
    )
    return True


def encode_all() -> bool:
    """Encode all.

    Returns:
        Whether the condition checked by this operation holds.

    Notes:
        Writes derived artifacts through folder.mkdir, once.
    """
    dataset.guard()
    dataset.checked("native_query_receipt.json")
    engine = None
    docs = load(ROOT / "catalog.public.json")
    profiles = {p["id"]: p for p in read_rows(ROOT / "profiles.public.jsonl")}
    config = region_configuration()
    records = []
    texts = []

    def encoder() -> QwenEmbedder | None:
        """Return encoder for this workflow.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        nonlocal engine
        if engine is None:
            if not admission.gpu_available():
                return None
            engine = QwenEmbedder()
            dataset.status("full_Qwen_blind_index_and_queries")
        return engine

    def views_for(
        path: Path, folder: Path, query: bool = False
    ) -> tuple[np.ndarray, list[dict]]:
        """Return views for for this workflow.

        Args:
            path: Path of the input or output artifact used by this operation.
            folder: Directory containing the operation artifacts.
            query: Query record or representation used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        target = folder / "views"
        i = 1
        while target.exists():
            target = folder / f"views_recovery_{i}"
            i += 1
        return encode_views(
            RegionalRetrievalEncodeViewsInputs(
                encoder(), dataset.Path(path), target, config, query=query
            )
        )

    try:
        for i, d in enumerate(docs):
            folder = ROOT / "index/items" / d["id"]
            p = folder / "vectors.npz"
            profile = profiles[d["id"]]
            text = (
                ""
                if profile["primary_unclear"]
                else "商品主体："
                + profile["product_type"]
                + "。可见结构："
                + profile["appearance"]
                + "。"
            )
            if not p.exists():
                active_engine = encoder()
                if active_engine is None:
                    return False
                start = len(active_engine.timings)
                views, details = views_for(d["image_path"], folder)
                title = active_engine.encode(text=d["title"])
                semantic = (
                    active_engine.encode(text=text)
                    if text
                    else np.zeros(2048, dtype=np.float32)
                )
                once(folder / "record.json", dict(d, views=details))
                np_once(
                    p,
                    regions=views,
                    title=title,
                    profile=semantic,
                    color=color_descriptor(dataset.Path(d["image_path"])),
                )
                once(folder / "runtime.json", dict(calls=active_engine.timings[start:]))
            records.append(load(folder / "record.json"))
            texts.append(text)
            if (i + 1) % 12 == 0:
                print(dict(blind_gallery_encoded=i + 1, total=len(docs)), flush=True)
        regions = np.zeros((len(docs), 7, 2048), np.float32)
        titles = []
        semantics = []
        colors = []
        for i, d in enumerate(docs):
            with np.load(
                ROOT / "index/items" / d["id"] / "vectors.npz", allow_pickle=False
            ) as b:
                regions[i, : len(b["regions"])] = b["regions"]
                titles.append(b["title"])
                semantics.append(b["profile"])
                colors.append(b["color"])
        once(ROOT / "index/records.json", records)
        once(ROOT / "index/profile_texts.json", texts)
        np_once(
            ROOT / "index/vectors.npz",
            regions=regions,
            titles=np.stack(titles),
            profiles=np.stack(semantics),
            colors=np.stack(colors),
        )
        for q in load(ROOT / "queries.public.json"):
            folder = ROOT / "query_preparation" / q["id"]
            folder.mkdir(parents=True, exist_ok=True)
            p = folder / "vectors.npz"
            if p.exists():
                continue
            active_engine = encoder()
            if active_engine is None:
                return False
            start = len(active_engine.timings)
            if q["image_path"]:
                views, details = views_for(q["image_path"], folder, True)
                old = load(folder / "old_description.json")["text"]
                new = load(folder / "new_description.json")["text"]
                oldcap = (
                    active_engine.encode(text=old, query=True)
                    if old
                    else np.zeros(2048, np.float32)
                )
                newcap = (
                    active_engine.encode(text=new, query=True)
                    if new
                    else np.zeros(2048, np.float32)
                )
                selected = load(folder / "grounding/result.json")["selected_image"]
                zoom = active_engine.encode(image=dataset.Path(selected), query=True)
                once(folder / "views.json", details)
            else:
                views = active_engine.encode(text=q["text"], query=True)[None, :]
                oldcap = newcap = zoom = np.zeros(2048, np.float32)
            np_once(p, views=views, old_caption=oldcap, new_caption=newcap, zoom=zoom)
            once(
                folder / "encoding_runtime.json",
                dict(calls=active_engine.timings[start:]),
            )
        once(
            ROOT / "embedding_runtime.json",
            engine.diagnostics() if engine else dict(resumed_no_new_forward=True),
        )
    finally:
        if engine:
            engine.close()
    dataset.receipt(
        "encoding_receipt.json",
        [
            *[p for p in (ROOT / "index").rglob("*") if p.is_file()],
            *[p for p in (ROOT / "query_preparation").rglob("*") if p.is_file()],
            ROOT / "embedding_runtime.json",
        ],
        gallery=96,
        queries=24,
        complete_official_Qwen=True,
    )
    return True


def index_inputs() -> (
    tuple[RegionalIndex, np.ndarray, np.ndarray, list[str], np.ndarray]
):
    """Return index inputs for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    with np.load(ROOT / "index/vectors.npz", allow_pickle=False) as b:
        values = {k: b[k].copy() for k in b.files}
    return (
        RegionalIndex(
            load(ROOT / "index/records.json"), values["regions"], region_configuration()
        ),
        values["titles"],
        values["profiles"],
        load(ROOT / "index/profile_texts.json"),
        values["colors"],
    )


def plan() -> None:
    """Apply the plan workflow step.

    Notes:
        Writes derived artifacts through once.
    """
    dataset.guard()
    dataset.checked("encoding_receipt.json")
    banks = index_inputs()
    index = banks[0]
    ids = [d["id"] for d in index.records]
    documents = {
        d["id"]: dict(id=d["id"], title=d["title"], image_path=d["image_path"])
        for d in index.records
    }
    proposals = []
    jobs = []
    for q in load(ROOT / "queries.public.json"):
        folder = ROOT / "query_preparation" / q["id"]
        with np.load(folder / "vectors.npz", allow_pickle=False) as b:
            views, oldcap, newcap, zoom = [
                b[k].copy() for k in ("views", "old_caption", "new_caption", "zoom")
            ]
        olddesc = (
            load(folder / "old_description.json")["text"] if q["image_path"] else ""
        )
        old = feature_values(
            GroundedBlindRuntimeFeatureValuesInputs(*banks, q, views, oldcap, olddesc)
        )
        if q["image_path"]:
            newdesc = load(folder / "new_description.json")["text"]
            new = feature_values(
                GroundedBlindRuntimeFeatureValuesInputs(
                    *banks, q, views, newcap, newdesc
                )
            )
            cosine = index.vectors[:, 0] @ zoom
            pool = proposal(ids, old["pool"], new["pool"], cosine)
            base = (
                0.375 * normalized_ranks(ids, old["base"])
                + 0.375 * normalized_ranks(ids, new["base"])
                + 0.25 * normalized_ranks(ids, cosine)
            )
            selected = load(folder / "grounding/result.json")["selected_image"]
            contexts = [
                ("original", q, olddesc, old["pool"]),
                ("whole", q, newdesc, pool),
                ("zoom", dict(q, image_path=selected), newdesc, pool),
            ]
        else:
            new = None
            pool = old["pool"]
            base = np.asarray(old["base"])
            contexts = [("original", q, "", pool)]
        proposals.append(
            dict(query=q, ids=ids, old=old, new=new, pool=pool, base=base.tolist())
        )
        for context, query, description, candidates in contexts:
            for cid in candidates:
                doc = documents[cid]
                jobs.append(
                    dict(
                        query_id=q["id"],
                        context=context,
                        candidate_id=cid,
                        query=query,
                        description=description,
                        document=doc,
                        identity=pair_identity(query, description, doc),
                    )
                )
    once(ROOT / "proposals.public.json", proposals)
    once(ROOT / "pair_plan.json", jobs)
    dataset.receipt(
        "plan_receipt.json",
        [ROOT / "proposals.public.json", ROOT / "pair_plan.json"],
        context_positions=len(jobs),
        unique_pairs=len({j["identity"]["sha256"] for j in jobs}),
        heldout_candidate_choices=1,
        grade_files_not_read=True,
    )


def infer() -> bool:
    """Run model inference using the configured inputs.

    Returns:
        Whether the condition checked by this operation holds.

    Notes:
        Writes derived artifacts through once.
    """
    dataset.guard()
    dataset.checked("plan_receipt.json")
    engine = None
    jobs = load(ROOT / "pair_plan.json")
    try:
        for i, j in enumerate(jobs):
            p = ROOT / "pairs" / (j["identity"]["sha256"] + ".json")
            if not p.exists():
                if engine is None:
                    if not admission.gpu_available():
                        dataset.status("waiting_natural_GPU_release_for_blind_reranker")
                        return False
                    engine = DescribedImageReranker()
                    dataset.status("single_frozen_candidate_blind_predictions_running")
                d = j["document"]
                docs = [dict(id=d["id"], image_path=d["image_path"], text=d["title"])]
                image = (
                    dataset.Path(j["query"]["image_path"])
                    if j["query"]["image_path"]
                    else None
                )
                scores = (
                    engine.score_described(
                        docs, query_image=image, description=j["description"]
                    )
                    if image and j["description"]
                    else engine.score(
                        docs, query_image=image, query_text=j["query"]["text"]
                    )
                )
                assert len(scores) == 1 and math.isfinite(float(scores[0]))
                once(
                    p,
                    dict(
                        identity=j["identity"],
                        logit=float(scores[0]),
                        timing=engine.adapter.timings[-1],
                    ),
                )
            r = load(p)
            assert r["identity"] == j["identity"] and math.isfinite(r["logit"])
            if (i + 1) % 100 == 0 or i + 1 == len(jobs):
                print(dict(blind_context_positions=i + 1, total=len(jobs)), flush=True)
        once(
            ROOT / "reranker_runtime.json",
            engine.diagnostics() if engine else dict(resumed_no_new_forward=True),
        )
    finally:
        if engine:
            engine.close()
    dataset.receipt(
        "reranker_receipt.json",
        [ROOT / "reranker_runtime.json", *sorted((ROOT / "pairs").glob("*.json"))],
        unique_actual_pairs=len(list((ROOT / "pairs").glob("*.json"))),
        models_reduced=False,
        missing_score_imputation=0,
    )
    return True


def orders() -> dict:
    """Return orders for this workflow.

    Returns:
        Result mapping produced by the workflow.
    """
    contexts: dict = {}
    for j in load(ROOT / "pair_plan.json"):
        contexts.setdefault(j["query_id"], {}).setdefault(j["context"], {})[
            j["candidate_id"]
        ] = load(ROOT / "pairs" / (j["identity"]["sha256"] + ".json"))["logit"]
    result = {}
    for p in load(ROOT / "proposals.public.json"):
        q = p["query"]
        qid = q["id"]
        old = p["old"]
        ids = p["ids"]
        matrix = np.asarray(old["features"])
        context = contexts[qid]
        base = np.asarray(old["base"])
        pool = old["pool"]
        offset = None if old["offsets"] is None else np.asarray(old["offsets"])
        cross = rerank_complete(
            ids,
            base,
            pool,
            np.asarray([context["original"][cid] for cid in pool]),
            offset,
        )
        primary = (
            fuse(
                GroundedSubjectRuntimeFuseInputs(
                    ids,
                    p["pool"],
                    p["base"],
                    context["whole"],
                    context["zoom"],
                    mode="dual",
                )
            )
            if q["image_path"]
            else cross
        )
        result[qid] = dict(
            global_=ranked_ids(ids, matrix[:, 0]),
            region_title=ranked_ids(ids, 0.8 * matrix[:, 1] + 0.2 * matrix[:, 3]),
            cross20=cross,
            **{PRIMARY: primary},
        )
        result[qid]["global"] = result[qid].pop("global_")
    return result


def prediction() -> None:
    """Apply the prediction workflow step.

    Notes:
        Writes derived artifacts through once.
    """
    dataset.guard()
    dataset.checked("reranker_receipt.json")
    once(ROOT / "predictions.json", orders())
    dataset.receipt(
        "prediction_receipt.json",
        [ROOT / "predictions.json"],
        methods=4,
        rankings=96,
        query_predictions=24,
        frozen_candidate_count=1,
        labels_used_in_runtime=False,
    )


def worker() -> None:
    """Apply the worker workflow step.

    Notes:
        Writes derived artifacts through p.parent.mkdir.
    """
    from multimodal_ecommerce import platform_lock as msvcrt

    p = ROOT / "runtime.lock"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a+b") as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            for name, stage in [
                ("native_query_receipt.json", native_queries),
                ("encoding_receipt.json", encode_all),
                ("plan_receipt.json", plan),
                ("reranker_receipt.json", infer),
                ("prediction_receipt.json", prediction),
            ]:
                if (ROOT / name).exists():
                    dataset.checked(name)
                    continue
                while stage() is False:
                    print(
                        dict(waiting_natural_GPU_release=True, stage=name), flush=True
                    )
                    time.sleep(30)
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["worker", "verify_predictions"])
    a = parser.parse_args().action
    if a == "worker":
        worker()
    else:
        dataset.guard()
        dataset.checked("prediction_receipt.json")
        assert orders() == load(ROOT / "predictions.json")
        print(dict(blind_prediction_replay_passed=True))
