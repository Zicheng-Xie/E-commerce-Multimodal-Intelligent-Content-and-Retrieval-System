"""Label-free native product descriptions and full models for all 19 additions.

This creates a new feature bank. It never changes an existing sealed experiment
or consults relation grades to select products or generate runtime descriptions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

import balanced_pair_review as review
import balanced_supplement as acquisition
import recover_balanced_media as recovery
import run  # noqa: F401
from compare_fgclip2_retrieval import text_key
from encode_primary_subjects import append_call, calls, save_arrays, save_npy
from learned_fusion_experiment import load, once, rel
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.fgclip_retrieval import FGClipEncoder
from multimodal_ecommerce.holdout_data import fingerprints
from multimodal_ecommerce.machine_teacher import resolve_local_teacher
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.precision_reranking import color_descriptor
from multimodal_ecommerce.regional_retrieval import (
    RegionalRetrievalEncodeViewsInputs,
    encode_views,
)
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.subject_roi import (
    SubjectDetector,
    save_subject_crop,
    subject_prompt,
)
from primary_subject_captions import PROMPT, SCHEMA

ROOT = PROJECT_ROOT / "artifacts/module2_balanced_supplement_features"
PARENT = PROJECT_ROOT / "artifacts/module2_primary_subjects"
FGROOT = PROJECT_ROOT / "artifacts/module2_fgclip2_comparison"
ROIROOT = PROJECT_ROOT / "artifacts/module2_subject_roi"


OPTIONS: dict[str, int] = dict(
    num_ctx=16384, num_predict=1024, temperature=0, seed=2026102413
)
PROFILE_PROMPT = PROMPT + "\n仅识别这一件候选商品。原始标题仅用于区分主售品和道具："
SOURCES = [
    "scripts/balanced_supplement_features.py",
    "tests/test_balanced_supplement_features.py",
    "scripts/balanced_pair_review.py",
    "scripts/balanced_supplement.py",
    "scripts/recover_balanced_media.py",
    "scripts/encode_primary_subjects.py",
    "scripts/compare_fgclip2_retrieval.py",
    "scripts/learned_fusion_experiment.py",
    "scripts/primary_subject_captions.py",
]


def selected() -> list:
    # The full technical-success set is fixed by acquisition, never by qrels.
    """Return selected for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    rows = read_rows(recovery.ROOT / "selected.evaluator.jsonl")
    assert len(rows) == len({r["id"] for r in rows}) == 19
    return [dict(id=r["id"], title=r["text"], image_path=r["image_path"]) for r in rows]


def profile_payload(doc: dict) -> tuple[dict, dict]:
    """Return profile payload for this workflow.

    Args:
        doc: Doc used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    assert set(doc) == {"id", "title", "image_path"}
    encoded, checksum = acquisition.encoded(doc["image_path"])
    prompt = PROFILE_PROMPT + doc["title"]
    request = dict(
        model=acquisition.MODEL,
        messages=[dict(role="user", content=prompt, images=[encoded])],
        format=SCHEMA,
        options=OPTIONS,
        stream=False,
        think=False,
        keep_alive="10m",
    )
    return request, dict(
        candidate_id=doc["id"],
        original_title=doc["title"],
        image_png_sha256=checksum,
        prompt=prompt,
        options=OPTIONS,
        schema=SCHEMA,
        model_digest=load(acquisition.ROOT / "model_manifest.json")["digest"],
        keep_alive="10m",
        queries_labels_and_rankings_supplied=False,
    )


def profile_value(raw: dict) -> Any:
    """Return profile value for this workflow.

    Args:
        raw: Raw used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    assert raw["model"] == acquisition.MODEL
    assert raw["done"] is True and raw["done_reason"] == "stop"
    for key, maximum in [
        ("prompt_eval_count", OPTIONS["num_ctx"] - OPTIONS["num_predict"]),
        ("eval_count", OPTIONS["num_predict"]),
    ]:
        assert type(raw[key]) is int and 0 <= raw[key] < maximum

    def unique(pairs: list[tuple[str, object]]) -> dict:
        """Return unique for this workflow.

        Args:
            pairs: Pairs used by this operation.

        Returns:
            Result mapping produced by the workflow.

        Raises:
            ValueError: Duplicate profile key.
        """
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate profile key")
            value[key] = item
        return value

    assert raw["message"]["role"] == "assistant"
    result = json.loads(raw["message"]["content"], object_pairs_hook=unique)
    assert set(result) == set(SCHEMA["required"])
    assert type(result["primary_unclear"]) is bool
    for key, limit in [("product", 60), ("appearance", 200), ("visible_text", 120)]:
        assert isinstance(result[key], str) and len(result[key]) <= limit
        assert key == "visible_text" or result[key].strip()
    return result


def receipt(name: str, paths: list[Path], **extra: object) -> None:
    """Record file hashes and provenance for the completed stage.

    Args:
        name: Name used by this operation.
        paths: Paths used by this operation.
        **extra: Named options forwarded to the underlying operation.

    Notes:
        Writes derived artifacts through once.
    """
    once(
        ROOT / name,
        dict(files={rel(p): sha256_file(p) for p in paths}, **extra),
    )


def guard_receipt(name: str) -> Any:
    """Return guard receipt for this workflow.

    Args:
        name: Name used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    result = load(ROOT / name)
    verify_files(PROJECT_ROOT, result["files"])
    return result


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Notes:
        Writes derived artifacts through once, write_json.
    """
    recovery.guard()
    review.guard()
    old = load(PARENT / "completion_receipt.json")
    verify_files(PARENT, old["files"])
    verify_files(PROJECT_ROOT, old["source_files"])
    verify_files(PROJECT_ROOT, old["project_files"])
    if (ROOT / "protocol.json").exists():
        guard()
        return
    assert load(ROOT / "engineering_checks.pre.json")["passed"]
    paths = [PROJECT_ROOT / s for s in SOURCES]
    paths += sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py"))
    paths += [
        ROOT / "engineering_checks.pre.json",
        recovery.ROOT / "protocol.json",
        recovery.ROOT / "acquisition_receipt.json",
        recovery.ROOT / "selected.evaluator.jsonl",
        recovery.ROOT / "post_acquisition_exclusion_snapshot.json",
        review.ROOT / "protocol.json",
        acquisition.ROOT / "model_manifest.json",
        PARENT / "completion_receipt.json",
        PARENT / "feature_paths.json",
        PARENT / "roi_feature_index.json",
        PARENT / "inputs.json",
        FGROOT / "model_manifest.json",
        ROIROOT / "detector_manifest.json",
        PROJECT_ROOT / "configs/embedding_model.json",
    ]
    for cohort in ("tune", "regression"):
        paths += sorted((PARENT / cohort / "index").glob("*.*"))
    paths += [Path(d["image_path"]) for d in selected()]
    once(ROOT / "catalog.public.json", selected())
    paths.append(ROOT / "catalog.public.json")
    once(
        ROOT / "protocol.json",
        dict(
            files={rel(p): sha256_file(p) for p in paths},
            additions=19,
            gallery_per_cohort=315,
            product_selection="All 19 nonduplicate acquisition successes, regardless of grades or strata",
            profile_source="New candidate-only original image and title; no query, relation judgment, ranking, or review-derived profile supplied",
            profile_prompt=PROFILE_PROMPT,
            profile_schema=SCHEMA,
            profile_options=OPTIONS,
            full_native_profile_images=True,
            profile_uncertainty="zero semantic vector and no detector request; retain original image/title and product",
            full_official_models=True,
            original_query_features_reused_exactly=True,
            original_296_gallery_features_reused_exactly=True,
            regions=load(PARENT / "tune/index/manifest.json")["regions"],
            GPU_admission="nvidia-smi finite nonnegative below 1500MiB, then explicit empty Ollama model list; natural expiry, never unload or shorten keep-alive",
            comparison_predictions_created=False,
            formal_metrics=None,
            independent_acceptance=False,
        ),
    )
    write_json(
        ROOT / "public_status.json",
        dict(
            phase="features_frozen_relation_review_pending",
            completed=False,
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
        ),
    )


def guard() -> Any:
    """Validate the frozen inputs before continuing the workflow.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    p = load(ROOT / "protocol.json")
    verify_files(PROJECT_ROOT, p["files"])
    assert p["profile_options"] == OPTIONS
    assert p["profile_prompt"] == PROFILE_PROMPT
    assert p["profile_schema"] == SCHEMA
    return p


def profiles() -> bool:
    """Return profiles for this workflow.

    Returns:
        Whether the condition checked by this operation holds.

    Notes:
        Writes derived artifacts through once, receipt, write_rows.
    """
    guard()
    labels = load(review.ROOT / "label_receipt.json")
    verify_files(review.ROOT, labels["files"])
    if (ROOT / "profile_receipt.json").exists():
        guard_receipt("profile_receipt.json")
        return True
    docs = selected()
    pending = [
        i for i in range(len(docs)) if not (ROOT / f"profiles/{i:03d}.json").exists()
    ]
    if pending and not acquisition.gpu_available():
        return False
    client = None
    try:
        if pending:
            client = OllamaClient(timeout=240)
            model = resolve_local_teacher(
                client,
                Path(load(acquisition.ROOT / "model_manifest.json")["model_root"]),
            )
            assert (
                model["digest"]
                == load(acquisition.ROOT / "model_manifest.json")["digest"]
            )
            once(ROOT / "profile_model_manifest.json", model)
        for i in pending:
            request, identity = profile_payload(docs[i])
            raw, value, error = None, None, None
            start = time.perf_counter()
            try:
                assert client is not None
                raw = client._request("/api/chat", request)
                value = profile_value(raw)
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Handled exception in profiles", exc_info=True
                )
                error = dict(type=type(exc).__name__, message=str(exc))
            record = dict(
                identity=identity,
                raw=raw,
                value=value,
                exception=error,
                seconds=time.perf_counter() - start,
            )
            folder = ROOT / f"profile_attempts/{i:03d}"
            path = folder / f"{len(list(folder.glob('*.json')))+1:03d}.json"
            once(path, record)
            if error is not None:
                raise RuntimeError(f"Preserved native profile failure: {rel(path)}")
            once(ROOT / f"profiles/{i:03d}.json", dict(record, attempt_file=rel(path)))
            print(dict(candidate_only_profiles=i + 1, total=len(docs)), flush=True)
    finally:
        if client:
            client.close()
    rows = []
    for i, doc in enumerate(docs):
        record = load(ROOT / f"profiles/{i:03d}.json")
        assert record["identity"] == profile_payload(doc)[1]
        value = profile_value(record["raw"])
        assert record["value"] == value and record["exception"] is None
        rows.append(
            dict(
                id=doc["id"],
                product_type=value["product"],
                appearance=value["appearance"],
                primary_unclear=value["primary_unclear"],
            )
        )
    path = ROOT / "profiles.candidate_only.jsonl"
    if path.exists():
        assert read_rows(path) == rows
    else:
        write_rows(path, rows)
    receipt(
        "profile_receipt.json",
        [
            path,
            ROOT / "profile_model_manifest.json",
            *sorted((ROOT / "profiles").glob("*.json")),
            *sorted((ROOT / "profile_attempts").rglob("*.json")),
        ],
        actual_full_native_requests=19,
        queries_or_labels_supplied=False,
        semantic_abstentions=sum(r["primary_unclear"] for r in rows),
        relation_label_receipt_sha256=sha256_file(review.ROOT / "label_receipt.json"),
    )
    return True


def full_models() -> bool:
    """Sequential PyTorch stages; called only after natural native-model expiry.

    Returns:
        Whether the condition checked by this operation holds.
    """
    guard()
    guard_receipt("profile_receipt.json")
    for name, action in [
        ("qwen_receipt.json", qwen),
        ("detection_receipt.json", detect),
        ("fg_receipt.json", fg),
    ]:
        if (ROOT / name).exists():
            guard_receipt(name)
            continue
        if not acquisition.gpu_available():
            return False
        action()
    return True


def profile_rows() -> dict:
    """Return profile rows for this workflow.

    Returns:
        Result mapping produced by the workflow.
    """
    return {r["id"]: r for r in read_rows(ROOT / "profiles.candidate_only.jsonl")}


def checked_vectors(
    bank: dict, *, abstain: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return checked vectors for this workflow.

    Args:
        bank: Bank used by this operation.
        abstain: Abstain used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    assert set(bank) == {"views", "title", "profile"}
    views, title, profile = (np.asarray(bank[k]) for k in ("views", "title", "profile"))
    assert views.ndim == 2 and 1 <= len(views) <= 7 and views.shape[1] == 2048
    assert title.shape == profile.shape == (2048,)
    assert all(np.isfinite(v).all() for v in (views, title, profile))
    assert (np.linalg.norm(views, axis=1) > 0).all() and np.linalg.norm(title) > 0
    assert np.linalg.norm(profile) == 0 if abstain else np.linalg.norm(profile) > 0
    padded = np.zeros((7, 2048), dtype=np.float32)
    padded[: len(views)] = views
    return padded, title.copy(), profile.copy()


class LoggedQwen:
    """Record actual Qwen embedding calls while reusing the pinned full encoder."""

    def __init__(self) -> None:
        """Initialize the component and validate its configured dependencies."""
        self.encoder = QwenEmbedder()

    def encode(
        self, *, text: str | None = None, image: Path | None = None, query: bool = False
    ) -> np.ndarray:
        """Encode the configured inputs into model feature representations.

        Args:
            text: Text input used by the operation.
            image: Image input used by the operation.
            query: Query record or representation used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        vector = self.encoder.encode(text=text, image=image, query=query)
        append_call(
            ROOT / "qwen_calls.jsonl",
            dict(
                text_sha256=text_key(text) if text is not None else None,
                image_sha256=sha256_file(image) if image else None,
                query=query,
                detail=self.encoder.diagnostics()["calls"][-1],
                vector_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),
            ),
        )
        return vector


def qwen() -> None:
    """Apply the qwen workflow step.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    guard_receipt("profile_receipt.json")
    encoder, records, vectors, texts, colors = LoggedQwen(), [], [], [], []
    profiles_by_id = profile_rows()
    try:
        for i, doc in enumerate(selected()):
            folder = ROOT / f"additions/index/products/{i:03d}"
            path = folder / "vectors.npz"
            row = profiles_by_id[doc["id"]]
            text = (
                ""
                if row["primary_unclear"]
                else "商品主体："
                + row["product_type"]
                + "。可见结构："
                + row["appearance"]
                + "。"
            )
            if not path.exists():
                views, metadata = encode_views(
                    RegionalRetrievalEncodeViewsInputs(
                        encoder,
                        Path(doc["image_path"]),
                        folder / "views",
                        guard()["regions"],
                    )
                )
                title = encoder.encode(text=doc["title"])
                semantic = (
                    encoder.encode(text=text)
                    if text
                    else np.zeros(2048, dtype=np.float32)
                )
                once(folder / "views.json", metadata)
                save_arrays(path, views=views, title=title, profile=semantic)
                once(path.with_suffix(".sha.json"), dict(sha256=sha256_file(path)))
            assert load(path.with_suffix(".sha.json"))["sha256"] == sha256_file(path)
            with np.load(path, allow_pickle=False) as b:
                vectors.append(checked_vectors(b, abstain=row["primary_unclear"]))
            records.append(dict(**doc, views=load(folder / "views.json")))
            texts.append(text)
            colors.append(color_descriptor(Path(doc["image_path"])))
            print(dict(Qwen_products=i + 1, total=19), flush=True)
        once(ROOT / "qwen_runtime.json", encoder.encoder.diagnostics())
    finally:
        encoder.encoder.close()
    folder = ROOT / "additions/index"
    arrays = dict(
        zip(("regions", "texts", "profiles"), [np.stack(v) for v in zip(*vectors)])
    )
    save_arrays(folder / "vectors.npz", **arrays)
    once(folder / "records.json", records)
    once(folder / "profile_texts.json", texts)
    save_npy(folder / "colors.npy", np.stack(colors))
    for cohort in ("tune", "regression"):
        old, new = PARENT / cohort / "index", ROOT / cohort / "index"
        with np.load(old / "vectors.npz", allow_pickle=False) as b:
            merged = {k: np.concatenate([b[k], arrays[k]]) for k in arrays}
        save_arrays(new / "vectors.npz", **merged)
        save_npy(
            new / "colors.npy",
            np.concatenate(
                [np.load(old / "colors.npy", allow_pickle=False), np.stack(colors)]
            ),
        )
        once(new / "records.json", load(old / "records.json") + records)
        once(new / "profile_texts.json", load(old / "profile_texts.json") + texts)
        once(
            new / "manifest.json",
            dict(
                files={
                    p.name: sha256_file(p)
                    for p in sorted(new.glob("*.*"))
                    if p.name != "manifest.json"
                },
                model=load(PROJECT_ROOT / "configs/embedding_model.json"),
                regions=guard()["regions"],
            ),
        )
    paths = [ROOT / "qwen_calls.jsonl", ROOT / "qwen_runtime.json"]
    paths += [
        p
        for directory in (folder, ROOT / "tune/index", ROOT / "regression/index")
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    ]
    receipt(
        "qwen_receipt.json",
        paths,
        actual_full_model_calls=len(calls(ROOT / "qwen_calls.jsonl")),
        original_vectors_preserved=True,
    )


def detect() -> None:
    """Apply the detect workflow step.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    profiles_by_id = profile_rows()
    detector = SubjectDetector(ROIROOT / "detector_manifest.json")
    try:
        for i, doc in enumerate(selected()):
            path = ROOT / f"detections/{i:03d}/result.json"
            if path.exists():
                continue
            row = profiles_by_id[doc["id"]]
            prompt = (
                None if row["primary_unclear"] else subject_prompt(row["product_type"])
            )
            result = (
                detector.detect(Path(doc["image_path"]), prompt)
                if prompt
                else dict(
                    prompt=None,
                    decision=dict(
                        status="global_fallback",
                        reason="uncertain or unsupported candidate-only subject",
                        selected=None,
                    ),
                    detections=[],
                    seconds=0,
                )
            )
            result.update(
                source_sha256=sha256_file(Path(doc["image_path"])),
                actual_model_call=bool(prompt),
            )
            if result["decision"]["status"] == "roi":
                result["crop"] = save_subject_crop(
                    Path(doc["image_path"]), result["decision"], path.parent
                )
            once(path, result)
        once(ROOT / "detector_runtime.json", detector.diagnostics())
    finally:
        detector.close()
    receipt(
        "detection_receipt.json",
        [
            ROOT / "detector_runtime.json",
            *[p for p in sorted((ROOT / "detections").rglob("*")) if p.is_file()],
        ],
        actual_calls=sum(
            load(p)["actual_model_call"]
            for p in (ROOT / "detections").rglob("result.json")
        ),
    )


def fg_image(encoder: FGClipEncoder, source: Path, path: Path) -> None:
    """Apply the fg image workflow step.

    Args:
        encoder: Encoder used by this operation.
        source: Source used by this operation.
        path: Path of the input or output artifact used by this operation.

    Notes:
        Writes derived artifacts through once.
    """
    if not path.exists():
        global_vector, dense, detail = encoder.image(source)
        append_call(
            ROOT / "fg_calls.jsonl",
            dict(source_sha256=sha256_file(source), detail=detail),
        )
        save_arrays(path, global_vector=global_vector, dense=dense)
        once(
            path.with_suffix(".json"),
            dict(source_sha256=sha256_file(source), sha256=sha256_file(path)),
        )
    assert load(path.with_suffix(".json")) == dict(
        source_sha256=sha256_file(source), sha256=sha256_file(path)
    )


def fg() -> None:
    """Apply the fg workflow step.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    guard_receipt("detection_receipt.json")
    encoder = FGClipEncoder(FGROOT / "model_manifest.json")
    rows = []
    try:
        for i, doc in enumerate(selected()):
            folder = ROOT / f"fg/{i:03d}"
            source = Path(doc["image_path"])
            fg_image(encoder, source, folder / "whole.npz")
            path = folder / "title.npz"
            if not path.exists():
                vector, detail = encoder.text(doc["title"], regional=False)
                append_call(
                    ROOT / "fg_calls.jsonl",
                    dict(source_sha256=text_key(doc["title"]), detail=detail),
                )
                save_arrays(path, vector=vector)
                once(
                    path.with_suffix(".json"),
                    dict(
                        source_sha256=text_key(doc["title"]), sha256=sha256_file(path)
                    ),
                )
            assert load(path.with_suffix(".json")) == dict(
                source_sha256=text_key(doc["title"]), sha256=sha256_file(path)
            )
            detection = load(ROOT / f"detections/{i:03d}/result.json")
            crop = ROOT / f"detections/{i:03d}/zoom.png"
            if detection["decision"]["status"] == "roi":
                fg_image(encoder, crop, folder / "roi.npz")
                roi_path = folder / "roi.npz"
            else:
                roi_path = folder / "whole.npz"
            rows.append(
                dict(
                    id=doc["id"],
                    image_key=sha256_file(source),
                    title_key=text_key(doc["title"]),
                    whole=rel(folder / "whole.npz"),
                    title=rel(path),
                    roi=rel(roi_path),
                    crop_sha256=sha256_file(crop) if crop.exists() else None,
                )
            )
            print(dict(FG_products=i + 1, total=19), flush=True)
        once(ROOT / "fg_runtime.json", encoder.diagnostics())
    finally:
        encoder.close()
    once(ROOT / "addition_FG_paths.json", rows)
    receipt(
        "fg_receipt.json",
        [
            ROOT / "fg_runtime.json",
            ROOT / "fg_calls.jsonl",
            ROOT / "addition_FG_paths.json",
            *[p for p in sorted((ROOT / "fg").rglob("*")) if p.is_file()],
        ],
        actual_full_model_requests=len(calls(ROOT / "fg_calls.jsonl")),
    )


def exposure() -> None:
    """Apply the exposure workflow step.

    Notes:
        Writes derived artifacts through once, write_json, write_rows.
    """
    for name in (
        "profile_receipt.json",
        "qwen_receipt.json",
        "detection_receipt.json",
        "fg_receipt.json",
    ):
        guard_receipt(name)
    paths = sorted((ROOT / "additions/index").rglob("*.png")) + sorted(
        (ROOT / "detections").rglob("*.png")
    )
    registry = PROJECT_ROOT / "data/module2_balanced_supplement_derived/records.jsonl"
    rows = [
        dict(
            id="balanced_derived/" + p.relative_to(ROOT).as_posix(),
            image_path=os.path.relpath(p, registry.parent).replace("\\", "/"),
            fingerprints=fingerprints(p),
            project_role="exposed_development_not_acceptance",
        )
        for p in paths
    ]
    if registry.exists():
        assert read_rows(registry) == rows
    else:
        write_rows(registry, rows)
    prior = load(review.ROOT / "post_review_exclusion_snapshot.json")
    once(
        ROOT / "post_features_exclusion_snapshot.json",
        dict(
            prior,
            files=dict(prior["files"], **{rel(registry): sha256_file(registry)}),
            image_fingerprints=prior["image_fingerprints"]
            + [r["fingerprints"] for r in rows],
            scope="Every prior exposure, all 48 metadata, all 19 raw recovered images and all supplemental derived views/ROIs; none reusable for independent acceptance",
        ),
    )
    write_json(
        ROOT / "public_status.json",
        dict(
            phase="full_features_completed_comparison_pending",
            completed=False,
            additions_encoded=19,
            gallery_per_cohort=315,
            derived_images=len(rows),
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["prepare", "profiles", "full_models", "exposure"]
    )
    args = parser.parse_args()
    if args.action == "prepare":
        # CPU-only freeze in a separate root. It does not touch review outputs,
        # resident models, or the native review's exclusive GPU execution lock.
        prepare()
    else:
        with recovery.locked() as acquired:
            if acquired:
                print(
                    dict(action=args.action, result=globals()[args.action]()),
                    flush=True,
                )
            else:
                print(dict(another_runner_holds_lock=True), flush=True)
