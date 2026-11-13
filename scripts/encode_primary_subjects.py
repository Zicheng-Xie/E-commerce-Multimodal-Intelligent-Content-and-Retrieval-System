"""Full pinned models for precise-subject development additions and new captions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import run  # noqa: F401
from acquire_primary_subjects import check as check_acquisition
from compare_fgclip2_retrieval import relative, text_key
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.fgclip_retrieval import FGClipEncoder
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.precision_reranking import color_descriptor
from multimodal_ecommerce.regional_retrieval import (
    RegionalRetrievalEncodeViewsInputs,
    encode_views,
)
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from multimodal_ecommerce.subject_roi import (
    SubjectDetector,
    save_subject_crop,
    subject_prompt,
)
from subject_roi_experiment import load, once

ROOT = PROJECT_ROOT / "artifacts/module2_primary_subjects"
PARENT = PROJECT_ROOT / "artifacts/module2_subject_coverage_expansion"
FGROOT = PROJECT_ROOT / "artifacts/module2_fgclip2_comparison"
ROIROOT = PROJECT_ROOT / "artifacts/module2_subject_roi"
COHORTS = ("tune", "regression", "fresh")
CATALOG = ROOT / "additions/catalog.public.json"
PROFILES = ROOT / "additions/profiles.public.jsonl"


def receipt(
    path: Path, files: list[Path], *, base: Path = ROOT, **extra: object
) -> None:
    """Record file hashes and provenance for the completed stage.

    Args:
        path: Path of the input or output artifact used by this operation.
        files: Files used by this operation.
        base: Base used by this operation.
        **extra: Named options forwarded to the underlying operation.

    Notes:
        Writes derived artifacts through once.
    """
    once(
        path,
        dict(
            files={p.relative_to(base).as_posix(): sha256_file(p) for p in files},
            **extra,
        ),
    )


def save_arrays(path: Path, **arrays: np.ndarray) -> None:
    """Never overwrite an already completed feature file.

    Args:
        path: Path of the input or output artifact used by this operation.
        **arrays: Named options forwarded to the underlying operation.

    Raises:
        FileExistsError: Inspect incomplete feature file before resuming.
    """
    if path.exists():
        with np.load(path, allow_pickle=False) as prior:
            assert set(prior.files) == set(arrays)
            assert all(np.array_equal(prior[k], v) for k, v in arrays.items())
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    if temporary.exists():
        raise FileExistsError("Inspect incomplete feature file before resuming")
    np.savez_compressed(temporary, allow_pickle=False, **arrays)
    temporary.replace(path)


def save_npy(path: Path, array: np.ndarray) -> None:
    """Persist npy.

    Args:
        path: Path of the input or output artifact used by this operation.
        array: Array used by this operation.

    Notes:
        Writes derived artifacts through np.save, path.parent.mkdir.
    """
    if path.exists():
        assert np.array_equal(np.load(path, allow_pickle=False), array)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def append_call(path: Path, value: object) -> None:
    """Apply the append call workflow step.

    Args:
        path: Path of the input or output artifact used by this operation.
        value: Input value to validate or transform.

    Notes:
        Writes derived artifacts through path.parent.mkdir.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def calls(path: Path) -> list[dict] | list:
    """Return calls for this workflow.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return read_rows(path) if path.exists() else []


def gpu_guard() -> None:
    """Do not evict a user-owned resident Ollama job to obtain GPU memory."""
    client = OllamaClient()
    try:
        active = client._request("/api/ps").get("models", [])
    finally:
        client.close()
    if active:
        raise RuntimeError(
            "Resident Ollama model preserved. Resume after its owner releases it "
            + "or its keep-alive expires naturally."
        )


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        FileExistsError: Preserve frozen feature protocol.

    Notes:
        Writes derived artifacts through once.
    """
    check_acquisition()
    if (ROOT / "feature_protocol.json").exists():
        raise FileExistsError("Preserve frozen feature protocol")
    verify_files(ROOT, load(ROOT / "acquisition_receipt.json")["files"])
    verify_files(ROOT, load(ROOT / "label_receipt.json")["files"])
    verify_files(PROJECT_ROOT, load(ROOT / "query_caption_receipt.json")["files"])
    execution = load(ROOT / "query_caption_execution_protocol.json")
    verify_files(PROJECT_ROOT, execution["files"])
    original_caption = load(ROOT / "query_caption_protocol.json")
    assert all(
        execution[key] == original_caption[key]
        for key in ["prompt", "schema", "options"]
    )
    parent = load(PARENT / "completion_receipt.json")
    verify_files(PARENT, parent["files"])
    verify_files(PROJECT_ROOT, parent["source_files"])
    verify_files(PROJECT_ROOT, parent["project_files"])
    docs = load(CATALOG)
    profiles = {r["id"]: r for r in read_rows(PROFILES)}
    assert len(docs) and len({d["id"] for d in docs}) == len(docs)
    assert set(profiles) == {d["id"] for d in docs}
    caption_count = 0
    for cohort in COHORTS:
        image_queries = {
            q["id"]: q
            for q in load(PARENT / cohort / "queries.public.json")
            if q["image_path"]
        }
        sources = {p.stem: p for p in (ROOT / "query_captions" / cohort).glob("*.json")}
        assert set(sources) == set(image_queries)
        for qid, source in sources.items():
            assert load(source)["source_sha256"] == sha256_file(
                Path(image_queries[qid]["image_path"])
            )
        caption_count += len(sources)
    assert caption_count == 24
    inputs = load(PARENT / "inputs.json")
    additions = []
    source_files = [
        Path(__file__),
        PROJECT_ROOT / "scripts/acquire_primary_subjects.py",
        ROOT / "acquisition_protocol.json",
        ROOT / "acquisition_receipt.json",
        ROOT / "label_receipt.json",
        ROOT / "query_caption_protocol.json",
        ROOT / "query_caption_execution_protocol.json",
        PROJECT_ROOT / "scripts/primary_subject_captions_safe.py",
        ROOT / "query_caption_receipt.json",
        ROOT / "caption_model_manifest.json",
        CATALOG,
        PROFILES,
        PARENT / "completion_receipt.json",
        PARENT / "inputs.json",
        PARENT / "feature_paths.json",
        PARENT / "roi_feature_index.json",
        ROIROOT / "detector_manifest.json",
        FGROOT / "model_manifest.json",
        PROJECT_ROOT / "configs/embedding_model.json",
        *sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")),
    ]
    source_files += sorted((ROOT / "query_captions").rglob("*.json"))
    for doc in docs:
        image = Path(doc["image_path"])
        source_files.append(image)
        image_key, title_key = sha256_file(image), text_key(doc["title"])
        inputs["images"][image_key] = relative(image)
        inputs["texts"][title_key] = doc["title"]
        hint = profiles[doc["id"]]["product_type"]
        identity = dict(image_key=image_key, prompt=subject_prompt(hint))
        key = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()
        inputs["roi_requests"][key] = dict(**identity, hint=hint)
        additions.append(
            dict(
                id=doc["id"],
                image_key=image_key,
                title_key=title_key,
                roi_request=key,
            )
        )
    for cohort, data in inputs["cohorts"].items():
        old_ids = [d["id"] for d in data["gallery"]]
        assert len(old_ids) == 193 and not set(old_ids) & set(profiles)
        data["gallery"] += additions
        data["labels"] = relative(ROOT / cohort / "judgments.ai.jsonl")
        data["original_predictions"] = relative(PARENT / cohort / "predictions.json")
        for query in data["queries"]:
            run_path = ROOT / cohort / "runs" / query["id"]
            query["original_features"] = relative(run_path / "features.json")
            query["original_cross"] = relative(run_path / "cross.json")
    once(ROOT / "inputs.json", inputs)
    once(ROOT / "addition_feature_requests.json", additions)
    source_files += [ROOT / "inputs.json", ROOT / "addition_feature_requests.json"]
    once(
        ROOT / "feature_protocol.json",
        dict(
            files={relative(p): sha256_file(p) for p in source_files},
            regions=load(PARENT / "tune/index/manifest.json")["regions"],
            additions=len(docs),
            gallery_per_cohort=193 + len(docs),
            full_official_models=True,
            original_query_features_reused=True,
            new_query_caption_vectors=24,
            empty_caption="zero-vector, explicit abstention; no NULL embedding",
            old_FG_query_and_ROI_features_reused_exactly=True,
            qrels_used_to_compute_features=False,
            formal_metrics=None,
        ),
    )
    print(dict(prepared=True, additions=len(docs), gallery=193 + len(docs)), flush=True)


def guard() -> Any:
    """Validate the frozen inputs before continuing the workflow.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    protocol = load(ROOT / "feature_protocol.json")
    verify_files(PROJECT_ROOT, protocol["files"])
    return protocol


class LoggedQwen:
    """Record every completed real call, including work before an interruption."""

    def __init__(self) -> None:
        """Initialize the component and validate its configured dependencies."""
        self.encoder = QwenEmbedder()
        self.ledger = ROOT / "qwen_call_ledger.jsonl"

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
        identity = dict(
            text_sha256=text_key(text) if text is not None else None,
            image_sha256=sha256_file(image) if image is not None else None,
            query=query,
        )
        append_call(
            self.ledger,
            dict(
                inputs=identity,
                detail=self.encoder.diagnostics()["calls"][-1],
                vector_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),
            ),
        )
        return vector

    def close(self) -> None:
        """Release the resources held by this component."""
        self.encoder.close()


def qwen() -> None:
    """Apply the qwen workflow step.

    Raises:
        FileExistsError: Preserve completed Qwen encoding.

    Notes:
        Writes derived artifacts through cache.mkdir, once, receipt, target.mkdir.
    """
    protocol = guard()
    if (ROOT / "encoding_receipt.json").exists():
        raise FileExistsError("Preserve completed Qwen encoding")
    gpu_guard()
    docs = load(CATALOG)
    profiles = {r["id"]: r for r in read_rows(PROFILES)}
    texts = [
        "商品主体："
        + profiles[d["id"]]["product_type"]
        + "。可见结构："
        + profiles[d["id"]]["appearance"]
        + "。"
        for d in docs
    ]
    folder = ROOT / "additions/index"
    encoder = LoggedQwen()
    records, regions, titles, semantic = [], [], [], []
    try:
        for i, (doc, text) in enumerate(zip(docs, texts), 1):
            cache = folder / "products" / f"{i:03d}"
            cache.mkdir(parents=True, exist_ok=True)
            vector_path = cache / "vectors.npz"
            if not vector_path.exists():
                views, metadata = encode_views(
                    RegionalRetrievalEncodeViewsInputs(
                        encoder,
                        Path(doc["image_path"]),
                        cache / "views",
                        protocol["regions"],
                    )
                )
                title, profile = encoder.encode(text=doc["title"]), encoder.encode(
                    text=text
                )
                once(cache / "views.json", metadata)
                save_arrays(vector_path, views=views, title=title, profile=profile)
                once(
                    cache / "vectors.sha256.json", dict(sha256=sha256_file(vector_path))
                )
            assert load(cache / "vectors.sha256.json")["sha256"] == sha256_file(
                vector_path
            )
            with np.load(vector_path, allow_pickle=False) as bank:
                padded = np.zeros((7, 2048), dtype=np.float32)
                padded[: len(bank["views"])] = bank["views"]
                regions.append(padded)
                titles.append(bank["title"].copy())
                semantic.append(bank["profile"].copy())
            records.append(dict(**doc, views=load(cache / "views.json")))
            if i % 8 == 0 or i == len(docs):
                print(dict(encoded=i, total=len(docs)), flush=True)
        for cohort in COHORTS:
            for source in sorted((ROOT / "query_captions" / cohort).glob("*.json")):
                query = load(source)
                path = ROOT / "new_query_features" / cohort / (source.stem + ".npz")
                if not path.exists():
                    value = (
                        encoder.encode(text=query["text"], query=True)
                        if query["text"]
                        else np.zeros(2048, dtype=np.float32)
                    )
                    save_arrays(path, caption=value)
                    once(
                        path.with_suffix(".json"),
                        dict(
                            source_caption_sha256=sha256_file(source),
                            feature_sha256=sha256_file(path),
                            abstain=not bool(query["text"]),
                            actual_model_call=bool(query["text"]),
                        ),
                    )
                else:
                    metadata = load(path.with_suffix(".json"))
                    assert metadata["source_caption_sha256"] == sha256_file(source)
                    assert metadata["feature_sha256"] == sha256_file(path)
        runtime = encoder.encoder.diagnostics()
    finally:
        encoder.close()
    save_arrays(
        folder / "vectors.npz",
        regions=np.array(regions),
        texts=np.array(titles),
        profiles=np.array(semantic),
    )
    save_npy(
        folder / "colors.npy",
        np.array([color_descriptor(Path(d["image_path"])) for d in docs]),
    )
    once(folder / "records.json", records)
    once(folder / "profile_texts.json", texts)
    once(
        folder / "runtime.json",
        dict(**runtime, all_completed_calls=calls(ROOT / "qwen_call_ledger.jsonl")),
    )
    for cohort in ["tune", "regression"]:
        parent, target = PARENT / cohort / "index", ROOT / cohort / "index"
        target.mkdir(parents=True, exist_ok=True)
        with (
            np.load(parent / "vectors.npz", allow_pickle=False) as a,
            np.load(folder / "vectors.npz", allow_pickle=False) as b,
        ):
            save_arrays(
                target / "vectors.npz",
                **{
                    k: np.concatenate([a[k], b[k]])
                    for k in ["regions", "texts", "profiles"]
                },
            )
        once(target / "records.json", load(parent / "records.json") + records)
        once(target / "profile_texts.json", load(parent / "profile_texts.json") + texts)
        save_npy(
            target / "colors.npy",
            np.concatenate(
                [
                    np.load(parent / "colors.npy", allow_pickle=False),
                    np.load(folder / "colors.npy", allow_pickle=False),
                ]
            ),
        )
        receipt(
            target / "manifest.json",
            [
                p
                for p in sorted(target.iterdir())
                if p.is_file() and p.name != "manifest.json"
            ],
            base=target,
            model=load(PROJECT_ROOT / "configs/embedding_model.json"),
            regions=protocol["regions"],
            upstream_files={
                relative(p): sha256_file(p)
                for p in [
                    parent / "manifest.json",
                    parent / "vectors.npz",
                    folder / "vectors.npz",
                ]
            },
        )
    files = [ROOT / "qwen_call_ledger.jsonl"]
    files += [
        p
        for base in [
            folder,
            ROOT / "tune/index",
            ROOT / "regression/index",
            ROOT / "new_query_features",
        ]
        for p in sorted(base.rglob("*"))
        if p.is_file()
    ]
    receipt(
        ROOT / "encoding_receipt.json",
        files,
        actual_full_embedding_calls=len(calls(ROOT / "qwen_call_ledger.jsonl")),
        additions=len(docs),
        gallery_per_cohort=193 + len(docs),
        original_vectors_reused_exactly=True,
        new_query_caption_features=len(
            list((ROOT / "new_query_features").glob("*/*.npz"))
        ),
    )
    print(
        dict(
            Qwen_completed=True,
            actual_full_calls=len(calls(ROOT / "qwen_call_ledger.jsonl")),
        ),
        flush=True,
    )


def detect() -> None:
    """Apply the detect workflow step.

    Raises:
        FileExistsError: Preserve completed subject detections.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    if (ROOT / "detection_receipt.json").exists():
        raise FileExistsError("Preserve completed subject detections")
    gpu_guard()
    inputs, additions = load(ROOT / "inputs.json"), load(
        ROOT / "addition_feature_requests.json"
    )
    detector = SubjectDetector(ROIROOT / "detector_manifest.json")
    try:
        for i, doc in enumerate(additions, 1):
            key = doc["roi_request"]
            path = ROOT / "detections" / key / "result.json"
            if path.exists():
                continue
            row = inputs["roi_requests"][key]
            source = PROJECT_ROOT / inputs["images"][row["image_key"]]
            if row["prompt"]:
                result = detector.detect(source, row["prompt"])
            else:
                result = dict(
                    prompt=None,
                    decision=dict(
                        status="global_fallback",
                        reason="unsupported public subject; image retained",
                        selected=None,
                    ),
                    detections=[],
                    seconds=0,
                )
            result.update(
                request_id=key,
                source_sha256=row["image_key"],
                actual_model_call=bool(row["prompt"]),
            )
            if result["decision"]["status"] == "roi":
                result["crop"] = save_subject_crop(
                    source, result["decision"], path.parent
                )
            once(path, result)
            if i % 16 == 0 or i == len(additions):
                print(dict(detected=i, total=len(additions)), flush=True)
        runtime = detector.diagnostics()
    finally:
        detector.close()
    once(ROOT / "detector_runtime.json", runtime)
    rows = [
        load(ROOT / "detections" / d["roi_request"] / "result.json") for d in additions
    ]
    receipt(
        ROOT / "detection_receipt.json",
        [
            ROOT / "detector_runtime.json",
            *[p for p in sorted((ROOT / "detections").rglob("*")) if p.is_file()],
        ],
        actual_full_detector_calls=sum(r["actual_model_call"] for r in rows),
        accepted=sum(r["decision"]["status"] == "roi" for r in rows),
        fallback=sum(r["decision"]["status"] != "roi" for r in rows),
        all_additions_retained=True,
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
        vector, dense, detail = encoder.image(source)
        append_call(
            ROOT / "fg_call_ledger.jsonl",
            dict(source_sha256=sha256_file(source), detail=detail),
        )
        save_arrays(path, global_vector=vector, dense=dense)
        once(
            path.with_suffix(".json"),
            dict(
                source_sha256=sha256_file(source),
                detail=detail,
                feature_sha256=sha256_file(path),
            ),
        )
    else:
        assert load(path.with_suffix(".json"))["feature_sha256"] == sha256_file(path)


def fg() -> None:
    """Apply the fg workflow step.

    Raises:
        FileExistsError: Preserve completed FG features.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    verify_files(ROOT, load(ROOT / "detection_receipt.json")["files"])
    if (ROOT / "fg_encoding_receipt.json").exists():
        raise FileExistsError("Preserve completed FG features")
    gpu_guard()
    inputs = load(ROOT / "inputs.json")
    additions = load(ROOT / "addition_feature_requests.json")
    paths, fmap = load(PARENT / "feature_paths.json"), load(
        PARENT / "roi_feature_index.json"
    )
    old_images = dict(paths["images"])
    old_crops = {r["crop_sha256"]: r["file"] for r in fmap.values() if r["crop_sha256"]}
    encoder = FGClipEncoder(FGROOT / "model_manifest.json")
    try:
        for i, doc in enumerate(additions, 1):
            imkey, titlekey = doc["image_key"], doc["title_key"]
            if imkey not in paths["images"]:
                path = ROOT / "fg_features/images" / (imkey + ".npz")
                fg_image(encoder, PROJECT_ROOT / inputs["images"][imkey], path)
                paths["images"][imkey] = relative(path)
            if titlekey not in paths["texts"]:
                path = ROOT / "fg_features/text" / (titlekey + "_long.npz")
                if not path.exists():
                    vector, detail = encoder.text(
                        inputs["texts"][titlekey], regional=False
                    )
                    append_call(
                        ROOT / "fg_call_ledger.jsonl",
                        dict(source_sha256=titlekey, detail=detail),
                    )
                    save_arrays(path, vector=vector)
                    once(
                        path.with_suffix(".json"),
                        dict(
                            source_sha256=titlekey,
                            detail=detail,
                            feature_sha256=sha256_file(path),
                        ),
                    )
                else:
                    assert load(path.with_suffix(".json"))[
                        "feature_sha256"
                    ] == sha256_file(path)
                paths["texts"][titlekey] = relative(path)
            key = doc["roi_request"]
            result = load(ROOT / "detections" / key / "result.json")
            if result["decision"]["status"] == "roi":
                crop = ROOT / "detections" / key / "zoom.png"
                checksum = sha256_file(crop)
                reused = old_images.get(checksum) or old_crops.get(checksum)
                if reused:
                    fmap[key] = dict(file=reused, crop_sha256=checksum)
                else:
                    path = ROOT / "fg_features/roi" / (checksum + ".npz")
                    fg_image(encoder, crop, path)
                    fmap[key] = dict(file=relative(path), crop_sha256=checksum)
            else:
                fmap[key] = dict(file=paths["images"][imkey], crop_sha256=None)
            if i % 16 == 0 or i == len(additions):
                print(dict(FG_encoded=i, total=len(additions)), flush=True)
        runtime = encoder.diagnostics()
    finally:
        encoder.close()
    ledger = calls(ROOT / "fg_call_ledger.jsonl")
    vision = sum(r["detail"].get("actual_vision_forward_calls", 0) for r in ledger)
    text = sum(r["detail"]["kind"] == "text" for r in ledger)
    once(ROOT / "feature_paths.json", paths)
    once(ROOT / "roi_feature_index.json", fmap)
    once(ROOT / "fg_runtime.json", dict(**runtime, all_completed_calls=ledger))
    receipt(
        ROOT / "fg_encoding_receipt.json",
        [
            ROOT / n
            for n in [
                "feature_paths.json",
                "roi_feature_index.json",
                "fg_runtime.json",
                "fg_call_ledger.jsonl",
            ]
        ]
        + [p for p in sorted((ROOT / "fg_features").rglob("*")) if p.is_file()],
        new_full_model_requests=len(ledger),
        actual_full_vision_forward_calls=vision,
        actual_full_text_forward_calls=text,
        actual_full_total_forward_calls=vision + text,
        old_global_and_query_ROI_features_reused_exactly=True,
    )
    print(
        dict(FG_completed=True, vision_forwards=vision, text_forwards=text), flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "qwen", "detect", "fg"])
    globals()[parser.parse_args().action]()
