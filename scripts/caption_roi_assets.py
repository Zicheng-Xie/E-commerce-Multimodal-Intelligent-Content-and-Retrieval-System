"""Frozen caption-guided query ROIs and exact existing gallery crop provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.contracts import FileDigests
from multimodal_ecommerce.holdout_data import fingerprints
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.safety import PROJECT_ROOT, ensure_output_path, sha256_file
from multimodal_ecommerce.subject_roi import POLICY, SubjectDetector, save_subject_crop
from primary_subject_ranker import query_signature

ROOT = PROJECT_ROOT / "artifacts/module2_caption_roi_development"
PARENT = PROJECT_ROOT / "artifacts/module2_dual_path_development"
WHOLE_PARENT = PROJECT_ROOT / "artifacts/module2_primary_subjects"
ROI_ROOTS = (
    PROJECT_ROOT / "artifacts/module2_subject_roi",
    PROJECT_ROOT / "artifacts/module2_subject_coverage_expansion",
    WHOLE_PARENT,
)
COHORTS = ("tune", "regression", "fresh")
REGISTRY = PROJECT_ROOT / "data/module2_caption_roi_derived/records.jsonl"
IDLE_BELOW_MIB = 1500


COARSE_PROMPTS: dict[str, str | dict[str, str]] = {
    "audio": {"overear": "headphones.", "default": "earphones."},
    "shoe": {"boot": "boot.", "default": "shoe."},
    "coat": {"vest": "vest.", "default": "jacket."},
    "bag": {"organizer": "bag organizer.", "default": "handbag."},
    "pillow": "pillow.",
    "cabinet": "cabinet.",
    "cup": "cup. bottle.",
    "fruit": "fruit.",
    "phone_case": "phone case.",
}


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    """Represent a workspace path relative to the project root.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Text produced by this operation.
    """
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def once(path: Path, value: object) -> None:
    """Persist a new result without replacing an existing artifact.

    Args:
        path: Path of the input or output artifact used by this operation.
        value: Input value to validate or transform.

    Notes:
        Writes derived artifacts through path.parent.mkdir.
    """
    path = ensure_output_path(path)
    if path.exists():
        assert load(path) == value, f"Preserve completed artifact: {path.name}"
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    if pending.exists():
        assert load(pending) == value
    else:
        with pending.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
    pending.replace(path)


def coarse_prompt(value: dict) -> str | None:
    """Return coarse prompt for this workflow.

    Args:
        value: Input value to validate or transform.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    signature = query_signature(value)
    mapping = COARSE_PROMPTS.get(signature["family"])
    if isinstance(mapping, dict):
        return mapping.get(signature["subtype"], mapping["default"])
    return mapping


def valid_image(path: Path) -> list:
    """Return valid image for this workflow.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
    assert min(width, height) > 0 and max(width, height) / min(width, height) <= 200
    return [width, height]


def sealed_file(path: Path, root: Path, seal: dict, files: FileDigests) -> Path:
    """Return sealed file for this workflow.

    Args:
        path: Path of the input or output artifact used by this operation.
        root: Root directory used to resolve the operation artifacts.
        seal: Seal used by this operation.
        files: Files used by this operation.

    Returns:
        Resolved artifact path.

    Notes:
        Updates the supplied files in place.
    """
    digest = sha256_file(path)
    assert seal["files"][path.relative_to(root).as_posix()] == digest
    files[relative(path)] = digest
    return path


@dataclass(frozen=True)
class CaptionRoiAssetsGalleryAssetInputs:
    """Input values for gallery asset.

    Attributes:
        doc: Doc used by this operation.
        route: Route used by this operation.
        inputs: Inputs used by this operation.
        fmap: Fmap used by this operation.
        feature_paths: Feature paths used by this operation.
        seals: Seals used by this operation.
        files: Files used by this operation.
    """

    doc: dict
    route: dict
    inputs: dict
    fmap: dict
    feature_paths: dict
    seals: dict
    files: FileDigests


def gallery_asset(options: CaptionRoiAssetsGalleryAssetInputs) -> dict:
    """Locate the original crop by request identity, never by an NPZ filename.

    Args:
        options: Validated input fields described by CaptionRoiAssetsGalleryAssetInputs.

    Returns:
        Mapping with fields: id, public_doc, source_image, source_sha256,
        roi_request, result_file, status, crop_sha256, native_image, native_sha256.
    """
    doc = options.doc
    route = options.route
    inputs = options.inputs
    fmap = options.fmap
    feature_paths = options.feature_paths
    seals = options.seals
    files = options.files
    key = route["roi_request"]
    request = inputs["roi_requests"][key]
    source = PROJECT_ROOT / inputs["images"][route["image_key"]]
    assert sha256_file(source) == request["image_key"] == route["image_key"]
    assert Path(doc["image_path"]).resolve() == source.resolve()
    files[relative(source)] = sha256_file(source)
    matches = [
        r for r in ROI_ROOTS if (r / "detections" / key / "result.json").exists()
    ]
    assert len(matches) == 1, f"Ambiguous or absent frozen crop: {key}"
    root = matches[0]
    path = sealed_file(
        root / "detections" / key / "result.json", root, seals[root], files
    )
    result = load(path)
    assert result["request_id"] == key
    assert result["source_sha256"] == route["image_key"]
    assert result["prompt"] == request["prompt"]
    asset = dict(
        id=doc["id"],
        public_doc=doc,
        source_image=relative(source),
        source_sha256=route["image_key"],
        roi_request=key,
        result_file=relative(path),
        status=result["decision"]["status"],
        crop_sha256=None,
        native_image=None,
        native_sha256=None,
    )
    if asset["status"] == "roi":
        for name in ["native", "zoom"]:
            crop = sealed_file(path.with_name(name + ".png"), root, seals[root], files)
            assert sha256_file(crop) == result["crop"][name + "_sha256"]
        selected = path.with_name("zoom.png")
        asset.update(
            crop_sha256=result["crop"]["zoom_sha256"],
            native_image=relative(path.with_name("native.png")),
            native_sha256=result["crop"]["native_sha256"],
        )
        assert fmap[key]["crop_sha256"] == asset["crop_sha256"]
    else:
        assert asset["status"] == "global_fallback"
        assert fmap[key]["crop_sha256"] is None
        assert fmap[key]["file"] == feature_paths["images"][route["image_key"]]
        selected = source
    asset.update(
        selected_image=relative(selected),
        selected_sha256=sha256_file(selected),
        selected_size=valid_image(selected),
    )
    return asset


def prepare() -> Any | dict:
    """Prepare the inputs and protocol for this development workflow.

    Returns:
        Mapping with fields: parent, whole_parent, cohorts.

    Notes:
        Writes derived artifacts through once.
    """
    if (ROOT / "crop_asset_protocol.json").exists():
        return guard()
    roots = [*ROI_ROOTS, PARENT]
    seals = {r: load(r / "completion_receipt.json") for r in roots}
    pinned = [
        Path(__file__),
        PROJECT_ROOT / "scripts/primary_subject_ranker.py",
        *[r / "completion_receipt.json" for r in roots],
        *sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")),
    ]
    files = {relative(p): sha256_file(p) for p in pinned}
    signature_path = PROJECT_ROOT / "scripts/primary_subject_ranker.py"
    assert seals[WHOLE_PARENT]["source_files"][relative(signature_path)] == sha256_file(
        signature_path
    )
    comparison = load(
        sealed_file(
            WHOLE_PARENT / "comparison_protocol.json",
            WHOLE_PARENT,
            seals[WHOLE_PARENT],
            files,
        )
    )
    for path in sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")):
        assert comparison["files"][relative(path)] == sha256_file(path)
    for name in ["inputs.json", "roi_feature_index.json", "feature_paths.json"]:
        sealed_file(WHOLE_PARENT / name, WHOLE_PARENT, seals[WHOLE_PARENT], files)
    manifest = ROI_ROOTS[0] / "detector_manifest.json"
    sealed_file(manifest, ROI_ROOTS[0], seals[ROI_ROOTS[0]], files)
    sealed_file(
        PARENT / "post_experiment_exclusion_snapshot.json", PARENT, seals[PARENT], files
    )
    inputs = load(WHOLE_PARENT / "inputs.json")
    fmap = load(WHOLE_PARENT / "roi_feature_index.json")
    feature_paths = load(WHOLE_PARENT / "feature_paths.json")
    cohorts = {}
    for cohort in COHORTS:
        public = {}
        for name in ["catalog.public.json", "queries.public.json"]:
            path = sealed_file(
                WHOLE_PARENT / cohort / name, WHOLE_PARENT, seals[WHOLE_PARENT], files
            )
            public[name] = load(path)
        docs = {d["id"]: d for d in public["catalog.public.json"]}
        assert len(docs) == len(public["catalog.public.json"]) == 296
        routes = inputs["cohorts"][cohort]["gallery"]
        assert list(docs) == [d["id"] for d in routes]
        gallery = [
            gallery_asset(
                CaptionRoiAssetsGalleryAssetInputs(
                    docs[d["id"]], d, inputs, fmap, feature_paths, seals, files
                )
            )
            for d in routes
        ]
        queries = []
        for query in public["queries.public.json"]:
            if not query["image_path"]:
                continue
            qid = query["id"]
            caption_path = sealed_file(
                WHOLE_PARENT / "query_captions" / cohort / (qid + ".json"),
                WHOLE_PARENT,
                seals[WHOLE_PARENT],
                files,
            )
            caption = load(caption_path)
            source = Path(query["image_path"])
            assert sha256_file(source) == caption["source_sha256"]
            files[relative(source)] = sha256_file(source)
            prompt = coarse_prompt(caption["value"])
            identity = dict(image_key=caption["source_sha256"], prompt=prompt)
            request_id = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode("utf-8")
            ).hexdigest()
            queries.append(
                dict(
                    id=qid,
                    public_query=query,
                    new_description=caption["text"],
                    caption_file=relative(caption_path),
                    caption_sha256=sha256_file(caption_path),
                    source_image=relative(source),
                    source_sha256=caption["source_sha256"],
                    signature=query_signature(caption["value"]),
                    prompt=prompt,
                    roi_request=request_id,
                    output_dir=relative(ROOT / "query_rois" / cohort / qid),
                )
            )
        assert len(queries) == 8
        cohorts[cohort] = dict(queries=queries, gallery=gallery)
    plan = dict(
        parent=relative(PARENT), whole_parent=relative(WHOLE_PARENT), cohorts=cohorts
    )
    once(ROOT / "crop_plan.json", plan)
    files[relative(ROOT / "crop_plan.json")] = sha256_file(ROOT / "crop_plan.json")
    once(
        ROOT / "crop_asset_protocol.json",
        dict(
            files=files,
            coarse_prompts=COARSE_PROMPTS,
            detector_policy=POLICY,
            detector_manifest=relative(manifest),
            query_requests=24,
            supported_query_requests=sum(
                q["prompt"] is not None for c in cohorts.values() for q in c["queries"]
            ),
            gallery_per_cohort=296,
            query_detection_repeated_even_when_old_prompt_matches=True,
            gallery_crops_reused_without_rebuilding=True,
            selected_crop="zoom.png; native.png retained as source evidence",
            full_official_model=True,
            qrels_or_quality_used=False,
            new_embedding_calls=0,
            new_FG_calls=0,
            formal_metrics=None,
        ),
    )
    return plan


def guard() -> Any:
    """Validate the frozen inputs before continuing the workflow.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    protocol = load(ROOT / "crop_asset_protocol.json")
    assert protocol["coarse_prompts"] == COARSE_PROMPTS
    assert protocol["detector_policy"] == POLICY
    verify_files(PROJECT_ROOT, protocol["files"])
    return load(ROOT / "crop_plan.json")


def gpu_available() -> bool:
    """Return gpu available for this workflow.

    Returns:
        Whether the condition checked by this operation holds.
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    memory = [
        float(line.strip()) for line in result.stdout.splitlines() if line.strip()
    ]
    if not memory or any(value >= IDLE_BELOW_MIB for value in memory):
        return False
    client = OllamaClient()
    try:
        return not client._request("/api/ps").get("models", [])
    finally:
        client.close()


def query_asset(query: dict, result_path: Path) -> dict:
    """Return query asset for this workflow.

    Args:
        query: Query record or representation used by this operation.
        result_path: Result path used by this operation.

    Returns:
        Mapping with fields: result_file, status, crop_sha256, native_image,
        native_sha256.
    """
    result = load(result_path)
    assert result["source_sha256"] == query["source_sha256"]
    assert result["request_id"] == query["roi_request"]
    assert result["prompt"] == query["prompt"]
    assert result["actual_model_call"] == (query["prompt"] is not None)
    assert result["caption_sha256"] == query["caption_sha256"]
    asset = dict(
        query,
        result_file=relative(result_path),
        status=result["decision"]["status"],
        crop_sha256=None,
        native_image=None,
        native_sha256=None,
    )
    if asset["status"] == "roi":
        for name in ["native", "zoom"]:
            assert (
                sha256_file(result_path.with_name(name + ".png"))
                == result["crop"][name + "_sha256"]
            )
        selected = result_path.with_name("zoom.png")
        asset.update(
            crop_sha256=result["crop"]["zoom_sha256"],
            native_image=relative(result_path.with_name("native.png")),
            native_sha256=result["crop"]["native_sha256"],
        )
    else:
        assert asset["status"] == "global_fallback"
        selected = PROJECT_ROOT / query["source_image"]
    asset.update(
        selected_image=relative(selected),
        selected_sha256=sha256_file(selected),
        selected_size=valid_image(selected),
    )
    return asset


def detect() -> bool:
    """Return detect for this workflow.

    Returns:
        Whether the condition checked by this operation holds.

    Notes:
        Writes derived artifacts through once.
    """
    plan = prepare()
    if (ROOT / "detection_receipt.json").exists():
        verify()
        return True
    pending = []
    for data in plan["cohorts"].values():
        for query in data["queries"]:
            target = PROJECT_ROOT / query["output_dir"] / "result.json"
            if target.exists():
                query_asset(query, target)
            else:
                pending.append(query)
    if any(q["prompt"] is not None for q in pending) and not gpu_available():
        return False
    detector, invocation = None, None
    try:
        for query in pending:
            source = PROJECT_ROOT / query["source_image"]
            assert sha256_file(source) == query["source_sha256"]
            target = PROJECT_ROOT / query["output_dir"] / "result.json"
            if query["prompt"] is None:
                result: dict = dict(
                    prompt=None,
                    detections=[],
                    seconds=0,
                    decision=dict(
                        status="global_fallback",
                        reason="Unresolved public coarse subject; retain whole query",
                        selected=None,
                    ),
                )
            else:
                if detector is None:
                    detector = SubjectDetector(ROI_ROOTS[0] / "detector_manifest.json")
                result = detector.detect(source, query["prompt"])
            result.update(
                request_id=query["roi_request"],
                source_sha256=query["source_sha256"],
                actual_model_call=query["prompt"] is not None,
                caption_sha256=query["caption_sha256"],
            )
            if result["decision"]["status"] == "roi":
                result["crop"] = save_subject_crop(
                    source, result["decision"], target.parent
                )
            once(target, result)
        if detector is not None:
            invocation = detector.diagnostics()
    finally:
        if detector is not None:
            detector.close()
    cohorts, completed = {}, []
    for cohort, data in plan["cohorts"].items():
        queries = []
        for q in data["queries"]:
            result_path = PROJECT_ROOT / q["output_dir"] / "result.json"
            queries.append(query_asset(q, result_path))
            result = load(result_path)
            completed.append(
                dict(
                    cohort=cohort,
                    query_id=q["id"],
                    actual_model_call=result["actual_model_call"],
                    seconds=result["seconds"],
                    result_file=relative(result_path),
                    result_sha256=sha256_file(result_path),
                )
            )
        cohorts[cohort] = dict(queries=queries, gallery=data["gallery"])
    files = dict(load(ROOT / "crop_asset_protocol.json")["files"])
    files[relative(ROOT / "crop_asset_protocol.json")] = sha256_file(
        ROOT / "crop_asset_protocol.json"
    )
    files.update(
        {
            relative(p): sha256_file(p)
            for p in sorted((ROOT / "query_rois").rglob("*"))
            if p.is_file()
        }
    )
    for data in cohorts.values():
        for asset in data["gallery"] + data["queries"]:
            for field in [
                "source_image",
                "selected_image",
                "native_image",
                "result_file",
            ]:
                if asset.get(field):
                    path = PROJECT_ROOT / asset[field]
                    files[relative(path)] = sha256_file(path)
    once(
        ROOT / "crop_inputs.json",
        dict(
            parent=relative(PARENT),
            whole_parent=relative(WHOLE_PARENT),
            cohorts=cohorts,
            files=files,
        ),
    )
    runtime = ROOT / "detector_runtime.json"
    if runtime.exists():
        assert load(runtime)["all_completed_calls"] == completed
    else:
        once(
            runtime,
            dict(
                invocation=invocation,
                all_completed_calls=completed,
                full_official_model=True,
            ),
        )
    receipt_files = [
        ROOT / "crop_inputs.json",
        runtime,
        *[p for p in sorted((ROOT / "query_rois").rglob("*")) if p.is_file()],
    ]
    queries = [q for c in cohorts.values() for q in c["queries"]]
    once(
        ROOT / "detection_receipt.json",
        dict(
            files={
                p.relative_to(ROOT).as_posix(): sha256_file(p) for p in receipt_files
            },
            query_requests=len(queries),
            actual_full_detector_calls=sum(c["actual_model_call"] for c in completed),
            accepted=sum(q["status"] == "roi" for q in queries),
            fallback=sum(q["status"] == "global_fallback" for q in queries),
            gallery_detector_calls=0,
            all_queries_and_gallery_retained=True,
            old_crops_changed=False,
            formal_metrics=None,
        ),
    )
    verify()
    return True


def verify() -> dict:
    """Verify the recorded artifacts and their consistency.

    Returns:
        Mapping with fields: passed, query_requests, actual_full_detector_calls,
        accepted, fallback, gallery_per_cohort, old_crops_rebuilt,
        qrels_or_quality_used, formal_metrics.

    Notes:
        Writes derived artifacts through once.
    """
    plan = guard()
    receipt = load(ROOT / "detection_receipt.json")
    verify_files(ROOT, receipt["files"])
    inputs = load(ROOT / "crop_inputs.json")
    verify_files(PROJECT_ROOT, inputs["files"])
    calls, accepted, total = 0, 0, 0
    for cohort in COHORTS:
        expected = plan["cohorts"][cohort]
        actual = inputs["cohorts"][cohort]
        assert actual["gallery"] == expected["gallery"]
        assert len(actual["gallery"]) == 296
        replay = []
        for query in expected["queries"]:
            path = PROJECT_ROOT / query["output_dir"] / "result.json"
            replay.append(query_asset(query, path))
            calls += load(path)["actual_model_call"]
        assert actual["queries"] == replay
        total += len(replay)
        accepted += sum(q["status"] == "roi" for q in replay)
    assert total == receipt["query_requests"] == 24
    assert calls == receipt["actual_full_detector_calls"]
    assert accepted == receipt["accepted"]
    assert total - accepted == receipt["fallback"]
    result = dict(
        passed=True,
        query_requests=total,
        actual_full_detector_calls=calls,
        accepted=accepted,
        fallback=total - accepted,
        gallery_per_cohort=296,
        old_crops_rebuilt=0,
        qrels_or_quality_used=False,
        formal_metrics=None,
    )
    once(ROOT / "crop_asset_verification.json", result)
    return result


def exposure() -> dict:
    """Return exposure for this workflow.

    Returns:
        Result mapping produced by the workflow.

    Raises:
        ValueError: Unexpected unregistered derived image.

    Notes:
        Writes derived artifacts through ensure_output_path(REGISTRY).parent.mkdir,
        once.
    """
    verify()
    rows = []
    for path in sorted((ROOT / "query_rois").rglob("*.png")):
        if path.name not in {"native.png", "zoom.png", "box.png"}:
            raise ValueError("Unexpected unregistered derived image")
        result = load(path.with_name("result.json"))
        rows.append(
            dict(
                id="caption_roi/" + path.relative_to(ROOT / "query_rois").as_posix(),
                source_image_sha256=result["source_sha256"],
                image_path=os.path.relpath(path, REGISTRY.parent).replace("\\", "/"),
                fingerprints=fingerprints(path),
                project_role="exposed_derived_development_ROI_not_acceptance",
                reusable_as_unseen=False,
            )
        )
    registry_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if REGISTRY.exists():
        assert REGISTRY.read_text(encoding="utf-8") == registry_text
    else:
        ensure_output_path(REGISTRY).parent.mkdir(parents=True, exist_ok=True)
        pending = REGISTRY.with_suffix(".jsonl.pending")
        if pending.exists():
            assert pending.read_text(encoding="utf-8") == registry_text
        else:
            with pending.open("x", encoding="utf-8", newline="") as handle:
                handle.write(registry_text)
        pending.replace(REGISTRY)
    prior_path = PARENT / "post_experiment_exclusion_snapshot.json"
    prior = load(prior_path)
    snapshot = dict(prior)
    snapshot.update(
        scope="All parent exposures and new caption-guided query crops/overlays; no image reusable as unseen",
        files=dict(prior["files"], **{relative(REGISTRY): sha256_file(REGISTRY)}),
        image_fingerprints=prior["image_fingerprints"]
        + [r["fingerprints"] for r in rows],
        new_caption_query_derived_images=len(rows),
        parent_snapshot_sha256=sha256_file(prior_path),
    )
    once(ROOT / "post_experiment_exclusion_snapshot.json", snapshot)
    result = dict(
        files={
            relative(REGISTRY): sha256_file(REGISTRY),
            relative(ROOT / "post_experiment_exclusion_snapshot.json"): sha256_file(
                ROOT / "post_experiment_exclusion_snapshot.json"
            ),
        },
        registered_derived_images=len(rows),
        native_zoom_and_box_registered=True,
        new_raw_images=0,
        new_metadata_records=0,
        original_ids_and_family_keys_preserved=True,
        pixels_and_dhash_recorded=True,
        formal_metrics=None,
    )
    once(ROOT / "exposure_receipt.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "detect", "verify", "exposure"])
    action = parser.parse_args().action
    value = globals()[action]()
    if action == "prepare":
        value = dict(
            prepared=True,
            image_queries=sum(len(c["queries"]) for c in value["cohorts"].values()),
        )
    print(json.dumps(value, ensure_ascii=False), flush=True)
