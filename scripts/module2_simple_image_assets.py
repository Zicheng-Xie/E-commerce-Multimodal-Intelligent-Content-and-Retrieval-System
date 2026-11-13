"""Uniform title-guided subject crops and persisted full-gallery image embeddings."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import run  # noqa: F401
from module2_development_benchmark import Evidence, load
from module2_shared_gallery_crop import DETECTOR
from module2_shared_gallery_retrieval import _checked_media, _vector_caches
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.safety import sha256_file, write_json
from multimodal_ecommerce.subject_roi import SubjectDetector, subject_prompt

EXTRA_SUBJECTS = (
    ("phone holder", ("手机支架", "手机架", "手机座", "手机导航架")),
    ("power bank", ("充电宝", "移动电源")),
    ("pants", ("裤",)),
    ("dress", ("连衣裙", "裙子", "襦裙")),
    ("chair", ("椅", "凳")),
    ("sofa", ("沙发",)),
    ("bed", ("床",)),
    ("laptop", ("笔记本电脑", "游戏本")),
    ("notebook", ("记事本", "本子", "日记本")),
    ("blanket", ("被子", "被芯", "包被", "抱毯", "棉被")),
    ("towel", ("浴巾", "毛巾")),
    ("lamp", ("台灯", "床头灯")),
    ("mouse", ("鼠标",)),
    ("keyboard", ("键盘",)),
    ("router", ("路由器",)),
    ("shelf", ("置物架", "收纳架")),
    ("handbag", ("包包", "水桶包", "斜挎包")),
)


def title_prompt(title: str) -> str:
    """Translate public title nouns, falling back to a generic object detector prompt.

    Args:
        title: Original published title without query or relevance information.

    Returns:
        Frozen English noun prompt used consistently for every gallery image.
    """
    known = subject_prompt(title)
    if known:
        return known
    for english, aliases in EXTRA_SUBJECTS:
        if any(alias in title for alias in aliases):
            return english + "."
    return "object."


def crop_original(source: Path, decision: dict, output: Path) -> str:
    """Save a geometrically valid padded subject rectangle from the exact original.

    Args:
        source: Original gallery image.
        decision: Detector result with EXIF-corrected size and selected rectangle.
        output: Derived PNG path inside the experiment directory.

    Returns:
        SHA256 of the saved crop, or the original when detection abstains.
    """
    selected = decision["decision"]["selected"]
    if selected is None:
        return sha256_file(source)
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    x1, y1, x2, y2 = selected["padded_box"]
    if list(image.size) != decision["source_size"] or not (
        0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height
    ):
        raise ValueError("Subject rectangle does not fit the original image")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.crop((x1, y1, x2, y2)).save(output)
    return sha256_file(output)


def detect_all(evidence: Evidence, output: Path, protocol: dict) -> None:
    """Check every gallery original with the same detector and frozen noun policy.

    Args:
        evidence: Development-only source and image verifier.
        output: Experiment output directory.
        protocol: Frozen product inputs and detector identity.
    """
    path = output / "detections.json"
    identity = sha256_file(output / "protocol.json")
    require_data_access([path], "development", evidence.registry)
    saved: dict = load(path) if path.exists() else dict(identity=identity, rows={})
    jobs = protocol["documents"]
    if saved["identity"] != identity or not set(saved["rows"]) <= set(jobs):
        raise ValueError("The detector cache has different source inputs")
    pending = [cid for cid in jobs if cid not in saved["rows"]]
    if pending:
        model = SubjectDetector(DETECTOR)
        try:
            for number, cid in enumerate(pending, 1):
                job = jobs[cid]
                source = _checked_media(evidence, job)
                result = model.detect(source, job["prompt"])
                crop = (
                    output
                    / "crops"
                    / (hashlib.sha256(cid.encode()).hexdigest() + ".png")
                )
                cropped = result["decision"]["selected"] is not None
                digest = crop_original(source, result, crop)
                saved["rows"][cid] = dict(
                    result=result,
                    image_path=str(crop if cropped else source),
                    image_sha256=digest,
                    crop_applied=cropped,
                )
                if number % 20 == 0 or number == len(pending):
                    temporary = path.with_suffix(".partial")
                    write_json(temporary, saved)
                    temporary.replace(path)
                    print(
                        f"Uniform subject checks: {len(saved['rows'])}/{len(jobs)}.",
                        flush=True,
                    )
            write_json(output / "detector_runtime.json", model.diagnostics())
        finally:
            model.close()
    evidence.digest(path)
    write_json(
        output / "crop_summary.json",
        dict(
            products=len(jobs),
            checked=len(saved["rows"]),
            applied=sum(row["crop_applied"] for row in saved["rows"].values()),
            statuses=dict(
                Counter(
                    row["result"]["decision"]["status"]
                    for row in saved["rows"].values()
                )
            ),
            generic_prompts=sum(job["prompt"] == "object." for job in jobs.values()),
            detections_sha256=sha256_file(path),
        ),
    )


def build_index(evidence: Evidence, output: Path, manifest: dict) -> None:
    """Encode all detected views and persist whole/crop/query matrices for reuse.

    Args:
        evidence: Development-only input verifier.
        output: Experiment output directory.
        manifest: Complete shared gallery and frozen image queries.
    """
    saved = evidence.json(output / "detections.json")
    catalog = {row["id"]: row for row in manifest["catalog"]}
    queries = {
        key: row
        for key, row in manifest["query_manifest"].items()
        if row["mode"] == "image"
    }
    if set(saved["rows"]) != set(catalog):
        raise ValueError("Check every original image before rebuilding the index")
    whole, query_vectors = _vector_caches(evidence, catalog, queries)
    crops: dict[str, np.ndarray] = {}
    model = QwenEmbedder()
    try:
        missing = sorted(set(catalog) - set(whole))
        for start in range(0, len(missing), 4):
            group = missing[start : start + 4]
            values = model.encode_images(
                [_checked_media(evidence, catalog[cid]) for cid in group]
            )
            whole.update(zip(group, values, strict=True))
        cropped = [cid for cid, row in saved["rows"].items() if row["crop_applied"]]
        for start in range(0, len(cropped), 4):
            group = cropped[start : start + 4]
            values = model.encode_images(
                [_checked_media(evidence, saved["rows"][cid]) for cid in group]
            )
            crops.update(zip(group, values, strict=True))
            if (start + len(group)) % 80 == 0 or start + len(group) == len(cropped):
                print(
                    f"Re-encoded subject crops: {start + len(group)}/{len(cropped)}.",
                    flush=True,
                )
        for key in sorted(set(queries) - set(query_vectors)):
            query_vectors[key] = model.encode(
                image=_checked_media(evidence, queries[key]), query=True
            )
        runtime = model.diagnostics()
    finally:
        model.close()
    ids, keys = sorted(catalog), sorted(queries)
    bank = dict(
        whole=np.stack([whole[cid] for cid in ids]),
        crop=np.stack([crops.get(cid, whole[cid]) for cid in ids]),
        queries=np.stack([query_vectors[key] for key in keys]),
    )
    if any(
        not np.isfinite(v).all()
        or v.shape[1:] != (2048,)
        or not np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-3)
        for v in bank.values()
    ):
        raise ValueError(
            "The rebuilt index must contain finite normalized model vectors"
        )
    folder = output / "index"
    folder.mkdir(exist_ok=True)
    np.savez_compressed(
        folder / "vectors.npz",
        whole=bank["whole"],
        crop=bank["crop"],
        queries=bank["queries"],
    )
    write_json(
        folder / "manifest.json",
        dict(
            products=ids,
            queries=keys,
            vectors_sha256=sha256_file(folder / "vectors.npz"),
            detections_sha256=sha256_file(output / "detections.json"),
            protocol_sha256=sha256_file(output / "protocol.json"),
            input_files=evidence.files,
        ),
    )
    write_json(output / "embedding_runtime.json", runtime)
