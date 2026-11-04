"""Module-two regional retrieval with frozen query-to-region cosine attention.

Image pixels and geometry select crops, never evaluation labels, masks or boxes.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import time
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from multimodal_ecommerce.embedding import model_configuration
from multimodal_ecommerce.retrieval import unit_rows
from multimodal_ecommerce.runtime_device import relocate_assets
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)

STRATEGIES = ("global", "mean", "attention", "attention_fused")


def region_configuration() -> dict:
    """Load and validate the versioned region policy.

    Returns:
        Result mapping produced by the workflow.

    Raises:
        ValueError: Invalid foreground fractions; Invalid foreground threshold; This
            policy supports seven views at edge 768.
    """
    config = json.loads(
        (PROJECT_ROOT / "configs/retrieval_regions_v1.json").read_text(
            encoding="utf-8-sig"
        )
    )
    for key, value in config.items():
        if key != "version" and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"Invalid region configuration: {key}")
    for key in ("center_fraction", "tile_fraction", "attention_temperature"):
        if not 0 < config[key] <= 1:
            raise ValueError(f"Invalid region configuration: {key}")
    for key in (
        "global_weight",
        "text_weight",
        "roi_padding",
        "category_bonus",
        "attribute_bonus",
    ):
        if not 0 <= config[key] <= 1:
            raise ValueError(f"Invalid region configuration: {key}")
    if (
        not 0
        <= config["foreground_min_fraction"]
        < config["foreground_max_fraction"]
        <= 1
    ):
        raise ValueError("Invalid foreground fractions")
    if not 0 <= config["foreground_threshold"] <= 255:
        raise ValueError("Invalid foreground threshold")
    if config["max_views"] != 7 or config["zoom_max_edge"] != 768:
        raise ValueError("This policy supports seven views at edge 768")
    return config


def pixel_roi(box: list[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Validate normalized x1,y1,x2,y2 ROI in the EXIF-oriented image.

    Args:
        box: Box used by this operation.
        size: Size used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: ROI requires four finite normalized coordinates; ROI must be ordered
            and within [0,1]; ROI is smaller than two pixels.
    """
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("ROI requires four finite normalized coordinates")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("ROI must be ordered and within [0,1]")
    width, height = size
    result = (
        int(x1 * width),
        int(y1 * height),
        min(width, math.ceil(x2 * width)),
        min(height, math.ceil(y2 * height)),
    )
    if result[2] - result[0] < 2 or result[3] - result[1] < 2:
        raise ValueError("ROI is smaller than two pixels")
    return result


def propose_views(picture: Image.Image, config: dict) -> list[dict]:
    """Propose a global, contrast ROI, center and four overlapping tile views.

    Border contrast is only a foreground heuristic. Uniform or broad foregrounds
    fall back to full image and fixed crops; no segmentation accuracy is claimed.

    Args:
        picture: Picture used by this operation.
        config: Configuration values for the selected operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Image is too small.
    """
    width, height = picture.size
    if min(width, height) < 2:
        raise ValueError("Image is too small")
    candidates = [("global", (0, 0, width, height))]
    small = picture.copy()
    small.thumbnail((192, 192))
    pixels = np.asarray(small.convert("RGB"), dtype=float)
    border = np.concatenate(
        [pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]], axis=0
    )
    background = np.median(border, axis=0)
    contrast = np.sqrt(np.mean((pixels - background) ** 2, axis=2))
    foreground = contrast > config["foreground_threshold"]
    fraction = float(foreground.mean())
    if (
        config["foreground_min_fraction"]
        <= fraction
        <= config["foreground_max_fraction"]
    ):
        ys, xs = np.where(foreground)
        sx, sy = width / pixels.shape[1], height / pixels.shape[0]
        padding = config["roi_padding"]
        dx, dy = (xs.max() - xs.min() + 1) * sx, (ys.max() - ys.min() + 1) * sy
        box = (
            max(0, int(xs.min() * sx - dx * padding)),
            max(0, int(ys.min() * sy - dy * padding)),
            min(width, math.ceil((xs.max() + 1) * sx + dx * padding)),
            min(height, math.ceil((ys.max() + 1) * sy + dy * padding)),
        )
        if (box[2] - box[0]) * (box[3] - box[1]) < width * height * 0.95:
            candidates.append(("contrast_roi", box))
    for name, fraction, ax, ay in [
        ("center", config["center_fraction"], 0.5, 0.5),
        ("top_left", config["tile_fraction"], 0, 0),
        ("top_right", config["tile_fraction"], 1, 0),
        ("bottom_left", config["tile_fraction"], 0, 1),
        ("bottom_right", config["tile_fraction"], 1, 1),
    ]:
        cw, ch = max(2, round(width * fraction)), max(2, round(height * fraction))
        left, top = round((width - cw) * ax), round((height - ch) * ay)
        candidates.append((name, (left, top, left + cw, top + ch)))
    seen, views = set(), []
    for kind, box in candidates:
        if box not in seen:
            seen.add(box)
            views.append({"kind": kind, "box_xyxy": list(box)})
    return views[: config["max_views"]]


@dataclass(frozen=True)
class RegionalRetrievalEncodeViewsInputs:
    """Input values for encode views.

    Attributes:
        embedder: Embedder used by this operation.
        image: Image input used by the operation.
        output: Output used by this operation.
        config: Configuration values for the selected operation.
        query: Query record or representation used by this operation.
        roi: Roi used by this operation.
    """

    embedder: Any
    image: Path
    output: Path
    config: dict
    _: KW_ONLY
    query: bool = False
    roi: list[float] | None = None


def encode_views(
    options: RegionalRetrievalEncodeViewsInputs,
) -> tuple[np.ndarray, list[dict]]:
    """Persist real crops, enlarge local views, and retain original global encoding.

    Args:
        options: Validated input fields described by RegionalRetrievalEncodeViewsInputs.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    embedder = options.embedder
    image = options.image
    output = options.output
    config = options.config
    query = options.query
    roi = options.roi
    output = ensure_output_path(output)
    if output.exists():
        raise FileExistsError(f"Use a new views directory: {output}")
    with Image.open(image) as source:
        picture = ImageOps.exif_transpose(source).convert("RGB")
    origin = (0, 0)
    if roi is not None:
        box = pixel_roi(roi, picture.size)
        origin = (box[0], box[1])
        picture = picture.crop(box)
    views = propose_views(picture, config)
    output.mkdir(parents=True)
    vectors = []
    overlay = picture.copy()
    drawer = ImageDraw.Draw(overlay)
    for number, view in enumerate(views):
        crop = picture.crop(tuple(view["box_xyxy"]))
        if number or roi is not None:
            scale = config["zoom_max_edge"] / max(crop.size)
            crop = crop.resize(
                tuple(max(2, round(edge * scale)) for edge in crop.size),
                Image.Resampling.LANCZOS,
            )
        path = output / f"{number:02}_{view['kind']}.png"
        crop.save(path)
        encode_path = image if number == 0 and roi is None else path
        vectors.append(embedder.encode(image=encode_path, query=query))
        view.update(
            {
                "image_path": str(path),
                "sha256": sha256_file(path),
                "encoded_size": list(crop.size),
                "original_box_xyxy": [
                    view["box_xyxy"][0] + origin[0],
                    view["box_xyxy"][1] + origin[1],
                    view["box_xyxy"][2] + origin[0],
                    view["box_xyxy"][3] + origin[1],
                ],
            }
        )
        if number:
            drawer.rectangle(tuple(view["box_xyxy"]), outline="#ff3b30", width=3)
            drawer.text(tuple(view["box_xyxy"][:2]), str(number), fill="#ff3b30")
    overlay.thumbnail((768, 768))
    overlay.save(output / "regions.png")
    return unit_rows(np.stack(vectors)), views


def attribute_evidence(record: dict, attributes: dict[str, str]) -> dict:
    """Record exact source-field matches, contradictions, and missing information.

    Args:
        record: Record used by this operation.
        attributes: Attributes used by this operation.

    Returns:
        Mapping with fields: matched, conflicts, unknown.
    """
    matched: list[str] = []
    conflicts: list[str] = []
    unknown: list[str] = []
    for key, wanted in attributes.items():
        source = record.get("attributes", {}).get(key)
        if source is None:
            unknown.append(key)
            continue
        values = source if isinstance(source, list) else [source]
        canonical = {str(value).strip().casefold() for value in values}
        (matched if wanted.strip().casefold() in canonical else conflicts).append(key)
    return {"matched": matched, "conflicts": conflicts, "unknown": unknown}


@dataclass(frozen=True)
class RegionalSearchRequest:
    """Input values for search.

    Attributes:
        query: Query record or representation used by this operation.
        top_k: Top k used by this operation.
        strategy: Strategy used by this operation.
        text_vector: Text vector used by this operation.
        text_bank: Text bank used by this operation.
        category: Category used by this operation.
        attributes: Attributes used by this operation.
        exclude_ids: Exclude ids used by this operation.
    """

    query: np.ndarray
    top_k: int = 10
    _: KW_ONLY
    strategy: str = "attention"
    text_vector: np.ndarray | None = None
    text_bank: np.ndarray | None = None
    category: str | None = None
    attributes: dict[str, str] | None = None
    exclude_ids: set[str] | None = None


class RegionalIndex:
    """Variable-count frozen regional features and explainable retrieval scores."""

    def __init__(self, records: list[dict], vectors: np.ndarray, config: dict) -> None:
        """Initialize the component and validate its configured dependencies.

        Args:
            records: Records used by this operation.
            vectors: Vectors used by this operation.
            config: Configuration values for the selected operation.

        Raises:
            ValueError: Region bank cardinality mismatch; Duplicate region record IDs;
                Invalid view order or count; Nonfinite vectors or nonzero padding.
        """
        self.records, self.config = records, config
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim != 3 or len(values) != len(records) or not records:
            raise ValueError("Region bank cardinality mismatch")
        if len({r["id"] for r in records}) != len(records):
            raise ValueError("Duplicate region record IDs")
        self.mask = np.zeros(values.shape[:2], dtype=bool)
        for i, row in enumerate(records):
            count = len(row["views"])
            if not 1 <= count <= values.shape[1] or row["views"][0]["kind"] != "global":
                raise ValueError("Invalid view order or count")
            self.mask[i, :count] = True
        if not np.isfinite(values).all() or np.any(values[~self.mask] != 0):
            raise ValueError("Nonfinite vectors or nonzero padding")
        self.vectors = np.zeros_like(values)
        self.vectors[self.mask] = unit_rows(values[self.mask])

    def search(self, options: RegionalSearchRequest) -> list[dict]:
        """Apply a=softmax(cos(Q,K)/tau), Z=sum(a*V), score=Q dot Z.

        K=V are frozen candidate-view embeddings. Retain the global branch.
        Metadata preferences come ONLY from explicit user inputs, never qrels.

        Args:
            options: Validated input fields described by RegionalSearchRequest.

        Returns:
            Values produced by this operation, in the stated return order.

        Raises:
            ValueError: Invalid search strategy or top_k; Query dimension mismatch;
                Attributes must map names to nonempty strings; Text fusion requires a
                text bank; Text bank dimension mismatch.
        """
        query = options.query
        top_k = options.top_k
        strategy = options.strategy
        text_vector = options.text_vector
        text_bank = options.text_bank
        category = options.category
        attributes = options.attributes
        exclude_ids = options.exclude_ids
        if strategy not in STRATEGIES or top_k < 1:
            raise ValueError("Invalid search strategy or top_k")
        queries = unit_rows(np.atleast_2d(query))
        if queries.shape[1] != self.vectors.shape[2]:
            raise ValueError("Query dimension mismatch")
        constraints = attributes or {}
        if not isinstance(constraints, dict) or any(
            not isinstance(k, str)
            or not isinstance(v, str)
            or not k.strip()
            or not v.strip()
            for k, v in constraints.items()
        ):
            raise ValueError("Attributes must map names to nonempty strings")
        sims = np.einsum("qd,nvd->nqv", queries, self.vectors)
        mask = self.mask[:, None, :]
        if strategy == "mean":
            weights = np.broadcast_to(mask, sims.shape).astype(float)
            weights /= weights.sum(axis=2, keepdims=True)
        else:
            logits = np.where(
                mask, sims / self.config["attention_temperature"], -np.inf
            )
            weights = np.exp(logits - np.max(logits, axis=2, keepdims=True))
            weights /= weights.sum(axis=2, keepdims=True)
        per_query = (weights * sims).sum(axis=2)
        global_scores = sims[:, 0, 0]
        regional_scores = per_query[:, 0]
        gw = self.config["global_weight"]
        if len(queries) > 1:
            regional_scores = gw * per_query[:, 0] + (1 - gw) * per_query[:, 1:].mean(
                axis=1
            )
        visual = gw * global_scores + (1 - gw) * regional_scores
        if strategy == "global":
            visual = global_scores.copy()
            weights = np.zeros_like(sims)
            weights[:, :, 0] = 1
        scores = visual.copy()
        text_scores = np.zeros(len(self.records))
        if strategy == "attention_fused" and text_vector is not None:
            if text_bank is None:
                raise ValueError("Text fusion requires a text bank")
            bank = unit_rows(text_bank)
            if bank.shape != (len(self.records), queries.shape[1]):
                raise ValueError("Text bank dimension mismatch")
            text_scores = bank @ unit_rows(np.asarray(text_vector).reshape(1, -1))[0]
            tw = self.config["text_weight"]
            scores = (1 - tw) * visual + tw * text_scores
        evidence = [attribute_evidence(row, constraints) for row in self.records]
        adjustments = np.zeros(len(self.records))
        if strategy == "attention_fused":
            for i, row in enumerate(self.records):
                if category is not None:
                    adjustments[i] += self.config["category_bonus"] * (
                        1 if row["category"] == category else -1
                    )
                if constraints:
                    item = evidence[i]
                    adjustments[i] += (
                        self.config["attribute_bonus"]
                        * (len(item["matched"]) - len(item["conflicts"]))
                        / len(constraints)
                    )
            scores += adjustments
        candidates = [
            i
            for i, row in enumerate(self.records)
            if row["id"] not in (exclude_ids or set())
        ]
        ranked = sorted(
            candidates, key=lambda i: (-float(scores[i]), self.records[i]["id"])
        )[:top_k]
        results = []
        for rank, i in enumerate(ranked, 1):
            focus = int(np.argmax(weights[i, 0]))
            results.append(
                {
                    "rank": rank,
                    **self.records[i],
                    "score": float(scores[i]),
                    "global_score": float(global_scores[i]),
                    "regional_score": float(regional_scores[i]),
                    "text_score": float(text_scores[i]),
                    "metadata_adjustment": float(adjustments[i]),
                    "attribute_evidence": evidence[i],
                    "region_weights": weights[
                        i, :, : len(self.records[i]["views"])
                    ].tolist(),
                    "focus_view": self.records[i]["views"][focus],
                    "strategy": strategy,
                }
            )
        return results

    @classmethod
    def load(cls, path: Path) -> tuple["RegionalIndex", np.ndarray | None, dict]:
        """Verify region assets and configured preprocessing before index use.

        Args:
            path: Path of the input or output artifact used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.

        Raises:
            ValueError: Model or region policy changed; build a new index; Incomplete
                regional manifest; Regional index checksum mismatch; Crop is missing
                from manifest; Persisted text dimensions disagree.
        """
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["model"] != model_configuration()
            or manifest["regions"] != region_configuration()
        ):
            raise ValueError("Model or region policy changed; build a new index")
        if not {"vectors.npz", "records.json"} <= set(manifest["files"]):
            raise ValueError("Incomplete regional manifest")
        for name, checksum in manifest["files"].items():
            file = (path / name).resolve()
            if not file.is_relative_to(path.resolve()) or sha256_file(file) != checksum:
                raise ValueError("Regional index checksum mismatch")
        records = relocate_assets(
            json.loads((path / "records.json").read_text(encoding="utf-8"))
        )
        for row in records:
            for view in row["views"]:
                crop = Path(view["image_path"]).resolve()
                relative = crop.relative_to(path.resolve()).as_posix()
                if (
                    relative not in manifest["files"]
                    or manifest["files"][relative] != view["sha256"]
                ):
                    raise ValueError("Crop is missing from manifest")
        with np.load(path / "vectors.npz", allow_pickle=False) as archive:
            index = cls(records, archive["regions"], manifest["regions"])
            texts = archive["texts"].copy() if "texts" in archive else None
            if texts is not None:
                texts = unit_rows(texts)
                if texts.shape != (len(records), index.vectors.shape[2]):
                    raise ValueError("Persisted text dimensions disagree")
        return index, texts, manifest


def build_regional_index(embedder: Any, dataset: Path, output: Path) -> dict:
    """Create an immutable region bank from eligible image and original-title data.

    Args:
        embedder: Embedder used by this operation.
        dataset: Dataset used by this operation.
        output: Output used by this operation.

    Returns:
        Mapping with fields: model, regions, dataset_sha256, count, view_count,
        build_seconds, attention, roi_source, implementation_sha256, files, diagnostics.

    Raises:
        FileExistsError: Use a new regional index directory.
        ValueError: Need nonempty unique eligible records; Changed or escaped source
            image; Mixed missing titles; split diagnostic and product banks.
    """
    output = ensure_output_path(output)
    if output.exists():
        raise FileExistsError("Use a new regional index directory")
    rows = [
        json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()
    ]
    eligible = [
        row for row in rows if row.get("evaluation_eligible") and row.get("image_path")
    ]
    if not eligible or len({r["id"] for r in eligible}) != len(eligible):
        raise ValueError("Need nonempty unique eligible records")
    for row in eligible:
        image = (dataset.parent / row["image_path"]).resolve()
        if (
            not image.is_relative_to(dataset.parent.resolve())
            or sha256_file(image) != row["image_sha256"]
        ):
            raise ValueError("Changed or escaped source image")
    text_mode = all(bool(row.get("text")) for row in eligible)
    if any(bool(row.get("text")) for row in eligible) and not text_mode:
        raise ValueError("Mixed missing titles; split diagnostic and product banks")
    config = region_configuration()
    output.mkdir(parents=True)
    started = time.perf_counter()
    records, encoded, texts = [], [], []
    for row in eligible:
        image = (dataset.parent / row["image_path"]).resolve()
        key = hashlib.sha256(row["id"].encode()).hexdigest()[:24]
        vectors, views = encode_views(
            RegionalRetrievalEncodeViewsInputs(
                embedder, image, output / "views" / key, config
            )
        )
        encoded.append(vectors)
        if text_mode:
            texts.append(embedder.encode(text=row["text"]))
        records.append(
            {
                "id": row["id"],
                "title": row.get("text") or row["id"],
                "category": row.get("category", ""),
                "attributes": row.get("attributes", {}),
                "image_path": str(image),
                "image_sha256": row["image_sha256"],
                "views": views,
            }
        )
        if len(records) % 20 == 0:
            print(f"Regional index {len(records)}/{len(eligible)}", flush=True)
    bank = np.zeros(
        (len(records), config["max_views"], encoded[0].shape[1]), dtype=np.float32
    )
    for i, vector in enumerate(encoded):
        bank[i, : len(vector)] = vector
    RegionalIndex(records, bank, config)
    if text_mode:
        np.savez(output / "vectors.npz", regions=bank, texts=np.stack(texts))
    else:
        np.savez(output / "vectors.npz", regions=bank)
    write_json(output / "records.json", records)
    manifest = {
        "model": embedder.config,
        "regions": config,
        "dataset_sha256": sha256_file(dataset),
        "count": len(records),
        "view_count": sum(len(row["views"]) for row in records),
        "build_seconds": time.perf_counter() - started,
        "attention": "Frozen cosine softmax cross-attention; Q=query view, K=V=gallery views; no trained projections",
        "roi_source": "Image border contrast and fixed crops; no source boxes or masks",
        "implementation_sha256": sha256_file(Path(__file__)),
        "files": {
            file.relative_to(output).as_posix(): sha256_file(file)
            for file in output.rglob("*")
            if file.is_file()
        },
        "diagnostics": embedder.diagnostics(),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def write_regional_gallery(path: Path, sections: list[dict]) -> None:
    """Show original images, focused crops, score parts, and view attention weights.

    Args:
        path: Path of the input or output artifact used by this operation.
        sections: Sections used by this operation.
    """
    path = ensure_output_path(path)
    esc = html.escape

    def image(source: str, caption: str) -> str:
        """Return image for this workflow.

        Args:
            source: Source used by this operation.
            caption: Caption used by this operation.

        Returns:
            Text produced by this operation.
        """
        relative = os.path.relpath(source, path.parent).replace("\\", "/")
        return f'<figure><img loading="lazy" src="{esc(relative, quote=True)}"><figcaption>{esc(caption)}</figcaption></figure>'

    blocks = []
    for section in sections:
        cards = []
        query_image = (
            image(section["query_image"], "查询图片")
            if section.get("query_image")
            else ""
        )
        for row in section["results"]:
            focus = row["focus_view"]
            weights = row["region_weights"][0]
            attention = ", ".join(
                f"{v['kind']} {w:.2f}" for v, w in zip(row["views"], weights)
            )
            cards.append(
                "<article>"
                + image(row["image_path"], "商品主图")
                + image(focus["image_path"], "主要匹配视图 " + focus["kind"])
                + f'<h3>#{row["rank"]} {esc(row["title"])}</h3><p>{esc(row["category"])}</p>'
                + f'<p>匹配分 {row["score"]:.4f} · 整图 {row["global_score"]:.4f} · 区域 {row["regional_score"]:.4f}</p>'
                + f'<details><summary>匹配依据</summary><p>文本 {row["text_score"]:.4f} · 条件加减分 {row["metadata_adjustment"]:.4f}</p>'
                + f'<p>{esc(attention)}</p><pre>{esc(json.dumps(row["attribute_evidence"], ensure_ascii=False))}</pre></details></article>'
            )
        blocks.append(
            f'<section><h2>{esc(section["query"])}</h2><p>策略：{esc(section["strategy"])}</p>{query_image}<div class="cards">'
            + "".join(cards)
            + "</div></section>"
        )
    content = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>模块二区域注意力检索</title>
<style>body{font-family:system-ui,sans-serif;max-width:1400px;margin:24px auto;padding:16px;background:#f5f7fb;color:#17304d}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:16px}article{background:white;border:1px solid #d9e2ef;padding:16px;border-radius:12px}figure{margin:0 0 12px}img{width:100%;height:210px;object-fit:contain}figcaption,details{color:#526477;font-size:14px}section{margin-top:32px}pre{white-space:pre-wrap;overflow-wrap:anywhere}h3,p{line-height:1.6;overflow-wrap:anywhere}section>figure{max-width:260px}</style>
<h1>模块二 多模态商品检索</h1><p>整图、ROI和局部视图共同参与匹配。展开卡片查看区域权重与属性依据。匹配分与区域权重均不是正确率或定位置信度。</p>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "".join(blocks) + "</html>", encoding="utf-8")
