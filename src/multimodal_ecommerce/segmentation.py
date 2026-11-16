"""Data contracts and lightweight evaluation utilities for module 3 segmentation.

The source annotation format is the FashionAI shop JSON format: one file per
image containing ``item*`` objects with polygons, boxes and audit metadata.
This module deliberately has no model dependency so conversion and regression
checks work on a CPU-only installation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class SegmentationRecord:
    image_id: str
    split: str
    image_path: str
    annotation_path: str
    source: str
    objects: tuple[dict[str, Any], ...]
    width: int = 0
    height: int = 0
    image_sha256: str | None = None
    annotation_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "image_id": self.image_id,
            "split": self.split,
            "image_path": self.image_path,
            "annotation_path": self.annotation_path,
            "source": self.source,
            "objects": list(self.objects),
            "image_sha256": self.image_sha256,
            "annotation_sha256": self.annotation_sha256,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_polygon(polygon: Any, width: int, height: int) -> bool:
    if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
        return False
    try:
        points = np.asarray(polygon, dtype=float).reshape(-1, 2)
    except (TypeError, ValueError):
        return False
    area = (
        abs(
            np.dot(points[:, 0], np.roll(points[:, 1], 1))
            - np.dot(points[:, 1], np.roll(points[:, 0], 1))
        )
        / 2
    )
    return bool(
        np.isfinite(points).all()
        and area > 0
        and (points >= 0).all()
        and (points[:, 0] <= width).all()
        and (points[:, 1] <= height).all()
    )


def _normalise_object(value: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    polygons = value.get("segmentation", [])
    bbox = value.get("bounding_box")
    issues = []
    normalized_bbox = None
    try:
        x0, y0, x1, y1 = map(float, bbox)
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or not (
            0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
        ):
            issues.append("invalid_bbox")
        normalized_bbox = [x0, y0, x1 - x0, y1 - y0]
    except (TypeError, ValueError):
        issues.append("invalid_bbox")
    if (
        not isinstance(polygons, list)
        or not polygons
        or not all(_valid_polygon(p, width, height) for p in polygons)
    ):
        issues.append("invalid_polygon")
    return {
        "category_id": int(value["category_id"]),
        "category_name": str(value["category_name"]),
        "bbox": normalized_bbox,
        "bbox_format": "xywh",
        "source_bbox": bbox,
        "polygons": polygons,
        "occlusion": value.get("occlusion"),
        "viewpoint": value.get("viewpoint"),
        "scale": value.get("scale"),
        "zoom_in": value.get("zoom_in"),
        "landmarks": value.get("landmarks", []),
        "annotation_quality": "invalid" if issues else "valid",
        "issues": issues,
        "eligible": not issues,
    }


def iter_records(
    root: Path, split: str, limit: int | None = None
) -> Iterator[SegmentationRecord]:
    """Yield validated records without loading the full dataset into memory."""
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown official split")
    image_root = root / split / "image"
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    ann_root = root / split / "annos"
    annotation_paths = {path.stem: path for path in ann_root.glob("*.json")}
    for number, image_path in enumerate(sorted(image_root.glob("*.jpg"))):
        if limit is not None and number >= limit:
            break
        annotation_path = annotation_paths.get(image_path.stem)
        if split != "test" and annotation_path is None:
            raise FileNotFoundError(f"Missing annotation: {image_path}")
        if split == "test" and annotation_path is not None:
            raise ValueError("Test must remain unlabeled")
        with Image.open(image_path) as image:
            width, height = image.size
        raw = (
            json.loads(annotation_path.read_text(encoding="utf-8"))
            if annotation_path
            else {}
        )
        objects = tuple(
            _normalise_object(value, width, height)
            for key, value in raw.items()
            if key.startswith("item") and isinstance(value, dict)
        )
        yield SegmentationRecord(
            image_id=f"{split}/{image_path.stem}",
            split=split,
            width=width,
            height=height,
            image_path=str(image_path),
            annotation_path=str(annotation_path) if annotation_path else "",
            source=str(raw.get("source", "unknown")),
            objects=objects,
            image_sha256=sha256_file(image_path),
            annotation_sha256=sha256_file(annotation_path) if annotation_path else None,
        )


def scan_dataset(
    root: Path,
    splits: tuple[str, ...] = ("train", "validation", "test"),
    limit: int | None = None,
) -> dict[str, Any]:
    """Scan source files and return reproducible counts and quality checks."""
    report: dict[str, Any] = {
        "schema_version": 1,
        "root": str(root.resolve()),
        "splits": {},
    }
    for split in splits:
        records = list(iter_records(root, split, limit))
        ann_root = root / split / "annos"
        image_stems = {Path(row.image_path).stem for row in records}
        orphan_annotations = sorted(
            path.name
            for path in ann_root.glob("*.json")
            if path.stem not in image_stems
        )
        categories = Counter(
            obj["category_name"] for row in records for obj in row.objects
        )
        quality = Counter(
            obj["annotation_quality"] for row in records for obj in row.objects
        )
        report["splits"][split] = {
            "records": len(records),
            "objects": sum(len(row.objects) for row in records),
            "categories": dict(sorted(categories.items())),
            "annotation_quality": dict(sorted(quality.items())),
            "annotated": split != "test",
            "orphan_annotations": orphan_annotations,
        }
    return report


def write_jsonl(records: Iterator[SegmentationRecord], output: Path) -> dict[str, Any]:
    """Write the internal contract and return a content manifest."""
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return {
        "schema_version": 1,
        "records": count,
        "sha256": sha256_file(output),
        "path": str(output),
    }


def polygon_mask(size: tuple[int, int], polygons: list[list[float]]) -> Any:
    """Rasterise polygons to a PIL L mask for deterministic metric checks."""
    width, height = size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        draw.polygon(
            [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)], fill=1
        )
    return mask


def mask_metrics(prediction: Any, target: Any) -> dict[str, float]:
    """Compute IoU and Dice for two binary PIL masks."""
    if prediction.size != target.size:
        raise ValueError("Mask sizes differ")
    p = np.asarray(prediction) > 0
    t = np.asarray(target) > 0
    intersection = int(np.logical_and(p, t).sum())
    union = int(np.logical_or(p, t).sum())
    total = int(p.sum() + t.sum())
    return {
        "iou": intersection / union if union else 1.0,
        "dice": 2 * intersection / total if total else 1.0,
    }


def bbox_iou(first: list[float], second: list[float]) -> float:
    """Intersection over union of xywh boxes."""
    x, y, w, h = first
    a, b, c, d = second
    inter = max(0, min(x + w, a + c) - max(x, a)) * max(
        0, min(y + h, b + d) - max(y, b)
    )
    union = w * h + c * d - inter
    return inter / union if union > 0 else 0.0
