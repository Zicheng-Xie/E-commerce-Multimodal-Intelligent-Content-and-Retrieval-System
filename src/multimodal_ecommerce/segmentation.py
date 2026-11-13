"""Data contracts and lightweight evaluation utilities for module 3 segmentation.

The source annotation format is the FashionAI shop JSON format: one file per
image containing ``item*`` objects with polygons, boxes and audit metadata.
This module deliberately has no model dependency so conversion and regression
checks work on a CPU-only installation.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class SegmentationRecord:
    image_id: str
    split: str
    image_path: str
    annotation_path: str
    source: str
    objects: tuple[dict[str, Any], ...]
    image_sha256: str | None = None
    annotation_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
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
    values = [float(v) for v in polygon]
    return all(0 <= v <= width + 1 and 0 <= values[i + 1] <= height + 1
               for i, v in enumerate(values) if i % 2 == 0)


def _normalise_object(value: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    polygons = value.get("segmentation", [])
    valid = [p for p in polygons if _valid_polygon(p, width, height)]
    bbox = value.get("bounding_box")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return {"category_id": int(value.get("category_id", -1)), "category_name": str(value.get("category_name", "unknown")), "bbox": bbox, "polygons": valid, "occlusion": value.get("occlusion"), "viewpoint": value.get("viewpoint"), "scale": value.get("scale"), "zoom_in": value.get("zoom_in"), "landmarks": value.get("landmarks", []), "annotation_quality": "invalid_bbox"}
    x0, y0, x1, y1 = map(float, bbox)
    if not (0 <= x0 <= x1 <= width and 0 <= y0 <= y1 <= height):
        return {"category_id": int(value.get("category_id", -1)), "category_name": str(value.get("category_name", "unknown")), "bbox": [x0, y0, x1, y1], "polygons": valid, "occlusion": value.get("occlusion"), "viewpoint": value.get("viewpoint"), "scale": value.get("scale"), "zoom_in": value.get("zoom_in"), "landmarks": value.get("landmarks", []), "annotation_quality": "invalid_bbox"}
    return {
        "category_id": int(value["category_id"]),
        "category_name": str(value["category_name"]),
        "bbox": [x0, y0, x1 - x0, y1 - y0],
        "polygons": valid,
        "occlusion": value.get("occlusion"),
        "viewpoint": value.get("viewpoint"),
        "scale": value.get("scale"),
        "zoom_in": value.get("zoom_in"),
        "landmarks": value.get("landmarks", []),
        "annotation_quality": "valid" if valid else "invalid_polygon",
    }


def iter_records(root: Path, split: str, limit: int | None = None) -> Iterator[SegmentationRecord]:
    """Yield validated records without loading the full dataset into memory."""
    image_root = root / split / "image"
    ann_root = root / split / "annos"
    annotation_paths = {path.stem: path for path in ann_root.glob("*.json")}
    for number, image_path in enumerate(sorted(image_root.glob("*.jpg"))):
        if limit is not None and number >= limit:
            break
        annotation_path = annotation_paths.get(image_path.stem)
        with Image.open(image_path) as image:
            width, height = image.size
        raw = json.loads(annotation_path.read_text(encoding="utf-8")) if annotation_path else {}
        objects = tuple(
            _normalise_object(value, width, height)
            for key, value in raw.items()
            if key.startswith("item") and isinstance(value, dict)
        )
        yield SegmentationRecord(
            image_id=f"{split}/{image_path.stem}",
            split=split,
            image_path=str(image_path),
            annotation_path=str(annotation_path) if annotation_path else "",
            source=str(raw.get("source", "unknown")),
            objects=objects,
            image_sha256=sha256_file(image_path),
            annotation_sha256=sha256_file(annotation_path) if annotation_path else None,
        )


def scan_dataset(root: Path, splits: tuple[str, ...] = ("train", "validation", "test"), limit: int | None = None) -> dict[str, Any]:
    """Scan source files and return reproducible counts and quality checks."""
    report: dict[str, Any] = {"schema_version": 1, "root": str(root.resolve()), "splits": {}}
    for split in splits:
        records = list(iter_records(root, split, limit))
        ann_root = root / split / "annos"
        image_stems = {Path(row.image_path).stem for row in records}
        orphan_annotations = sorted(path.name for path in ann_root.glob("*.json") if path.stem not in image_stems)
        categories = Counter(obj["category_name"] for row in records for obj in row.objects)
        quality = Counter(obj["annotation_quality"] for row in records for obj in row.objects)
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
    return {"schema_version": 1, "records": count, "sha256": sha256_file(output), "path": str(output)}


def polygon_mask(size: tuple[int, int], polygons: list[list[float]]) -> Any:
    """Rasterise polygons to a PIL L mask for deterministic metric checks."""
    width, height = size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        draw.polygon([(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)], fill=1)
    return mask


def mask_metrics(prediction: Any, target: Any) -> dict[str, float]:
    """Compute IoU and Dice for two binary PIL masks."""
    pred = prediction.convert("1")
    truth = target.convert("1")
    p = list(pred.getdata())
    t = list(truth.getdata())
    intersection = sum(a and b for a, b in zip(p, t))
    pred_area = sum(p)
    target_area = sum(t)
    union = pred_area + target_area - intersection
    return {
        "iou": intersection / union if union else 1.0,
        "dice": 2 * intersection / (pred_area + target_area) if pred_area + target_area else 1.0,
    }
