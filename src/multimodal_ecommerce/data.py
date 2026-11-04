"""Read protected datasets and create small, traceable working copies."""

from __future__ import annotations

import csv
import heapq
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import ijson
from PIL import Image, ImageDraw

from multimodal_ecommerce import safety

PRODUCT_RULES = {
    "3C数码": (
        "手机",
        "平板",
        "笔记本",
        "电脑",
        "键盘",
        "鼠标",
        "耳机",
        "相机",
        "摄像",
        "智能手表",
        "存储",
        "充电器",
        "移动电源",
        "数码",
        "显示器",
        "路由器",
        "U盘",
        "硬盘",
    ),
    "服饰穿搭": (
        "T恤",
        "衬衫",
        "裙",
        "裤",
        "外套",
        "羽绒",
        "毛衣",
        "针织",
        "卫衣",
        "西装",
        "夹克",
        "风衣",
        "马甲",
        "服饰",
        "鞋",
        "靴",
        "帽",
        "围巾",
        "婚纱",
        "汉服",
        "舞蹈服",
        "斜挎包",
        "双肩包",
    ),
    "家居日用": (
        "家居",
        "家具",
        "餐桌",
        "书桌",
        "茶几",
        "椅",
        "沙发",
        "床",
        "柜",
        "收纳",
        "台灯",
        "吊灯",
        "壁灯",
        "窗帘",
        "地毯",
        "枕",
        "被",
        "餐具",
        "杯",
        "碗",
        "果盘",
        "餐盘",
        "毛巾",
        "浴巾",
        "置物",
        "书架",
        "拖把",
        "衣物用刷",
    ),
}
PRODUCT_SCAN_LIMIT = 50_000


def parse_pv(value: str) -> dict[str, list[str]]:
    """Preserve repeated product property keys and values in source order.

    Args:
        value: Input value to validate or transform.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    result: dict[str, list[str]] = defaultdict(list)
    for item in value.split("#;#"):
        if "#:#" in item:
            key, content = item.split("#:#", 1)
            if key.strip():
                result[key.strip()].append(content.strip())
    return dict(result)


def classify_product(record: dict[str, Any]) -> tuple[str, dict[str, str]] | None:
    """Match source label; use title only when the label itself is absent.

    Args:
        record: Record used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    fields = ("label",) if record.get("label") else ("title",)
    for field in fields:
        value = str(record.get(field) or "").casefold()
        for category, keywords in PRODUCT_RULES.items():
            for keyword in keywords:
                if keyword.casefold() in value:
                    return category, {"field": field, "keyword": keyword}
    return None


def _no_links(path: Path) -> None:
    for part in (path, *path.parents):
        if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
            raise ValueError(f"Symlink/junction paths are not allowed: {part}")


def _source(root: Path, candidate: Path) -> Path:
    _no_links(candidate)
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Source path escapes the protected root: {candidate}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


class _Builder:
    def __init__(self, source_root: Path, output_root: Path) -> None:
        """Initialize the component and validate its configured dependencies.

        Args:
            source_root: Source root used by this operation.
            output_root: Output root used by this operation.

        Raises:
            ValueError: Output must not be an ancestor of the source root.

        Notes:
            Writes derived artifacts through self.output.mkdir.
        """
        _no_links(source_root)
        _no_links(output_root)
        self.source = source_root.resolve()
        self.output = safety.ensure_output_path(output_root, self.source)
        if self.source.is_relative_to(self.output):
            raise ValueError("Output must not be an ancestor of the source root")
        self.output.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self.files: list[dict[str, Any]] = []

    def target(self, relative: str) -> Path:
        """Return target for this workflow.

        Args:
            relative: Relative used by this operation.

        Returns:
            Resolved artifact path.

        Notes:
            Writes derived artifacts through result.parent.mkdir.
        """
        candidate = self.output / relative
        _no_links(candidate)
        result = safety.ensure_output_path(candidate, self.source)
        if not result.is_relative_to(self.output):
            raise ValueError(f"Output path escapes the output directory: {relative}")
        result.parent.mkdir(parents=True, exist_ok=True)
        return result

    def copy(self, source: Path, relative: str) -> str:
        """Return copy for this workflow.

        Args:
            source: Source used by this operation.
            relative: Relative used by this operation.

        Returns:
            Text produced by this operation.
        """
        source = _source(self.source, source)
        target = self.target(relative)
        digest = safety.sha256_file(source)
        shutil.copy2(source, target)
        if safety.sha256_file(target) != digest:
            raise OSError(f"Copy checksum mismatch: {source}")
        self.files.append(
            {
                "source_path": str(source),
                "path": relative,
                "sha256": digest,
                "size_bytes": target.stat().st_size,
            }
        )
        return relative

    def image(self, source: Path, relative: str) -> tuple[str, int, int]:
        """Return image for this workflow.

        Args:
            source: Source used by this operation.
            relative: Relative used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        copied = self.copy(source, relative)
        with Image.open(self.output / copied) as image:
            width, height = image.size
            image.verify()
        return copied, width, height

    def write(self, relative: str, data: Any) -> None:
        """Apply the write workflow step.

        Args:
            relative: Relative used by this operation.
            data: Data used by this operation.
        """
        safety.write_json(self.target(relative), data, self.source)


def _products(builder: _Builder, count: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for split in ("train", "test"):
        filename = f"product1m_product5m_{split}_id_label.json"
        source = _source(builder.source, builder.source / "多模态电商数据" / filename)
        counts = dict.fromkeys(PRODUCT_RULES, 0)
        scanned = 0
        if count:
            with source.open("rb") as handle:
                for source_id, product in ijson.kvitems(handle, ""):
                    scanned += 1
                    matched = classify_product(product)
                    if matched and counts[matched[0]] < count:
                        category, rule = matched
                        counts[category] += 1
                        builder.records.append(
                            {
                                "id": f"product/{split}/{source_id}",
                                "dataset": "product",
                                "split": split,
                                "source_id": source_id,
                                "source_path": str(source),
                                "image_path": None,
                                "media_status": "pending_download",
                                "image_url": product.get("url"),
                                "video_url": product.get("video"),
                                "text": product.get("title", ""),
                                "label": product.get("label", ""),
                                "category": category,
                                "category_rule": rule,
                                "attributes": parse_pv(product.get("pv") or ""),
                                "pv_raw": product.get("pv", ""),
                                "provenance": {"source_split": split},
                            }
                        )
                    if all(value >= count for value in counts.values()):
                        break
                    if scanned >= PRODUCT_SCAN_LIMIT:
                        break
        results[split] = {
            "scanned": scanned,
            "selected_by_category": counts,
            "requested_per_category": count,
            "shortfalls": {key: count - n for key, n in counts.items() if n < count},
            "scan_limit": PRODUCT_SCAN_LIMIT,
        }
    return results


def _read_attribute_values(source: Path) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    current: str | None = None
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        match = re.fullmatch(r"#+\s+([a-z_]+_labels)\s*", line.strip())
        if match:
            current = match.group(1)
            values[current] = []
        elif current and line.strip().startswith("- "):
            values[current].append(line.strip()[2:].strip())
    if not values or any(not choices for choices in values.values()):
        raise ValueError(f"Missing attribute definitions: {source}")
    return values


def _fashion(builder: _Builder, count: int) -> dict[str, Any]:
    root = builder.source / "服饰细分属性" / "round1_fashionAI_attributes_test_a"
    readme = _source(builder.source, root / "README.md")
    values = _read_attribute_values(readme)
    source = _source(
        builder.source, root / "Tests" / "round1_fashionAI_attributes_answer_a.csv"
    )
    counts = dict.fromkeys(values, 0)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        for relative, key, code in csv.reader(handle):
            if key not in values:
                raise ValueError(f"Unknown attribute: {key}")
            if counts[key] >= count:
                continue
            if len(code) != len(values[key]) or set(code) - set("nym"):
                raise ValueError(f"Invalid attribute code: {key} {code}")
            if code.count("y") != 1:
                raise ValueError(f"Expected one positive attribute: {key} {code}")
            image_source = _source(builder.source, root / relative)
            image, width, height = builder.image(
                image_source, f"images/fashionai/test_a/{key}/{image_source.name}"
            )
            counts[key] += 1
            label = values[key][code.index("y")]
            builder.records.append(
                {
                    "id": f"fashionai/test_a/{key}/{image_source.stem}",
                    "dataset": "fashionai",
                    "split": "test_a",
                    "source_id": relative,
                    "source_path": str(source),
                    "source_image_path": str(image_source),
                    "image_path": image,
                    "width": width,
                    "height": height,
                    "text": None,
                    "attribute_key": key,
                    "label_code": code,
                    "label_values": values[key],
                    "category": label,
                    "attributes": {key: label},
                    "provenance": {
                        "source_split": "test_a",
                        "usage": "held_out_evaluation",
                        "m_semantics": "unverified_preserved_verbatim",
                        "text_status": "no_natural_language_caption_in_source",
                    },
                }
            )
            if all(n >= count for n in counts.values()):
                break
    builder.write("annotations/fashionai_attribute_values.json", values)
    return {
        "split": "test_a",
        "selected_by_attribute": counts,
        "requested_per_attribute": count,
        "shortfalls": {key: count - n for key, n in counts.items() if n < count},
        "note": "Official test_a is retained as held-out data; m is preserved.",
    }


def _instances(
    annotation: dict[str, Any], width: int, height: int
) -> list[dict[str, Any]]:
    result = []
    for key, item in sorted(annotation.items()):
        if not key.startswith("item"):
            continue
        x1, y1, x2, y2 = item["bounding_box"]
        if x2 < x1 or y2 < y1:
            raise ValueError(f"Invalid bounding box for {key}")
        polygons = item.get("segmentation", [])
        mask = Image.new("1", (width, height))
        drawer = ImageDraw.Draw(mask)
        for polygon in polygons:
            if len(polygon) < 6 or len(polygon) % 2:
                raise ValueError(f"Invalid segmentation polygon for {key}")
            drawer.polygon(list(zip(polygon[::2], polygon[1::2])), fill=1)
        area = mask.histogram()[1]
        result.append(
            {
                **item,
                "source_instance_id": key,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": area,
                "area_method": "pixel_union_of_segmentation_polygons",
                "segmentation": polygons,
            }
        )
    return result


def _deepfashion(builder: _Builder, count: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for split in ("train", "validation"):
        root = builder.source / "服饰分割" / split
        _no_links(root / "image")
        selected = heapq.nsmallest(count, (root / "image").glob("*.jpg"))
        coco: dict[str, Any] = {"images": [], "annotations": [], "categories": []}
        categories: dict[int, str] = {}
        for index, image_source in enumerate(selected, 1):
            source = _source(
                builder.source, root / "annos" / f"{image_source.stem}.json"
            )
            annotation = json.loads(source.read_text(encoding="utf-8"))
            image, width, height = builder.image(
                image_source, f"images/deepfashion/{split}/{image_source.name}"
            )
            annotation_path = builder.copy(
                source, f"annotations/source/deepfashion/{split}/{source.name}"
            )
            instances = _instances(annotation, width, height)
            record_id = f"deepfashion/{split}/{image_source.stem}"
            builder.records.append(
                {
                    "id": record_id,
                    "dataset": "deepfashion",
                    "split": split,
                    "source_id": image_source.stem,
                    "source_path": str(source),
                    "source_image_path": str(image_source),
                    "image_path": image,
                    "annotation_path": annotation_path,
                    "width": width,
                    "height": height,
                    "text": None,
                    "category": [i["category_name"] for i in instances],
                    "attributes": {},
                    "instances": instances,
                    "pair_id": annotation.get("pair_id"),
                    "image_source": annotation.get("source"),
                    "provenance": {
                        "source_split": split,
                        "text_status": "no_natural_language_caption_in_source",
                    },
                }
            )
            coco["images"].append(
                {
                    "id": index,
                    "file_name": image,
                    "width": width,
                    "height": height,
                    "source_record_id": record_id,
                }
            )
            for item in instances:
                category_id = item["category_id"]
                categories[category_id] = item["category_name"]
                coco["annotations"].append(
                    {
                        "id": len(coco["annotations"]) + 1,
                        "image_id": index,
                        "category_id": category_id,
                        "bbox": item["bbox"],
                        "segmentation": item["segmentation"],
                        "area": item["area"],
                        "area_method": item["area_method"],
                        "iscrowd": 0,
                        "source_instance_id": item["source_instance_id"],
                    }
                )
        coco["categories"] = [
            {"id": key, "name": value, "supercategory": "clothes"}
            for key, value in sorted(categories.items())
        ]
        builder.write(f"annotations/deepfashion_{split}.json", coco)
        results[split] = {
            "selected": len(selected),
            "requested": count,
            "shortfall": max(0, count - len(selected)),
            "instances": len(coco["annotations"]),
        }
    return results


def prepare_samples(
    source_root: Path,
    output_root: Path,
    *,
    product_per_category: int = 8,
    fashion_per_attribute: int = 2,
    deepfashion_per_split: int = 4,
) -> dict[str, Any]:
    """Copy traceable samples without writing to protected source datasets.

    Official splits remain intact. Product records have remote image URLs only;
    their media status stays pending_download. Sampling is deterministic in
    source order and is intended for pipeline checks, not unbiased evaluation.

    Args:
        source_root: Source root used by this operation.
        output_root: Output root used by this operation.
        product_per_category: Product per category used by this operation.
        fashion_per_attribute: Fashion per attribute used by this operation.
        deepfashion_per_split: Deepfashion per split used by this operation.

    Returns:
        Mapping with fields: schema_version, source_root, output_root, product,
        fashionai, deepfashion, sampling_method, category_rules.

    Raises:
        ValueError: Sample counts must be non-negative integers.
    """
    for count in (product_per_category, fashion_per_attribute, deepfashion_per_split):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Sample counts must be non-negative integers")
    builder = _Builder(Path(source_root), Path(output_root))
    summary = {
        "schema_version": 1,
        "source_root": str(builder.source),
        "output_root": str(builder.output),
        "product": _products(builder, product_per_category),
        "fashionai": _fashion(builder, fashion_per_attribute),
        "deepfashion": _deepfashion(builder, deepfashion_per_split),
        "sampling_method": "deterministic_first_match_within_official_split",
        "category_rules": {
            "fields_in_priority_order": ["label", "title_only_if_label_missing"],
            "category_priority": list(PRODUCT_RULES),
            "keywords": PRODUCT_RULES,
            "limitation": "Substring heuristic; requires manual category review.",
        },
    }
    samples = builder.target("samples.jsonl")
    with samples.open("w", encoding="utf-8", newline="\n") as handle:
        for record in builder.records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary["total_records"] = len(builder.records)
    summary["copied_files"] = len(builder.files)
    summary["verified_local_images"] = sum(
        record["image_path"] is not None for record in builder.records
    )
    summary["pending_product_images"] = sum(
        record["dataset"] == "product" for record in builder.records
    )
    builder.write("manifest.json", {"schema_version": 1, "files": builder.files})
    builder.write("summary.json", summary)
    return summary
