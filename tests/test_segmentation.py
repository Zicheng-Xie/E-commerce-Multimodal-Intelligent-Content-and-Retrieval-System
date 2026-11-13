from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from multimodal_ecommerce.segmentation import iter_records, mask_metrics, polygon_mask, scan_dataset


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "fashion"
    (root / "train" / "image").mkdir(parents=True)
    (root / "train" / "annos").mkdir()
    Image.new("RGB", (10, 10), "white").save(root / "train" / "image" / "000001.jpg")
    (root / "train" / "annos" / "000001.json").write_text(json.dumps({
        "source": "shop",
        "item1": {"category_id": 1, "category_name": "shirt", "bounding_box": [1, 1, 8, 8], "segmentation": [[1, 1, 8, 1, 8, 8, 1, 8]], "occlusion": 1},
    }))
    return root


def test_contract_and_scan(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    row = next(iter_records(root, "train"))
    assert row.objects[0]["bbox"] == [1.0, 1.0, 7.0, 7.0]
    assert scan_dataset(root, ("train",))["splits"]["train"]["objects"] == 1


def test_unannotated_test_images_are_kept_without_fake_labels(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    (root / "test" / "image").mkdir(parents=True)
    Image.new("RGB", (10, 10), "white").save(root / "test" / "image" / "000002.jpg")
    row = next(iter_records(root, "test"))
    assert row.objects == ()
    assert row.annotation_path == ""


def test_mask_metrics() -> None:
    target = polygon_mask((10, 10), [[1, 1, 8, 1, 8, 8, 1, 8]])
    assert mask_metrics(target, target) == {"iou": 1.0, "dice": 1.0}
