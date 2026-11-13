"""Meaningful checks for source protection and heterogeneous sample parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from multimodal_ecommerce import safety
from multimodal_ecommerce.data import classify_product, parse_pv, prepare_samples


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 30), color="red").save(path)


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create miniature fixtures with the same source layouts as real data.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.

    Returns:
        Resolved artifact path.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "protected"
    products: dict = {
        "phone": {"title": "手机", "label": "手机", "pv": "颜色#:#红#;#颜色#:#蓝"},
        "shirt": {"title": "衬衫", "label": "衬衫", "pv": "品牌#:#示例"},
        "desk": {"title": "桌子", "label": "家具", "pv": ""},
    }
    for split in ("train", "test"):
        _json(
            source / "多模态电商数据" / f"product1m_product5m_{split}_id_label.json",
            products,
        )
    root = source / "服饰细分属性" / "round1_fashionAI_attributes_test_a"
    root.mkdir(parents=True)
    (root / "README.md").write_text(
        "# collar_design_labels\n- Invisible\n- Shirt Collar\n- Peter Pan\n",
        encoding="utf-8",
    )
    answer = root / "Tests" / "round1_fashionAI_attributes_answer_a.csv"
    answer.parent.mkdir()
    answer.write_text(
        "Images/collar_design_labels/a.jpg,collar_design_labels,myn\n",
        encoding="utf-8",
    )
    _image(root / "Images" / "collar_design_labels" / "a.jpg")
    for split in ("train", "validation"):
        root = source / "服饰分割" / split
        _image(root / "image" / "000001.jpg")
        _json(
            root / "annos" / "000001.json",
            {
                "source": "user",
                "pair_id": 1,
                "item1": {
                    "category_id": 1,
                    "category_name": "short sleeve top",
                    "bounding_box": [2, 3, 12, 20],
                    "segmentation": [[2, 3, 12, 3, 12, 20, 2, 20]],
                },
            },
        )
    return source


def test_multivalue_and_label_first_category() -> None:
    """Repeated properties survive and label matches outrank title matches."""
    assert parse_pv("颜色#:#红#;#颜色#:#蓝#;#无效") == {"颜色": ["红", "蓝"]}
    result = classify_product({"label": "衬衫", "title": "手机同款上衣"})
    assert result == ("服饰穿搭", {"field": "label", "keyword": "衬衫"})
    assert classify_product({"label": "其他", "title": "普通商品"}) is None


def test_copies_traceable_samples_without_modifying_source(
    dataset: Path, tmp_path: Path
) -> None:
    """Copies preserve hashes, official splits, unknown labels and COCO boxes.

    Args:
        dataset: Dataset used by this operation.
        tmp_path: Isolated temporary directory supplied by pytest.
    """
    before: dict = {
        str(path): (path.stat().st_mtime_ns, safety.sha256_file(path))
        for path in dataset.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "prepared"
    summary = prepare_samples(
        dataset,
        output,
        product_per_category=1,
        fashion_per_attribute=2,
        deepfashion_per_split=1,
    )
    records = [
        json.loads(line)
        for line in (output / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["total_records"] == 9
    assert summary["verified_local_images"] == 3
    products = [r for r in records if r["dataset"] == "product"]
    assert {r["split"] for r in products} == {"train", "test"}
    assert all(r["image_path"] is None for r in products)
    assert products[0]["attributes"]["颜色"] == ["红", "蓝"]
    fashion = next(r for r in records if r["dataset"] == "fashionai")
    assert fashion["label_code"] == "myn"
    assert fashion["category"] == "Shirt Collar"
    assert summary["fashionai"]["shortfalls"] == {"collar_design_labels": 1}
    coco = json.loads((output / "annotations" / "deepfashion_train.json").read_text())
    assert coco["annotations"][0]["bbox"] == [2, 3, 10, 17]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        assert safety.sha256_file(output / entry["path"]) == entry["sha256"]
    after: dict = {
        str(path): (path.stat().st_mtime_ns, safety.sha256_file(path))
        for path in dataset.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize("relative", ["protected/output", "."])
def test_rejects_output_inside_source_or_source_ancestor(
    dataset: Path, tmp_path: Path, relative: str
) -> None:
    """Neither protected files nor a containing directory is an output target.

    Args:
        dataset: Dataset used by this operation.
        tmp_path: Isolated temporary directory supplied by pytest.
        relative: Relative used by this operation.
    """
    with pytest.raises(ValueError):
        prepare_samples(dataset, tmp_path / relative)


def test_rejects_escaped_fashion_image(dataset: Path, tmp_path: Path) -> None:
    """CSV-relative paths cannot read arbitrary files outside the source.

    Args:
        dataset: Dataset used by this operation.
        tmp_path: Isolated temporary directory supplied by pytest.
    """
    answer = (
        dataset
        / "服饰细分属性"
        / "round1_fashionAI_attributes_test_a"
        / "Tests"
        / "round1_fashionAI_attributes_answer_a.csv"
    )
    answer.write_text("../../../outside.jpg,collar_design_labels,myn\n")
    _image(tmp_path / "outside.jpg")
    with pytest.raises(ValueError, match="escapes"):
        prepare_samples(dataset, tmp_path / "prepared", product_per_category=0)
