"""Eligible releases must keep source geometry and remove dangling COCO joins."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from multimodal_ecommerce import safety
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.fashion_audit import audit_fashion
from multimodal_ecommerce.fashion_release import release_fashion
from multimodal_ecommerce.safety import sha256_file, write_json


def fixture_data(root: Path) -> tuple[Path, Path]:
    """Return fixture data for this workflow.

    Args:
        root: Root directory used to resolve the operation artifacts.

    Returns:
        Values produced by this operation, in the stated return order.

    Notes:
        Writes derived artifacts through Image.new('RGB', (20, 30), 'blue').save,
        dataset.mkdir, write_json, write_rows.
    """
    dataset = root / "dataset"
    dataset.mkdir()
    rows = []
    files = []
    for number, bad in ((1, False), (2, True)):
        image = dataset / f"{number}.png"
        Image.new("RGB", (20, 30), "blue").save(image)
        files.append({"path": image.name, "sha256": sha256_file(image)})
        rows.append(
            {
                "id": f"train/{number}",
                "source_id": str(number),
                "dataset": "deepfashion",
                "split": "train",
                "image_path": image.name,
                "width": 20,
                "height": 30,
                "pair_id": 1,
                "instances": [
                    {
                        "category_id": 1,
                        "category_name": "top",
                        "source_instance_id": "item1",
                        "bbox": [0, 0, 10, 20],
                        "area": 100,
                        "area_method": "test_fixture",
                        "segmentation": [[-0.2 if bad else 0, 0, 10, 0, 10, 20]],
                    }
                ],
            }
        )
    write_rows(dataset / "samples.jsonl", rows)
    write_json(dataset / "manifest.json", {"files": files})
    audit = root / "audit"
    audit_fashion(dataset / "samples.jsonl", audit)
    return dataset, audit


def test_release_excludes_invalid_image_and_keeps_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that release excludes invalid image and keeps geometry.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    dataset, audit = fixture_data(tmp_path)
    source_hash = sha256_file(dataset / "samples.jsonl")
    output = tmp_path / "release"
    result = release_fashion(dataset / "samples.jsonl", audit, output)
    assert result["eligible_records"] == 1
    assert sha256_file(dataset / "samples.jsonl") == source_hash
    kept = read_rows(output / "samples.jsonl")
    assert kept[0]["instances"] == read_rows(dataset / "samples.jsonl")[0]["instances"]
    assert kept[0]["split_scoped_pair_key"] == "train/1"
    coco = json.loads((output / "annotations/deepfashion_train.json").read_text())
    assert {row["id"] for row in coco["images"]} == {1}
    assert {row["image_id"] for row in coco["annotations"]} == {1}
    assert not (output / "2.png").exists()


def test_release_refuses_stale_audit_or_inconsistent_eligibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that release refuses stale audit or inconsistent eligibility.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.

    Notes:
        Writes derived artifacts through write_json.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    dataset, audit = fixture_data(tmp_path)
    flags = json.loads((audit / "eligibility.json").read_text())
    flags[1]["annotation_valid"] = True
    write_json(audit / "eligibility.json", flags)
    with pytest.raises(ValueError, match="Eligibility contradicts"):
        release_fashion(dataset / "samples.jsonl", audit, tmp_path / "invalid")
    with (dataset / "samples.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="different data"):
        release_fashion(dataset / "samples.jsonl", audit, tmp_path / "stale")
