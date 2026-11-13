"""Exercise geometry checks without inferring identity from scoped pair IDs."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from multimodal_ecommerce import safety
from multimodal_ecommerce.dataset_release import write_rows
from multimodal_ecommerce.fashion_audit import audit_fashion
from multimodal_ecommerce.safety import sha256_file, write_json


def test_invalid_polygon_and_pair_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A negative coordinate is invalid; matching pair IDs alone prove no leak.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    dataset = tmp_path / "copies"
    dataset.mkdir()
    records = []
    for split, color in [("train", "red"), ("validation", "blue")]:
        path = dataset / f"{split}.png"
        Image.new("RGB", (20, 30), color).save(path)
        records.append(
            {
                "id": split,
                "dataset": "deepfashion",
                "split": split,
                "image_path": path.name,
                "width": 20,
                "height": 30,
                "pair_id": 1,
                "instances": [
                    {
                        "source_instance_id": "item1",
                        "category_name": "top",
                        "bbox": [0, 0, 10, 20],
                        "area": 100,
                        "segmentation": [
                            [-0.2 if split == "train" else 0, 0, 10, 0, 10, 20]
                        ],
                    }
                ],
            }
        )
    write_rows(dataset / "samples.jsonl", records)
    write_json(
        dataset / "manifest.json",
        {
            "files": [
                {
                    "path": f"{split}.png",
                    "sha256": sha256_file(dataset / f"{split}.png"),
                }
                for split in ("train", "validation")
            ]
        },
    )
    report = audit_fashion(dataset / "samples.jsonl", tmp_path / "audit")
    assert report["issue_counts"] == {"polygon_invalid_or_out_of_bounds": 1}
    assert len(report["cross_split_groups"]) == 1
    assert (
        report["cross_split_groups"][0]["status"]
        == "identifier_collision_global_scope_unverified"
    )
