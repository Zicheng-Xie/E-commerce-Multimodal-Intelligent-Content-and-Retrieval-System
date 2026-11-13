"""Test deterministic sampling, leakage grouping and immutable release copies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from multimodal_ecommerce import safety
from multimodal_ecommerce.dataset_release import (
    DatasetReleaseFreezeReleaseInputs,
    DatasetReleaseSampleCandidatesInputs,
    duplicate_groups,
    freeze_release,
    read_rows,
    sample_candidates,
    write_rows,
)
from multimodal_ecommerce.media import inspect_image


def test_sampling_reproducible_and_excludes_old(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing source ordering cannot change hash-ranked selected IDs.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "source"
    data = source / "多模态电商数据"
    data.mkdir(parents=True)
    prior = tmp_path / "prior.jsonl"
    write_rows(prior, [{"dataset": "product", "source_id": "0"}])
    products: dict = {
        str(n): {
            "title": f"手机{n}",
            "label": "手机",
            "url": "https://img.alicdn.com/a",
            "pv": "品牌#:#样例",
        }
        for n in range(10)
    }
    for split in ("train", "test"):
        (data / f"product1m_product5m_{split}_id_label.json").write_text(
            json.dumps(products), encoding="utf-8"
        )
    sample_candidates(
        DatasetReleaseSampleCandidatesInputs(
            source, prior, tmp_path / "v1", per_group=3
        )
    )
    for path in data.glob("*.json"):
        path.write_text(
            json.dumps(dict(reversed(list(products.items())))), encoding="utf-8"
        )
    sample_candidates(
        DatasetReleaseSampleCandidatesInputs(
            source, prior, tmp_path / "v2", per_group=3
        )
    )
    first = read_rows(tmp_path / "v1/samples.jsonl")
    second = read_rows(tmp_path / "v2/samples.jsonl")
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert len(first) == 6
    assert all(row["source_id"] != "0" for row in first)
    with pytest.raises(FileExistsError):
        sample_candidates(
            DatasetReleaseSampleCandidatesInputs(source, prior, tmp_path / "v1")
        )


def test_transitive_duplicate_groups() -> None:
    """ID, exact title and near-image links propagate to the whole component."""
    rows: list[dict] = [
        {"id": "a", "source_id": "old"},
        {"id": "b", "source_id": "old", "normalized_title": "衣服"},
        {"id": "c", "normalized_title": "衣服"},
        {"id": "d"},
        {"id": "e"},
    ]
    groups, edges = duplicate_groups(rows, {"c": 0, "d": 3, "e": 2**64 - 1})
    assert any(set(group) == {"a", "b", "c", "d"} for group in groups)
    assert ["e"] in groups
    assert any(edge["kind"] == "dhash_candidate" for edge in edges)


def test_freeze_copies_and_blocks_prior_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exclude known titles without rewriting source bytes; freeze new images.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    local = tmp_path / "local"
    local.mkdir()
    image = local / "a.png"
    Image.new("RGB", (40, 40), "red").save(image)
    details = inspect_image(image.read_bytes())
    old = tmp_path / "old.jsonl"
    write_rows(
        old,
        [
            {
                "dataset": "product",
                "id": "old",
                "source_id": "old",
                "text": "旧 衣服",
                "attributes": {"颜色": ["红"]},
                "category": "服饰穿搭",
                "image_path": None,
            }
        ],
    )
    prior_local = tmp_path / "prior_local.jsonl"
    write_rows(prior_local, [])
    rows: list[dict] = [
        {
            "dataset": "product",
            "id": "new",
            "source_id": "new",
            "text": "新衣服",
            "split": "test",
            "category": "服饰穿搭",
            "attributes": {"颜色": ["红", "蓝"]},
            "image_path": "a.png",
            "image_sha256": details["sha256"],
            "pixel_sha256": details["pixel_sha256"],
        },
        {
            "dataset": "product",
            "id": "old_copy",
            "source_id": "other",
            "text": "旧衣服",
            "split": "train",
            "category": "服饰穿搭",
            "attributes": {"颜色": ["红"]},
            "image_path": None,
        },
    ]
    dataset = local / "products.jsonl"
    write_rows(dataset, rows)
    before = dataset.read_bytes()
    output = tmp_path / "release"
    report = freeze_release(
        DatasetReleaseFreezeReleaseInputs(
            dataset, old, prior_local, output, per_group=1
        )
    )
    assert report["selected"] == 1
    assert report["decisions"]["matches_previous_development_data"] == 1
    assert report["roles"]["heldout"] == 1
    assert (output / "images/a.png").read_bytes() == image.read_bytes()
    assert dataset.read_bytes() == before
    assert read_rows(output / "products.jsonl")[0]["quality_flags"] == [
        "multi_value_attributes_require_sku_review"
    ]
    with pytest.raises(FileExistsError):
        freeze_release(
            DatasetReleaseFreezeReleaseInputs(dataset, old, prior_local, output)
        )
