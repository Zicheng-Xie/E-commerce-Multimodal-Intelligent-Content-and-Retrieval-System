"""Realistic acceptance contamination and near-duplicate fold checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from multimodal_ecommerce import development_expansion as expansion
from multimodal_ecommerce.business_acceptance import product_family_keys
from multimodal_ecommerce.holdout_data import fingerprints


def fixture_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, dict, list[dict]]:
    """Return fixture product for this workflow.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.

    Returns:
        Values produced by this operation, in the stated return order.

    Notes:
        Writes derived artifacts through
        Image.fromarray(np.random.default_rng(42).integers(0, 256, (96, 96, 3),
        dtype=np.uint8)).save.
    """
    monkeypatch.setattr(expansion, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "development.png"
    Image.fromarray(
        np.random.default_rng(42).integers(0, 256, (96, 96, 3), dtype=np.uint8)
    ).save(path)
    document: dict = dict(
        id="dev", title="商品原始标题", image_path=str(path), grade=1, bucket="answer"
    )
    raw: dict = {
        "dev": dict(
            source_id="dev",
            text="商品原始标题",
            image_url="https://example.invalid/dev",
        )
    }
    reserved: list[dict] = [
        dict(
            id="reserved",
            family_keys=["id:reserved"],
            fingerprints=dict(sha256="other", pixels="other", dhash=0),
        )
    ]
    return document, raw, reserved


@pytest.mark.parametrize("contamination", ["id", "family", "pixels"])
def test_guard_rejects_reserved_before_any_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contamination: object
) -> None:
    """Verify that guard rejects reserved before any model.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
        contamination: Contamination used by this operation.
    """
    document, raw, reserved = fixture_product(tmp_path, monkeypatch)
    if contamination == "id":
        reserved[0]["id"] = "dev"
    elif contamination == "family":
        reserved[0]["family_keys"] = list(product_family_keys(raw["dev"]))
    else:
        reserved[0]["fingerprints"] = fingerprints(tmp_path / "development.png")
    with pytest.raises(ValueError, match="Reserved acceptance"):
        expansion.acceptance_guard([document], raw, reserved)


def test_inference_input_has_no_judgment_or_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that inference input has no judgment or bucket.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    document, raw, reserved = fixture_product(tmp_path, monkeypatch)
    clean = expansion.clean_document(document)
    assert set(clean) == {"id", "title", "image_path"}
    guard = expansion.acceptance_guard([document], raw, reserved)
    assert guard["reserved_images_opened"] == 0 and guard["acceptance_predictions"] == 0


def test_same_pixels_remain_in_one_fold_and_query_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that same pixels remain in one fold and query family.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    document, raw, reserved = fixture_product(tmp_path, monkeypatch)
    other: dict = dict(document, id="other")
    signature = fingerprints(tmp_path / "development.png")
    guard: dict = dict(
        signatures={"dev": signature, "other": signature},
        family_keys={"dev": ["id:dev"], "other": ["id:other"]},
    )
    query: dict = dict(
        id="image", text=None, image_path=document["image_path"], exclude_id="dev"
    )
    metadata = expansion.fold_metadata([document, other], [query], guard)
    assert metadata["family_count"] == 1
    assert metadata["gallery_families"] == ["dev", "dev"]
    assert metadata["query_families"] == ["dev"]
    assert len(set(metadata["gallery_folds"])) == 1
