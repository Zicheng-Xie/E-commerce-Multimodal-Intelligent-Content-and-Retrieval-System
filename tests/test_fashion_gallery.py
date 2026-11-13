"""Validate training-only expansion, exact real-vector commits and interrupted resume."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from multimodal_ecommerce import fashion_gallery, safety
from multimodal_ecommerce.expanded_showcase import ExpandedShowcase
from multimodal_ecommerce.fashion_gallery import (
    FashionGallery,
    public_title,
    training_paths,
)
from multimodal_ecommerce.holdout_data import fingerprints


@pytest.fixture
def gallery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FashionGallery:
    """Create actual tiny raster files and fake model identity, without real datasets.

    Args:
        tmp_path: Isolated synthetic source/output root.
        monkeypatch: Scoped fixture dependency replacement.

    Returns:
        Synthetic training catalog with no real heldout input.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(fashion_gallery, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(fashion_gallery, "consumed_exclusions", lambda: (set(), []))
    source = tmp_path / "train"
    (source / "image").mkdir(parents=True)
    (source / "annos").mkdir()
    for i in range(1, 5):
        Image.new("RGB", (13 + i, 17), (i * 30, 0, 0)).save(
            source / "image" / f"{i:06}.jpg"
        )
        (source / "annos" / f"{i:06}.json").write_text(
            json.dumps(dict(item1=dict(category_id=1), pair_id=i))
        )
    return FashionGallery(dict(synthetic=True), [], set(), tmp_path / "derived", source)


def _vectors(paths: list[Path]) -> np.ndarray:
    return np.repeat(np.eye(1, 2048, dtype=np.float32), len(paths), axis=0)


def test_commits_are_restorable_and_tampering_is_rejected(
    gallery: FashionGallery,
) -> None:
    """Only committed vectors become ready, and changed images fail reload.

    Args:
        gallery: Synthetic training-only catalog.
    """
    records, vectors = gallery.step(_vectors, 2)
    assert gallery.ready == gallery.cursor == 2 and vectors.shape == (2, 2048)
    other = FashionGallery(
        dict(synthetic=True), [], set(), gallery.output, gallery.source
    )
    restored = other.restore()
    assert other.cursor == 2 and restored[0][0] == records
    assert len(training_paths(gallery.source)) == 4
    image = gallery.output / "media/000001.jpg"
    image.write_bytes(b"altered")
    with pytest.raises(ValueError, match="image changed"):
        FashionGallery(
            dict(synthetic=True), [], set(), gallery.output, gallery.source
        ).restore()


def test_uncommitted_invalid_vectors_do_not_advance_cursor(
    gallery: FashionGallery,
) -> None:
    """A failed inference never becomes a fake completed item or lost resume position.

    Args:
        gallery: Synthetic catalog.
    """
    with pytest.raises(ValueError, match="normalized"):
        gallery.step(lambda paths: np.zeros((len(paths), 2048)), 2)
    assert gallery.cursor == gallery.ready == 0
    assert not list((gallery.output / "shards").glob("*.json"))
    gallery.step(_vectors, 2)
    assert gallery.ready == 2


def test_reservations_and_consumed_ids_are_not_encoded(gallery: FashionGallery) -> None:
    """Protected images and known consumed identities are rejected before encoding.

    Args:
        gallery: Synthetic catalog.
    """
    gallery.blocked_ids.add("deepfashion/train/000001")
    gallery.protected = [fingerprints(gallery.paths[1])]
    records, _ = gallery.step(_vectors, 2)
    assert not records and gallery.excluded == 2 and gallery.ready == 0
    gallery.protected = []
    gallery.blocked_ids.add("deepfashion/pair/3")
    assert gallery._prepare(gallery.paths[2]) is None


def test_only_train_is_accepted_and_native_titles_do_not_invent_retail_facts(
    gallery: FashionGallery,
) -> None:
    """Validation/test source roots fail before images are opened.

    Args:
        gallery: Synthetic catalog.
    """
    with pytest.raises(ValueError, match="training split"):
        FashionGallery(
            {}, [], set(), gallery.output, gallery.source.parent / "validation"
        )
    title = public_title(dict(item1=dict(category_id=1), item2=dict(category_id=8)))
    assert "训练图片" in title and "长裤" in title and "短袖上衣" in title
    assert "价格" not in title
    redirect = gallery.source.parent / "redirect/train"
    redirect.mkdir(parents=True)
    (redirect / "image").symlink_to(gallery.source / "image")
    with pytest.raises(ValueError, match="redirect"):
        training_paths(redirect)


def test_newly_protected_existing_shard_is_denied(gallery: FashionGallery) -> None:
    """A later exclusion prevents reusing a previously committed media record.

    Args:
        gallery: Synthetic training index.
    """
    gallery.step(_vectors, 2)
    protected = [fingerprints(gallery.paths[0])]
    other = FashionGallery(
        dict(synthetic=True), protected, set(), gallery.output, gallery.source
    )
    with pytest.raises(PermissionError, match="protected"):
        other.restore()


def test_shared_embedding_similarity_includes_every_new_image() -> None:
    """Additional photos participate in both query modes without fake title vectors."""
    engine = object.__new__(ExpandedShowcase)
    engine.whole = engine.crop = engine.titles = np.repeat(np.eye(1, 2048), 3, axis=0)
    engine.fashion_vectors = [np.repeat(np.eye(1, 2048), 2, axis=0)]
    for image in [True, False]:
        scores, titles = engine._similarities(np.eye(1, 2048)[0], image)
        assert scores.shape == titles.shape == (5,)
        assert np.allclose(scores, 1) and np.allclose(titles, 1)
