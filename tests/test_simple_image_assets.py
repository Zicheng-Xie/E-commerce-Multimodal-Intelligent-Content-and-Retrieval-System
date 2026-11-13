"""Ensure subject cropping uses EXIF-corrected originals and never overwrites inputs."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from module2_simple_image_assets import crop_original
from multimodal_ecommerce.safety import sha256_file


def test_crop_uses_display_orientation_and_preserves_original(tmp_path: Path) -> None:
    """Detector coordinates must apply after rotation recorded in EXIF.

    Args:
        tmp_path: Temporary originals and derived output directory.
    """
    source, output = tmp_path / "source.jpg", tmp_path / "crop.png"
    image = Image.new("RGB", (60, 30), "red")
    exif = image.getexif()
    exif[274] = 6
    image.save(source, exif=exif)
    original_hash = sha256_file(source)
    decision = dict(
        source_size=[30, 60], decision=dict(selected=dict(padded_box=[2, 5, 22, 45]))
    )
    assert crop_original(source, decision, output) == sha256_file(output)
    with Image.open(output) as crop:
        assert crop.size == (20, 40)
    assert sha256_file(source) == original_hash
    decision["source_size"] = [60, 30]
    with pytest.raises(ValueError, match="original image"):
        crop_original(source, decision, tmp_path / "wrong.png")


def test_detector_abstention_keeps_original_and_creates_no_crop(tmp_path: Path) -> None:
    """Missing confident boxes must preserve the gallery image and its bytes.

    Args:
        tmp_path: Temporary originals and derived output directory.
    """
    source, output = tmp_path / "source.png", tmp_path / "crop.png"
    Image.new("RGB", (50, 40)).save(source)
    assert crop_original(
        source, dict(decision=dict(selected=None)), output
    ) == sha256_file(source)
    assert not output.exists()
