"""Migration keeps frozen inputs unchanged while resolving host-local assets."""

from __future__ import annotations

from pathlib import Path

import pytest

from multimodal_ecommerce.runtime_device import (
    local_model_config,
    project_path,
    relocate_assets,
)
from multimodal_ecommerce.safety import PROJECT_ROOT


def test_relocate_windows_asset_preserves_input() -> None:
    """Verify that relocate windows asset preserves input."""
    original: dict = {
        "views": [{"image_path": r"E:\code\Multimodal_E-commerce\data\sample.jpg"}]
    }
    relocated = relocate_assets(original)
    assert relocated["views"][0]["image_path"] == str(PROJECT_ROOT / "data/sample.jpg")
    assert original["views"][0]["image_path"].startswith("E:")


def test_reject_escaped_historical_path() -> None:
    """Verify that reject escaped historical path."""
    with pytest.raises(ValueError, match="escapes"):
        project_path("E:/code/Multimodal_E-commerce/../../outside.jpg")


def test_unrelated_path_is_not_redirected() -> None:
    """Verify that unrelated path is not redirected."""
    assert project_path("/tmp/example.jpg") == Path("/tmp/example.jpg")


def test_mac_model_config_keeps_frozen_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that mac model config keeps frozen metadata.

    Args:
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    original: dict = {
        "path": r"E:\LLM_for_Ollama\external\model",
        "repository": "Qwen/model",
        "revision": "fixed",
    }
    config = local_model_config(original)
    assert config["path"] == str(PROJECT_ROOT / ".models/model")
    assert config["revision"] == original["revision"]
    assert original["path"].startswith("E:")
