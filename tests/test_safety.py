"""Regression tests for protected destinations and path traversal."""

from __future__ import annotations

from pathlib import Path

import pytest

from multimodal_ecommerce import safety


def test_write_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject source descendants, ancestors and destinations outside the project.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "source"
    for destination in [source / "out.json", tmp_path, tmp_path / "../outside.json"]:
        with pytest.raises(ValueError):
            safety.write_json(destination, {}, source)
    output = tmp_path / "outputs/report.json"
    safety.write_json(output, {"ok": True}, source)
    assert output.exists()
