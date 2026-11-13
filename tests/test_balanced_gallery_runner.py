"""Observe an existing review without duplicate workers or invented grades."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import pytest

import balanced_gallery_runner as runner


def test_engineering_failure_remains_evidence_semantic_null_is_not_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that engineering failure remains evidence semantic null is not failure.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.

    Notes:
        Writes derived artifacts through (folder / '001.json').write_text, (folder /
        '002.json').write_text, folder.mkdir.
    """
    monkeypatch.setattr(runner.review, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "rel", lambda path: path.name)
    folder = tmp_path / "attempts/000_00"
    folder.mkdir(parents=True)
    (folder / "001.json").write_text(
        json.dumps(dict(exception=None, result=dict(grades=[None]))), encoding="utf-8"
    )
    assert runner.review_failure() is None
    (folder / "002.json").write_text(
        json.dumps(
            dict(
                exception=dict(type="TimeoutError", message="native error"), result=None
            )
        ),
        encoding="utf-8",
    )
    failure = runner.review_failure()
    assert failure is not None
    assert (
        failure["file"] == "002.json" and failure["exception"]["type"] == "TimeoutError"
    )


def test_missing_receipt_only_observes_files_never_starts_another_reviewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that missing receipt only observes files never starts another reviewer.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.

    Notes:
        Writes derived artifacts through (tmp_path / 'native').mkdir, (tmp_path /
        'native/000_00.json').write_text.
    """
    monkeypatch.setattr(runner.review, "ROOT", tmp_path)
    (tmp_path / "native").mkdir()
    (tmp_path / "native/000_00.json").write_text("{}", encoding="utf-8")
    observed = []
    monkeypatch.setattr(
        runner, "public_status", lambda phase, **extra: observed.append((phase, extra))
    )

    def forbidden() -> NoReturn:
        """Apply the forbidden workflow step.

        Raises:
            AssertionError: Must not touch GPU, model server or another reviewer.
        """
        raise AssertionError("Must not touch GPU, model server or another reviewer")

    monkeypatch.setattr(runner, "guard", forbidden)
    monkeypatch.setattr(runner.recovery, "locked", forbidden)
    assert runner.step() == "waiting_existing_native_review"
    assert observed == [
        (
            "waiting_existing_native_review",
            dict(native_responses=1, planned_native_requests=532),
        )
    ]
