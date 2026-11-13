"""Self-contradiction abstains without modifying frozen model or retrying."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import pytest

import grounded_subject_experiment as original
import grounded_subject_recovery as recovery


def test_recovery_namespace_keeps_original_source_runtime_unmodified() -> None:
    """Verify that recovery namespace keeps original source runtime unmodified."""
    previous = original.localize
    module = recovery.runtime()
    assert module.localize is recovery.safe_localize
    assert original.localize is previous and original.guard is not module.guard


def test_contradiction_preserves_raw_review_and_returns_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that contradiction preserves raw review and returns whole.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.

    Notes:
        Writes derived artifacts through Image.new('RGB', (300, 200)).save.
    """
    import json

    from PIL import Image

    core = recovery.core
    image = tmp_path / "source.png"
    Image.new("RGB", (300, 200)).save(image)
    values: dict = {
        "grounding.native.json": dict(
            product="内胆包",
            appearance="开口",
            visible_text="",
            primary_unclear=False,
            box=[500, 100, 950, 600],
            reason="可见主体",
        ),
        "review.native.json": dict(
            decision="accept",
            correct_sale_item=None,
            complete_visible_item=True,
            reason="未能确定主售对象",
        ),
    }
    records: dict = {
        name: dict(
            raw=dict(
                model=core.MODEL,
                done=True,
                done_reason="stop",
                prompt_eval_count=10,
                eval_count=10,
                message=dict(role="assistant", content=json.dumps(value)),
            )
        )
        for name, value in values.items()
    }

    def fail(*args: object) -> NoReturn:
        """Apply the fail workflow step.

        Args:
            *args: Extra arguments accepted to match the patched dependency.

        Raises:
            ValueError: Native accept contradicts required checks.
        """
        raise ValueError("Native accept contradicts required checks")

    saved: dict = {}
    monkeypatch.setattr(core, "localize", fail)
    result = recovery.safe_localize(
        image,
        tmp_path / "job",
        None,  # type: ignore[arg-type]  # Localization is replaced by the failing test double.
        lambda p, v: saved.update({p.name: v}),
        lambda p: records[p.name],
    )
    assert result["accepted"] is False and result["selected_image"] == str(
        image.resolve()
    )
    assert result["review"] == values["review.native.json"]
    assert saved["result.json"] == result and "semantic_abstention" in result
