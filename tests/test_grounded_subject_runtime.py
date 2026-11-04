"""Query-only crop validity, abstention, complete evidence and retained catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from grounded_subject_runtime import (
    GroundedSubjectRuntimeFuseInputs,
    fuse,
    geometry,
    proposal,
)


def test_native_review_rejection_keeps_original_without_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that native review rejection keeps original without retries.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.

    Notes:
        Writes derived artifacts through Image.new('RGB', (300, 200), 'white').save.
    """
    import json

    from PIL import Image

    import grounded_subject_runtime as runtime

    source = tmp_path / "source.png"
    Image.new("RGB", (300, 200), "white").save(source)
    responses = iter(
        [
            dict(
                product="内胆包",
                appearance="上方开口",
                visible_text="",
                primary_unclear=False,
                box=[500, 100, 950, 600],
                reason="主售主体",
            ),
            dict(
                decision="reject",
                correct_sale_item=False,
                complete_visible_item=True,
                reason="裁到展示道具",
            ),
        ]
    )

    def request(*args: object) -> tuple[dict, float]:
        """Return request for this workflow.

        Args:
            *args: Extra arguments accepted to match the patched dependency.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        value = next(responses)
        return (
            dict(
                model=runtime.MODEL,
                done=True,
                done_reason="stop",
                prompt_eval_count=10,
                eval_count=10,
                message=dict(role="assistant", content=json.dumps(value)),
            ),
            0.1,
        )

    def once(path: Path, value: object) -> None:
        """Persist a new result without replacing an existing artifact.

        Args:
            path: Path of the input or output artifact used by this operation.
            value: Input value to validate or transform.

        Notes:
            Writes derived artifacts through path.parent.mkdir, path.write_text.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    monkeypatch.setattr(runtime, "native_request", request)

    def load(path: Path) -> Any:
        """Read the serialized input from the supplied path.

        Args:
            path: Path of the input or output artifact used by this operation.

        Returns:
            Decoded JSON value; the caller must validate its expected schema.
        """
        return json.loads(path.read_text(encoding="utf-8"))

    # Native requests are patched; a real model client must never be used here.
    result = runtime.localize(source, tmp_path / "query", None, once, load)  # type: ignore[arg-type]
    assert not result["accepted"] and result["selected_image"] == str(source.resolve())
    assert (tmp_path / "query/zoom.png").exists()
    assert runtime.localize(source, tmp_path / "query", None, once, load) == result  # type: ignore[arg-type]


def test_grounding_geometry_and_uncertainty_keep_whole_image() -> None:
    """Verify that grounding geometry and uncertainty keep whole image."""
    assert geometry(
        dict(primary_unclear=False, box=[600, 100, 900, 500]), (1000, 500)
    ) == [576, 34, 924, 266]
    assert (
        geometry(dict(primary_unclear=True, box=[600, 100, 900, 500]), (1000, 500))
        is None
    )
    assert (
        geometry(dict(primary_unclear=False, box=[0, 0, 1000, 1000]), (1000, 500))
        is None
    )
    assert (
        geometry(dict(primary_unclear=False, box=[900, 100, 600, 500]), (1000, 500))
        is None
    )


def test_union_retains_old_candidates_and_missing_ce_never_imputed() -> None:
    """Verify that union retains old candidates and missing ce never imputed."""
    ids = [str(i) for i in range(30)]
    base = np.arange(30, dtype=float)
    pool = proposal(ids, ids[:20], ids[5:25], base)
    assert set(ids[:25]) <= set(pool) and len(pool) == 30
    whole = dict.fromkeys(pool, 0.1)
    zoom = dict.fromkeys(pool, 0.2)
    result = fuse(
        GroundedSubjectRuntimeFuseInputs(ids, pool, base, whole, zoom, mode="dual")
    )
    assert len(result) == len(set(result)) == 30 and set(result) == set(ids)
    del zoom[pool[0]]
    with pytest.raises(ValueError):
        fuse(
            GroundedSubjectRuntimeFuseInputs(ids, pool, base, whole, zoom, mode="dual")
        )
