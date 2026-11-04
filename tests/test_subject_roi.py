"""Critical ROI geometry, modality routing and protected-source checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from multimodal_ecommerce.safety import PROTECTED_ROOT
from multimodal_ecommerce.subject_roi import (
    SubjectRoiImageOnlyOrderInputs,
    choose_box,
    image_only_order,
    save_subject_crop,
    subject_prompt,
)


def test_text_route_does_not_inspect_image_scores() -> None:
    """Verify that text route does not inspect image scores."""
    original = ["z", "b", "a"]
    result = image_only_order(
        SubjectRoiImageOnlyOrderInputs(
            original,
            is_image=False,
            ids=[],
            base=np.array([]),
            pool=[],
            logits=np.array([]),
            whole_scores=np.array([np.nan]),
            roi_scores=np.array([np.nan]),
            offsets=None,
        )
    )
    assert result == original and result is not original


def test_main_subject_translation_ignores_background_words() -> None:
    """Verify that main subject translation ignores background words."""
    assert (
        subject_prompt("商品主体：吸管水杯。可见结构：旁边摆着鞋和枕头。")
        == "cup. bottle."
    )
    assert (
        subject_prompt("商品主体：耳机升级线。可见结构：连接耳机。")
        == "earphone cable."
    )
    assert subject_prompt("商品主体：无法识别。可见结构：有枕头。") is None


def test_box_selection_rejects_tiny_boxes_and_preserves_fallback() -> None:
    """Verify that box selection rejects tiny boxes and preserves fallback."""
    detections: list[dict] = [
        dict(score=0.99, box=[0, 0, 1, 1]),
        dict(score=0.8, box=[20, 20, 120, 120]),
        dict(score=0.7, box=[0, 0, 300, 300]),
    ]
    value = choose_box(detections, (400, 400))
    assert value["selected"]["box"] == [20, 20, 120, 120]
    assert value["selected"]["padded_box"] == [12, 12, 128, 128]
    assert (
        choose_box([dict(score=0.99, box=[0, 0, 400, 400])], (400, 400))["status"]
        == "global_fallback"
    )


def test_crop_writer_rejects_protected_source_before_file_io() -> None:
    """Verify that crop writer rejects protected source before file io."""
    with pytest.raises(ValueError, match="read-only source"):
        save_subject_crop(
            Path("unused.png"),
            dict(status="roi"),
            PROTECTED_ROOT / "should_never_be_created",
        )
