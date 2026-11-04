"""Primary noun ordering and multi-instance geometry are ranking-independent."""

from __future__ import annotations

from business_subject_views import noun, union_box


def test_subject_head_handles_context_words_without_title_substrings() -> None:
    """Verify that subject head handles context words without title substrings."""
    assert noun("榴莲果肉") == "durian."
    assert noun("茶水杯柜") == "cabinet."
    assert noun("耳机盒保护套") == "earphone case."
    assert noun("笔记本电脑内胆包") == "laptop bag."
    assert noun("打火机耳机组合") is None


def test_union_keeps_both_instances_and_rejects_whole_frame() -> None:
    """Verify that union keeps both instances and rejects whole frame."""
    rows: list[dict] = [
        {"score": 0.8, "box": [10, 10, 40, 50]},
        {"score": 0.4, "box": [60, 20, 90, 60]},
    ]
    box, valid = union_box(rows, (100, 100))
    assert box is not None
    assert len(valid) == 2 and box[0] <= 10 and box[2] >= 90
    box, _ = union_box([{"score": 0.9, "box": [0, 0, 100, 100]}], (100, 100))
    assert box is None
