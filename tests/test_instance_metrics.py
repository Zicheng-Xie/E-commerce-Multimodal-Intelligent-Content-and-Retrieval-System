import numpy as np
import pytest
from multimodal_ecommerce.instance_metrics import instance_ap


def row(labels=(1,), scores=(0.9,)):
    m = np.ones((2, 2), dtype=bool)
    return dict(
        gt_labels=[1],
        gt_masks=[m],
        labels=labels,
        scores=scores,
        masks=[m for _ in labels],
    )


def test_perfect_missing_and_wrong_class():
    assert instance_ap([row()])["mask_map"] == 1
    assert instance_ap([row((), ())])["mask_map"] == 0
    assert instance_ap([row((2,))])["mask_map"] == 0


def test_false_positive_before_true_positive():
    r = row((1, 1), (0.9, 0.8))
    r["masks"][0] = np.zeros((2, 2), dtype=bool)
    assert instance_ap([r])["mask_ap50"] == 0.5


def test_duplicate_cannot_match_second_gt():
    r = row((1, 1), (0.9, 0.8))
    r["gt_masks"].append(np.zeros((2, 2), dtype=bool))
    r["gt_labels"].append(1)
    assert instance_ap([r])["mask_ap50"] == pytest.approx(51 / 101)


def test_thirteen_class_oracle():
    examples = []
    for c in range(1, 14):
        r = row((c,))
        r["gt_labels"] = [c]
        examples.append(r)
    result = instance_ap(examples)
    assert len(result["per_class"]) == 13
    assert all(v["ap"] == v["ap50"] == 1 for v in result["per_class"].values())


def test_empty_predictions_and_empty_targets():
    assert instance_ap([row((), ())])["mask_map"] == 0
    assert instance_ap([])["mask_map"] is None
