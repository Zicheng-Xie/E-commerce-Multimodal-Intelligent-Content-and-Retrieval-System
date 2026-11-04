"""Product-role constraints must abstain on uncertain/ambiguous requests."""

from __future__ import annotations

import numpy as np

from multimodal_ecommerce.business_constraints import constrain_scores, query_intent


def test_accessory_titles_do_not_outrank_requested_main_product() -> None:
    """Verify that accessory titles do not outrank requested main product."""
    scores, evidence = constrain_scores(
        "无线蓝牙耳机",
        ["耳机保护套", "无线蓝牙耳机带保护套", "USB有线耳机"],
        [
            "商品主体：耳机保护套。可见外观：黑色",
            "商品主体：耳机。可见外观：黑色",
            "商品主体：耳机。可见外观：黑色",
        ],
        np.array([0.8, 0.5, 0.6]),
    )
    assert scores[1] > scores[0] and scores[1] > scores[2]
    assert evidence[0]["state"] == "explicit_product_or_attribute_conflict"
    assert evidence[2]["attributes"]["conflicts"] == ["wireless"]


def test_accessory_request_remains_valid_and_unknown_is_not_negative() -> None:
    """Verify that accessory request remains valid and unknown is not negative."""
    scores, why = constrain_scores(
        "手机保护壳",
        ["手机壳", "手机", "其他商品"],
        ["手机保护壳", "手机", "主体不清楚"],
        np.zeros(3),
    )
    assert scores[0] > scores[2] > scores[1]
    assert why[2]["state"] == "unknown_product_type"
    assert query_intent("无线蓝牙耳机，不要保护套")[0] == "headphone"
    assert query_intent("手机和耳机")[0] is None


def test_image_query_is_unchanged_and_mixed_attribute_titles_are_unknown() -> None:
    """Verify that image query is unchanged and mixed attribute titles are unknown."""
    raw = np.array([0.2, 0.4])
    scores, why = constrain_scores(None, ["a", "b"], ["a", "b"], raw)
    assert np.array_equal(raw, scores)
    _, why = constrain_scores("无线耳机", ["有线无线耳机"], ["耳机"], np.array([0.4]))
    assert why[0]["attributes"]["unknown"] == ["wireless"]
