"""Incomplete sale-item/structure evidence always keeps the original view."""

from __future__ import annotations

from business_mask_verifier import selected_view


def test_mask_selection_requires_all_checks_and_preserves_box_fallback() -> None:
    """Verify that mask selection requires all checks and preserves box fallback."""
    value: dict = {
        "box_correct_sale_item": True,
        "box_complete_visible_item": True,
        "mask_complete_without_props": True,
    }
    assert selected_view(value) == "masked_native"
    value["mask_complete_without_props"] = False
    assert selected_view(value) == "box_native"
    value["box_complete_visible_item"] = None
    assert selected_view(value) == "whole"
    value["box_complete_visible_item"] = True
    value["box_correct_sale_item"] = False
    assert selected_view(value) == "whole"
