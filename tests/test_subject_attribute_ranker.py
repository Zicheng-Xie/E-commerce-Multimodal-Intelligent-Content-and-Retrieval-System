"""Public feature abstention and family-masked full-evidence fusion invariants."""

from __future__ import annotations

import numpy as np
import pytest

from learned_fusion_model import FEATURE_NAMES as ORIGINAL_FEATURES
from learned_fusion_model import fit as original_fit
from subject_attribute_experiment import new_family_groups
from subject_attribute_ranker import (
    ATTRIBUTE_NAMES,
    FEATURE_NAMES,
    attribute_vector,
    fit,
    public_descriptor,
)


def public(product: str, text: str = "", unclear: bool = False) -> dict:
    """Return public for this workflow.

    Args:
        product: Product used by this operation.
        text: Text input used by the operation.
        unclear: Unclear used by this operation.

    Returns:
        Mapping with fields: product, appearance, visible_text, primary_unclear.
    """
    return dict(
        product=product, appearance="", visible_text=text, primary_unclear=unclear
    )


def test_explicit_model_conflict_is_soft_and_missing_model_abstains() -> None:
    """Verify that explicit model conflict is soft and missing model abstains."""
    original = np.zeros(20)
    q = public("手机壳", "华为 Mate10 裸机手感")
    d = public("手机壳", "华为Mate20Pro手机壳")
    features, evidence = attribute_vector(original, q, d)
    assert evidence["query"]["phone_models"] == ["huawei:mate10"]
    assert features[ATTRIBUTE_NAMES.index("phone_model_conflict")] == 1
    features, _ = attribute_vector(original, public("手机壳"), d)
    assert all(
        features[ATTRIBUTE_NAMES.index(name)] == 0
        for name in ("phone_models_known", "phone_model_match", "phone_model_conflict")
    )
    with pytest.raises(ValueError):
        attribute_vector(original, dict(q, grade=1), d)


def test_sale_item_is_distinguished_from_material_laptop_sleeve_and_scene_bag() -> None:
    """Verify that sale item is distinguished from material laptop sleeve and scene
    bag.
    """
    q = public_descriptor(public("内胆包", "LV speedy nano枕头包 专属内胆包"))
    assert q["family"] == "bag" and q["subtype"] == "organizer"
    laptop = public_descriptor(public("内胆包", "联想笔记本电脑内胆包"))
    assert laptop["subtype"] == "laptop"
    material = public_descriptor(public("手机壳", "闪粉手机壳水晶滴胶材料DIY"))
    assert material["role"] == "material" and material["family"] == "craft_material"
    unknown = public_descriptor(public("手机壳", "华为mate10", True))
    assert unknown["family"] is None and unknown["phone_models"] == []


def test_refit_recipe_matches_original_ranknet_and_ignores_held_or_null_cells() -> None:
    """Verify that refit recipe matches original ranknet and ignores held or null
    cells.
    """
    rng = np.random.default_rng(17)
    x = rng.normal(size=(4, 6, 20))
    labels = np.tile([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], (4, 1))
    labels[2, 4] = np.nan
    allowed = np.ones((4, 6), bool)
    allowed[0] = False
    allowed[:, 0] = False
    base = rng.normal(size=(4, 6))
    expected = original_fit(x, labels, allowed, base, pairwise=True)
    actual = fit(x, labels, allowed, base, names=ORIGINAL_FEATURES)
    for key in ("mean", "scale", "weights", "score_location", "score_scale"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-14)
    poisoned_x = x.copy()
    poisoned_y = labels.copy()
    poisoned_base = base.copy()
    excluded = ~allowed | ~np.isfinite(labels)
    poisoned_x[excluded] = 1e9
    poisoned_base[excluded] = -1e9
    poisoned_y[~allowed] = 1 - labels[~allowed]
    poisoned = fit(
        poisoned_x, poisoned_y, allowed, poisoned_base, names=ORIGINAL_FEATURES
    )
    assert actual == poisoned
    extended = np.concatenate(
        [x, rng.normal(size=(4, 6, len(FEATURE_NAMES) - 20))], axis=-1
    )
    structured = fit(extended, labels, allowed, base, names=FEATURE_NAMES)
    assert len(structured["weights"]) == len(FEATURE_NAMES)


def test_new_source_family_links_are_transitive_before_fold_assignment() -> None:
    """Verify that new source family links are transitive before fold assignment."""
    rows: list[dict] = [
        dict(id="a", family_keys=["model:x"]),
        dict(id="b", family_keys=["model:x", "title:y"]),
        dict(id="c", family_keys=["title:y"]),
        dict(id="d", family_keys=["model:z"]),
    ]
    assert new_family_groups(rows) == dict(a="a", b="a", c="a", d="d")
