"""Same complete public feature pipeline works on previously unseen opaque IDs."""

from __future__ import annotations

import numpy as np

from grounded_blind_dataset import classify
from grounded_blind_runtime import (
    GroundedBlindRuntimeFeatureValuesInputs,
    feature_values,
)
from multimodal_ecommerce.regional_retrieval import (
    RegionalIndex,
    region_configuration,
)


def test_blind_sampler_does_not_turn_known_accessory_words_into_main_products() -> None:
    """Verify that blind sampler does not turn known accessory words into main
    products.
    """
    assert classify("头戴耳机耳罩套配件") is None
    assert classify("笔记本电脑内胆包") == "laptop_sleeve"
    assert classify("托特包内胆包") == "organizer"
    assert classify("羽绒马甲") == "down_vest"
    assert classify("手机壳滴胶材料") is None


def test_public_feature_pipeline_handles_new_gallery_size_and_text_without_labels() -> (
    None
):
    """Verify that public feature pipeline handles new gallery size and text without
    labels.
    """
    rng = np.random.default_rng(15)
    n = 24
    dim = 2048
    records = [
        dict(
            id=f"opaque{i}",
            title="手机壳" if i % 2 else "头戴耳机",
            views=[dict(kind="global")],
        )
        for i in range(n)
    ]
    vectors = rng.normal(size=(n, 1, dim)).astype(np.float32)
    titles = rng.normal(size=(n, dim))
    profiles = rng.normal(size=(n, dim))
    q = rng.normal(size=(1, dim))
    q /= np.linalg.norm(q)
    index = RegionalIndex(records, vectors, region_configuration())
    result = feature_values(
        GroundedBlindRuntimeFeatureValuesInputs(
            index,
            titles,
            profiles,
            ["商品主体"] * n,
            np.zeros((n, 48)),
            dict(id="newtext", text="手机壳", image_path=None),
            q,
            np.zeros(dim),
            "",
        )
    )
    assert result["ids"] == [d["id"] for d in records] and len(result["pool"]) == 20
    assert len(result["features"]) == n and len(result["features"][0]) == 13
    assert len(result["offsets"]) == n and result["description"] == ""
