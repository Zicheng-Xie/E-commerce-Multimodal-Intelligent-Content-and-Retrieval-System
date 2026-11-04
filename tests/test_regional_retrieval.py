"""Verify attention math, view safety, metadata evidence, and immutable banks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from multimodal_ecommerce import regional_retrieval as regions
from multimodal_ecommerce import safety
from multimodal_ecommerce.regional_retrieval import (
    RegionalRetrievalEncodeViewsInputs,
    RegionalSearchRequest,
)


def make_index() -> regions.RegionalIndex:
    """Return make index for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    records: list[dict] = [
        {
            "id": "a",
            "category": "服饰",
            "attributes": {"颜色": ["黑色", "白色"]},
            "views": [{"kind": "global"}, {"kind": "center"}],
        },
        {
            "id": "b",
            "category": "家居",
            "attributes": {"颜色": ["蓝色"]},
            "views": [{"kind": "global"}],
        },
    ]
    return regions.RegionalIndex(
        records,
        np.array([[[0.6, 0.8], [1, 0]], [[0.7, np.sqrt(0.51)], [0, 0]]]),
        regions.region_configuration(),
    )


def test_attention_masks_padding_and_matches_qkv_formula() -> None:
    """Verify that attention masks padding and matches qkv formula."""
    index = make_index()
    results = index.search(RegionalSearchRequest(np.array([1, 0]), 2))
    row = next(r for r in results if r["id"] == "a")
    sims = np.array([0.6, 1.0])
    weights = np.exp((sims - sims.max()) / index.config["attention_temperature"])
    weights /= weights.sum()
    expected = 0.6 * 0.6 + 0.4 * np.dot(weights, sims)
    assert row["score"] == pytest.approx(expected)
    assert row["region_weights"][0] == pytest.approx(weights)
    assert results[0]["id"] == "a"
    assert next(r for r in results if r["id"] == "b")["region_weights"] == [[1.0]]
    assert (
        index.search(RegionalSearchRequest(np.array([1, 0]), 2, strategy="global"))[0][
            "id"
        ]
        == "b"
    )
    assert (
        index.search(RegionalSearchRequest(np.array([1, 0]), exclude_ids={"a"}))[0][
            "id"
        ]
        == "b"
    )


def test_metadata_is_explicit_and_unknown_is_neutral() -> None:
    """Verify that metadata is explicit and unknown is neutral."""
    index = make_index()
    results = index.search(
        RegionalSearchRequest(
            np.array([1, 0]),
            2,
            strategy="attention_fused",
            category="服饰",
            attributes={"颜色": "黑色", "材质": "棉"},
        )
    )
    row = next(r for r in results if r["id"] == "a")
    assert row["attribute_evidence"] == {
        "matched": ["颜色"],
        "conflicts": [],
        "unknown": ["材质"],
    }
    assert row["metadata_adjustment"] == pytest.approx(0.055)
    assert next(r for r in results if r["id"] == "b")[
        "metadata_adjustment"
    ] == pytest.approx(-0.055)
    plain = index.search(RegionalSearchRequest(np.array([1, 0]), strategy="attention"))
    preference = index.search(
        RegionalSearchRequest(
            np.array([1, 0]),
            strategy="attention",
            category="家居",
            attributes={"颜色": "蓝色"},
        )
    )
    assert [r["score"] for r in plain] == [r["score"] for r in preference]
    with pytest.raises(ValueError):
        index.search(
            RegionalSearchRequest(np.array([1, 0]), attributes={"颜色": ["黑色"]})  # type: ignore[dict-item]  # Exercise rejection of a non-string attribute.
        )


def test_mean_and_text_fusion_are_independently_switchable() -> None:
    """Verify that mean and text fusion are independently switchable."""
    index = make_index()
    row = next(
        r
        for r in index.search(
            RegionalSearchRequest(np.array([1, 0]), 2, strategy="mean")
        )
        if r["id"] == "a"
    )
    assert row["score"] == pytest.approx(0.6 * 0.6 + 0.4 * 0.8)
    text = np.array([[0, 1], [1, 0]])
    fused = index.search(
        RegionalSearchRequest(
            np.array([1, 0]),
            2,
            strategy="attention_fused",
            text_vector=np.array([1, 0]),
            text_bank=text,
        )
    )
    assert fused[0]["id"] == "b"
    with pytest.raises(ValueError, match="text bank"):
        index.search(
            RegionalSearchRequest(
                np.array([1, 0]),
                strategy="attention_fused",
                text_vector=np.array([1, 0]),
            )
        )


def test_proposals_detect_simple_foreground_and_uniform_fallback() -> None:
    """Verify that proposals detect simple foreground and uniform fallback."""
    config = regions.region_configuration()
    picture = Image.new("RGB", (200, 160), "white")
    ImageDraw.Draw(picture).rectangle((70, 40, 130, 120), fill="black")
    views = regions.propose_views(picture, config)
    roi = next(v for v in views if v["kind"] == "contrast_roi")["box_xyxy"]
    assert roi[0] <= 70 and roi[1] <= 40 and roi[2] >= 130 and roi[3] >= 120
    assert len(views) <= 7 and views[0]["kind"] == "global"
    assert len({tuple(v["box_xyxy"]) for v in views}) == len(views)
    assert all(
        v["kind"] != "contrast_roi"
        for v in regions.propose_views(Image.new("RGB", (100, 100), "white"), config)
    )


@pytest.mark.parametrize(
    "box",
    [
        [0.7, 0, 0.2, 1],
        [-0.1, 0, 1, 1],
        [0, 0, 2, 1],
        [0, 0, float("nan"), 1],
        [0, 0, 0.001, 0.001],
    ],
)
def test_invalid_roi_rejected(box: object) -> None:
    """Verify that invalid roi rejected.

    Args:
        box: Box used by this operation.
    """
    with pytest.raises(ValueError):
        regions.pixel_roi(box, (100, 100))  # type: ignore[arg-type]  # Intentionally invalid ROI input.


def test_region_encoding_exif_origin_and_global_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that region encoding exif origin and global preserved.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.

    Notes:
        Writes derived artifacts through Image.new('RGB', (120, 80), 'white').save.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    image = tmp_path / "input.png"
    Image.new("RGB", (120, 80), "white").save(image)
    calls = []

    class Encoder:
        """Supply deterministic vectors for the isolated retrieval scenario under
        test.
        """

        def encode(self, *, image: object, query: dict) -> np.ndarray:
            """Encode the configured inputs into model feature representations.

            Args:
                image: Image input used by the operation.
                query: Query record or representation used by this operation.

            Returns:
                Computed array in the order of the supplied records.
            """
            calls.append(image)
            return np.array([1, 0])

    _, views = regions.encode_views(
        RegionalRetrievalEncodeViewsInputs(
            Encoder(), image, tmp_path / "views", regions.region_configuration()
        )
    )
    assert calls[0] == image
    assert views[1]["encoded_size"] == [768, 512]
    _, roi_views = regions.encode_views(
        RegionalRetrievalEncodeViewsInputs(
            Encoder(),
            image,
            tmp_path / "roi",
            regions.region_configuration(),
            roi=[0.25, 0.25, 0.75, 0.75],
        )
    )
    assert roi_views[0]["original_box_xyxy"] == [30, 20, 90, 60]
    with pytest.raises(FileExistsError):
        regions.encode_views(
            RegionalRetrievalEncodeViewsInputs(
                Encoder(), image, tmp_path / "views", regions.region_configuration()
            )
        )


def test_bank_tampering_and_untrusted_html(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that bank tampering and untrusted html.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.

    Notes:
        Writes derived artifacts through Image.new('RGB', (32, 32), 'white').save,
        crop.write_bytes, dataset.write_text.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    config = regions.region_configuration()
    monkeypatch.setattr(regions, "region_configuration", lambda: config)
    monkeypatch.setattr(regions, "model_configuration", lambda: {})
    picture = tmp_path / "source.png"
    Image.new("RGB", (32, 32), "white").save(picture)
    dataset = tmp_path / "products.jsonl"
    import json

    dataset.write_text(
        json.dumps(
            {
                "id": "a",
                "evaluation_eligible": True,
                "image_path": "source.png",
                "image_sha256": safety.sha256_file(picture),
                "text": "<script>alert(1)</script>",
            }
        )
        + "\n"
    )

    class Encoder:
        """Supply deterministic vectors for the isolated retrieval scenario under
        test.
        """

        config: dict = {}

        def encode(self, **kwargs: object) -> np.ndarray:
            """Encode the configured inputs into model feature representations.

            Args:
                **kwargs: Extra arguments accepted to match the patched dependency.

            Returns:
                Computed array in the order of the supplied records.
            """
            return np.array([1, 0])

        def diagnostics(self) -> dict:
            """Describe the active backend and recorded runtime measurements.

            Returns:
                Result mapping produced by the workflow.
            """
            return {}

    output = tmp_path / "index"
    regions.build_regional_index(Encoder(), dataset, output)
    index, _, _ = regions.RegionalIndex.load(output)
    results = index.search(RegionalSearchRequest(np.array([1, 0])))
    regions.write_regional_gallery(
        tmp_path / "results.html",
        [{"query": "<q>", "strategy": "attention", "results": results}],
    )
    markup = (tmp_path / "results.html").read_text(encoding="utf-8")
    assert "<script>" not in markup and "&lt;script&gt;" in markup
    crop = Path(index.records[0]["views"][0]["image_path"])
    crop.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        regions.RegionalIndex.load(output)
