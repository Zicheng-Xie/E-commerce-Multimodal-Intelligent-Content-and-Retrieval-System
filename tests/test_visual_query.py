"""Test description abstention and the newly connected two-modality data path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from multimodal_ecommerce.intent_retrieval import normalized_ranks, recall_channels
from multimodal_ecommerce.pair_reranker import PairReranker
from multimodal_ecommerce.query_fusion import (
    QueryFusionVisualChannelsInputs,
    visual_channels,
)
from multimodal_ecommerce.visual_query import DescribedImageReranker, caption_text


def test_unclear_description_abstains_and_rejects_extra_evaluator_fields() -> None:
    """Verify that unclear description abstains and rejects extra evaluator fields."""
    assert (
        caption_text(dict(product="主体不明", appearance="轮廓遮挡", unclear=True))
        == ""
    )
    with pytest.raises(ValueError, match="fields"):
        caption_text(dict(product="柜子", appearance="双门", unclear=False, grade=1))


def test_multimodal_cross_receives_both_query_fields_and_blocks_labels() -> None:
    """Verify that multimodal cross receives both query fields and blocks labels."""
    seen: list[dict] = []

    def process(item: dict) -> list[float]:
        """Record the exact backend payload and return a fixed pair score.

        Args:
            item: Multimodal request received by the patched backend.

        Returns:
            One deterministic score for the single candidate.
        """
        seen.append(item)
        return [1.25]

    scorer = object.__new__(DescribedImageReranker)
    # This test replaces the adapter without loading a checkpoint.
    scorer.adapter = cast(
        PairReranker,
        SimpleNamespace(
            picture=lambda path: "pixels:" + str(path),
            runtime=SimpleNamespace(synchronize=lambda: None),
            config=dict(instruction="same fixed instruction"),
            backend=SimpleNamespace(process=process),
            timings=[],
        ),
    )
    documents: list[dict] = [dict(id="doc", image_path="doc.png", text="原始标题")]
    result = scorer.score_described(
        documents, query_image=Path("query.png"), description="双层壁挂瓶架"
    )
    assert result.tolist() == [1.25]
    assert seen[0]["query"] == dict(image="pixels:query.png", text="双层壁挂瓶架")
    assert seen[0]["documents"] == [dict(image="pixels:doc.png", text="原始标题")]
    assert scorer.adapter.timings[0]["query_modalities"] == ["image", "text"]
    with pytest.raises(ValueError, match="evaluator fields"):
        scorer.score_described(
            [dict(documents[0], grade=1)],
            query_image=Path("query.png"),
            description="双层壁挂瓶架",
        )
    assert len(seen) == 1


def test_empty_caption_contributes_neither_tie_broken_ids_nor_a_caption_channel() -> (
    None
):
    """Verify that empty caption contributes neither tie broken IDs nor a caption
    channel.
    """
    ids = ["z", "a", "b"]
    index = SimpleNamespace(
        records=[dict(id=cid) for cid in ids],
        search=lambda vector, **kwargs: [
            dict(id=cid, score=score) for cid, score in zip(ids, [0.2, 0.9, 0.5])
        ],
    )
    features = np.random.default_rng(42).random((3, 13))
    result = visual_channels(
        QueryFusionVisualChannelsInputs(
            index,
            np.ones((3, 4)),
            np.ones((3, 4)),
            features,
            np.ones((2, 4)),
            np.zeros(4),
            "",
            np.arange(3),
        )
    )
    base, _ = recall_channels(ids, features)
    expected = 0.5 * normalized_ranks(ids, base) + 0.25 * normalized_ranks(
        ids, np.array([0.2, 0.9, 0.5])
    )
    assert result["description_channel_active"] is False
    assert result["caption_scores"] == [0, 0, 0]
    assert result["fused_scores"] == pytest.approx(expected)
    assert set(result["pool"]) == set(ids)


def test_nonempty_caption_requires_real_nonzero_vector() -> None:
    """Verify that nonempty caption requires real nonzero vector."""
    index = SimpleNamespace(
        records=[dict(id="a")],
        search=lambda vector, **kwargs: [dict(id="a", score=0.5)],
    )
    with pytest.raises(ValueError, match="nonzero"):
        visual_channels(
            QueryFusionVisualChannelsInputs(
                index,
                np.ones((1, 4)),
                np.ones((1, 4)),
                np.ones((1, 13)),
                np.ones((2, 4)),
                np.zeros(4),
                "柜体",
                np.arange(1),
            )
        )
