"""Critical scoring and candidate-boundary checks, not quality measurements."""

from __future__ import annotations

import numpy as np

from multimodal_ecommerce.fgclip_retrieval import (
    FgclipRetrievalAuxiliaryScoresInputs,
    FgclipRetrievalFusePoolInputs,
    auxiliary_scores,
    fuse_pool,
)
from multimodal_ecommerce.intent_retrieval import normalized_ranks, rerank_complete


def vector(x: float, y: int = 0) -> np.ndarray:
    """Return vector for this workflow.

    Args:
        x: X used by this operation.
        y: Y used by this operation.

    Returns:
        Computed array in the order of the supplied records.
    """
    out = np.zeros(768, dtype=np.float32)
    out[:2] = [x, y]
    return out / np.linalg.norm(out)


def test_local_support_requires_more_than_one_accidental_patch() -> None:
    """Verify that local support requires more than one accidental patch."""
    query = vector(1)
    isolated = np.stack([query, *[vector(0, 1) for _ in range(20)]])
    supported = np.stack([vector(0.7, np.sqrt(0.51)) for _ in range(21)])
    result = auxiliary_scores(
        FgclipRetrievalAuxiliaryScoresInputs(
            ["isolated", "supported"],
            np.stack([query, query]),
            np.stack([query, query]),
            [isolated, supported],
            query_image=None,
            query_text=query,
            query_box=query,
        )
    )
    assert result["local_scores"][1] > result["local_scores"][0]


def test_no_description_abstains_from_local_and_caption_votes() -> None:
    """Verify that no description abstains from local and caption votes."""
    ids = ["a", "b"]
    images = np.stack([vector(1), vector(0, 1)])
    common: dict = dict(
        ids=ids,
        gallery_images=images,
        gallery_titles=images[::-1],
        query_image=vector(1),
        query_text=None,
        query_box=None,
    )
    first = auxiliary_scores(
        FgclipRetrievalAuxiliaryScoresInputs(gallery_dense=[images, images], **common)
    )
    second = auxiliary_scores(
        FgclipRetrievalAuxiliaryScoresInputs(
            gallery_dense=[images[::-1], images[::-1]], **common
        )
    )
    assert first == second
    assert not first["local_active"]
    np.testing.assert_array_equal(
        first["fused_scores"], normalized_ranks(ids, np.array(first["global_scores"]))
    )


def test_new_vote_changes_only_the_same_twenty_candidates() -> None:
    """Verify that new vote changes only the same twenty candidates."""
    ids = [f"item{i:02}" for i in range(25)]
    pool = ids[:20]
    base = np.arange(25, 0, -1, dtype=float)
    logits = np.arange(20, 0, -1, dtype=float)
    aux = np.zeros(25)
    aux[0] = -100
    aux[1] = 100
    aux[-1] = 1e6
    baseline = rerank_complete(ids, base, pool, logits)
    actual = fuse_pool(
        FgclipRetrievalFusePoolInputs(ids, base, pool, logits, aux, None)
    )
    assert actual != baseline
    assert actual.index(ids[1]) < baseline.index(ids[1])
    assert actual[20:] == baseline[20:]
    assert set(actual[:20]) == set(pool)
    assert len(actual) == len(set(actual)) == len(ids)
