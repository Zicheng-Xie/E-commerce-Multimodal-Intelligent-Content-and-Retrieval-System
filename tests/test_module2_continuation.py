"""Critical resident input validation, consensus integrity and label-origin flow."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from multimodal_ecommerce.human_business_pipeline import validate_consensus
from multimodal_ecommerce.resident_retrieval import validate_search
from multimodal_ecommerce.semantic_ranker import FEATURE_NAMES
from multimodal_ecommerce.supervised_pipeline import (
    SupervisedPipelineFitSupervisedBundleInputs,
    fit_supervised_bundle,
)


@pytest.mark.parametrize(
    "query",
    [
        {"text": "x", "image_id": "a"},
        {"text": "x", "category": "answer"},
        {"text": "x", "top_k": True},
        {"image_id": "a", "roi": [0, 0, 2, 1]},
        {"text": "x", "roi": [0, 0, 1, 1]},
        {"image_id": "a", "roi": [False, 0, 1, 1]},
    ],
)
def test_resident_rejects_ambiguous_queries_labels_and_invalid_roi(query: dict) -> None:
    """Verify that resident rejects ambiguous queries labels and invalid roi.

    Args:
        query: Query record or representation used by this operation.
    """
    with pytest.raises(ValueError):
        validate_search(query)


def test_resident_accepts_fixed_default_top10_and_valid_roi() -> None:
    """Verify that resident accepts fixed default top10 and valid roi."""
    assert validate_search({"text": "商品"})["top_k"] == 10
    assert validate_search({"image_id": "a", "roi": [0.1, 0.1, 0.9, 0.9]})["roi"] == [
        0.1,
        0.1,
        0.9,
        0.9,
    ]


def test_no_human_labels_never_train_or_generate_metrics() -> None:
    """Verify that no human labels never train or generate metrics."""
    result = fit_supervised_bundle(
        SupervisedPipelineFitSupervisedBundleInputs(
            np.zeros((1, 1, 13)),
            np.full((1, 1), np.nan),
            np.ones((1, 1), dtype=bool),
            np.zeros(1),
            np.zeros(1),
            np.zeros((1, 1)),
            {},
            {},
            "declared_human_business_development",
        )
    )
    assert result["trained"] is False and "models" not in result


def test_complete_group_training_keeps_real_origin_without_claiming_validation() -> (
    None
):
    """Verify that complete group training keeps real origin without claiming
    validation.
    """
    rng = np.random.default_rng(42)
    x = rng.normal(size=(4, 6, len(FEATURE_NAMES)))
    labels = np.tile([0, 1, 0, 1, 0, 1], (4, 1)).astype(float)
    valid = np.ones((4, 6), dtype=bool)
    metadata: dict = dict(
        query_ids=["q0", "q1", "q2", "q3"],
        query_families=[None] * 4,
        gallery_families=[f"g{i}" for i in range(6)],
    )
    result = fit_supervised_bundle(
        SupervisedPipelineFitSupervisedBundleInputs(
            x,
            labels,
            valid,
            np.array([0, 0, 1, 2]),
            np.array([0, 1, 2, 0, 1, 2]),
            rng.random((4, 6)),
            metadata,
            dict(steps=5, l2=0.05, learning_rate=0.1),
            "declared_human_business_development",
        )
    )
    assert result["trained"] and len(result["models"]) == 18
    assert all(
        m["label_origin"] == "declared_human_business_development"
        and m["human_quality_validated"] is False
        for m in result["candidates"].values()
    )


def test_human_consensus_rejects_injected_single_person_judgment() -> None:
    """Verify that human consensus rejects injected single person judgment."""
    row: dict = dict(
        id="a",
        packet_id="r",
        role="development_training_only",
        query_id="q",
        candidate_id="c",
        query_image="",
        query_text="需求",
        query_rule="用途相符",
        candidate_image="c.jpg",
        candidate_title="商品",
        grade=None,
        note="",
        reviewer="",
        review_origin=None,
    )
    manifest: dict = dict(release_id="r", batches=[dict(id="b", row_ids=["a"])])
    inputs: dict = dict(primary=[], adjudications=[])
    assert (
        validate_consensus(manifest, [row], inputs, [row])["status"]["completed"] == 0
    )
    changed = copy.deepcopy(row)
    changed.update(grade=1, reviewer="invented", review_origin="human")
    with pytest.raises(ValueError, match="recomputed consensus"):
        validate_consensus(manifest, [row], inputs, [changed])


def test_webp_media_response_is_image_even_without_windows_mime_registration(
    tmp_path: Path,
) -> None:
    """Verify that webp media response is image even without windows mime
    registration.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.

    Notes:
        Writes derived artifacts through picture.write_bytes.
    """
    import threading

    import httpx

    from multimodal_ecommerce.retrieval_http import RetrievalHTTPServer

    picture = tmp_path / "sample.webp"
    picture.write_bytes(b"RIFF-test-WEBP")

    class Engine:
        """Supply a minimal resident engine for HTTP media and request-contract
        tests.
        """

        assets: dict = {"asset": picture}

    # Media-only fixture; inference methods must never run in this scenario.
    server = RetrievalHTTPServer(("127.0.0.1", 0), Engine())  # type: ignore[arg-type]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(trust_env=False) as client:
            response = client.get(f"http://127.0.0.1:{server.server_port}/media/asset")
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "image/webp"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
