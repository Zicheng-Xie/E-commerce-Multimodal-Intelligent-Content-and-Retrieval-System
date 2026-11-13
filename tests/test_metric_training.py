"""Critical training isolation and real human-label handling checks."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from multimodal_ecommerce.metric_training import (
    METRIC_POLICY,
    metric_transform,
    train_metric,
)
from multimodal_ecommerce.precision_labels import (
    human_training_entries,
    validate_precision_labels,
)
from multimodal_ecommerce.rank_correction import RankCorrectionRankCorrectionInputs


def packet() -> list:
    """Return packet for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return [
        {
            "id": str(i),
            "packet_id": "fixed",
            "role": "development_training_only",
            "query_id": "q",
            "candidate_id": f"g{i}",
            "query_image": "q.jpg",
            "candidate_image": f"g{i}.jpg",
            "grade": None,
            "note": "",
            "reviewer": "",
            "review_origin": None,
        }
        for i in range(3)
    ]


def test_pending_relations_never_become_human_negatives() -> None:
    """Verify that pending relations never become human negatives."""
    blank = packet()
    assert validate_precision_labels(blank, copy.deepcopy(blank))["completed"] == 0
    reviewed = copy.deepcopy(blank)
    for i in [0, 1]:
        reviewed[i].update(
            grade=1 - i, reviewer="actual-reviewer", review_origin="human"
        )
    checked = validate_precision_labels(blank, reviewed)
    entry: dict = {
        "query_id": "q",
        "group": "query-group",
        "candidate_ids": ["g0", "g1", "g2"],
        "candidate_groups": ["a", "b", "c"],
        "features": [[1], [2], [3]],
        "teacher": [1, 2, 3],
        "labels": [True, True, False],
    }
    filtered = human_training_entries([entry], checked["completed_rows"])[0]
    assert filtered["candidate_ids"] == ["g0", "g1"]
    assert filtered["labels"] == [True, False]
    assert (
        checked["pending"] == 1 and checked["full_gallery_recall_computable"] is False
    )


@pytest.mark.parametrize(
    "change",
    [
        {"grade": True},
        {"grade": 0.0},
        {"grade": 1},
        {"query_image": "other.jpg"},
        {"review_origin": "model"},
    ],
)
def test_human_import_rejects_wrong_types_unsigned_or_tampered_fields(
    change: object,
) -> None:
    """Verify that human import rejects wrong types unsigned or tampered fields.

    Args:
        change: Change used by this operation.
    """
    blank = packet()
    rows = copy.deepcopy(blank)
    rows[0].update(change)
    with pytest.raises(ValueError):
        validate_precision_labels(blank, rows)


def test_human_import_rejects_duplicate_or_wrong_packet() -> None:
    """Verify that human import rejects duplicate or wrong packet."""
    blank = packet()
    rows = copy.deepcopy(blank)
    rows[1] = rows[0]
    with pytest.raises(ValueError):
        validate_precision_labels(blank, rows)
    rows = copy.deepcopy(blank)
    rows[0]["packet_id"] = "other"
    with pytest.raises(ValueError):
        validate_precision_labels(blank, rows)


def training_problem() -> tuple[list, np.ndarray]:
    """Return training problem for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    records = []
    for group in range(4):
        for source in ["user", "shop"]:
            records.append(
                {
                    "pair_id": str(group),
                    "identity": str(group),
                    "source": source,
                    "project_role": (
                        "training" if group < 3 else "development_validation"
                    ),
                }
            )
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(8, 5)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return records, vectors


def test_metric_training_ignores_development_vectors_and_identity_labels() -> None:
    """Verify that metric training ignores development vectors and identity labels."""
    records, vectors = training_problem()
    policy: dict = {**METRIC_POLICY, "steps": 3}
    first = train_metric(vectors, records, policy)
    vectors[6:] = 1000
    records[6]["identity"] = "changed"
    second = train_metric(vectors, records, policy)
    assert first["weights"] == second["weights"]
    assert set(first["training_query_groups"]) == {"0", "1", "2"}
    assert first["excluded_development_groups"] == ["3"]
    transformed = metric_transform(vectors[:6], np.array(first["weights"]))
    assert np.allclose(np.linalg.norm(transformed, axis=1), 1)


def test_metric_training_rejects_group_leakage() -> None:
    """Verify that metric training rejects group leakage."""
    records, vectors = training_problem()
    records[-1]["pair_id"] = "0"
    with pytest.raises(ValueError, match="crosses"):
        train_metric(vectors, records, {**METRIC_POLICY, "steps": 1})


def test_trained_attention_and_query_adapter_do_not_fit_development_vectors() -> None:
    """Verify that trained attention and query adapter do not fit development
    vectors.
    """
    from multimodal_ecommerce.trained_attention import (
        ATTENTION_POLICY,
        adapt_queries,
        fit_attention,
    )

    records, global_vectors = training_problem()
    vectors = np.repeat(global_vectors[:, None], 3, axis=1)
    mask = np.ones((len(records), 3), dtype=bool)
    policy: dict = {**ATTENTION_POLICY, "steps": 3}
    first = fit_attention(vectors, mask, records, policy)
    vectors[6:] = 200
    records[6]["identity"] = "changed"
    second = fit_attention(vectors, mask, records, policy)
    assert first == second
    assert 0.02 <= first["temperature"] <= 0.15
    assert 0.2 <= first["global_weight"] <= 0.8
    assert np.all(
        (np.array(first["channel_weights"]) >= 0.5)
        & (np.array(first["channel_weights"]) <= 2)
    )
    assert np.allclose(np.linalg.norm(adapt_queries(global_vectors, first), axis=1), 1)


def test_masked_attention_ignores_padding_and_matches_original_with_one_view() -> None:
    """Verify that masked attention ignores padding and matches original with one
    view.
    """
    from multimodal_ecommerce.trained_attention import attention_scores

    q = np.eye(3, dtype=np.float32)
    gallery = np.repeat(q[:, None], 3, axis=1)
    mask = np.zeros((3, 3), dtype=bool)
    mask[:, 0] = True
    first = attention_scores(q, gallery, mask)
    gallery[:, 1:] = 1000
    assert np.allclose(first, attention_scores(q, gallery, mask))
    assert np.allclose(first, q @ q.T)


def test_rank_correction_keeps_anchors_gallery_and_tail() -> None:
    """Verify that rank correction keeps anchors gallery and tail."""
    from multimodal_ecommerce.rank_correction import rank_correction

    base = ["a", "b", "c", "d", "e"]
    adapted = ["a", "d", "c", "b", "e"]
    result = rank_correction(
        RankCorrectionRankCorrectionInputs(
            base, base, adapted, weight=1, anchors=1, pool_size=4
        )
    )
    assert result == ["a", "d", "c", "b", "e"]
    assert (
        rank_correction(
            RankCorrectionRankCorrectionInputs(
                base, adapted, adapted, anchors=2, pool_size=4
            )
        )
        == base
    )
    with pytest.raises(ValueError):
        rank_correction(RankCorrectionRankCorrectionInputs(base, base, adapted[:-1]))
