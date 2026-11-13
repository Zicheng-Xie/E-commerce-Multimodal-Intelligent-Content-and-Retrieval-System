"""Input preservation and ranking-contract checks for the precision work."""

from __future__ import annotations

from types import SimpleNamespace
from typing import NoReturn

import numpy as np
import pytest

from multimodal_ecommerce.local_visual import select_descriptors
from multimodal_ecommerce.pair_reranker import strict_tokenize
from multimodal_ecommerce.precision_boundary import boundary_refine
from multimodal_ecommerce.precision_reranking import (
    fit_pairwise,
    pairwise_scores,
    pareto_gate,
    refine_pool,
)


def test_pair_vision_failure_is_visible() -> None:
    """Verify that pair vision failure is visible."""
    processor = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "chat")
    pairs = [[{"content": [{"type": "image"}, {"type": "image"}]}]]

    def broken(*args: object, **kwargs: object) -> NoReturn:
        """Apply the broken workflow step.

        Args:
            *args: Extra arguments accepted to match the patched dependency.
            **kwargs: Extra arguments accepted to match the patched dependency.

        Raises:
            OSError: damaged source image.
        """
        raise OSError("damaged source image")

    with pytest.raises(OSError, match="damaged"):
        strict_tokenize(processor, pairs, broken, 10240)
    with pytest.raises(ValueError, match="lost vision"):
        strict_tokenize(
            processor, pairs, lambda *args, **kwargs: ([object()], None, {}), 10240
        )


@pytest.mark.parametrize(
    "tokens,grids,error", [(10241, 2, "token limit"), (20, 1, "misaligned")]
)
def test_no_silent_pair_truncation_or_missing_image(
    tokens: int, grids: int, error: str
) -> None:
    """Verify that no silent pair truncation or missing image.

    Args:
        tokens: Tokens used by this operation.
        grids: Grids used by this operation.
        error: Error used by this operation.
    """

    class Processor:
        """Build controlled multimodal tensors for token and image-alignment tests."""

        def apply_chat_template(self, *args: object, **kwargs: object) -> str:
            """Return apply chat template for this workflow.

            Args:
                *args: Extra arguments accepted to match the patched dependency.
                **kwargs: Extra arguments accepted to match the patched dependency.

            Returns:
                Text produced by this operation.
            """
            return "chat"

        def __call__(self, **kwargs):
            assert kwargs["truncation"] is False
            return {
                "input_ids": np.zeros((1, tokens)),
                "pixel_values": np.zeros((2, 4)),
                "image_grid_thw": np.zeros((grids, 3)),
            }

    pairs = [[{"content": [{"type": "image"}, {"type": "image"}]}]]
    with pytest.raises(ValueError, match=error):
        strict_tokenize(
            Processor(), pairs, lambda *a, **k: ([object(), object()], None, {}), 10240
        )


def test_rerank_more_than_ten_and_keep_complete_tail_and_anchors() -> None:
    """Verify that rerank more than ten and keep complete tail and anchors."""
    base = [str(i) for i in range(20)]
    pool = base[:15]
    result = refine_pool(base, pool, np.arange(15, dtype=float), anchors=2)
    assert result[:2] == base[:2]
    assert result[15:] == base[15:]
    assert set(result) == set(base) and len(result) == len(base)
    positives = {"14"}
    assert sum(x in positives for x in result[:10]) / 10 == 0.1
    assert sum(x in positives for x in base[:10]) / 10 == 0
    with pytest.raises(ValueError, match="prefix"):
        refine_pool(base, list(reversed(pool)), np.arange(15))


def test_group_cv_excludes_query_and_gallery_validation_labels() -> None:
    """Verify that group cv excludes query and gallery validation labels."""
    entries: list[dict] = [
        {
            "group": "train",
            "candidate_groups": ["train", "train", "validation"],
            "features": [
                [1.0, 0.0, 1.0, 0.9, 1.0],
                [-1.0, 0.0, -1.0, 0.1, 0.0],
                [10000.0, 0.0, 0.0, 0.0, 0.0],
            ],
            "labels": [True, False, True],
            "teacher": [2.0, -2.0, 100.0],
        },
        {
            "group": "validation",
            "candidate_groups": ["validation", "train"],
            "features": [
                [500.0, 500.0, 500.0, 500.0, 500.0],
                [-500.0, -500.0, -500.0, -500.0, -500.0],
            ],
            "labels": [False, True],
            "teacher": [50.0, -50.0],
        },
    ]
    policy: dict = {
        "hard_negative_limit": 8,
        "distillation_temperature": 2.0,
        "distillation_mix": 0.5,
        "pairwise_steps": 500,
        "pairwise_l2": 0.1,
        "pairwise_lr": 0.1,
    }
    first = fit_pairwise(entries, policy, excluded_groups={"validation"}, distill=True)
    assert first["training_query_groups"] == ["train"]
    assert first["training_gallery_groups"] == ["train"]
    assert first["mean"][0] == 0
    entries[0]["labels"][2] = False
    entries[0]["features"][2] = [-90000.0] * 5
    entries[1]["features"] = [[0.0] * 5, [0.0] * 5]
    entries[1]["labels"] = [True, False]
    second = fit_pairwise(entries, policy, excluded_groups={"validation"}, distill=True)
    assert first == second
    assert pairwise_scores(entries[0], first)[0] > pairwise_scores(entries[0], first)[1]


def test_precision_gate_rejects_improvement_with_other_metric_regression() -> None:
    """Verify that precision gate rejects improvement with other metric regression."""
    base: dict = {key: 0.8 for key in ("recall@10", "ap", "ndcg@10", "mrr", "hit@1")}
    base["precision@10"] = 0.4
    candidate: dict = {**base, "precision@10": 0.44, "ap": 0.79}
    failed = pareto_gate(candidate, base)
    assert not failed["passes_development_metric_gate"]
    assert failed["regressed_metrics"] == ["ap"]
    candidate["ap"] = 0.81
    passed = pareto_gate(candidate, base)
    assert passed["passes_development_metric_gate"]
    assert not passed["formal_acceptance"]


def test_boundary_uses_only_scored_window_and_keeps_all_other_ranks() -> None:
    """Verify that boundary uses only scored window and keeps all other ranks."""
    base = [str(i) for i in range(30)]
    result = boundary_refine(
        base, np.arange(10, dtype=float), anchors=5, count=10, blend=1.0
    )
    assert result[:5] == base[:5]
    assert result[15:] == base[15:]
    assert set(result) == set(base)
    assert result[5:15] == list(reversed(base[5:15]))
    with pytest.raises(ValueError, match="window logits"):
        boundary_refine(base, np.ones(30), anchors=5, count=10)


def test_attention_patch_selection_keeps_alignment_and_unit_norm() -> None:
    """Verify that attention patch selection keeps alignment and unit norm."""
    tokens = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    selected, indices = select_descriptors(tokens, np.array([0.1, 0.8, 0.2]), 2)
    assert indices.tolist() == [1, 2]
    np.testing.assert_allclose(np.linalg.norm(selected, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(selected[0], [0.0, 1.0])


def test_precision_cli_rejects_consumed_blind_index() -> None:
    """Verify that precision cli rejects consumed blind index."""
    import subprocess
    import sys
    import uuid
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    output = root / "tmp" / ("precision_guard_" + uuid.uuid4().hex)
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/retrieve_precision.py"),
            "--index",
            "artifacts/module2_blind/predictions/index",
            "--cached-query-number",
            "0",
            "--mode",
            "boundary",
            "--output",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode != 0
    assert "Consumed blind assets" in result.stderr
    assert not output.exists()
