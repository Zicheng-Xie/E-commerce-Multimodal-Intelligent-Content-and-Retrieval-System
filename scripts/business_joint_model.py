"""STIR-inspired business adapter on full frozen DINOv2 joint-image attention.

This is a separately specified adaptation, not the STIR paper checkpoint or
its end-to-end training recipe. Both images interact inside all 12 frozen
visual Transformer layers; only the complete business head is fitted.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TrainingPolicy(TypedDict):
    """Frozen optimizer, representation and provenance fields for this model."""

    backbone: str
    backbone_layers: int
    original_image_processing: str
    joint_input: str
    joint_pixels: list[int]
    all_patch_tokens: int
    representation: str
    backbone_trainable: bool
    business_head: str
    steps: int
    learning_rate: float
    weight_decay: float
    seed: int
    head_device: str
    threads: int
    full_backbone_dtype: str
    parent_rank_weight: float
    auxiliary_rank_weight: float
    masks: str
    upstream_idea: str
    attribution: str


POLICY: TrainingPolicy = dict(
    backbone="complete pinned facebook/dinov2-small",
    backbone_layers=12,
    original_image_processing="official existing 256-short-edge/224-center-crop RGB normalization per image",
    joint_input="query above candidate; concatenate pixel tensors vertically before the full backbone",
    joint_pixels=[3, 448, 224],
    all_patch_tokens=512,
    representation="joint CLS, 384 dimensions; no patch selection",
    backbone_trainable=False,
    business_head="LayerNorm384, Linear384->192, GELU, Linear192->1",
    steps=400,
    learning_rate=0.001,
    weight_decay=0.0001,
    seed=2026102506,
    head_device="cpu",
    threads=4,
    full_backbone_dtype="float32",
    parent_rank_weight=0.8,
    auxiliary_rank_weight=0.2,
    masks="disabled globally because candidate/query-only semantic quality assessment failed",
    upstream_idea="https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/postprocessing/pairwise_postprocessing",
    attribution="Inspired by concatenated-pixel Siamese Transformer reranking; distinct DINOv2 frozen-backbone business adaptation, not a paper reproduction",
)


class PairHead(nn.Module):
    """Map frozen 384-dimensional joint-image features to a business relevance logit."""

    def __init__(self) -> None:
        """Initialize the component and validate its configured dependencies."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(384), nn.Linear(384, 192), nn.GELU(), nn.Linear(192, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return forward for this workflow.

        Args:
            x: X used by this operation.

        Returns:
            Computed array in the order of the supplied records.
        """
        return self.layers(x).squeeze(-1)


def training_rows(
    labels: np.ndarray, allowed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return training rows for this workflow.

    Args:
        labels: Judgments used for the stated development or evaluation protocol.
        allowed: Allowed used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Known binary business training cells required.
    """
    known = allowed & np.isfinite(labels)
    if not np.isin(labels[known], [0.0, 1.0]).all() or not known.any():
        raise ValueError("Known binary business training cells required")
    weights = np.zeros(labels.shape, dtype=np.float32)
    for q in range(len(labels)):
        for grade in (0.0, 1.0):
            cells = known[q] & (labels[q] == grade)
            if cells.any():
                weights[q, cells] = 0.5 / cells.sum()
    weights /= weights.sum()
    return known, weights[known]


def fit_head(
    features: np.ndarray,
    labels: np.ndarray,
    allowed: np.ndarray,
    valid: np.ndarray,
    seed: int,
) -> tuple[PairHead, np.ndarray, list]:
    """Fit head.

    Args:
        features: Feature array aligned with the candidate or query records.
        labels: Judgments used for the stated development or evaluation protocol.
        allowed: Allowed used by this operation.
        valid: Valid used by this operation.
        seed: Seed used to reproduce randomized ordering.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Missing real pair attention features; Nonfinite joint head loss.
    """
    torch.set_num_threads(POLICY["threads"])
    torch.manual_seed(seed)
    known, weights = training_rows(labels, allowed & valid)
    x = torch.tensor(features[known], dtype=torch.float32)
    if not torch.isfinite(x).all():
        raise ValueError("Missing real pair attention features")
    target = torch.tensor(labels[known], dtype=torch.float32)
    weight = torch.tensor(weights)
    model = PairHead()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=POLICY["learning_rate"],
        weight_decay=POLICY["weight_decay"],
    )
    trace = []
    for step in range(POLICY["steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss = (
            F.binary_cross_entropy_with_logits(model(x), target, reduction="none")
            * weight
        ).sum()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite joint head loss")
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == POLICY["steps"] - 1:
            trace.append({"step": step + 1, "loss": float(loss.detach())})
    model.eval()
    scores = np.full(labels.shape, np.nan, dtype=np.float32)
    with torch.no_grad():
        scores[valid] = model(
            torch.tensor(features[valid], dtype=torch.float32)
        ).numpy()
    return model, scores, trace


def blend(pool: list[str], parent: list[str], aux: dict[str, float]) -> list:
    """Return blend for this workflow.

    Args:
        pool: Pool used by this operation.
        parent: Parent used by this operation.
        aux: Aux used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Same actually Qwen-scored parent pool and full auxiliary evidence
            required.
    """
    if (
        len(pool) != len(set(pool))
        or not set(pool) <= set(parent)
        or set(pool) != set(aux)
    ):
        raise ValueError(
            "Same actually Qwen-scored parent pool and full auxiliary evidence required"
        )
    order = [c for c in parent if c in set(pool)]
    n = max(1, len(pool) - 1)
    pr = {c: 1 - i / n for i, c in enumerate(order)}
    ar = {c: 1 - i / n for i, c in enumerate(sorted(pool, key=lambda c: (-aux[c], c)))}
    return sorted(
        pool, key=lambda c: (-(0.8 * pr[c] + 0.2 * ar[c]), parent.index(c))
    ) + [c for c in parent if c not in set(pool)]
