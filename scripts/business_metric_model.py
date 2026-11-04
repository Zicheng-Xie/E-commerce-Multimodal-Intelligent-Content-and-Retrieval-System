"""Business-relation Multi-Similarity learning on complete frozen embeddings.

Equations/mining idea: KevinMusgrave/pytorch-metric-learning (MIT),
multi_similarity_loss.py and multi_similarity_miner.py. Unlike identity-label
examples, positive/negative masks here are explicit query-product business grades.
This projection is a new auxiliary component, not a reduced Qwen/STIR model.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TrainingPolicy(TypedDict):
    """Frozen optimizer, representation and provenance fields for this model."""

    input_dim: int
    hidden_dim: int
    output_dim: int
    steps: int
    learning_rate: float
    weight_decay: float
    alpha: float
    beta: float
    base: float
    mining_epsilon: float
    seed: int
    device: str
    threads: int
    backbone: str
    normalization: str
    labels: str
    role: str


POLICY: TrainingPolicy = dict(
    input_dim=2048,
    hidden_dim=256,
    output_dim=256,
    steps=400,
    learning_rate=0.001,
    weight_decay=0.0001,
    alpha=2.0,
    beta=50.0,
    base=0.5,
    mining_epsilon=0.1,
    seed=2026102505,
    device="cpu",
    threads=4,
    backbone="complete frozen original Qwen embeddings, no backbone changes",
    normalization="per-vector LayerNorm/L2 only; no training statistics from held-out cells",
    labels="known explicit business pairs, never different-SKU negatives",
    role="auxiliary metric component; not STIR or final retrieval system",
)


class Projection(nn.Module):
    """Project 2048-dimensional frozen features into normalized 256-dimensional
    vectors.
    """

    def __init__(self) -> None:
        """Initialize the component and validate its configured dependencies."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2048, 256), nn.GELU(), nn.LayerNorm(256), nn.Linear(256, 256)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return forward for this workflow.

        Args:
            x: X used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        return F.normalize(self.layers(x), dim=-1)


def business_loss(
    sim: torch.Tensor, labels: torch.Tensor, allowed: torch.Tensor
) -> torch.Tensor:
    """Return business loss for this workflow.

    Args:
        sim: Sim used by this operation.
        labels: Judgments used for the stated development or evaluation protocol.
        allowed: Allowed used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: No allowed query with both known business classes.
    """
    known = allowed & torch.isfinite(labels)
    pos = known & (labels == 1)
    neg = known & (labels == 0)
    eligible = pos.any(1) & neg.any(1)
    if not eligible.any():
        raise ValueError("No allowed query with both known business classes")
    # Detached mining depends only on allowed train cells; no held/unknown scores.
    max_neg = sim.detach().masked_fill(~neg, -torch.inf).max(1).values
    min_pos = sim.detach().masked_fill(~pos, torch.inf).min(1).values
    hard_pos = pos & (sim.detach() < max_neg[:, None] + POLICY["mining_epsilon"])
    hard_neg = neg & (sim.detach() > min_pos[:, None] - POLICY["mining_epsilon"])
    zero = torch.zeros((len(sim), 1), device=sim.device, dtype=sim.dtype)
    positive = (-POLICY["alpha"] * (sim - POLICY["base"])).masked_fill(
        ~hard_pos, -torch.inf
    )
    negative = (POLICY["beta"] * (sim - POLICY["base"])).masked_fill(
        ~hard_neg, -torch.inf
    )
    losses = (
        torch.logsumexp(torch.cat([zero, positive], 1), 1) / POLICY["alpha"]
        + torch.logsumexp(torch.cat([zero, negative], 1), 1) / POLICY["beta"]
    )
    return losses[eligible].mean()


def fit(
    q: np.ndarray, g: np.ndarray, labels: np.ndarray, allowed: np.ndarray, seed: int
) -> tuple[Projection, np.ndarray, list]:
    """Return fit for this workflow.

    Args:
        q: Q used by this operation.
        g: G used by this operation.
        labels: Judgments used for the stated development or evaluation protocol.
        allowed: Allowed used by this operation.
        seed: Seed used to reproduce randomized ordering.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Nonfinite full metric training loss; Nonfinite projection gradients.
    """
    torch.set_num_threads(POLICY["threads"])
    torch.manual_seed(seed)
    model = Projection()
    q_tensor, g_tensor, labels_tensor = [
        torch.as_tensor(v, dtype=torch.float32) for v in (q, g, labels)
    ]
    allowed_tensor = torch.as_tensor(allowed, dtype=torch.bool)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=POLICY["learning_rate"],
        weight_decay=POLICY["weight_decay"],
    )
    trace = []
    model.train()
    for step in range(POLICY["steps"]):
        optimizer.zero_grad(set_to_none=True)
        sim = model(q_tensor) @ model(g_tensor).T
        loss = business_loss(sim, labels_tensor, allowed_tensor)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite full metric training loss")
        loss.backward()
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        ):
            raise ValueError("Nonfinite projection gradients")
        optimizer.step()
        if step % 50 == 0 or step == POLICY["steps"] - 1:
            trace.append({"step": step + 1, "loss": float(loss.detach())})
    model.eval()
    with torch.no_grad():
        scores = (model(q_tensor) @ model(g_tensor).T).numpy()
    return model, scores, trace
