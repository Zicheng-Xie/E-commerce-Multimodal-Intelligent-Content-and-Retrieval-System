"""Loss-gradient isolation is critical with many unknown/held business pairs."""

from __future__ import annotations

import torch

from business_metric_model import business_loss


def test_loss_cannot_use_unknown_or_held_cells_for_mining_or_gradients() -> None:
    """Verify that loss cannot use unknown or held cells for mining or gradients."""
    scores = torch.tensor([[0.2, 0.8, 0.99, 0.99]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0, float("nan"), 0.0]])
    allowed = torch.tensor([[True, True, True, False]])
    loss = business_loss(scores, labels, allowed)
    loss.backward()
    assert scores.grad is not None
    assert scores.grad[0, 0] < 0 and scores.grad[0, 1] > 0
    assert scores.grad[0, 2] == 0 and scores.grad[0, 3] == 0
    changed = scores.detach().clone()
    changed[0, 2:] = -1
    assert torch.allclose(loss.detach(), business_loss(changed, labels, allowed))
