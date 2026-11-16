"""Explicit placement must fail visibly, never switch devices silently."""

import pytest
import torch
from multimodal_ecommerce.segmentation_device import (
    select_device,
    move_target,
    cpu_prediction,
    memory_sample,
)


def test_cpu_and_target_preserve_labels_and_dtype():
    device = select_device("cpu")
    target = {
        "labels": torch.tensor([13]),
        "masks": torch.ones(1, 2, 2, dtype=torch.uint8),
    }
    moved = move_target(target, device)
    assert moved["labels"].tolist() == [13]
    assert moved["masks"].dtype == torch.uint8
    assert (
        cpu_prediction({"scores": torch.ones(1, requires_grad=True)})[
            "scores"
        ].requires_grad
        is False
    )
    assert memory_sample(device)["current_allocated_bytes"] is None


def test_unavailable_mps_is_error(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS unavailable"):
        select_device("mps")
    with pytest.raises(ValueError):
        select_device("cuda")
