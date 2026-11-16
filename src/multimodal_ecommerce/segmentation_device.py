"""Explicit device selection and synchronization; never silently fall back."""

import torch


def select_device(name):
    if name not in ("cpu", "mps"):
        raise ValueError("Expected cpu or mps")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS unavailable in this process; verify macOS GPU access outside the sandbox"
        )
    return torch.device(name)


def synchronize(device):
    if torch.device(device).type == "mps":
        torch.mps.synchronize()


def move_target(target, device):
    return {k: v.to(device) for k, v in target.items()}


def cpu_prediction(prediction):
    return {k: v.detach().cpu() for k, v in prediction.items()}


def memory_sample(device):
    if torch.device(device).type == "mps":
        return dict(
            current_allocated_bytes=torch.mps.current_allocated_memory(),
            driver_allocated_bytes=torch.mps.driver_allocated_memory(),
        )
    return dict(current_allocated_bytes=None, driver_allocated_bytes=None)
