"""Resolve host-local assets and provide an explicit CUDA or Apple MPS runtime."""

from __future__ import annotations

import os
import platform
import re
from pathlib import Path
from types import ModuleType
from typing import TypeVar, cast

AssetT = TypeVar("AssetT")


def local_model_config(config: dict) -> dict:
    """Copy a frozen model configuration and resolve its Mac checkpoint directory.

    Args:
        config: Model metadata containing repository and path fields.

    Returns:
        A shallow copy with a project-local path on Mac for historical E: paths.
        The supplied configuration remains unchanged for frozen-index checks.

    Raises:
        ValueError: The repository suffix is not a safe model directory name.
    """
    resolved = dict(config)
    if platform.system() == "Darwin" and str(resolved.get("path", "")).startswith("E:"):
        name = resolved["repository"].rsplit("/", 1)[-1]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise ValueError("Expected a safe model repository directory name")
        resolved["path"] = str(Path(__file__).resolve().parents[2] / ".models" / name)
    return resolved


class RuntimeDevice:
    """Select one available GPU backend without silently falling back to the CPU.

    Attributes:
        device: Torch device selected by availability or MULTIMODAL_DEVICE.
        name: CUDA device name or the Apple Silicon backend label.
        torch: Torch module supplied by the model adapter.
    """

    def __init__(self, torch: ModuleType) -> None:
        """Resolve the requested backend before a model allocates GPU weights.

        Args:
            torch: Imported PyTorch module used by the calling model adapter.

        Raises:
            RuntimeError: Neither an available CUDA nor MPS backend is selected.
        """
        self.torch = torch
        requested = os.environ.get("MULTIMODAL_DEVICE")
        selected = requested or (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else None
        )
        if selected not in {"cuda", "mps"}:
            raise RuntimeError("An available CUDA or Apple MPS GPU is required")
        if selected == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("The requested CUDA backend is unavailable")
        if selected == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("The requested Apple MPS backend is unavailable")
        self.device = torch.device(selected)
        self.name = (
            torch.cuda.get_device_name() if selected == "cuda" else "Apple Silicon MPS"
        )

    def synchronize(self) -> None:
        """Wait for GPU work so the adapter's latency measurement includes execution."""
        backend = self.torch.cuda if self.device.type == "cuda" else self.torch.mps
        backend.synchronize()

    def reset_peak(self) -> None:
        """Reset CUDA peak statistics; MPS exposes current memory through this
        adapter.
        """
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats()

    def allocated(self) -> int:
        """Read the allocation statistic available on the selected GPU backend.

        Returns:
            CUDA peak allocation or MPS current allocation, in bytes.
        """
        return int(
            self.torch.cuda.max_memory_allocated()
            if self.device.type == "cuda"
            else self.torch.mps.current_allocated_memory()
        )

    def reserved(self) -> int:
        """Read the reservation statistic available on the selected GPU backend.

        Returns:
            CUDA peak reservation or MPS driver allocation, in bytes.
        """
        return int(
            self.torch.cuda.max_memory_reserved()
            if self.device.type == "cuda"
            else self.torch.mps.driver_allocated_memory()
        )

    def empty_cache(self) -> None:
        """Release unused allocator cache after the owning adapter drops its model."""
        backend = self.torch.cuda if self.device.type == "cuda" else self.torch.mps
        backend.empty_cache()


def project_path(value: str | Path) -> Path:
    """Relocate a historical project path and reject traversal outside the workspace.

    Args:
        value: A path, optionally using the original Windows project prefix.

    Returns:
        The equivalent local path; unrelated paths retain their original location.

    Raises:
        ValueError: A path using the historical prefix escapes the current project.
    """
    normalized = str(value).replace("\\", "/")
    prefix = "E:/code/Multimodal_E-commerce/"
    root = Path(__file__).resolve().parents[2]
    if normalized.startswith(prefix):
        candidate = (root / normalized[len(prefix) :]).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("Historical project path escapes workspace")
        return candidate
    return Path(normalized)


def relocate_assets(value: AssetT) -> AssetT:
    """Copy a metadata tree while relocating historical project-path strings.

    Args:
        value: Dictionary, list, string or scalar from verified index metadata.

    Returns:
        A tree with the same container and scalar types. Only historical project
        path strings change; the source tree remains available for hash checks.
    """
    if isinstance(value, dict):
        return cast(AssetT, {key: relocate_assets(item) for key, item in value.items()})
    if isinstance(value, list):
        return cast(AssetT, [relocate_assets(item) for item in value])
    if isinstance(value, str) and value.replace("\\", "/").startswith(
        "E:/code/Multimodal_E-commerce/"
    ):
        return cast(AssetT, str(project_path(value)))
    return value
