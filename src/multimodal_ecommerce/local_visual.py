"""Compact visual features and attention-selected local descriptors for retrieval."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from multimodal_ecommerce.runtime_device import RuntimeDevice, local_model_config
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file


def select_descriptors(
    tokens: np.ndarray, attention: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Select descriptors.

    Args:
        tokens: Tokens used by this operation.
        attention: Attention used by this operation.
        top_k: Top k used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Misaligned or nonfinite patch descriptors; Invalid patch selection
            size; Zero patch descriptor.
    """
    if (
        tokens.ndim != 2
        or attention.shape != (len(tokens),)
        or not np.isfinite(tokens).all()
        or not np.isfinite(attention).all()
    ):
        raise ValueError("Misaligned or nonfinite patch descriptors")
    if top_k < 1 or top_k > len(tokens):
        raise ValueError("Invalid patch selection size")
    indices = np.lexsort((np.arange(len(tokens)), -attention))[:top_k]
    vectors = tokens[indices]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Zero patch descriptor")
    return (vectors / norms).astype(np.float32), indices


class LocalVisualEncoder:
    """Extract frozen DINOv2 global and attention-selected patch descriptors."""

    def __init__(self) -> None:
        """Initialize the component and validate its configured dependencies.

        Raises:
            ValueError: Changed DINO checkpoint.
        """
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.runtime = RuntimeDevice(torch)
        self.torch = torch
        self.config = local_model_config(
            json.loads(
                (PROJECT_ROOT / "configs/dino_model.json").read_text(encoding="utf-8")
            )
        )
        for name, digest in self.config["files"].items():
            if sha256_file(Path(self.config["path"]) / name) != digest:
                raise ValueError("Changed DINO checkpoint")
        self.processor = AutoImageProcessor.from_pretrained(
            self.config["path"], local_files_only=True, use_fast=False
        )
        self.model = (
            AutoModel.from_pretrained(
                self.config["path"], local_files_only=True, attn_implementation="eager"
            )
            .to(self.runtime.device)
            .eval()
        )
        self.model.requires_grad_(False)

    def encode(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode the configured inputs into model feature representations.

        Args:
            path: Path of the input or output artifact used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        with Image.open(path) as original:
            picture = ImageOps.exif_transpose(original).convert("RGB")
        inputs = self.processor(images=picture, return_tensors="pt").to(
            self.runtime.device
        )
        with self.torch.inference_mode():
            result = self.model(**inputs, output_attentions=True)
        tokens = result.last_hidden_state[0].float().cpu().numpy()
        attention = result.attentions[-1][0, :, 0, 1:].mean(dim=0).float().cpu().numpy()
        patches, indices = select_descriptors(
            tokens[1:], attention, self.config["patch_top_k"]
        )
        global_vector = tokens[0] / np.linalg.norm(tokens[0])
        return global_vector.astype(np.float32), patches, indices

    def close(self) -> None:
        """Release the resources held by this component."""
        del self.model
        gc.collect()
        self.runtime.empty_cache()


def dense_similarity(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Symmetric patch MaxSim on attention-selected normalized local features.

    Args:
        query: Query record or representation used by this operation.
        gallery: Gallery used by this operation.

    Returns:
        Computed array in the order of the supplied records.
    """
    import torch

    runtime = RuntimeDevice(torch)
    with torch.inference_mode():
        q = torch.as_tensor(query, device=runtime.device)
        bank = torch.as_tensor(gallery, device=runtime.device)
        sim = torch.einsum("id,njd->nij", q, bank)
        result = (
            sim.max(dim=2).values.mean(dim=1) + sim.max(dim=1).values.mean(dim=1)
        ) / 2
    return result.cpu().numpy()
