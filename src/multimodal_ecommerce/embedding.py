"""Strict local adapter around Qwen's pinned multimodal embedding reference."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from multimodal_ecommerce.runtime_device import RuntimeDevice, local_model_config
from multimodal_ecommerce.safety import PROJECT_ROOT


def model_configuration() -> dict[str, Any]:
    """Read the downloaded model's pinned revision and preprocessing policy.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return json.loads(
        (PROJECT_ROOT / "configs/embedding_model.json").read_text(encoding="utf-8")
    )


class QwenEmbedder:
    """Encode local text or images in a shared 2048-dimensional space."""

    def __init__(self) -> None:
        """Initialize the component and validate its configured dependencies.

        Raises:
            RuntimeError: Vendored Qwen source does not match the pinned checksum.
            ImportError: Cannot load Qwen reference module.
        """
        import torch
        from qwen_vl_utils import process_vision_info

        self.config = local_model_config(model_configuration())
        self.torch = torch
        self.runtime = RuntimeDevice(torch)
        code = PROJECT_ROOT / "vendor/qwen3_vl_embedding/qwen3_vl_embedding.py"
        if (
            hashlib.sha256(code.read_bytes()).hexdigest()
            != self.config["vendor_sha256"][code.name]
        ):
            raise RuntimeError(
                "Vendored Qwen source does not match the pinned checksum"
            )
        spec = importlib.util.spec_from_file_location("qwen_embedding_reference", code)
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load Qwen reference module")
        reference = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = reference
        spec.loader.exec_module(reference)

        # The revision-pinned upstream module is loaded dynamically, without stubs.
        class StrictEmbedder(reference.Qwen3VLEmbedder):  # type: ignore[name-defined]
            # Upstream catches preprocessing errors and encodes NULL. Retrieval
            # must fail visibly instead of quietly indexing an unrelated vector.
            """Enforce token and image-tensor checks on the pinned Qwen embedder."""

            def _preprocess_inputs(self, conversations):
                text = self.processor.apply_chat_template(
                    conversations, add_generation_prompt=True, tokenize=False
                )
                images, videos, kwargs = process_vision_info(
                    conversations,
                    image_patch_size=16,
                    return_video_metadata=True,
                    return_video_kwargs=True,
                )
                if videos is not None:
                    raise ValueError(
                        "This baseline supports still images and text only"
                    )
                inputs = self.processor(
                    text=text,
                    images=images,
                    truncation=False,
                    padding=True,
                    do_resize=False,
                    return_tensors="pt",
                    **kwargs,
                )
                if inputs["input_ids"].shape[1] > self.max_length:
                    raise ValueError(
                        "Embedding input exceeds the configured token limit"
                    )
                if images and "pixel_values" not in inputs:
                    raise RuntimeError("Image input produced no vision tensor")
                return inputs

        torch.manual_seed(42)
        self.runtime.reset_peak()
        started = time.perf_counter()
        self.backend = StrictEmbedder(
            self.config["path"],
            max_length=self.config["max_length"],
            min_pixels=4096,
            max_pixels=self.config["max_pixels"],
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        self.backend.model.to(self.runtime.device)
        self.load_seconds = time.perf_counter() - started
        self.timings: list[dict] = []

    def encode(
        self, *, text: str | None = None, image: Path | None = None, query: bool = False
    ) -> np.ndarray:
        """Encode exactly one modality, preserving real image features.

        Args:
            text: Unicode text, mutually exclusive with image.
            image: Existing local image path; no URLs are fetched.
            query: Use the official retrieval query instruction.

        Returns:
            Computed array in the order of the supplied records.

        Raises:
            ValueError: Provide exactly one of text or image; Embedding text cannot be
                empty.
            RuntimeError: Invalid embedding shape or non-finite values; Zero embedding.
        """
        if (text is None) == (image is None):
            raise ValueError("Provide exactly one of text or image")
        item: dict[str, Any] = {
            "instruction": self.config[
                "query_instruction" if query else "document_instruction"
            ]
        }
        if image is not None:
            with Image.open(image) as picture:
                item["image"] = ImageOps.exif_transpose(picture).convert("RGB")
        else:
            if not text or not text.strip():
                raise ValueError("Embedding text cannot be empty")
            item["text"] = text
        self.runtime.synchronize()
        started = time.perf_counter()
        vector = self.backend.process([item])[0].float().cpu().numpy()
        self.runtime.synchronize()
        self.timings.append(
            {
                "modality": "image" if image else "text",
                "query": query,
                "seconds": time.perf_counter() - started,
            }
        )
        if vector.shape != (2048,) or not np.isfinite(vector).all():
            raise RuntimeError("Invalid embedding shape or non-finite values")
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            raise RuntimeError("Zero embedding")
        return vector / norm

    def diagnostics(self) -> dict[str, Any]:
        """Return backend, latency and measured PyTorch memory statistics.

        Returns:
            Mapping with fields: config, device, dtype, torch, gpu,
            memory_stats_kind, load_seconds, calls, peak_allocated_mib,
            peak_reserved_mib.
        """
        return {
            "config": self.config,
            "device": str(self.backend.model.device),
            "dtype": str(next(self.backend.model.parameters()).dtype),
            "torch": self.torch.__version__,
            "gpu": self.runtime.name,
            "memory_stats_kind": (
                "peak_cuda"
                if self.runtime.device.type == "cuda"
                else "current_mps_not_peak"
            ),
            "load_seconds": self.load_seconds,
            "calls": self.timings,
            "peak_allocated_mib": self.runtime.allocated() / 2**20,
            "peak_reserved_mib": self.runtime.reserved() / 2**20,
        }

    def close(self) -> None:
        """Release the model's GPU allocation after an experiment."""
        del self.backend
        self.runtime.empty_cache()
