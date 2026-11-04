"""Strict local image-pair reranker using Qwen's pinned full 2B checkpoint."""

from __future__ import annotations

import ast
import gc
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from multimodal_ecommerce.runtime_device import RuntimeDevice, local_model_config
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file


def reranker_configuration() -> dict:
    """Return reranker configuration for this workflow.

    Returns:
        Result mapping produced by the workflow.
    """
    return json.loads(
        (PROJECT_ROOT / "configs/reranker_model.json").read_text(encoding="utf-8")
    )


def reference_module(code: Path) -> types.ModuleType:
    """Omit upstream's unused scipy.special import, retaining all model code.

    The source stays byte-identical on disk. Refuse this compatibility patch if
    upstream starts actually using special, rather than emulating that library.

    Args:
        code: Code used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Upstream now uses scipy.special; review dependencies.
    """
    expected = PROJECT_ROOT / "vendor/qwen3_vl_reranker/qwen3_vl_reranker.py"
    expected_digest = reranker_configuration()["vendor_sha256"][expected.name]
    if code.resolve() != expected.resolve() or sha256_file(code) != expected_digest:
        raise ValueError("Only the pinned vendored reference may be compiled")
    tree = ast.parse(code.read_text(encoding="utf-8"))
    if any(isinstance(n, ast.Name) and n.id == "special" for n in ast.walk(tree)):
        raise ValueError("Upstream now uses scipy.special; review dependencies")
    tree.body = [
        n
        for n in tree.body
        if not (isinstance(n, ast.ImportFrom) and n.module == "scipy")
    ]
    module = types.ModuleType("qwen_reranker_reference")
    module.__file__ = str(code)
    sys.modules["qwen_reranker_reference"] = module
    exec(compile(tree, str(code), "exec"), module.__dict__)
    return module


def strict_tokenize(processor: Any, pairs: list, vision: Any, max_length: int) -> Any:
    """Fail on damaged vision inputs, excess tokens or missing images.

    Args:
        processor: Processor used by this operation.
        pairs: Pairs used by this operation.
        vision: Vision used by this operation.
        max_length: Max length used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.

    Raises:
        ValueError: Image pair lost vision input; Reranker token limit exceeded; no
            silent truncation; Missing or misaligned image tensors.
    """
    expected = sum(
        part["type"] == "image"
        for pair in pairs
        for msg in pair
        for part in msg["content"]
    )
    text = processor.apply_chat_template(
        pairs, tokenize=False, add_generation_prompt=True
    )
    images, videos, kwargs = vision(
        pairs, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
    )
    if videos is not None or not images or len(images) != expected:
        raise ValueError("Image pair lost vision input")
    inputs = processor(
        text=text,
        images=images,
        truncation=False,
        padding=True,
        do_resize=False,
        return_tensors="pt",
        **kwargs,
    )
    if inputs["input_ids"].shape[1] > max_length:
        raise ValueError("Reranker token limit exceeded; no silent truncation")
    if "pixel_values" not in inputs or inputs["image_grid_thw"].shape[0] != expected:
        raise ValueError("Missing or misaligned image tensors")
    return inputs


class PairReranker:
    """Full image-pair cross-encoder; return yes-minus-no logits, not confidence."""

    def __init__(self) -> None:
        """Initialize the component and validate its configured dependencies.

        Raises:
            ValueError: Changed upstream reranker source.
        """
        import torch
        from qwen_vl_utils import process_vision_info

        self.config = local_model_config(reranker_configuration())
        self.runtime = RuntimeDevice(torch)
        self.torch = torch
        directory = Path(self.config["path"])
        for name, digest in self.config["files"].items():
            if sha256_file(directory / name) != digest:
                raise ValueError(f"Changed reranker checkpoint file: {name}")
        code = PROJECT_ROOT / "vendor/qwen3_vl_reranker/qwen3_vl_reranker.py"
        if sha256_file(code) != self.config["vendor_sha256"][code.name]:
            raise ValueError("Changed upstream reranker source")
        reference = reference_module(code)

        # The pinned reference is populated by compilation at runtime.
        class StrictReranker(reference.Qwen3VLReranker):  # type: ignore[name-defined]  # pylint: disable=no-member
            """Score complete image pairs without silent NULL or truncation
            fallbacks.
            """

            def tokenize(self, pairs: list[dict], **kwargs: object) -> dict:
                """Return tokenize for this workflow.

                Args:
                    pairs: Pairs used by this operation.
                    **kwargs: Named options forwarded to the underlying operation.

                Returns:
                    Decoded JSON value; the caller must validate its expected schema.
                """
                return strict_tokenize(
                    self.processor, pairs, process_vision_info, self.max_length
                )

            @torch.inference_mode()
            def compute_scores(self, inputs: dict) -> list[float]:
                """Return compute scores for this workflow.

                Args:
                    inputs: Inputs used by this operation.

                Returns:
                    Values produced by this operation, in the stated return order.
                """
                hidden = self.model(**inputs, use_cache=False).last_hidden_state[:, -1]
                # Avoid BF16 sigmoid saturation/ties; this preserves exact score order.
                return self.score_linear(hidden).float().squeeze(-1).cpu().tolist()

        started = time.perf_counter()
        self.backend = StrictReranker(
            model_name_or_path=self.config["path"],
            max_length=self.config["max_length"],
            min_pixels=self.config["min_pixels"],
            max_pixels=self.config["max_pixels"],
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        self.backend.model.to(self.runtime.device)
        self.backend.device = self.runtime.device
        self.backend.score_linear.to(self.runtime.device)
        self.load_seconds = time.perf_counter() - started
        self.runtime.reset_peak()
        self.timings: list[dict] = []

    @staticmethod
    def picture(path: Path) -> Image.Image:
        """Return picture for this workflow.

        Args:
            path: Path of the input or output artifact used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        with Image.open(path) as source:
            return ImageOps.exif_transpose(source).convert("RGB")

    def score(
        self,
        documents: list[dict],
        *,
        query_image: Path | None = None,
        query_text: str | None = None,
    ) -> np.ndarray:
        """Return score for this workflow.

        Args:
            documents: Documents used by this operation.
            query_image: Query image used by this operation.
            query_text: Query text used by this operation.

        Returns:
            Computed array in the order of the supplied records.

        Raises:
            ValueError: Need exactly one query modality and nonempty documents; Document
                has labels, unsupported fields or no image; Invalid image-pair reranker
                output.
        """
        if (query_image is None) == (query_text is None) or not documents:
            raise ValueError("Need exactly one query modality and nonempty documents")
        query = (
            {"image": self.picture(query_image)}
            if query_image is not None
            else {"text": query_text}
        )
        scores = []
        for document in documents:
            if not set(document) <= {"id", "image_path", "text"} or not document.get(
                "image_path"
            ):
                raise ValueError("Document has labels, unsupported fields or no image")
            pair = {"image": self.picture(Path(document["image_path"]))}
            if document.get("text"):
                pair["text"] = document["text"]
            self.runtime.synchronize()
            started = time.perf_counter()
            values = self.backend.process(
                {
                    "instruction": self.config["instruction"],
                    "query": query,
                    "documents": [pair],
                }
            )
            self.runtime.synchronize()
            elapsed = time.perf_counter() - started
            if len(values) != 1 or not np.isfinite(values[0]):
                raise ValueError("Invalid image-pair reranker output")
            scores.append(values[0])
            self.timings.append({"document_id": document["id"], "seconds": elapsed})
        return np.asarray(scores, dtype=np.float64)

    def diagnostics(self) -> dict:
        """Describe the active backend and recorded runtime measurements.

        Returns:
            Mapping with fields: model, load_seconds, score, source_compatibility,
            input_safety, torch, gpu, memory_stats_kind, peak_allocated_mib, timings.
        """
        return {
            "model": self.config,
            "load_seconds": self.load_seconds,
            "score": "unnormalized yes-minus-no logit, not calibrated probability",
            "source_compatibility": "unused scipy.special import omitted only",
            "input_safety": "no NULL fallback, no silent truncation, EXIF-oriented RGB",
            "torch": self.torch.__version__,
            "gpu": self.runtime.name,
            "memory_stats_kind": (
                "peak_cuda"
                if self.runtime.device.type == "cuda"
                else "current_mps_not_peak"
            ),
            "peak_allocated_mib": self.runtime.allocated() / 2**20,
            "timings": self.timings,
        }

    def close(self) -> None:
        """Release the resources held by this component."""
        del self.backend
        gc.collect()
        self.runtime.empty_cache()
