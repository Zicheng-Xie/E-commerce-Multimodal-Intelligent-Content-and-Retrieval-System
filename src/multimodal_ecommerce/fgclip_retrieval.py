"""Frozen FG-CLIP 2 auxiliary scoring; never uses relevance judgments."""

from __future__ import annotations

import json
import math
import time
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from multimodal_ecommerce.intent_retrieval import POLICY as ORIGINAL_POLICY
from multimodal_ecommerce.intent_retrieval import normalized_ranks
from multimodal_ecommerce.runtime_device import RuntimeDevice
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from multimodal_ecommerce.semantic_ranker import ranked_ids

POLICY: dict[str, Any] = dict(
    name="fgclip2_local_fusion20",
    role="exposed_AI_development_comparison_only",
    repository="qihoo360/fg-clip2-base",
    revision="430fbc8a912c86fd4de601381b6245a0edab22f0",
    dtype="float32",
    batch_size=1,
    global_text_walk="long",
    global_text_tokens=196,
    regional_text_walk="box",
    regional_text_tokens=64,
    text_truncation="official 196/64-token limits; every truncation logged",
    image_patch_budget="official input-size policy: 128,256,576,784,1024",
    local_top_fraction=0.05,
    local_weight=0.25,
    auxiliary_cross_vote_weight=0.25,
    text_image_weight=0.8,
    image_image_weight=0.6,
    image_caption_image_weight=0.2,
    image_caption_title_weight=0.2,
    rerank_pool=20,
    fixed_k=10,
    original_public_type_offsets_unchanged=True,
    text_and_image_branch_abstain_when_no_query_text=True,
    qrels_used_in_runtime=False,
    frozen_full_model=True,
    model_finetuning=False,
    formal_acceptance=False,
    default_promotion_allowed=False,
)


def normalized(values: np.ndarray) -> np.ndarray:
    """Reject invalid real feature tensors before cosine scoring.

    Args:
        values: Values used by this operation.

    Returns:
        Computed array in the order of the supplied records.

    Raises:
        ValueError: Expected FG-CLIP 2 768-dimensional vectors; Nonfinite or zero
            FG-CLIP features.
    """
    array = np.asarray(values, dtype=np.float32)
    if array.ndim not in (1, 2) or array.shape[-1] != 768:
        raise ValueError("Expected FG-CLIP 2 768-dimensional vectors")
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if not np.isfinite(array).all() or np.any(norms < 1e-8):
        raise ValueError("Nonfinite or zero FG-CLIP features")
    return array / norms


def patch_budget(size: tuple[int, int]) -> int:
    """Match the official model card's adaptive image-size recipe.

    Args:
        size: Size used by this operation.

    Returns:
        Count or index produced by this operation.
    """
    count = (size[0] // 16) * (size[1] // 16)
    for threshold, result in [(784, 1024), (576, 784), (256, 576), (128, 256)]:
        if count > threshold:
            return result
    return 128


class FGClipEncoder:
    """Load the complete pinned checkpoint locally; no CPU or NULL fallback."""

    def __init__(self, manifest_path: Path) -> None:
        """Initialize the component and validate its configured dependencies.

        Args:
            manifest_path: Manifest path used by this operation.

        Raises:
            ValueError: FG-CLIP model identity changed.
        """
        import torch
        from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["repository"], manifest["revision"]) != (
            POLICY["repository"],
            POLICY["revision"],
        ):
            raise ValueError("FG-CLIP model identity changed")
        root = PROJECT_ROOT / manifest["path"]
        for name, expected in manifest["files"].items():
            if sha256_file(root / name) != expected:
                raise ValueError(f"Changed checkpoint or remote code: {name}")
        self.runtime = RuntimeDevice(torch)
        torch.manual_seed(42)
        self.runtime.reset_peak()
        start = time.perf_counter()
        loaded: Any = AutoModelForCausalLM.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.float32,
            output_loading_info=True,
        )
        self.model: Any
        self.model, info = loaded
        if any(
            info.get(key)
            for key in [
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            ]
        ):
            raise RuntimeError(f"Checkpoint failed complete loading: {info}")
        self.model = self.model.eval().to(self.runtime.device)
        self.processor = AutoImageProcessor.from_pretrained(root, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        self.torch = torch
        self.loading = info
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.load_seconds = time.perf_counter() - start
        self.timings: list[dict[str, Any]] = []

    def image(self, path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
        """Use the official global and dense-feature APIs, removing only padding.

        Args:
            path: Path of the input or output artifact used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.

        Raises:
            RuntimeError: Dense patch mask/geometry mismatch.
        """
        with Image.open(path) as source:
            picture = ImageOps.exif_transpose(source).convert("RGB")
        budget = patch_budget(picture.size)
        inputs = self.processor(
            images=picture, max_num_patches=budget, return_tensors="pt"
        ).to(self.runtime.device)
        self.runtime.synchronize()
        start = time.perf_counter()
        with self.torch.inference_mode():
            global_feature = self.model.get_image_features(**inputs)
            dense = self.model.get_image_dense_feature(**inputs)
        self.runtime.synchronize()
        shape = inputs["spatial_shapes"][0].tolist()
        valid = int(shape[0] * shape[1])
        if int(inputs["pixel_attention_mask"].sum()) != valid:
            raise RuntimeError("Dense patch mask/geometry mismatch")
        global_array = normalized(global_feature[0].float().cpu().numpy())
        dense_array = normalized(dense[0, :valid].float().cpu().numpy())
        detail = dict(
            kind="image_global_and_dense",
            seconds=time.perf_counter() - start,
            source_size=list(picture.size),
            max_num_patches=budget,
            spatial_shapes=shape,
            valid_patches=valid,
            actual_vision_forward_calls=2,
        )
        self.timings.append(detail)
        return global_array, dense_array, detail

    def text(self, value: str, *, regional: bool = False) -> tuple[np.ndarray, dict]:
        """Return text for this workflow.

        Args:
            value: Input value to validate or transform.
            regional: Regional used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.

        Raises:
            ValueError: Empty text must abstain, not be embedded.
        """
        if not value.strip():
            raise ValueError("Empty text must abstain, not be embedded")
        walk = "box" if regional else "long"
        length = 64 if regional else 196
        count = len(self.tokenizer(value, truncation=False)["input_ids"])
        inputs = self.tokenizer(
            value.lower(),
            padding="max_length",
            max_length=length,
            truncation=True,
            return_tensors="pt",
        ).to(self.runtime.device)
        self.runtime.synchronize()
        start = time.perf_counter()
        with self.torch.inference_mode():
            vector = self.model.get_text_features(**inputs, walk_type=walk)
        self.runtime.synchronize()
        detail = dict(
            kind="text",
            walk_type=walk,
            input_token_count=count,
            max_length=length,
            truncated=count > length,
            seconds=time.perf_counter() - start,
        )
        self.timings.append(detail)
        return normalized(vector[0].float().cpu().numpy()), detail

    def diagnostics(self) -> dict:
        """Describe the active backend and recorded runtime measurements.

        Returns:
            Mapping with fields: repository, revision, dtype, device,
            parameter_count, full_checkpoint_loading, load_seconds, torch, gpu,
            memory_stats_kind, calls, peak_allocated_mib, peak_reserved_mib.
        """
        return dict(
            repository=POLICY["repository"],
            revision=POLICY["revision"],
            dtype=str(next(self.model.parameters()).dtype),
            device=str(next(self.model.parameters()).device),
            parameter_count=self.parameter_count,
            full_checkpoint_loading=self.loading,
            load_seconds=self.load_seconds,
            torch=self.torch.__version__,
            gpu=self.runtime.name,
            memory_stats_kind=(
                "peak_cuda"
                if self.runtime.device.type == "cuda"
                else "current_mps_not_peak"
            ),
            calls=self.timings,
            peak_allocated_mib=self.runtime.allocated() / 2**20,
            peak_reserved_mib=self.runtime.reserved() / 2**20,
        )

    def close(self) -> None:
        """Release the resources held by this component."""
        del self.model
        self.runtime.empty_cache()


@dataclass(frozen=True)
class FgclipRetrievalAuxiliaryScoresInputs:
    """Input values for auxiliary scores.

    Attributes:
        ids: Identifiers in the order expected by the associated arrays.
        gallery_images: Gallery images used by this operation.
        gallery_titles: Gallery titles used by this operation.
        gallery_dense: Gallery dense used by this operation.
        query_image: Query image used by this operation.
        query_text: Query text used by this operation.
        query_box: Query box used by this operation.
    """

    ids: list[str]
    gallery_images: np.ndarray
    gallery_titles: np.ndarray
    gallery_dense: list[np.ndarray]
    _: KW_ONLY
    query_image: np.ndarray | None
    query_text: np.ndarray | None
    query_box: np.ndarray | None


def auxiliary_scores(options: FgclipRetrievalAuxiliaryScoresInputs) -> dict:
    """Compute global and locally supported semantic evidence, without labels.

    Args:
        options: Validated input fields described by
            FgclipRetrievalAuxiliaryScoresInputs.

    Returns:
        Mapping with fields: global_scores, local_scores, fused_scores, local_active.

    Raises:
        ValueError: Gallery features must align by stable candidate ID; At least one
            actual query modality is required; Text global/local features must abstain
            together.
    """
    ids = options.ids
    gallery_images = options.gallery_images
    gallery_titles = options.gallery_titles
    gallery_dense = options.gallery_dense
    query_image = options.query_image
    query_text = options.query_text
    query_box = options.query_box
    images, titles = normalized(gallery_images), normalized(gallery_titles)
    if (
        images.shape != titles.shape
        or len(ids) != len(images)
        or len(ids) != len(gallery_dense)
    ):
        raise ValueError("Gallery features must align by stable candidate ID")
    if query_image is None and query_text is None:
        raise ValueError("At least one actual query modality is required")
    if (query_text is None) != (query_box is None):
        raise ValueError("Text global/local features must abstain together")
    if query_image is None:
        assert query_text is not None
        q = normalized(query_text)
        global_score = 0.8 * (images @ q) + 0.2 * (titles @ q)
    else:
        global_score = images @ normalized(query_image)
        if query_text is not None:
            q = normalized(query_text)
            global_score = 0.6 * global_score + 0.2 * (images @ q) + 0.2 * (titles @ q)
    local = np.zeros(len(ids))
    if query_box is not None:
        qbox = normalized(query_box)
        for index, patches in enumerate(gallery_dense):
            values = normalized(patches) @ qbox
            count = max(1, math.ceil(len(values) * POLICY["local_top_fraction"]))
            local[index] = np.partition(values, len(values) - count)[-count:].mean()
        fused = 0.75 * normalized_ranks(ids, global_score) + 0.25 * normalized_ranks(
            ids, local
        )
    else:
        fused = normalized_ranks(ids, global_score)
    return dict(
        global_scores=global_score.tolist(),
        local_scores=local.tolist(),
        fused_scores=fused.tolist(),
        local_active=query_box is not None,
    )


@dataclass(frozen=True)
class FgclipRetrievalFusePoolInputs:
    """Input values for fuse pool.

    Attributes:
        ids: Identifiers in the order expected by the associated arrays.
        base: Base used by this operation.
        pool: Pool used by this operation.
        cross_logits: Cross logits used by this operation.
        auxiliary: Auxiliary used by this operation.
        offsets: Offsets used by this operation.
    """

    ids: list[str]
    base: np.ndarray
    pool: list[str]
    cross_logits: np.ndarray
    auxiliary: np.ndarray
    offsets: np.ndarray | None


def fuse_pool(options: FgclipRetrievalFusePoolInputs) -> list[str]:
    """Add a fixed auxiliary cross vote; retain candidate budget and complete tail.

    Args:
        options: Validated input fields described by FgclipRetrievalFusePoolInputs.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: This experiment requires 20 distinct candidates; Auxiliary scores
            must cover the unchanged full gallery; Need all 20 real cross-encoder
            scores; Finite aligned source scores required; Invalid original
            public-evidence offsets.
    """
    ids = options.ids
    base = options.base
    pool = options.pool
    cross_logits = options.cross_logits
    auxiliary = options.auxiliary
    offsets = options.offsets
    if len(pool) != 20 or len(set(ids)) != len(ids) or len(set(pool)) != len(pool):
        raise ValueError("This experiment requires 20 distinct candidates")
    if not set(pool) <= set(ids) or len(ids) != len(auxiliary):
        raise ValueError("Auxiliary scores must cover the unchanged full gallery")
    if cross_logits.shape != (20,) or not np.isfinite(cross_logits).all():
        raise ValueError("Need all 20 real cross-encoder scores")
    if (
        base.shape != (len(ids),)
        or not np.isfinite(base).all()
        or not np.isfinite(auxiliary).all()
    ):
        raise ValueError("Finite aligned source scores required")
    position = {cid: i for i, cid in enumerate(ids)}
    selected = np.array([position[cid] for cid in pool])
    alpha = POLICY["auxiliary_cross_vote_weight"]
    weight = ORIGINAL_POLICY["cross_encoder_weight"]
    score = weight * (
        (1 - alpha) * normalized_ranks(pool, cross_logits)
        + alpha * normalized_ranks(pool, auxiliary[selected])
    )
    score += (1 - weight) * normalized_ranks(ids, base)[selected]
    if offsets is not None:
        if offsets.shape != (len(ids),) or not np.isfinite(offsets).all():
            raise ValueError("Invalid original public-evidence offsets")
        score += offsets[selected]
    return ranked_ids(pool, score) + [
        cid for cid in ranked_ids(ids, base) if cid not in set(pool)
    ]
