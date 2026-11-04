"""GroundingDINO subject crops and image-only experimental routing."""

from __future__ import annotations

import json
import math
import time
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from multimodal_ecommerce.fgclip_retrieval import (
    FgclipRetrievalFusePoolInputs,
    fuse_pool,
)
from multimodal_ecommerce.intent_retrieval import normalized_ranks
from multimodal_ecommerce.runtime_device import RuntimeDevice
from multimodal_ecommerce.safety import PROJECT_ROOT, ensure_output_path, sha256_file

# Public noun translations, frozen before detector inference; not relevance labels.
SUBJECTS = [
    ("ear pad", ["耳罩", "耳垫", "耳机套"]),
    ("earphone cable", ["耳机线", "升级线"]),
    ("earphone case", ["耳机盒", "耳机收纳", "耳机保护盒"]),
    ("phone case", ["手机背壳", "手机壳"]),
    ("bag organizer", ["内胆包", "包内胆", "包中包"]),
    ("pillow case", ["枕套"]),
    ("headphones", ["头戴", "后挂耳机"]),
    ("earphones", ["耳机", "耳塞"]),
    ("boot", ["靴"]),
    ("shoe", ["鞋"]),
    ("vest", ["马甲", "背心"]),
    ("jacket", ["羽绒", "外套", "棉服", "夹克", "开衫"]),
    ("shirt", ["衬衫", "短袖", "T恤", "t恤"]),
    ("pillow", ["枕"]),
    ("cabinet", ["柜"]),
    ("cup. bottle", ["杯", "奶瓶"]),
    ("handbag", ["手提包", "手袋", "挎包"]),
    ("table", ["桌"]),
    ("fruit", ["榴莲", "水果", "果肉"]),
    ("bread", ["面包", "糕点"]),
]
POLICY: dict[str, Any] = dict(
    primary_method="image_roi_fusion20",
    repository="IDEA-Research/grounding-dino-tiny",
    revision="a2bb814dd30d776dcf7e30523b00659f4f141c71",
    dtype="float32",
    disable_custom_kernels=True,
    detector_threshold=0.30,
    text_threshold=0.25,
    min_box_fraction=0.02,
    max_box_fraction=0.95,
    max_padded_fraction=0.98,
    min_box_edge=8,
    padding=0.08,
    crop_long_edge=768,
    choose_box="highest score among geometrically valid boxes; coordinate tie break",
    prompt_sources="frozen query-only description / frozen public gallery product_type",
    unknown_subject="retain global features without detector inference",
    local_and_global_roi_vote=0.5,
    text_routing="return original cross20 exactly, with no FG or ROI scoring",
    same_original_twenty_candidates=True,
    original_Qwen_scores_unchanged=True,
    qrels_used_in_runtime=False,
    formal_acceptance=False,
    default_promotion_allowed=False,
)


def subject_prompt(description: str) -> str | None:
    """Return subject prompt for this workflow.

    Args:
        description: Description used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    head = description.split("商品主体：", 1)[-1].split("。", 1)[0]
    if any(word in head for word in ["不明", "无法", "组合", "展示道具"]):
        return None
    for english, aliases in SUBJECTS:
        if any(alias in head for alias in aliases):
            return english + "."
    return None


def choose_box(detections: list[dict], size: tuple[int, int]) -> dict:
    """Apply label-free geometry checks; never drop the image on failure.

    Args:
        detections: Detections used by this operation.
        size: Size used by this operation.

    Returns:
        Mapping with fields: status, reason, selected.
    """
    width, height = size
    valid = []
    for row in detections:
        score = row["score"]
        box = row["box"]
        if len(box) != 4 or not np.isfinite([score, *box]).all():
            continue
        if score < POLICY["detector_threshold"]:
            continue
        x1, y1, x2, y2 = [
            max(0, min(value, limit))
            for value, limit in zip(box, [width, height, width, height])
        ]
        dx, dy = x2 - x1, y2 - y1
        fraction = dx * dy / (width * height)
        if (
            min(dx, dy) < POLICY["min_box_edge"]
            or not POLICY["min_box_fraction"] <= fraction <= POLICY["max_box_fraction"]
        ):
            continue
        padding = POLICY["padding"]
        padded = [
            max(0, math.floor(x1 - dx * padding)),
            max(0, math.floor(y1 - dy * padding)),
            min(width, math.ceil(x2 + dx * padding)),
            min(height, math.ceil(y2 + dy * padding)),
        ]
        if (padded[2] - padded[0]) * (padded[3] - padded[1]) / (
            width * height
        ) > POLICY["max_padded_fraction"]:
            continue
        valid.append(
            dict(
                box=[x1, y1, x2, y2],
                padded_box=padded,
                score=score,
                phrase=row.get("phrase"),
                box_fraction=fraction,
            )
        )
    if not valid:
        return dict(
            status="global_fallback",
            reason="no geometrically valid confident subject box",
            selected=None,
        )
    selected = min(valid, key=lambda row: (-row["score"], row["box"]))
    return dict(status="roi", reason="frozen score/geometry policy", selected=selected)


class SubjectDetector:
    """Complete official detector with the supported pure-PyTorch backend."""

    def __init__(self, manifest_path: Path) -> None:
        """Initialize the component and validate its configured dependencies.

        Args:
            manifest_path: Manifest path used by this operation.

        Raises:
            ValueError: Detector source/checkpoint changed.
        """
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert (
            manifest["repository"] == POLICY["repository"]
            and manifest["revision"] == POLICY["revision"]
        )
        root = PROJECT_ROOT / manifest["path"]
        for name, digest in manifest["files"].items():
            if sha256_file(root / name) != digest:
                raise ValueError("Detector source/checkpoint changed")
        self.runtime = RuntimeDevice(torch)
        self.runtime.reset_peak()
        start = time.perf_counter()
        loaded: Any = AutoModelForZeroShotObjectDetection.from_pretrained(
            root,
            local_files_only=True,
            dtype=torch.float32,
            disable_custom_kernels=True,
            output_loading_info=True,
        )
        self.model: Any
        self.model, info = loaded
        if any(
            info.get(k)
            for k in [
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            ]
        ):
            raise RuntimeError(f"Incomplete detector loading: {info}")
        self.model = self.model.eval().to(self.runtime.device)
        self.processor = AutoProcessor.from_pretrained(root, local_files_only=True)
        self.torch = torch
        self.loading = info
        self.load_seconds = time.perf_counter() - start
        self.timings: list[dict] = []

    def detect(self, image_path: Path, prompt: str) -> dict:
        """Return detect for this workflow.

        Args:
            image_path: Image path used by this operation.
            prompt: Prompt used by this operation.

        Returns:
            Mapping with fields: prompt, source_size, detections, decision, seconds,
            input_pixel_values.
        """
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.runtime.device
        )
        self.runtime.synchronize()
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model(**inputs)
        self.runtime.synchronize()
        results = self.processor.post_process_grounded_object_detection(
            output,
            inputs.input_ids,
            threshold=POLICY["detector_threshold"],
            text_threshold=POLICY["text_threshold"],
            target_sizes=[image.size[::-1]],
        )[0]
        phrases = (
            results["text_labels"] if "text_labels" in results else results["labels"]
        )
        detections = [
            dict(box=box, score=score, phrase=str(phrase))
            for box, score, phrase in zip(
                results["boxes"].cpu().tolist(),
                results["scores"].cpu().tolist(),
                phrases,
            )
        ]
        info = dict(
            prompt=prompt,
            source_size=list(image.size),
            detections=detections,
            decision=choose_box(detections, image.size),
            seconds=time.perf_counter() - start,
            input_pixel_values=list(inputs["pixel_values"].shape),
        )
        self.timings.append(dict(seconds=info["seconds"], detections=len(detections)))
        return info

    def diagnostics(self) -> dict:
        """Describe the active backend and recorded runtime measurements.

        Returns:
            Mapping with fields: repository, revision, parameter_count, dtype,
            device, full_checkpoint_loading, disable_custom_kernels, load_seconds,
            calls, peak_allocated_mib, peak_reserved_mib.
        """
        return dict(
            repository=POLICY["repository"],
            revision=POLICY["revision"],
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            dtype=str(next(self.model.parameters()).dtype),
            device=str(next(self.model.parameters()).device),
            full_checkpoint_loading=self.loading,
            disable_custom_kernels=self.model.config.disable_custom_kernels,
            load_seconds=self.load_seconds,
            calls=self.timings,
            peak_allocated_mib=self.runtime.allocated() / 2**20,
            peak_reserved_mib=self.runtime.reserved() / 2**20,
        )

    def close(self) -> None:
        """Release the resources held by this component."""
        del self.model
        self.runtime.empty_cache()


def save_subject_crop(source_path: Path, decision: dict, folder: Path) -> dict:
    """Persist subject crop.

    Args:
        source_path: Source path used by this operation.
        decision: Decision used by this operation.
        folder: Directory containing the operation artifacts.

    Returns:
        Mapping with fields: native_sha256, zoom_sha256, native_size, zoom_size.

    Raises:
        ValueError: Only accepted boxes create crops.

    Notes:
        Writes derived artifacts through crop.save, folder.mkdir, overlay.save,
        resized.save.
    """
    if decision["status"] != "roi":
        raise ValueError("Only accepted boxes create crops")
    folder = ensure_output_path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    box = decision["selected"]["padded_box"]
    crop = image.crop(tuple(box))
    native = folder / "native.png"
    zoom = folder / "zoom.png"
    crop.save(native)
    scale = POLICY["crop_long_edge"] / max(crop.size)
    resized = crop.resize(
        tuple(max(2, round(v * scale)) for v in crop.size), Image.Resampling.LANCZOS
    )
    resized.save(zoom)
    overlay = image.copy()
    ImageDraw.Draw(overlay).rectangle(tuple(box), outline="#ff3030", width=3)
    overlay.thumbnail((768, 768))
    overlay.save(folder / "box.png")
    return dict(
        native_sha256=sha256_file(native),
        zoom_sha256=sha256_file(zoom),
        native_size=list(crop.size),
        zoom_size=list(resized.size),
    )


@dataclass(frozen=True)
class SubjectRoiImageOnlyOrderInputs:
    """Input values for image only order.

    Attributes:
        original: Original used by this operation.
        is_image: Is image used by this operation.
        ids: Identifiers in the order expected by the associated arrays.
        base: Base used by this operation.
        pool: Pool used by this operation.
        logits: Logits used by this operation.
        whole_scores: Whole scores used by this operation.
        roi_scores: Roi scores used by this operation.
        offsets: Offsets used by this operation.
    """

    original: list[str]
    _: KW_ONLY
    is_image: bool
    ids: list[str]
    base: np.ndarray
    pool: list[str]
    logits: np.ndarray
    whole_scores: np.ndarray
    roi_scores: np.ndarray | None
    offsets: np.ndarray | None


def image_only_order(options: SubjectRoiImageOnlyOrderInputs) -> list[str]:
    """Route text exactly to the original; fuse whole-image and ROI evidence.

    Args:
        options: Validated input fields described by SubjectRoiImageOnlyOrderInputs.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    original = options.original
    is_image = options.is_image
    ids = options.ids
    base = options.base
    pool = options.pool
    logits = options.logits
    whole_scores = options.whole_scores
    roi_scores = options.roi_scores
    offsets = options.offsets
    if not is_image:
        return list(original)
    auxiliary = whole_scores
    if roi_scores is not None:
        auxiliary = 0.5 * normalized_ranks(ids, whole_scores) + 0.5 * normalized_ranks(
            ids, roi_scores
        )
    return fuse_pool(
        FgclipRetrievalFusePoolInputs(ids, base, pool, logits, auxiliary, offsets)
    )
