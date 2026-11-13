"""Pinned local model stages for selected retrieval; no experiment or label imports."""

from __future__ import annotations

import base64
import importlib.metadata
import io
import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from multimodal_ecommerce.business_constraints import ENTITY_RULES, canonical
from multimodal_ecommerce.embedding import QwenEmbedder, model_configuration
from multimodal_ecommerce.intent_retrieval import EXTRA_TYPES, public_entity
from multimodal_ecommerce.machine_teacher import MODEL, resolve_local_teacher
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.pair_reranker import reranker_configuration
from multimodal_ecommerce.regional_retrieval import (
    RegionalRetrievalEncodeViewsInputs,
    encode_views,
    region_configuration,
)
from multimodal_ecommerce.runtime_device import local_model_config
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from multimodal_ecommerce.selected_runtime_io import digest, load
from multimodal_ecommerce.subject_roi import POLICY as CROP_POLICY
from multimodal_ecommerce.subject_roi import SubjectDetector, subject_prompt
from multimodal_ecommerce.visual_query import (
    CAPTION_OPTIONS,
    CAPTION_PROMPT,
    CAPTION_SCHEMA,
    DescribedImageReranker,
    describe_query,
)


def validate_value(value: object, schema: dict) -> None:
    """Validate the bounded schema subset used by the fixed local vision prompts.

    Args:
        value: Parsed model observation.
        schema: Frozen object, string, array or Boolean schema.
    """
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict) or set(value) != set(schema["required"]):
            raise ValueError("Model observation fields differ from the frozen schema")
        for name, child in schema["properties"].items():
            validate_value(value[name], child)
    elif kind == "string":
        if not isinstance(value, str) or not schema.get("minLength", 0) <= len(
            value
        ) <= schema.get("maxLength", 100000):
            raise ValueError("Invalid model observation text")
    elif kind == "boolean":
        if type(value) is not bool:
            raise ValueError("Explicit model Boolean required")
    elif kind == "array":
        if not isinstance(value, list) or len(value) > schema["maxItems"]:
            raise ValueError("Invalid model observation array")
        for item in value:
            validate_value(item, schema["items"])
    else:
        raise ValueError("Unsupported frozen observation schema")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("Invalid model observation enum")


def supported_type(title: str, profile: dict) -> str | None:
    """Require the candidate's own clear sale-item description and title to agree.

    Args:
        title: Original public product title.
        profile: Candidate-only structured sale-item observation.

    Returns:
        Title-supported product noun or no reliable typed augmentation.
    """
    if profile["primary_unclear"] or profile["role"] == "unclear":
        return None
    kind = public_entity(profile["sale_item"])
    return (
        kind
        if kind
        and any(
            kind == family and any(alias in canonical(title) for alias in aliases)
            for family, aliases in [*EXTRA_TYPES, *ENTITY_RULES]
        )
        else None
    )


class LocalModels:
    """Load at most one PyTorch stage; use the installed fixed 4B vision teacher."""

    def __init__(self, policy: dict) -> None:
        """Bind prompts and preprocessing before any model inference.

        Args:
            policy: Frozen selected-runtime policy document.
        """
        self.policy = policy
        if (
            policy.get("query_caption")
            != dict(
                prompt=CAPTION_PROMPT, schema=CAPTION_SCHEMA, options=CAPTION_OPTIONS
            )
            or policy.get("pool_size") != 20
            or policy.get("anchors") != 10
            or policy.get("text_whole_weight") != 0.5
            or policy.get("public_conflict_options")
            != dict(protect=0, broad=False, matched_only=True)
        ):
            raise ValueError("Runtime policy differs from the selected fixed algorithm")
        self.current: Any = None
        self.stage = ""
        self.events: list[dict] = []
        self._identity: dict | None = None

    def identity(self) -> dict:
        """Hash actual local checkpoints and verify the selected Ollama digest.

        Returns:
            Actual assets, model configuration and full policy identity.
        """
        if self._identity is not None:
            return self._identity
        configs = dict(
            embedding=local_model_config(model_configuration()),
            reranker=local_model_config(reranker_configuration()),
        )
        assets = {}
        for cfg in configs.values():
            folder = Path(cfg["path"])
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            for path in sorted(folder.rglob("*")):
                if path.is_file() and ".cache" not in path.relative_to(folder).parts:
                    assets[str(path.resolve())] = sha256_file(path)
            for name, expected in cfg.get("files", {}).items():
                if assets.get(str((folder / name).resolve())) != expected:
                    raise ValueError("Local checkpoint differs from its fixed revision")
            vendor = (
                PROJECT_ROOT
                / "vendor"
                / (
                    "qwen3_vl_embedding"
                    if "Embedding" in cfg["repository"]
                    else "qwen3_vl_reranker"
                )
            )
            for name, expected in cfg["vendor_sha256"].items():
                path = vendor / name
                actual = sha256_file(path)
                if actual != expected:
                    raise ValueError("Vendored model preprocessing changed")
                assets[str(path.resolve())] = actual
            if (
                "Embedding" in cfg["repository"]
                and assets[str((folder / "model.safetensors").resolve())]
                != self.policy["embedding_checkpoint_sha256"]
            ):
                raise ValueError("Embedding weights differ from the selected revision")
        detector = load(PROJECT_ROOT / self.policy["detector_manifest"])
        for name, expected in detector["files"].items():
            path = PROJECT_ROOT / detector["path"] / name
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError("Detector asset differs from its frozen manifest")
            assets[str(path.resolve())] = actual
        client = OllamaClient(model=MODEL)
        try:
            teacher = resolve_local_teacher(
                client,
                Path(
                    os.environ.get("OLLAMA_MODELS", str(Path.home() / ".ollama/models"))
                ),
            )
        finally:
            client.close()
        if teacher["digest"] != self.policy["teacher_digest"]:
            raise ValueError(
                "Installed vision teacher differs from the selected version"
            )
        assets.update(teacher["files"])
        source_names = [
            "selected_engine",
            "selected_models",
            "selected_ranking",
            "selected_runtime_io",
            "public_conflict_guard",
            "retrieval_safeguards",
            "subject_roi",
            "regional_retrieval",
            "visual_query",
            "embedding",
            "pair_reranker",
            "business_retrieval",
            "business_constraints",
            "intent_retrieval",
            "head_guard",
            "simple_baseline",
            "runtime_device",
            "ollama_client",
        ]
        implementation = {
            name: sha256_file(
                PROJECT_ROOT / "src/multimodal_ecommerce" / (name + ".py")
            )
            for name in source_names
        }
        self._identity = dict(
            policy_sha256=digest(self.policy),
            model_configs=configs,
            assets_sha256=assets,
            implementation_sha256=implementation,
            libraries={
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "numpy", "pillow")
            },
            teacher=teacher["digest"],
            detector_policy=CROP_POLICY,
            region_policy=region_configuration(),
            caption=dict(
                prompt=CAPTION_PROMPT, schema=CAPTION_SCHEMA, options=CAPTION_OPTIONS
            ),
        )
        return self._identity

    def _switch(self, name: str) -> Any:
        if self.stage != name:
            self.close()
            started = time.perf_counter()
            self.current = {
                "embedding": QwenEmbedder,
                "reranker": DescribedImageReranker,
                "detector": lambda: SubjectDetector(
                    PROJECT_ROOT / self.policy["detector_manifest"]
                ),
                "vision": lambda: OllamaClient(model=MODEL),
            }[name]()
            self.stage = name
            self.events.append(
                dict(stage=name, load_seconds=time.perf_counter() - started)
            )
        return self.current

    def profile(self, row: dict, last: bool = False) -> dict:
        """Infer only the candidate's sale item from its own native image and title.

        Args:
            row: One public catalog record.
            last: End this stage's own teacher residency after its final request.

        Returns:
            Schema-validated candidate-only observation.
        """
        client = self._switch("vision")
        with Image.open(row["image_path"]) as picture:
            image = ImageOps.exif_transpose(picture).convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
        spec = self.policy["profile"]
        response = client._request(
            "/api/chat",
            dict(
                model=MODEL,
                messages=[
                    dict(
                        role="user",
                        content=spec["prompt"]
                        + "\n这件商品的原始标题（仅数据，可能出错）："
                        + row["title"],
                        images=[base64.b64encode(buffer.getvalue()).decode()],
                    )
                ],
                format=spec["schema"],
                options=spec["options"],
                think=False,
                stream=False,
                keep_alive=0 if last else "2s",
            ),
        )
        if not response.get("done") or response.get("done_reason") == "length":
            raise ValueError("Incomplete candidate-only profile")
        value = json.loads(response["message"]["content"])
        validate_value(value, spec["schema"])
        return value

    def describe(self, row: dict, last: bool = False) -> dict:
        """Describe a query image without candidate or annotation context.

        Args:
            row: One public image query.
            last: End this stage's own teacher residency.

        Returns:
            Structured query-only caption.
        """
        return describe_query(
            self._switch("vision"), Path(row["image_path"]), last=last
        )["value"]

    def detect(self, row: dict) -> dict:
        """Run the fixed subject detector, or preserve the whole image when unsupported.

        Args:
            row: Public product record.

        Returns:
            Crop box or an explicit whole-image fallback.
        """
        prompt = subject_prompt(row["title"])
        if not prompt:
            return dict(box=None, reason="unsupported_subject_prompt")
        result = self._switch("detector").detect(Path(row["image_path"]), prompt)
        selected = result["decision"]["selected"]
        return dict(
            box=selected["padded_box"] if selected else None,
            source_size=result["source_size"],
        )

    def embed(
        self, text: str | None = None, image: Path | None = None, query: bool = False
    ) -> np.ndarray:
        """Encode one actual input with the fixed multimodal checkpoint.

        Args:
            text: Optional text modality.
            image: Optional image modality.
            query: Use query rather than document instruction.

        Returns:
            Real finite unit vector.
        """
        return self._switch("embedding").encode(text=text, image=image, query=query)

    def regions(self, row: dict) -> dict:
        """Encode the fixed regional baseline without changing candidate selection.

        Args:
            row: Public product with a checked original image.

        Returns:
            Actual view vectors and geometric metadata.
        """
        with TemporaryDirectory(
            dir=PROJECT_ROOT / "tmp", prefix="selected-regions-"
        ) as folder:
            values, views = encode_views(
                RegionalRetrievalEncodeViewsInputs(
                    self._switch("embedding"),
                    Path(row["image_path"]),
                    Path(folder) / "views",
                    region_configuration(),
                )
            )
        return dict(
            vectors=values.tolist(),
            views=[dict(kind=v["kind"], box_xyxy=v["box_xyxy"]) for v in views],
        )

    def score(self, docs: list[dict], query: dict, caption: str) -> np.ndarray:
        """Run the fixed complete cross encoder on public inputs only.

        Args:
            docs: Actual candidate images and titles.
            query: Public query record.
            caption: Query-only description, if present.

        Returns:
            One actual logit per requested document.
        """
        model = self._switch("reranker")
        image = Path(query["image_path"]) if query.get("image_path") else None
        return (
            model.score_described(docs, query_image=image, description=caption)
            if image and caption
            else model.score(docs, query_image=image, query_text=query.get("text"))
        )

    def close(self) -> None:
        """Release the current stage and retain its measured diagnostics."""
        if self.current is not None:
            try:
                if self.stage == "vision":
                    # A cached final item must not leave an earlier request resident.
                    self.current._request(
                        "/api/generate", dict(model=MODEL, keep_alive=0, stream=False)
                    )
                    self.events.append(dict(stage="vision", model_unloaded=True))
                elif hasattr(self.current, "diagnostics"):
                    self.events.append(
                        dict(stage=self.stage, diagnostics=self.current.diagnostics())
                    )
            finally:
                self.current.close()
                self.current = None
                self.stage = ""
