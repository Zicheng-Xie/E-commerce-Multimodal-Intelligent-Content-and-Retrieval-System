"""Pinned local vision teacher for explicitly machine-origin development signals."""

from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from multimodal_ecommerce.model_interfaces import ModelInspector
from multimodal_ecommerce.ollama_client import OllamaClient, OllamaError
from multimodal_ecommerce.safety import sha256_file

MODEL = "qwen3.5:4b-q4_K_M"


OPTIONS: dict[str, int] = {
    "num_ctx": 8192,
    "num_predict": 256,
    "temperature": 0,
    "seed": 2026102205,
}


def resolve_local_teacher(client: ModelInspector, root: Path) -> dict:
    """Resolve the server-selected OCI platform manifest, including the projector.

    Args:
        client: Configured local model client used for inference.
        root: Root directory used to resolve the operation artifacts.

    Returns:
        Mapping with fields: model, digest, server_version, details, files,
        manifest_resolution.

    Raises:
        ValueError: Exactly one installed local vision teacher is required; Invalid
            installed teacher digest; Resolved manifest checksum mismatch; Server
            manifest is not linked from configured local tag; Cannot resolve the actual
            server-selected local checkpoint; Teacher checkpoint layer changed; Teacher
            model/projector binding is ambiguous; Server FROM and selected platform
            manifest disagree.
    """
    import platform

    if platform.system() == "Darwin" and str(root).startswith("E:"):
        root = Path.home() / ".ollama/models"
    health = client.health()
    matches = [m for m in health["models"] if m["name"] == MODEL]
    if len(matches) != 1 or "vision" not in matches[0].get("capabilities", []):
        raise ValueError("Exactly one installed local vision teacher is required")
    tag = matches[0]
    manifest_path = root / "manifests/registry.ollama.ai/library/qwen3.5/4b-q4_K_M"
    parent = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = tag["digest"].removeprefix("sha256:")
    if not re.fullmatch("[a-f0-9]{64}", digest):
        raise ValueError("Invalid installed teacher digest")
    selected = root / "blobs" / ("sha256-" + digest)
    if selected.is_file():
        if sha256_file(selected) != digest:
            raise ValueError("Resolved manifest checksum mismatch")
        refs = [r["digest"].removeprefix("sha256:") for r in parent.get("layers", [])]
        if digest not in refs and sha256_file(manifest_path) != digest:
            raise ValueError("Server manifest is not linked from configured local tag")
        resolved = json.loads(selected.read_text(encoding="utf-8"))
    elif sha256_file(manifest_path) == digest:
        selected = manifest_path
        resolved = parent
    else:
        raise ValueError("Cannot resolve the actual server-selected local checkpoint")
    files = {
        str(manifest_path): sha256_file(manifest_path),
        str(selected): sha256_file(selected),
    }
    models = []
    projectors = []
    for layer in resolved["layers"]:
        blob = root / "blobs" / layer["digest"].replace(":", "-", 1)
        if (
            not blob.is_file()
            or blob.stat().st_size != layer["size"]
            or sha256_file(blob) != layer["digest"].removeprefix("sha256:")
        ):
            raise ValueError("Teacher checkpoint layer changed")
        files[str(blob)] = sha256_file(blob)
        if layer["mediaType"].endswith(".model"):
            models.append(blob.resolve())
        if layer["mediaType"].endswith(".projector"):
            projectors.append(blob.resolve())
    show = client._request("/api/show", {"model": MODEL})
    match = re.search(r"^FROM\s+(.+)$", show.get("modelfile", ""), re.M)
    if match is None or len(models) != 1 or len(projectors) != 1:
        raise ValueError("Teacher model/projector binding is ambiguous")
    active = Path(match.group(1).strip().strip('"')).resolve()
    if not active.is_relative_to(root.resolve() / "blobs") or active != models[0]:
        raise ValueError("Server FROM and selected platform manifest disagree")
    return dict(
        model=MODEL,
        digest=digest,
        server_version=health["version"],
        details=show.get("details"),
        files=files,
        manifest_resolution="server-selected platform child manifest verified against parent and all local blobs",
    )


def strict_grades(value: object, size: int) -> list[int | None]:
    """Return strict grades for this workflow.

    Args:
        value: Input value to validate or transform.
        size: Size used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Machine grade count mismatch; Machine grades must be integer -1/0/1.
    """
    if not isinstance(value, list) or len(value) != size:
        raise ValueError("Machine grade count mismatch")
    if any(type(v) is not int or v not in (-1, 0, 1) for v in value):
        raise ValueError("Machine grades must be integer -1/0/1")
    return [None if v == -1 else v for v in value]


def infer_teacher(
    client: OllamaClient, prompt: str, schema: dict, image: Path | None = None
) -> dict:
    """Return infer teacher for this workflow.

    Args:
        client: Configured local model client used for inference.
        prompt: Prompt used by this operation.
        schema: Validation schema for the supplied serialized value.
        image: Image input used by the operation.

    Returns:
        Mapping with fields: result, response.

    Raises:
        OllamaError: Machine teacher returned incomplete output; Teacher prompt too
            close to context limit.
        ValueError: Teacher output must be an object.

    Notes:
        Writes derived artifacts through picture.save.
    """
    message: dict[str, Any] = {"role": "user", "content": prompt}
    if image:
        with Image.open(image) as source:
            picture = ImageOps.exif_transpose(source).convert("RGB")
            picture.thumbnail((512, 512))
            buffer = io.BytesIO()
            picture.save(buffer, format="JPEG", quality=92)
        message["images"] = [base64.b64encode(buffer.getvalue()).decode("ascii")]
    start = time.perf_counter()
    response = client._request(
        "/api/chat",
        {
            "model": MODEL,
            "messages": [message],
            "stream": False,
            "think": False,
            "format": schema,
            "keep_alive": "10m",
            "options": OPTIONS,
        },
    )
    response["wall_seconds"] = time.perf_counter() - start
    if not response.get("done") or response.get("done_reason") == "length":
        raise OllamaError("Machine teacher returned incomplete output")
    if (
        response.get("prompt_eval_count", 0)
        >= OPTIONS["num_ctx"] - OPTIONS["num_predict"]
    ):
        raise OllamaError("Teacher prompt too close to context limit")
    content = response.get("message", {}).get("content", "")
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("Teacher output must be an object")
    return {"result": result, "response": response}
