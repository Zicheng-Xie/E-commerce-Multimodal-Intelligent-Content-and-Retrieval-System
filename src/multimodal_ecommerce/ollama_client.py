"""Local-only Ollama inference and model storage verification."""

from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from PIL import Image, ImageOps

from multimodal_ecommerce.exceptions import EcommerceError

CHAT_OPTIONS: dict[str, int] = {
    "num_ctx": 4096,
    "num_predict": 512,
    "temperature": 0,
    "seed": 42,
}


class OllamaError(EcommerceError, RuntimeError):
    """An actionable local model or service failure."""


class OllamaClient:
    """Access local Qwen without automatically downloading any model."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3.5:4b-q4_K_M",
        timeout: float = 240,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialize the component and validate its configured dependencies.

        Args:
            base_url: Base url used by this operation.
            model: Model used by this operation.
            timeout: Timeout used by this operation.
            client: Configured local model client used for inference.

        Raises:
            ValueError: Only a local Ollama service is supported; Use an unauthenticated
                local HTTP endpoint; Cloud models are outside this local-only project.
        """
        parsed = urlparse(base_url)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Only a local Ollama service is supported.")
        if parsed.scheme != "http" or parsed.username or parsed.password:
            raise ValueError("Use an unauthenticated local HTTP endpoint.")
        if "cloud" in model.lower():
            raise ValueError("Cloud models are outside this local-only project.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = client or httpx.Client(timeout=timeout, trust_env=False)

    def close(self) -> None:
        """Release HTTP connections."""
        self.client.close()

    def _request(self, endpoint: str, payload: dict | None = None) -> dict:
        try:
            if payload is None:
                response = self.client.get(self.base_url + endpoint)
            else:
                response = self.client.post(self.base_url + endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise OllamaError(f"Ollama {endpoint} failed: {error}") from error
        if not isinstance(result, dict) or result.get("error"):
            raise OllamaError(f"Invalid Ollama response: {result}")
        return result

    def health(self) -> dict[str, Any]:
        """Report service version and installed models without loading weights.

        Returns:
            Mapping with fields: version, base_url, model, model_installed, models.
        """
        version = self._request("/api/version")
        models = self._request("/api/tags").get("models", [])
        return {
            "version": version.get("version"),
            "base_url": self.base_url,
            "model": self.model,
            "model_installed": any(item["name"] == self.model for item in models),
            "models": models,
        }

    def verify_model_storage(
        self, expected_root: Path, reference_model: str | None = "llama3:8b"
    ) -> dict[str, Any]:
        """Cross-check the running server's FROM path and on-disk manifest.

        Args:
            expected_root: Expected root used by this operation.
            reference_model: Reference model used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        root = expected_root.resolve()
        report = self.health()
        for label, model in (("reference", reference_model), ("model", self.model)):
            if model is None:
                continue
            if label == "model" and not report["model_installed"]:
                continue
            details = self._request("/api/show", {"model": model})
            match = re.search(r"^FROM\s+(.+)$", details.get("modelfile", ""), re.M)
            if match is None:
                raise OllamaError(f"Cannot verify FROM for {model}")
            blob = Path(match.group(1).strip().strip('"')).resolve()
            if not blob.is_relative_to(root / "blobs") or not blob.is_file():
                raise OllamaError(f"Unexpected model storage: {blob}")
            name, tag = model.rsplit(":", 1)
            manifest = root / "manifests/registry.ollama.ai/library" / name / tag
            document = json.loads(manifest.read_text(encoding="utf-8"))
            digests = {
                layer["digest"].replace(":", "-") for layer in document["layers"]
            }
            if blob.name not in digests:
                raise OllamaError(f"Manifest and server blob disagree: {model}")
            report[f"{label}_blob"] = str(blob)
            report[f"{label}_manifest"] = str(manifest)
            report[f"{label}_details"] = details.get("details", {})
        report["expected_root"] = str(root)
        return report

    def chat(
        self,
        prompt: str,
        image_path: Path | None = None,
        json_schema: dict | None = None,
    ) -> dict[str, Any]:
        """Run deterministic bounded inference; resize image in memory only.

        Args:
            prompt: Prompt used by this operation.
            image_path: Image path used by this operation.
            json_schema: Json schema used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.

        Raises:
            OllamaError: Generation was incomplete or truncated; Model returned no
                visible answer.
        """
        message: dict[str, Any] = {"role": "user", "content": prompt}
        if image_path is not None:
            with Image.open(image_path) as original:
                picture = ImageOps.exif_transpose(original).convert("RGB")
                picture.thumbnail((768, 768))
                buffer = io.BytesIO()
                picture.save(buffer, format="JPEG", quality=90)
            message["images"] = [base64.b64encode(buffer.getvalue()).decode("ascii")]
        payload = {
            "model": self.model,
            "messages": [message],
            "stream": False,
            "think": False,
            "keep_alive": "2m",
            "options": dict(CHAT_OPTIONS),
        }
        if json_schema is not None:
            payload["format"] = json_schema
        started = time.perf_counter()
        result = self._request("/api/chat", payload)
        result["wall_seconds"] = time.perf_counter() - started
        if not result.get("done") or result.get("done_reason") == "length":
            raise OllamaError("Generation was incomplete or truncated.")
        if not result.get("message", {}).get("content", "").strip():
            raise OllamaError("Model returned no visible answer.")
        return result
