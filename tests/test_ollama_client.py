"""Validate local-only requests, incomplete output and storage mismatches."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from multimodal_ecommerce.ollama_client import OllamaClient, OllamaError


def test_reject_remote_or_cloud() -> None:
    """Local data must not be routed to a cloud endpoint or model."""
    with pytest.raises(ValueError):
        OllamaClient(base_url="https://example.com")
    with pytest.raises(ValueError):
        OllamaClient(model="qwen3.5:cloud")


def test_vision_payload_and_source_unchanged(tmp_path: Path) -> None:
    """Vision requests contain images, bounded context and disabled thinking.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
    """
    picture = tmp_path / "image.png"
    Image.new("RGB", (1600, 1000), "red").save(picture)
    before = picture.read_bytes()
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Return handler for this workflow.

        Args:
            request: Request used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"done": True, "done_reason": "stop", "message": {"content": "红色"}},
        )

    client = OllamaClient(client=httpx.Client(transport=httpx.MockTransport(handler)))
    result = client.chat("颜色？", picture)
    client.close()
    assert result["message"]["content"] == "红色"
    assert calls[0]["messages"][0]["images"]
    assert calls[0]["options"]["num_ctx"] == 4096
    assert calls[0]["think"] is False
    assert picture.read_bytes() == before


@pytest.mark.parametrize(
    "response",
    [
        {"done": False},
        {"done": True, "done_reason": "length", "message": {"content": "partial"}},
        {"done": True, "message": {"content": ""}},
    ],
)
def test_reject_incomplete_generation(response: dict) -> None:
    """Truncation and empty messages cannot pass the availability check.

    Args:
        response: Response used by this operation.
    """
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=response))
    client = OllamaClient(client=httpx.Client(transport=transport))
    with pytest.raises(OllamaError):
        client.chat("test")
    client.close()


def test_storage_refuses_wrong_root(tmp_path: Path) -> None:
    """Do not pull a model when the reference blob belongs to another root.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        """Return handler for this workflow.

        Args:
            request: Request used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        routes: dict = {
            "/api/version": {"version": "test"},
            "/api/tags": {"models": []},
            "/api/show": {"modelfile": f"FROM {tmp_path / 'elsewhere/blob'}"},
        }
        return httpx.Response(200, json=routes[request.url.path])

    client = OllamaClient(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(OllamaError, match="Unexpected model storage"):
        client.verify_model_storage(tmp_path / "expected")
    client.close()
