"""Verify content validation, resumption and split leakage."""

from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from multimodal_ecommerce import safety
from multimodal_ecommerce.media import fetch_image, image_url, localize_products


def _image() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), "blue").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/a",
        "https://img.alicdn.com.evil.test/a",
        "file:///a",
        "https://user@img.alicdn.com/a",
    ],
)
def test_refuse_non_image_hosts(url: str) -> None:
    """Reject private and arbitrary hosts in dataset URLs.

    Args:
        url: Url used by this operation.
    """
    with pytest.raises(ValueError):
        image_url(url)


def test_redirect_destination_checked() -> None:
    """Validate redirects before another request is made."""
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                302, headers={"location": "http://localhost/a"}
            )
        )
    )
    with pytest.raises(ValueError):
        fetch_image(client, "https://img.alicdn.com/a")
    client.close()


def test_download_cache_failure_and_cross_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache avoids requests; exact train/test duplicates are excluded from eval.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "samples.jsonl"
    records = [
        {
            "id": f"product/{split}/{number}",
            "source_id": str(number),
            "split": split,
            "dataset": "product",
            "category": "家居日用",
            "image_url": f"http://img.alicdn.com/{number}",
        }
        for split, number in [("train", 1), ("test", 2), ("test", 3)]
    ]
    source.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    before = source.read_bytes()
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Return handler for this workflow.

        Args:
            request: Request used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        calls.append(str(request.url))
        content = b"not an image" if request.url.path == "/3" else _image()
        return httpx.Response(200, content=content)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    output = tmp_path / "localized"
    report = localize_products(source, output, client)
    assert report["downloaded"] == 2
    assert len(report["failures"]) == 1
    assert report["excluded_test_ids"] == ["product/test/2"]
    assert report["duplicates"][0]["cross_split"] is True
    repeated = localize_products(source, output, client)
    assert repeated["cache_hits"] == 2
    assert len(calls) == 4
    assert source.read_bytes() == before
    rows = [
        json.loads(line)
        for line in (output / "products.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[1]["evaluation_eligible"] is False
    assert rows[2]["image_path"] is None
    assert list((output / "images").glob("*.png"))
    client.close()
