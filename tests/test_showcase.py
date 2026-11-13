"""Exercise live public search, exact observation caching and the browser API."""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np
import pytest
from PIL import Image

from multimodal_ecommerce import safety, showcase_engine
from multimodal_ecommerce.safety import sha256_file
from multimodal_ecommerce.selected_runtime_io import atomic_json, digest, load
from multimodal_ecommerce.showcase_engine import ShowcaseEngine
from multimodal_ecommerce.showcase_http import ShowcaseServer


class _Access:
    """Allow only synthetic temporary fixture paths in place of production policy."""

    @staticmethod
    def check(path: Path) -> Path:
        """Resolve a synthetic fixture path.

        Args:
            path: Test fixture location.

        Returns:
            Absolute fixture location.
        """
        return path.resolve()

    output = check


class _Models:
    """Deterministic synthetic observations, never actual quality measurements."""

    def __init__(self) -> None:
        """Track calls so exact cache reuse can be verified."""
        self.calls = 0

    def identity(self) -> dict:
        """Return the fixture model identity.

        Returns:
            Empty synthetic identity.
        """
        return {}

    def embed(self, **kwargs: Any) -> np.ndarray:
        """Provide a finite unit query vector.

        Args:
            **kwargs: Public synthetic query content.

        Returns:
            A unit vector with the actual embedding dimension.
        """
        assert kwargs.get("query")
        self.calls += 1
        return np.eye(1, 2048, dtype=np.float32)[0]

    def describe(self, row: dict, last: bool) -> dict:
        """Provide a query-only caption.

        Args:
            row: Synthetic image record.
            last: Stage release flag.

        Returns:
            Valid caption without answers.
        """
        assert row["image_path"] and last
        self.calls += 1
        return dict(product="耳机", appearance="黑色无线耳机", unclear=False)

    def score(self, docs: list, query: dict, description: str) -> list[float]:
        """Score title content independently of arbitrary product IDs.

        Args:
            docs: Public candidate records.
            query: Public query.
            description: Public optional caption.

        Returns:
            Deterministic synthetic logits.
        """
        assert query["mode"] == "text" or description
        self.calls += 1
        return [float(d["text"].split()[-1]) for d in docs]

    def close(self) -> None:
        """Synthetic backend has no model assets to release."""


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ShowcaseEngine:
    """Create an entirely synthetic public bundle with 24 distinct product images.

    Args:
        tmp_path: Isolated fixture root.
        monkeypatch: Scoped dependency substitution.

    Returns:
        Search engine using synthetic observations and no protected real data.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(showcase_engine, "RuntimeAccess", _Access)
    monkeypatch.setattr(showcase_engine, "reserved_signatures", lambda: [])
    records = []
    for i in range(24):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (9, 9), (i, i * 2, i * 3)).save(path)
        records.append(
            dict(
                id=f"p{i:02}",
                title=f"无线耳机 {i}",
                image_path=str(path),
                image_sha256=sha256_file(path),
            )
        )
    folder = tmp_path / "public_index"
    folder.mkdir()
    bank = np.repeat(np.eye(1, 2048, dtype=np.float32), 24, axis=0)
    np.savez_compressed(folder / "vectors.npz", whole=bank, crop=bank, titles=bank)
    manifest = dict(
        products=records,
        samples=[],
        labels_exported=False,
        supported_types={r["id"]: "headphone" for r in records},
        model_policy={},
        content_id="synthetic_public_catalog",
        vectors_sha256=sha256_file(folder / "vectors.npz"),
    )
    manifest["manifest_sha256"] = digest(manifest)
    atomic_json(folder / "manifest.json", manifest)
    return ShowcaseEngine(tmp_path, _Models())


@pytest.mark.parametrize("mode", ["text", "image"])
def test_live_text_image_search_and_exact_cache(
    engine: ShowcaseEngine, mode: str
) -> None:
    """Both modes rank the whole gallery; exact repeat observations avoid model calls.

    Args:
        engine: Synthetic search instance.
        mode: Query mode.
    """
    request = (
        {"text": "无线耳机"}
        if mode == "text"
        else {"image_token": engine.tokens["p00"]}
    )
    first = engine.search(request, lambda _: None)
    calls = cast(_Models, engine.models).calls
    second = engine.search(request, lambda _: None)
    assert len(first["results"]) == 10
    assert (
        first["ranking"] == second["ranking"]
        and cast(_Models, engine.models).calls == calls
    )
    assert len(first["ranking"]) == (24 if mode == "text" else 23)
    assert "p00" not in first["ranking"] if mode == "image" else True
    assert first["formal_metrics"] is None and calls >= 21


def test_private_fields_and_arbitrary_paths_cannot_be_query_inputs(
    engine: ShowcaseEngine,
) -> None:
    """Answers, file paths and unknown image tokens never reach inference.

    Args:
        engine: Synthetic search instance.
    """
    for request in (
        {"text": "耳机", "human_grade": 1},
        {"image_path": "/etc/passwd"},
        {"image_token": "not_a_registered_image"},
    ):
        with pytest.raises(ValueError):
            engine.search(request, lambda _: None)
    assert cast(_Models, engine.models).calls == 0


def test_sealed_upload_is_denied_before_parsing_or_model_calls(
    engine: ShowcaseEngine,
) -> None:
    """Known sealed bytes are blocked before image decoding and are never persisted.

    Args:
        engine: Synthetic search instance.
    """
    path = engine.assets[engine.tokens["p00"]]
    engine.protected = [dict(sha256=sha256_file(path))]
    with pytest.raises(ValueError, match="封存"):
        engine.upload(path.read_bytes())
    assert (
        not (engine.root / "uploads").exists()
        and cast(_Models, engine.models).calls == 0
    )


def test_uploaded_raster_and_tampered_bundle(engine: ShowcaseEngine) -> None:
    """Normal raster upload yields a bounded token; modified public metadata fails.

    Args:
        engine: Synthetic search instance.
    """
    image = io.BytesIO()
    Image.new("RGB", (20, 20), "red").save(image, format="PNG")
    result = engine.upload(image.getvalue())
    assert result["token"] in engine.assets and result["image_url"].startswith(
        "/media/"
    )
    path = engine.index / "manifest.json"
    manifest = load(path)
    manifest["products"][0]["title"] = "tampered"
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match="manifest changed"):
        ShowcaseEngine(engine.root, _Models())


def test_loopback_browser_catalog_query_jobs_and_origin_isolation(
    engine: ShowcaseEngine,
) -> None:
    """Serve actual API contracts and reject cross-origin/file-path requests.

    Args:
        engine: Synthetic search instance.
    """
    server = ShowcaseServer(("127.0.0.1", 0), engine)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{server.server_port}", trust_env=False
        ) as client:
            assert client.get("/health").json()["products"] == 24
            assert (
                len(client.get("/api/catalog?page=2&size=20").json()["products"]) == 4
            )
            assert (
                client.get(
                    "/api/catalog", headers={"Origin": "http://foreign.invalid"}
                ).status_code
                == 403
            )
            assert client.get("/media/not_a_file").status_code == 404
            assert (
                client.post(
                    "/api/search", json={"image_path": "/etc/passwd"}
                ).status_code
                == 400
            )
            response = client.post("/api/search", json={"text": "耳机"})
            assert response.status_code == 202
            for _ in range(100):
                job = client.get("/api/jobs/" + response.json()["job_id"]).json()
                if job["state"] == "done":
                    break
                time.sleep(0.01)
            assert len(job["result"]["results"]) == 10
    finally:
        server.shutdown()
        server.server_close()
        server.executor.shutdown(wait=True)
        thread.join()
