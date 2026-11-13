"""Machine provenance, null uncertainty and platform-bound local checkpoint checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from multimodal_ecommerce.contracts import OllamaLayer
from multimodal_ecommerce.machine_teacher import resolve_local_teacher, strict_grades
from multimodal_ecommerce.model_interfaces import ModelInspector


def test_machine_unknown_is_null_and_bool_is_not_a_label() -> None:
    """Verify that machine unknown is null and bool is not a label."""
    assert strict_grades([1, -1, 0], 3) == [1, None, 0]
    for value in [[True, 0, 1], ["1", 0, 1], [0.5, 0, 1], [1, 0]]:
        with pytest.raises(ValueError):
            strict_grades(value, 3)


def local_fixture(tmp_path: Path) -> tuple[Path, ModelInspector, str, list]:
    """Return local fixture for this workflow.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.

    Returns:
        Values produced by this operation, in the stated return order.

    Notes:
        Writes derived artifacts through (root / 'blobs' / ('sha256-' +
        digest)).write_bytes, (root / 'blobs').mkdir, manifest.parent.mkdir,
        manifest.write_text.
    """
    root = tmp_path / "models"
    (root / "blobs").mkdir(parents=True)
    layers: list[OllamaLayer] = []
    for kind, content in [("model", b"weights"), ("projector", b"vision")]:
        digest = hashlib.sha256(content).hexdigest()
        (root / "blobs" / ("sha256-" + digest)).write_bytes(content)
        layers.append(
            dict(
                mediaType="application/vnd.ollama.image." + kind,
                digest="sha256:" + digest,
                size=len(content),
            )
        )
    content = json.dumps(dict(layers=layers)).encode()
    digest = hashlib.sha256(content).hexdigest()
    (root / "blobs" / ("sha256-" + digest)).write_bytes(content)
    manifest = root / "manifests/registry.ollama.ai/library/qwen3.5/4b-q4_K_M"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(dict(layers=[dict(digest="sha256:" + digest)])), encoding="utf-8"
    )

    class FakeClient:
        """Return a controlled local model manifest for checkpoint provenance tests."""

        def health(self) -> dict:
            """Return health for this workflow.

            Returns:
                Mapping with fields: version, models.
            """
            return dict(
                version="test",
                models=[
                    dict(
                        name="qwen3.5:4b-q4_K_M", digest=digest, capabilities=["vision"]
                    )
                ],
            )

        def _request(self, endpoint: str, payload: dict | None = None) -> dict:
            """Return the controlled model-show payload for this checkpoint fixture."""
            return dict(
                modelfile="FROM "
                + str(root / "blobs" / layers[0]["digest"].replace(":", "-", 1)),
                details={},
            )

    return root, FakeClient(), digest, layers


def test_child_platform_model_and_projector_are_verified(tmp_path: Path) -> None:
    """Verify that child platform model and projector are verified.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
    """
    root, client, digest, _ = local_fixture(tmp_path)
    actual = resolve_local_teacher(client, root)
    assert actual["digest"] == digest
    assert len(actual["files"]) == 4


def test_wrong_local_projector_bytes_are_rejected(tmp_path: Path) -> None:
    """Verify that wrong local projector bytes are rejected.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.

    Notes:
        Writes derived artifacts through (root / 'blobs' /
        layers[1]['digest'].replace(':', '-', 1)).write_bytes.
    """
    root, client, _, layers = local_fixture(tmp_path)
    (root / "blobs" / layers[1]["digest"].replace(":", "-", 1)).write_bytes(b"changed")
    with pytest.raises(ValueError, match="layer changed"):
        resolve_local_teacher(client, root)
