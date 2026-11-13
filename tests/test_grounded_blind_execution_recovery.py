"""Verify actual runtime argument contract and isolated recovery namespace."""

from __future__ import annotations

from pathlib import Path

import pytest

import grounded_blind_execution_recovery as recovery
from multimodal_ecommerce.regional_retrieval import RegionalRetrievalEncodeViewsInputs


def test_corrected_abi_preserves_real_configuration_and_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify that corrected abi preserves real configuration and output.

    Args:
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
        tmp_path: Isolated temporary directory supplied by pytest.
    """
    received = []

    def record(
        options: RegionalRetrievalEncodeViewsInputs,
    ) -> str:
        """Return record for this workflow.

        Args:
            options: The complete request forwarded to the real view encoder.

        Returns:
            Text produced by this operation.
        """
        received.append(
            (
                options.embedder,
                options.image,
                options.output,
                options.config,
                options.query,
            )
        )
        return "actual result"

    monkeypatch.setattr(recovery, "encode_views", record)
    config: dict = {"zoom_max_edge": 768}
    image = tmp_path / "original.png"
    output = tmp_path / "views"
    engine = object()
    assert (
        recovery.ordered_views(
            RegionalRetrievalEncodeViewsInputs(
                engine, image, output, config, query=True
            )
        )
        == "actual result"
    )
    assert received == [(engine, image, output, config, True)]


def test_execution_namespace_does_not_mutate_frozen_original() -> None:
    """Verify that execution namespace does not mutate frozen original."""
    previous = recovery.original.encode_views
    guard = recovery.original.dataset.guard
    module = recovery.runtime()
    assert module.encode_views is recovery.ordered_views
    assert recovery.original.encode_views is previous
    assert (
        recovery.original.dataset.guard is guard
        and module.dataset is not recovery.original.dataset
    )
