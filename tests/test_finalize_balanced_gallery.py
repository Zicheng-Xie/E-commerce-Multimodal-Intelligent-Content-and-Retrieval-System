"""Report export consumes real metric schema, preserves unknown endpoints/order."""

from __future__ import annotations

import pytest

import finalize_balanced_gallery as finalization


def test_report_uses_ranges_and_explicit_six_metric_order() -> None:
    """Verify that report uses ranges and explicit six metric order."""
    ranges: dict = {
        m: [i / 100, (i + 1) / 100] for i, m in enumerate(finalization.METRICS)
    }
    reversed_ranges: dict = dict(reversed(list(ranges.items())))
    cells = finalization.report_cells(dict(ranges=reversed_ranges))
    assert cells == [f"{i:.2f} / {i+1:.2f}" for i in range(6)]
    with pytest.raises(KeyError):
        finalization.report_cells(dict(macro_ranges=ranges))


def test_undefined_recall_endpoint_is_not_exported_as_zero() -> None:
    """Verify that undefined recall endpoint is not exported as zero."""
    ranges: dict = {m: [0.0, 0.0] for m in finalization.METRICS}
    ranges["recall@10"] = [None, None]
    cells = finalization.report_cells(dict(ranges=ranges))
    assert (
        cells[list(finalization.METRICS).index("recall@10")] == "undefined / undefined"
    )
    assert len(cells) == 6
