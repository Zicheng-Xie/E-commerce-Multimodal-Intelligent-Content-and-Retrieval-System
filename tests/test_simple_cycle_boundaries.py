"""Check human export integrity and holdout access boundaries without models."""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.human_audit_ui import ASSETS, render_review


def test_consumed_and_reserved_paths_reject_tuning_and_inference() -> None:
    """Consumed labels and new final candidates cannot enter development."""
    registry = dict(
        rules=[
            dict(path_prefix="artifacts/dev/", role="exposed_development"),
            dict(path_prefix="data/used/", role="consumed_business_heldout"),
            dict(
                path_prefix="artifacts/new/",
                role="reserved_final_acceptance",
                model_inference_allowed=False,
                formal_acceptance_ready=False,
            ),
        ],
        exclusion_metadata_allowlist=["data/used/records.jsonl"],
    )
    require_data_access([Path("artifacts/dev/catalog.json")], "development", registry)
    for purpose in ["development", "training", "acceptance_inference"]:
        for name in ["data/used/judgments.jsonl", "artifacts/new/gallery.json"]:
            with pytest.raises(PermissionError):
                require_data_access([Path(name)], purpose, registry)
    require_data_access(
        [Path("data/used/records.jsonl")], "exclusion_metadata", registry
    )
    with pytest.raises(PermissionError):
        require_data_access(
            [Path("data/used/judgments.jsonl")], "exclusion_metadata", registry
        )


def test_pages_hide_machine_judgments_and_escape_original_text(tmp_path: Path) -> None:
    """The human packet must not leak stored AI answers or execute raw titles.

    Args:
        tmp_path: Isolated temporary review folder supplied by pytest.
    """
    task = dict(
        task_id="task-one",
        grade="SECRET_AI_ANSWER",
        reason="SECRET_AI_REASON",
        score=0.987654321,
        query=dict(text="<script>bad()</script>", image_path=None),
        candidate=dict(title="商品 & 属性", image_path=None),
    )
    render_review([task], tmp_path)
    page = (tmp_path / "batch_001.html").read_text()
    assert "SECRET_AI" not in page and "0.987654321" not in page
    assert "<script>bad()" not in page
    assert "&lt;script&gt;bad()" in page
    assert 'id="human"' in page and 'value="uncertain"' in page
    assert not (tmp_path / "human_answers.csv").exists()


def test_browser_csv_preserves_unicode_quotes_and_abstention() -> None:
    """Run the actual JavaScript exporter and parse its CSV with Python."""
    row = dict(
        task_id="one",
        reviewer_type="human",
        reviewer="测试者",
        reviewed_at="2026-11-05T00:00:00Z",
        human_grade="uncertain",
        reason='用途不明, 标题含"引号"\n需核实',
    )
    command = (
        "const api=require(process.argv[1]);const row=JSON.parse(process.argv[2]);"
        + "process.stdout.write(api.exportCsv([row,{...row,human_grade:''}]));"
    )
    node = shutil.which("node")
    assert node is not None, "Node is required for browser CSV export verification"
    result = subprocess.run(
        [
            node,
            "-e",
            command,
            str(ASSETS / "human_audit.js"),
            json.dumps(row),
        ],
        check=True,
        capture_output=True,
    )
    decoded = result.stdout.decode("utf-8-sig")
    assert list(csv.DictReader(io.StringIO(decoded))) == [row]


def test_image_batch_rejects_invalid_model_outputs(tmp_path: Path) -> None:
    """The adapter must fail on incomplete or zero embeddings instead of indexing.

    Args:
        tmp_path: Temporary location for a valid minimal test image.
    """
    from unittest.mock import Mock

    image = tmp_path / "image.png"
    Image.new("RGB", (32, 32)).save(image)
    encoder = object.__new__(QwenEmbedder)
    encoder.config = dict(document_instruction="doc", query_instruction="query")
    encoder.runtime = Mock()
    encoder.backend = Mock()
    encoder.timings = []
    with pytest.raises(ValueError):
        encoder.encode_images([])
    for values in [torch.zeros(1, 2048), torch.ones(2, 2048)]:
        encoder.backend.process.return_value = values
        with pytest.raises(RuntimeError):
            encoder.encode_images([image])
    encoder.backend.process.return_value = torch.ones(1, 2048)
    actual = encoder.encode_images([image], query=True)
    assert np.allclose(np.linalg.norm(actual, axis=1), 1)
    assert encoder.backend.process.call_args[0][0][0]["instruction"] == "query"
