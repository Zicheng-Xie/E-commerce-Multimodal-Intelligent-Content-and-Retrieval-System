"""Regression checks for import side effects, trusted loading and error
compatibility.
"""

from __future__ import annotations

import importlib
import sys
import urllib.request
from pathlib import Path
from typing import NoReturn

import httpx
import pytest

from multimodal_ecommerce import module_loading
from multimodal_ecommerce.copy_policy import SourceConflictError
from multimodal_ecommerce.exceptions import EcommerceError
from multimodal_ecommerce.ollama_client import OllamaError
from multimodal_ecommerce.pair_reranker import reference_module


@pytest.mark.parametrize(
    "module_name",
    [
        "verify_business_review_setup",
        "prepare_semantic_runtime",
        "prepare_dino_metadata",
        "serve_retrieval",
        "business_query_review",
        "human_business_pipeline",
        "business_review_jobs",
        "download_dino",
        "download_reranker",
        "benchmark_resident",
        "evaluate_business_constraints",
        "train_adjudicated_ranker",
        "prepare_reranker_metadata",
    ],
)
def test_cli_import_does_not_start_work(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Importing a CLI must not write artifacts, contact services or start inference.

    Args:
        module_name: Reviewed command-line module to import.
        monkeypatch: Fixture restoring the import cache and patched operations.
    """

    def _reject(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("CLI import attempted an external operation")

    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", _reject)
    monkeypatch.setattr(httpx, "Client", _reject)
    monkeypatch.setattr(Path, "mkdir", _reject)
    monkeypatch.setattr(Path, "write_text", _reject)
    monkeypatch.setattr(Path, "write_bytes", _reject)

    module = importlib.import_module(module_name)

    assert callable(module.main)


def test_dynamic_loader_registers_dataclass_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Future annotations on dataclasses require registration before module
    execution.

    Args:
        tmp_path: Isolated project root.
        monkeypatch: Fixture restoring the trusted-source root.
    """
    monkeypatch.setattr(module_loading, "PROJECT_ROOT", tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    source = scripts / "stage.py"
    source.write_text(
        "from __future__ import annotations\n"
        + "from dataclasses import dataclass\n"
        + '@dataclass\nclass Stage:\n    """Controlled stage fixture."""\n'
        + "    count: int = 3\nvalue = Stage()\n",
        encoding="utf-8",
    )
    module = module_loading.load_project_module("test_dataclass", source)
    try:
        assert module.value.count == 3
        assert sys.modules[module.__name__] is module
    finally:
        sys.modules.pop(module.__name__, None)


def test_dynamic_loader_rejects_source_outside_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Untrusted source must be rejected before its code can execute.

    Args:
        tmp_path: Isolated project root.
        monkeypatch: Fixture restoring the trusted-source root.
    """
    monkeypatch.setattr(module_loading, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "untrusted.py"
    source.write_text('raise AssertionError("untrusted source executed")\n')

    with pytest.raises(ValueError, match="trusted project script"):
        module_loading.load_project_module("test_escape", source)


def test_dynamic_loader_rolls_back_failed_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing module must not leave a partially initialized import-cache entry.

    Args:
        tmp_path: Isolated project root.
        monkeypatch: Fixture restoring the trusted-source root.
    """
    monkeypatch.setattr(module_loading, "PROJECT_ROOT", tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    source = scripts / "broken.py"
    source.write_text('raise RuntimeError("controlled import failure")\n')

    with pytest.raises(RuntimeError, match="controlled import failure"):
        module_loading.load_project_module("test_failure", source)

    assert "ecommerce_runtime_test_failure" not in sys.modules


def test_reference_compilation_rejects_unpinned_source(tmp_path: Path) -> None:
    """The vendor compatibility compiler must not execute arbitrary Python files.

    Args:
        tmp_path: Isolated location for the deliberately untrusted source.
    """
    source = tmp_path / "not_vendor.py"
    source.write_text('raise AssertionError("untrusted vendor code executed")\n')

    with pytest.raises(ValueError, match="pinned vendored reference"):
        reference_module(source)


def test_business_error_hierarchy_preserves_existing_catches() -> None:
    """Business-wide catches and the original standard exception catches both work."""
    assert issubclass(SourceConflictError, EcommerceError)
    assert issubclass(SourceConflictError, ValueError)
    assert issubclass(OllamaError, EcommerceError)
    assert issubclass(OllamaError, RuntimeError)
