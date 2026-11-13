"""Load an isolated implementation namespace from trusted project-owned source."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from multimodal_ecommerce.safety import PROJECT_ROOT


def load_project_module(name: str, path: Path) -> Any:
    """Execute a workflow module inside the project's scripts directory.

    Args:
        name: Identifier distinguishing this isolated execution namespace.
        path: Existing Python source below the current project's scripts root.

    Returns:
        The loaded module. Its symbols are deliberately late-bound because the
        recovery workflows patch different, receipt-verified implementation ABIs.
        This import boundary cannot expose a single static service interface.

    Raises:
        ValueError: The namespace is invalid or the source escapes the scripts root.
        ImportError: Python cannot create a loader for the selected source.
        FileNotFoundError: The requested source file does not exist.
    """
    source = path.resolve()
    scripts_root = (PROJECT_ROOT / "scripts").resolve()
    if not name.isidentifier() or not source.is_relative_to(scripts_root):
        raise ValueError("Expected an identifier and trusted project script source")
    if source.suffix != ".py" or not source.is_file():
        raise FileNotFoundError(f"No project Python implementation at {source}")
    module_name = "ecommerce_runtime_" + name
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create a workflow loader for {source}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Import machinery must roll back registration even after an interrupt.
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module
