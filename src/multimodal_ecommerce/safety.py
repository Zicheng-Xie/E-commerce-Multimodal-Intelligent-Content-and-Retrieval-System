"""Constrain generated files to the project and keep source assets read-only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_ROOT = PROJECT_ROOT / "Alibaba阿里算法实习"


def ensure_output_path(path: Path, source_root: Path = PROTECTED_ROOT) -> Path:
    """Validate a destination, resolving junctions and symbolic links.

    Args:
        path: Intended output file or directory.
        source_root: Additional source directory that must remain read-only.

    Returns:
        Absolute destination within this project.

    Raises:
        ValueError: Destination escapes the project or intersects protected data.
    """
    destination = path.resolve()
    project = PROJECT_ROOT.resolve()
    if not destination.is_relative_to(project):
        raise ValueError(f"Output must stay inside project: {destination}")
    for protected in (PROTECTED_ROOT.resolve(), source_root.resolve()):
        if destination.is_relative_to(protected) or protected.is_relative_to(
            destination
        ):
            raise ValueError(f"Output intersects read-only source: {destination}")
    return destination


def sha256_file(path: Path) -> str:
    """Read a file in bounded chunks and return its SHA256 digest.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Text produced by this operation.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any, source_root: Path = PROTECTED_ROOT) -> None:
    """Write UTF-8 JSON only to a validated project destination.

    Args:
        path: Path of the input or output artifact used by this operation.
        data: Data used by this operation.
        source_root: Source root used by this operation.
    """
    destination = ensure_output_path(path, source_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
