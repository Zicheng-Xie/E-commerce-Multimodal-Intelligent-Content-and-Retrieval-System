"""Reproduce the complete locked CUDA environment without PowerShell scripts."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Synchronize exact dependency versions; keep downloads and temp files on E:."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = str(ROOT / ".cache/uv")
    environment["TEMP"] = str(ROOT / "tmp")
    environment["TMP"] = str(ROOT / "tmp")
    (ROOT / "tmp").mkdir(exist_ok=True)
    command = [
        "uv",
        "pip",
        "sync",
        "--python",
        str(ROOT / ".venv312/Scripts/python.exe"),
        str(ROOT / "requirements-retrieval.lock"),
        "--index",
        "https://download.pytorch.org/whl/cu128",
        "--index-strategy",
        "unsafe-best-match",
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.offline:
        command.append("--offline")
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
