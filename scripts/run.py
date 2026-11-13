"""Run the project without changing PowerShell execution policies."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from multimodal_ecommerce.evaluation_access import check_invoked_command  # noqa: E402

check_invoked_command(Path(sys.argv[0]))

for variable, relative in {
    "UV_CACHE_DIR": ".cache/uv",
    "PIP_CACHE_DIR": ".cache/pip",
    "HF_HOME": ".cache/huggingface",
    "TORCH_HOME": ".cache/torch",
    "TORCHINDUCTOR_CACHE_DIR": ".cache/torchinductor",
    "TRITON_CACHE_DIR": ".cache/triton",
    "CUDA_CACHE_PATH": ".cache/cuda",
    "PYLINTHOME": ".cache/pylint",
    "TEMP": "tmp",
    "TMP": "tmp",
}.items():
    os.environ[variable] = str(ROOT / relative)
(ROOT / "tmp").mkdir(exist_ok=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from multimodal_ecommerce.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
