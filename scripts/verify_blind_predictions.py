"""Replay every frozen blind ranking without reading the answer labels."""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import run  # noqa: F401,E402
from multimodal_ecommerce.blind_verification import replay_predictions  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(replay_predictions(), ensure_ascii=False, indent=2))
