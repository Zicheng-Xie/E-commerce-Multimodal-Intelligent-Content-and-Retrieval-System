"""Independent heldout preparation and one-shot prediction/evaluation."""

from __future__ import annotations

import argparse
import json
import os

# Set before numpy/torch imports so small CPU matrix jobs do not oversubscribe.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import run  # noqa: F401,E402
from multimodal_ecommerce.blind_evaluation import (  # noqa: E402
    SUITE,
    freeze_recipe,
    predict,
    prepare,
    score,
    verify_recipe,
)
from multimodal_ecommerce.holdout_data import audit_exposure  # noqa: E402


def main() -> None:
    """Run the command-line workflow defined by this module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("audit", "freeze", "prepare", "predict", "score", "verify")
    )
    args = parser.parse_args()
    if args.action == "audit":
        result = audit_exposure(SUITE / "exposure.json")["counts"]
    elif args.action == "freeze":
        result = {"frozen_at": freeze_recipe()["frozen_at"]}
    elif args.action == "prepare":
        result = prepare()
    elif args.action == "predict":
        result = predict()
    elif args.action == "score":
        report = score()
        result = {
            "state": "consumed",
            "formal_module2_acceptance": False,
            "queries": report["queries"],
            "internal_report": str(SUITE / "internal_report.json"),
        }
    else:
        result = {"verified": bool(verify_recipe(SUITE))}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
