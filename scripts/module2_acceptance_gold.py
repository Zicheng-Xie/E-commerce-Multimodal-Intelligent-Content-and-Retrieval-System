"""Independent human authoring tools; do not invoke on real heldout data during development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.selected_gold import (
    freeze_gold,
    prepare_packet,
    text_template,
)


def main() -> None:
    """Prepare blinded human work only after a verified engineering release exists."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("text-template", "packet", "freeze"))
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "text-template": text_template,
        "packet": prepare_packet,
        "freeze": freeze_gold,
    }[args.action](args.release, args.holdout, args.output)
    print(
        json.dumps(
            result or dict(text_template_prepared=True), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
