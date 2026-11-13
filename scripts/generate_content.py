"""Generate a factual draft and an explicitly separate controlled marketing line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.content_contract import MARKETING_LINES, generate_content
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.safety import PROJECT_ROOT, ensure_output_path, write_json


def main() -> None:
    """Use the existing local model with source checks and missing-field abstention.

    Raises:
        FileExistsError: Choose a new output path to preserve prior drafts.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", type=Path, required=True)
    parser.add_argument("--style", choices=list(MARKETING_LINES), default="neutral")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = ensure_output_path(PROJECT_ROOT / args.output)
    if output.exists():
        raise FileExistsError("Choose a new output path to preserve prior drafts")
    product = json.loads((PROJECT_ROOT / args.product).read_text(encoding="utf-8-sig"))
    client = OllamaClient()
    try:
        result = generate_content(client, product, args.style)
    finally:
        client.close()
    write_json(output, result)
    print(
        json.dumps(
            {"status": result["status"], "output": str(output)}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
