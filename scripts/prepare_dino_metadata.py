"""Inspect a compact complementary visual encoder before downloading weights."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path


def main() -> None:
    """Inspect a compact complementary visual encoder before downloading weights."""
    root = Path(__file__).resolve().parents[1]
    with urllib.request.urlopen(
        "https://huggingface.co/api/models/facebook/dinov2-small?blobs=true", timeout=60
    ) as response:
        meta = json.load(response)
    result = {
        "repository": "facebook/dinov2-small",
        "revision": meta["sha"],
        "files": meta["siblings"],
    }
    (root / "artifacts/module2_precision/dino_metadata.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "revision": meta["sha"],
                "files": [
                    {"name": x["rfilename"], "size": x.get("size")}
                    for x in meta["siblings"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
