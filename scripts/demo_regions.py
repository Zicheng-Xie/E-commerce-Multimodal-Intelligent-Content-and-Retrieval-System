"""Exercise Chinese text, image and user ROI queries against the product bank."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.regional_retrieval import (
    RegionalIndex,
    RegionalRetrievalEncodeViewsInputs,
    RegionalSearchRequest,
    encode_views,
    write_regional_gallery,
)
from multimodal_ecommerce.safety import PROJECT_ROOT, ensure_output_path, write_json


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Use a new demo directory.

    Notes:
        Writes derived artifacts through output.mkdir, write_json.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=Path, default=Path("data/index_regions_products_v1")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/module2_product_demo_v1")
    )
    args = parser.parse_args()
    output = ensure_output_path(PROJECT_ROOT / args.output)
    if output.exists():
        raise FileExistsError("Use a new demo directory")
    index, texts, _ = RegionalIndex.load(PROJECT_ROOT / args.index)
    subprocess.run(
        ["ollama", "stop", "qwen3.5:4b-q4_K_M"], check=True, capture_output=True
    )
    embedder = QwenEmbedder()
    try:
        output.mkdir(parents=True)
        sections: list[dict] = []
        for text in ("黑色马丁靴", "红魔游戏手机", "儿童绘本收纳书架"):
            started = time.perf_counter()
            query = embedder.encode(text=text, query=True)
            for strategy in ("global", "attention", "attention_fused"):
                ranked = index.search(
                    RegionalSearchRequest(
                        query, strategy=strategy, text_vector=query, text_bank=texts
                    )
                )
                sections.append(
                    {
                        "query": text,
                        "strategy": strategy,
                        "results": ranked,
                        "seconds_since_query_encoding_start": time.perf_counter()
                        - started,
                    }
                )
        # Known gallery image for a nonself integration check, never a quality score.
        row = index.records[0]
        image = Path(row["image_path"])
        excluded = {
            r["id"] for r in index.records if r["image_sha256"] == row["image_sha256"]
        }
        for name, roi in (
            ("whole_image", None),
            ("user_center_roi", [0.2, 0.2, 0.8, 0.8]),
        ):
            started = time.perf_counter()
            query, views = encode_views(
                RegionalRetrievalEncodeViewsInputs(
                    embedder, image, output / name, index.config, query=True, roi=roi
                )
            )
            ranked = index.search(
                RegionalSearchRequest(
                    query,
                    strategy="attention_fused",
                    text_vector=query[0],
                    text_bank=texts,
                    exclude_ids=excluded,
                )
            )
            sections.append(
                {
                    "query": "图片检索 " + name,
                    "query_image": str(image),
                    "roi": roi,
                    "query_views": views,
                    "strategy": "attention_fused",
                    "results": ranked,
                    "query_and_search_seconds": time.perf_counter() - started,
                }
            )
        write_json(
            output / "results.json",
            {
                "scope": "Integration examples only; no relevance labels or quality scores",
                "index_count": len(index.records),
                "sections": sections,
                "diagnostics": embedder.diagnostics(),
            },
        )
        write_regional_gallery(output / "results.html", sections)
        print(
            json.dumps(
                {
                    "sections": len(sections),
                    "index_count": len(index.records),
                    "output": str(output),
                },
                indent=2,
            )
        )
    finally:
        embedder.close()


if __name__ == "__main__":
    main()
