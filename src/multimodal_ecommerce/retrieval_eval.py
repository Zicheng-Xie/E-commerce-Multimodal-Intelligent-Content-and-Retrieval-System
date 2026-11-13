"""Retrieval diagnostics and a local HTML viewer without claimed ground truth."""

from __future__ import annotations

import html
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.retrieval import VectorIndex
from multimodal_ecommerce.safety import ensure_output_path, write_json

DEMO_QUERIES = [
    "儿童绘本收纳书架",
    "黑色马丁靴",
    "红魔游戏手机",
    "苹果平板键盘",
    "木质水果盘展示架",
    "儿童彩色小挎包",
]


def ranking_metrics(
    ranked_ids: list[str], relevant_ids: set[str], k: int = 10
) -> dict[str, float]:
    """Compute PDF-defined precision/recall only when explicit qrels exist.

    Args:
        ranked_ids: Ranked ids used by this operation.
        relevant_ids: Relevant ids used by this operation.
        k: K used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Need positive k, known relevant IDs and unique results.
    """
    if k < 1 or not relevant_ids or len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("Need positive k, known relevant IDs and unique results")
    hits = len(set(ranked_ids[:k]) & relevant_ids)
    return {f"precision@{k}": hits / k, f"recall@{k}": hits / len(relevant_ids)}


def write_gallery(path: Path, sections: list[dict]) -> None:
    """Render escaped local result cards with images, scores, IDs and titles.

    Args:
        path: Path of the input or output artifact used by this operation.
        sections: Sections used by this operation.
    """
    path = ensure_output_path(path)
    blocks = []
    for section in sections:
        cards = []
        mode = {"image": "图片向量", "text": "标题向量", "fused": "图文等权融合"}.get(
            section.get("target", "image"), "图片向量"
        )
        query_preview = ""
        if section.get("query_image"):
            query_path = os.path.relpath(section["query_image"], path.parent).replace(
                "\\", "/"
            )
            query_preview = (
                '<p>查询图片</p><img style="width:160px;height:160px" src="'
                + html.escape(query_path, quote=True)
                + '">'
            )
        for result in section["results"]:
            relative = os.path.relpath(result["image_path"], path.parent).replace(
                "\\", "/"
            )
            cards.append(
                '<article><img loading="lazy" src="'
                + html.escape(relative, quote=True)
                + '"><b>'
                + html.escape(f"#{result['rank']}  {result['score']:.4f}")
                + "</b><p>"
                + html.escape(result["title"])
                + "</p><small>"
                + html.escape(result["id"])
                + "</small></article>"
            )
        blocks.append(
            "<section><h2>"
            + html.escape(section["query"])
            + '</h2><p class="note">排序：'
            + mode
            + "</p>"
            + query_preview
            + '<div class="cards">'
            + "".join(cards)
            + "</div></section>"
        )
    content = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>本地图文检索结果</title>
<style>body{font-family:system-ui,sans-serif;background:#f4f6f8;color:#17293e;margin:0;padding:32px;max-width:1200px}h1{margin:0 0 12px}.note{color:#526477;line-height:1.8}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}article{background:white;border:1px solid #dce3eb;padding:16px;border-radius:12px}img{width:100%;height:230px;object-fit:contain}b{display:block;margin-top:12px}small{overflow-wrap:anywhere;color:#617186}p{line-height:1.6}section{margin-top:32px}</style>
<h1>千问图文检索基线</h1><p class="note">本地图片真实嵌入 · 2048维 · 默认按图片余弦相似度排序。分数不是概率。示例用于工程检查，不是正式质量评测。图片可能含过期促销信息。</p>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "".join(blocks) + "</html>", encoding="utf-8")


def run_diagnostics(
    embedder: QwenEmbedder, directory: Path, output: Path
) -> dict[str, Any]:
    """Run Chinese text queries, paired diagnostics and non-self image searches.

    Args:
        embedder: Embedder used by this operation.
        directory: Directory used by this operation.
        output: Output used by this operation.

    Returns:
        Mapping with fields: indexed_products, paired_diagnostic, paired_ranks,
        index_self_lookup_hits, scope, examples, diagnostics.

    Raises:
        FileExistsError: Use a new diagnostics output directory.
    """
    output = ensure_output_path(output)
    if output.exists():
        raise FileExistsError("Use a new diagnostics output directory")
    output.mkdir(parents=True)
    index = VectorIndex.load(directory)
    sections = []
    for query in DEMO_QUERIES:
        started = time.perf_counter()
        vector = embedder.encode(text=query, query=True)
        results = index.search(vector, top_k=3, target="image")
        sections.append(
            {
                "kind": "text_to_image",
                "query": query,
                "seconds": time.perf_counter() - started,
                "results": results,
            }
        )
    paired = []
    for record in index.records:
        vector = embedder.encode(text=record["title"], query=True)
        results = index.search(vector, top_k=len(index.records))
        rank = next(item["rank"] for item in results if item["id"] == record["id"])
        paired.append(
            {"id": record["id"], "split": record["split"], "paired_image_rank": rank}
        )
    for record in index.records[:3]:
        vector = embedder.encode(image=Path(record["image_path"]), query=True)
        sections.append(
            {
                "kind": "image_to_image_nonself",
                "query": "相似商品（已排除自身）: " + record["id"],
                "query_image": record["image_path"],
                "results": index.search(vector, top_k=3, exclude_id=record["id"]),
            }
        )
    self_hits = sum(
        index.search(vector, top_k=1)[0]["id"] == record["id"]
        for vector, record in zip(index.images, index.records)
    )
    metrics = {}
    for split in sorted({item["split"] for item in paired}):
        subset = [item for item in paired if item["split"] == split]
        metrics[split] = {
            "count": len(subset),
            **{
                f"paired_hit@{k}": float(
                    np.mean([item["paired_image_rank"] <= k for item in subset])
                )
                for k in (1, 5, 10)
            },
        }
    report = {
        "indexed_products": len(index.records),
        "paired_diagnostic": metrics,
        "paired_ranks": paired,
        "index_self_lookup_hits": self_hits,
        "scope": "Paired-title engineering diagnostic. No independently judged relevance qrels; not PDF Recall@10 or Precision@10.",
        "examples": sections,
        "diagnostics": embedder.diagnostics(),
    }
    write_json(output / "report.json", report)
    write_gallery(output / "results.html", sections)
    return report
