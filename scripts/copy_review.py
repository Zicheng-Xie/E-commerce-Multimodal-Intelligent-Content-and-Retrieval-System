"""Render a frozen copy run for human review without changing its evidence."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.evaluation import verify_files
from multimodal_ecommerce.safety import PROJECT_ROOT, ensure_output_path


def main() -> None:
    """Create an escaped read-only page with source facts, outputs and flags.

    Raises:
        FileExistsError: Use a new review page.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = PROJECT_ROOT / args.run_dir
    verify_files(
        directory, json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    )
    output = ensure_output_path(PROJECT_ROOT / args.output)
    if output.exists():
        raise FileExistsError("Use a new review page")
    cards = []
    statuses = {
        "draft_requires_human_review": "待人工评审草稿",
        "blocked_by_rules": "规则标记，待处理",
        "blocked_source_conflict": "来源冲突，未生成",
        "generation_failed": "生成失败",
    }
    for item in read_rows(directory / "results.jsonl"):
        copy = {
            key: value for key, value in item.get("copy", {}).items() if key != "_meta"
        }
        cards.append(
            "<article><h2>"
            + html.escape(item["id"])
            + "</h2><p>"
            + html.escape(item["category"] + " · " + statuses[item["status"]])
            + "</p><div class='columns'><section><h3>来源</h3><pre>"
            + html.escape(json.dumps(item["source"], ensure_ascii=False, indent=2))
            + "</pre></section><section><h3>生成结果</h3><pre>"
            + html.escape(
                json.dumps(
                    copy
                    or {
                        "error": item.get("error"),
                        "reason": item["source_policy"]["findings"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            + "</pre></section></div><details><summary>自动规则与来源约束（不等于人工评分）</summary><pre>"
            + html.escape(
                json.dumps(
                    {
                        "source_policy": item["source_policy"],
                        "output_checks": item.get("copy", {})
                        .get("_meta", {})
                        .get("fact_check"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            + "</pre></details></article>"
        )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>文案基线人工复核</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1200px;margin:auto;padding:28px;background:#f4f6f8;color:#17293e}article{background:white;padding:22px;margin:24px 0;border:1px solid #dce3eb;border-radius:10px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 system-ui,sans-serif}h2{font-size:18px;overflow-wrap:anywhere}details{border-top:1px solid #dce3eb;padding-top:16px}@media(max-width:700px){.columns{grid-template-columns:1fr}}</style><h1>文案基线人工复核</h1><p>核对标题、详情、三条卖点与来源事实。先独立判断，再展开自动规则作辅助。此页面只读；评分填写在从human_scores.template.jsonl复制的文件中。规则未标记不表示事实正确，来源本身也可能有误。</p>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page + "".join(cards) + "</html>", encoding="utf-8")
    print(
        json.dumps({"records": len(cards), "output": str(output)}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
