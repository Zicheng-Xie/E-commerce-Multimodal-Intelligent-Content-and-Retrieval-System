"""展示生成后的标题、详情和卖点，支持终端查看及HTML导出。"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import TypedDict


class GeneratedCopy(TypedDict):
    """只包含三类生成内容。"""

    title: str
    description: str
    selling_points: list[str]


PROJECT_DIR = Path(__file__).resolve().parents[1]


def find_latest_run() -> Path:
    """查找项目中最近修改且包含记录的生成目录。"""
    base = PROJECT_DIR / "outputs" / "copywriting_v3"
    candidates = (
        [
            folder
            for folder in base.iterdir()
            if folder.is_dir()
            and (folder / "records").is_dir()
            and any((folder / "records").glob("*.json"))
        ]
        if base.is_dir()
        else []
    )
    if not candidates:
        raise ValueError("没有找到生成记录，请指定结果目录。")
    return max(candidates, key=lambda folder: (folder / "records").stat().st_mtime_ns)


def read_copywriting(path: Path) -> list[GeneratedCopy]:
    """只读取result中的文案，跳过尚未生成文案的记录。"""
    if path.is_file():
        files = [path]
    else:
        records = path / "records" if (path / "records").is_dir() else path
        if not records.is_dir():
            raise ValueError(f"找不到记录目录：{path}")
        files = sorted(records.glob("*.json"))
    results: list[GeneratedCopy] = []
    for file in files:
        try:
            record = json.loads(file.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as error:
            raise ValueError(f"无法读取记录：{file}；{error}") from error
        result = record.get("result") if isinstance(record, dict) else None
        if not isinstance(result, dict):
            continue
        title = result.get("title")
        description = result.get("description")
        points = result.get("selling_points")
        if (
            not isinstance(title, str)
            or not isinstance(description, str)
            or not isinstance(points, list)
            or not all(isinstance(point, str) for point in points)
        ):
            raise ValueError(f"生成文案字段格式不完整：{file}")
        results.append(
            {"title": title, "description": description, "selling_points": points}
        )
    if not results:
        raise ValueError("该位置没有可展示的生成文案，请确认目录及生成进度。")
    return results


def render_text(results: list[GeneratedCopy]) -> str:
    """输出三类文案，不包含输入属性、规则评分或耗时。"""
    blocks = []
    for result in results:
        points = "\n".join(f"- {point}" for point in result["selling_points"])
        blocks.append(
            f"商品标题：{result['title']}\n\n详情文案：{result['description']}\n\n核心卖点：\n{points}"
        )
    return "\n\n" + ("\n\n" + "─" * 50 + "\n\n").join(blocks) + "\n"


def render_html(results: list[GeneratedCopy]) -> str:
    """生成离线可打开的页面，转义文案以保证按原文显示。"""
    cards = []
    for result in results:
        title = html.escape(str(result["title"]))
        description = html.escape(str(result["description"]))
        points = "".join(
            f"<li>{html.escape(point)}</li>" for point in result["selling_points"]
        )
        cards.append(
            f"<article><h2>{title}</h2><h3>详情文案</h3><p>{description}</p><h3>核心卖点</h3><ul>{points}</ul></article>"
        )
    return (
        """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>生成文案</title>
<style>body{margin:0;background:#f3f5f7;color:#202735;font:16px/1.8 system-ui,'Microsoft YaHei',sans-serif}main{max-width:880px;margin:32px auto;padding:0 18px}article{background:white;border:1px solid #e0e5eb;border-radius:12px;padding:24px 30px;margin-bottom:24px;overflow-wrap:anywhere}h2{font-size:23px;margin:0 0 20px}h3{font-size:16px;margin:18px 0 6px;color:#586174}p{margin:0;white-space:pre-wrap}ul{margin:4px 0;padding-left:24px}li{white-space:pre-wrap}</style></head><body><main>"""
        + "".join(cards)
        + "</main></body></html>"
    )


def main() -> int:
    """读取指定结果目录，默认显示终端文案，可选导出HTML。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        help="生成目录、records目录或单条记录；省略时查找最近修改的生成目录",
    )
    parser.add_argument(
        "--html", type=Path, metavar="输出文件", help="将同样的文案导出为离线HTML页面"
    )
    args = parser.parse_args()
    try:
        results = read_copywriting(
            args.path if args.path is not None else find_latest_run()
        )
        if args.html:
            args.html.parent.mkdir(parents=True, exist_ok=True)
            args.html.write_text(render_html(results), encoding="utf-8")
        else:
            print(render_text(results), end="")
    except (OSError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
