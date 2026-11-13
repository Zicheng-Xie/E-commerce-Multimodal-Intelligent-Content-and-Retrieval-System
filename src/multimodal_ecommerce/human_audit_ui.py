"""Render local human-only review forms without exposing machine judgments."""

from __future__ import annotations

import html
import os
import shutil
from pathlib import Path

from multimodal_ecommerce.runtime_device import project_path

ASSETS = Path(__file__).with_name("web_assets")


def picture(value: str | None, root: Path) -> str:
    """Resolve a development original into escaped local image markup.

    Args:
        value: Original image path or empty path for a text query.
        root: Review page directory used for relative image references.

    Returns:
        Lazy image markup containing only the original image location.
    """
    if not value:
        return ""
    relative = os.path.relpath(project_path(value), root)
    return '<img loading="lazy" src="' + html.escape(relative, quote=True) + '">'


def render_review(
    tasks: list[dict],
    root: Path,
    *,
    title: str = "20% 真实人工复核",
    instructions: str | None = None,
) -> None:
    """Render frozen relations as paged forms; preserve all actual answer files.

    Args:
        tasks: Frozen private sampled relations; only identity and originals render.
        root: Local packet folder; no remote service is used.
        title: Display name for this review packet.
        instructions: Optional instructions for a targeted development review.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "answers").mkdir(exist_ok=True)
    for name in ["human_audit.js", "human_audit.css"]:
        shutil.copyfile(ASSETS / name, root / name)
    header = (
        '<!doctype html><meta charset="utf-8"><meta name="viewport" '
        + 'content="width=device-width,initial-scale=1"><link rel="stylesheet" '
        + 'href="human_audit.css"><title>'
        + html.escape(title)
        + "</title>"
    )
    pages = []
    for batch, start in enumerate(range(0, len(tasks), 30), 1):
        name = f"batch_{batch:03d}.html"
        pages.append(name)
        cards = []
        for task in tasks[start : start + 30]:
            query, candidate = task["query"], task["candidate"]
            key = html.escape(task["task_id"], quote=True)
            choices = "".join(
                '<label><input type="radio" name="grade_'
                + key
                + '" value="'
                + value
                + '">'
                + label
                + "</label>"
                for value, label in [
                    ("1", "1 相关"),
                    ("0", "0 不相关"),
                    ("uncertain", "证据不足"),
                ]
            )
            cards.append(
                '<article data-task="'
                + key
                + '"><code>'
                + key
                + "</code><div class=pair><section><h3>查询：</h3><p>"
                + html.escape(query["text"] or "原始图片查询")
                + "</p>"
                + picture(query["image_path"], root)
                + "</section><section><h3>候选原标题：</h3><p>"
                + html.escape(candidate["title"])
                + "</p>"
                + picture(candidate["image_path"], root)
                + "</section></div><fieldset><legend>你的判断</legend>"
                + choices
                + '</fieldset><label>判断理由（必填）<input class="reason" '
                + 'placeholder="例如：商品主体不同 / 用途与明确属性匹配"></label>'
                + '<small class="saved"></small></article>'
            )
        (root / name).write_text(
            header
            + f'<body data-batch="{batch:03d}"'
            + (
                f' data-packet="{html.escape(root.parent.name, quote=True)}"'
                if instructions
                else ""
            )
            + '><header><a href="index.html">任务目录</a>'
            + f"<h1>第 {batch}/{(len(tasks) + 29) // 30} 批 · {len(cards)} 条</h1>"
            + '<label>审核者姓名或固定代号<input id="reviewer" autocomplete="name">'
            + '</label><label><input type="checkbox" id="human">'
            + "我本人查看原图并独立判断，没有让 AI 代填</label>"
            + '<button id="export">导出本批已完成答案 CSV</button>'
            + '<p id="progress" aria-live="polite"></p></header>'
            + "<p>按查询的商品主体、用途及明确属性判断相关性。相关=1，不相关=0，"
            + "证据不足=uncertain。这里不显示 AI 标签、分数或排序。浏览器尝试本地保存；"
            + "请每批及时导出 CSV 备份。将导出的文件放入本目录 answers 文件夹；"
            + "同一批修改后覆盖原文件，不要保留重复副本。</p>"
            + "".join(cards)
            + '<script src="human_audit.js"></script></body>',
            encoding="utf-8",
        )
    original_instructions = (
        "<p>全部完成后，在项目终端运行：<code>.venv-mac/bin/python "
        + "scripts/simple_cycle_review.py score</code>。评分器会合并 answers/ 中的 CSV "
        + "和原始 human_answers.csv；拒绝重复答案。全部复核前，整体准确率保持空值。</p>"
        + "<p>“证据不足”会保留为不确定，AI 未判定项单独统计。"
        + "低于 80% 时需要扩充人工复核或重新标注，不能把 AI 指标直接当作真实效果。</p>"
    )
    (root / "index.html").write_text(
        header
        + f"<h1>{html.escape(title)}</h1><p>由你本人复核，共 "
        + str(len(tasks))
        + " 条，每批 30 条。点击批次填写姓名、确认本人判断，"
        + "选择相关性并填写理由。完成一批后导出 CSV，保存到本目录 answers/ 文件夹。"
        + "同批再次导出请覆盖旧文件。浏览器清理缓存可能丢失未导出的进度。</p>"
        + (
            f"<p>{html.escape(instructions)}</p>"
            if instructions
            else original_instructions
        )
        + "".join(
            f'<p><a href="{name}">第 {i + 1} 批（{i * 30 + 1}–'
            + f"{min((i + 1) * 30, len(tasks))}）</a></p>"
            for i, name in enumerate(pages)
        ),
        encoding="utf-8",
    )
