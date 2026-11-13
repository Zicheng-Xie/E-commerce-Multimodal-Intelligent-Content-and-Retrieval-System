"""Present the existing immutable 780-pair packet as an offline human review grid."""

from __future__ import annotations

import json
import os
import shutil

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import file_map, verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

PACKET = PROJECT_ROOT / "artifacts/module2_precision/review_packet"
OUT = PROJECT_ROOT / "artifacts/module2_precision60/review"
HTML = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>模块二开发标注网格</title>
<style>body{font:15px system-ui,sans-serif;background:#f2f5fa;color:#152f49;margin:0}header{padding:16px 24px;background:white;border-bottom:1px solid #d7dfea}h1{font-size:23px;margin:0 0 8px}.layout{display:grid;grid-template-columns:280px 1fr;gap:20px;padding:20px}aside{position:sticky;top:12px;align-self:start;background:white;padding:14px;border-radius:10px}aside img{width:100%;height:270px;object-fit:contain}button,input,select{font:inherit;padding:7px;border:1px solid #aebccc;border-radius:5px;margin:3px}button{cursor:pointer;background:white}.grid{display:grid;grid-template-columns:repeat(4,minmax(170px,1fr));gap:12px}.card{background:white;padding:9px;border:2px solid #d5deea;border-radius:8px}.card img{width:100%;height:180px;object-fit:contain}.card input{box-sizing:border-box;width:96%}.positive{border-color:#25834b}.negative{border-color:#8693a3}.chosen{background:#d6e9ff}#status{color:#84431c}small{color:#586b7e}@media(max-width:1050px){.grid{grid-template-columns:repeat(2,minmax(160px,1fr))}.layout{grid-template-columns:230px 1fr}}@media(max-width:650px){.layout{display:block}aside{position:static}aside img{height:180px}.grid{margin-top:15px}}</style>
<header><h1>模块二开发相关性标注：网格视图</h1><p>只用于旧开发数据训练。判断商品类型、用途和可见外观是否相关；背景或人物相似不算。看不清时保持待定并说明原因。该Top30包不能用于正式验收或计算全图库Recall。</p><label>评审代号<input id="reviewer"></label><label><input type="checkbox" id="human">由我本人逐项判断</label><button id="export">导出全部进度</button><label>恢复进度<input id="restore" type="file" accept=".jsonl"></label><p id="status" role="status"></p></header>
<div class="layout"><aside><img id="queryImage" alt="查询商品"><p id="progress"></p><select id="querySelect" aria-label="选择查询"></select><div><button id="prev">上一页</button><button id="next">下一页</button></div><p><button id="bulk">已逐项核对：本页待定项均不相关</button></p><small>批量按钮仅影响本页待定项，保留已选正例。未操作项一直为null。页面没有后台保存，请及时导出。</small></aside><main id="grid" class="grid"></main></div>
<script>window.REVIEW_TEMPLATE=__ROWS__;window.REVIEW_IMAGE_PREFIX=__PREFIX__;</script><script src="review.js"></script></html>"""


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve the original review grid.

    Notes:
        Writes derived artifacts through (OUT / 'index.html').write_text, OUT.mkdir,
        write_json.
    """
    if OUT.exists():
        raise FileExistsError("Preserve the original review grid")
    manifest = json.loads((PACKET / "manifest.json").read_text(encoding="utf-8"))
    verify_files(PACKET, manifest["files"])
    rows = read_rows(PACKET / "judgments.blank.jsonl")
    assert all(r["grade"] is None for r in rows)
    for row in rows:
        for key in ["query_image", "candidate_image"]:
            assert (PACKET / row[key]).is_file()
    OUT.mkdir()
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    prefix = os.path.relpath(PACKET, OUT).replace("\\", "/") + "/"
    (OUT / "index.html").write_text(
        HTML.replace("__ROWS__", data).replace("__PREFIX__", json.dumps(prefix)),
        encoding="utf-8",
    )
    shutil.copyfile(PROJECT_ROOT / "assets/precision_review_grid.js", OUT / "review.js")
    write_json(
        OUT / "manifest.json",
        {
            "role": "development_training_only",
            "rows": len(rows),
            "queries": len({r["query_id"] for r in rows}),
            "human_labels_completed": 0,
            "automatic_grading": False,
            "source_packet_manifest_sha256": sha256_file(PACKET / "manifest.json"),
            "export_schema": "Original immutable 780-row packet; unchanged packet/id/image fields",
            "full_gallery_recall_computable": False,
            "formal_acceptance": False,
            "files": file_map(OUT),
        },
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "human_labels_completed": 0,
                "images_checked": 2 * len(rows),
            }
        )
    )


if __name__ == "__main__":
    main()
