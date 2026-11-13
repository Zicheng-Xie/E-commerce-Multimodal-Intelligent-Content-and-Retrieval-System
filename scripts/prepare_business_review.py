"""Create exhaustive blank business judgments for the exposed development catalog."""

from __future__ import annotations

import hashlib
import json
import os
import shutil

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import file_map, verify_files
from multimodal_ecommerce.dataset_release import write_rows
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from prepare_precision_grid import HTML

ROOT = PROJECT_ROOT / "artifacts/module2_precision60/business"
OUT = ROOT / "review"


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve the complete business review packet.

    Notes:
        Writes derived artifacts through (OUT / 'index.html').write_text, OUT.mkdir,
        write_json, write_rows.
    """
    if OUT.exists():
        raise FileExistsError("Preserve the complete business review packet")
    manifest = json.loads((ROOT / "input_manifest.json").read_text(encoding="utf-8"))
    verify_files(ROOT, manifest["files"])
    context = json.loads((ROOT / "review_context.json").read_text(encoding="utf-8"))
    allowed = json.loads((ROOT / "allowed_candidates.json").read_text(encoding="utf-8"))
    candidates = {r["id"]: r for r in context["catalog"]}
    packet = sha256_file(ROOT / "input_manifest.json")
    rows = []
    for query in context["queries"]:
        for candidate in allowed[query["id"]]:
            row = candidates[candidate]
            rows.append(
                {
                    "id": hashlib.sha256(
                        (packet + "|" + query["id"] + "|" + candidate).encode()
                    ).hexdigest()[:24],
                    "packet_id": packet,
                    "role": "development_training_only",
                    "query_id": query["id"],
                    "candidate_id": candidate,
                    "query_image": (
                        os.path.relpath(query["image_path"], OUT).replace("\\", "/")
                        if query["image_path"]
                        else ""
                    ),
                    "query_text": query["text"] or "",
                    "query_rule": query["relevance_rule"],
                    "candidate_image": os.path.relpath(row["image_path"], OUT).replace(
                        "\\", "/"
                    ),
                    "candidate_title": row["title"],
                    "grade": None,
                    "note": "",
                    "reviewer": "",
                    "review_origin": None,
                }
            )
    rows.sort(
        key=lambda r: hashlib.sha256(
            ("20261022:business:" + r["id"]).encode()
        ).hexdigest()
    )
    OUT.mkdir()
    write_rows(OUT / "judgments.blank.jsonl", rows)
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    html = HTML.replace("__ROWS__", data).replace("__PREFIX__", '""')
    html = html.replace(
        '<img id="queryImage" alt="查询商品">',
        '<h2 id="queryText"></h2><img id="queryImage" alt="查询商品"><p id="queryRule"></p>',
    )
    html = html.replace(
        "只用于旧开发数据训练。判断商品类型、用途和可见外观是否相关；背景或人物相似不算。看不清时保持待定并说明原因。该Top30包不能用于正式验收或计算全图库Recall。",
        "只用于已暴露三品类开发数据。请依据左侧查询标准，结合商品原始标题和图片判断；不显示排名和模型分数。覆盖每条查询的全部允许候选，全部完成后才能计算完整开发指标。它仍不是独立验收集。",
    )
    (OUT / "index.html").write_text(html, encoding="utf-8")
    shutil.copyfile(PROJECT_ROOT / "assets/precision_review_grid.js", OUT / "review.js")
    for row in rows:
        assert (OUT / row["candidate_image"]).is_file()
        if row["query_image"]:
            assert (OUT / row["query_image"]).is_file()
    write_json(
        OUT / "manifest.json",
        {
            "role": "development_training_only",
            "queries": len(context["queries"]),
            "rows": len(rows),
            "coverage": "Every allowed non-self/non-identical-pixel candidate, all 18 old queries retained",
            "human_labels_completed": 0,
            "automatic_grading": False,
            "formal_acceptance": False,
            "query_approval": "Legacy development query criteria; not independent human query collection",
            "input_manifest_sha256": packet,
            "files": file_map(OUT),
        },
    )
    print(
        json.dumps(
            {
                "queries": len(context["queries"]),
                "blank_relations": len(rows),
                "human_labels_completed": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
