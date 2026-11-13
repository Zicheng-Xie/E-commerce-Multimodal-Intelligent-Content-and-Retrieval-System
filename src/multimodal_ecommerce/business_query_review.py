"""Independent human query authoring before business acceptance judgments."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)

ROOT = PROJECT_ROOT / "artifacts/module2_business_acceptance"
EDITABLE = {"text", "relevance_rule", "approved", "reviewer", "review_origin", "note"}


def validate_query_authoring(
    templates: list[dict], value: dict, release_id: str
) -> dict:
    """Validate query authoring.

    Args:
        templates: Templates used by this operation.
        value: Input value to validate or transform.
        release_id: Release id used by this operation.

    Returns:
        Mapping with fields: query_slots, human_approved_queries, pending_queries,
        all_queries_approved, human_relevance_labels, model_inference_allowed,
        formal_acceptance, formal_metrics, next_step.

    Raises:
        ValueError: Query authoring release or schema mismatch; Named human query author
            declaration required; All fixed query slots must be retained; Unknown or
            duplicate query slot; Immutable query scope or image changed; Query
            approval, rule and reviewer types invalid; Only declared human query
            authoring is allowed; Image query must not acquire a title/text model hint;
            Text query must be text or pending null; Every edited query needs the
            declared author identity; Approved queries need actual text and a relevance
            rule.
    """
    if (
        set(value) != {"release_id", "role", "reviewer", "declared_human", "rows"}
        or value["release_id"] != release_id
        or value["role"] != "acceptance_query_author"
    ):
        raise ValueError("Query authoring release or schema mismatch")
    if (
        value["declared_human"] is not True
        or not isinstance(value["reviewer"], str)
        or not value["reviewer"].strip()
    ):
        raise ValueError("Named human query author declaration required")
    expected = {r["id"]: r for r in templates}
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != len(templates):
        raise ValueError("All fixed query slots must be retained")
    seen = set()
    approved = 0
    for row in rows:
        original = expected.get(row.get("id"))
        if original is None or row["id"] in seen:
            raise ValueError("Unknown or duplicate query slot")
        seen.add(row["id"])
        if set(row) != set(original) or any(
            row[k] != original[k] for k in original if k not in EDITABLE
        ):
            raise ValueError("Immutable query scope or image changed")
        if type(row["approved"]) is not bool or not all(
            isinstance(row[k], str) for k in ("relevance_rule", "reviewer", "note")
        ):
            raise ValueError("Query approval, rule and reviewer types invalid")
        if row["review_origin"] not in (None, "human"):
            raise ValueError("Only declared human query authoring is allowed")
        if row["kind"] == "image" and row["text"] is not None:
            raise ValueError("Image query must not acquire a title/text model hint")
        if (
            row["kind"] == "text"
            and row["text"] is not None
            and not isinstance(row["text"], str)
        ):
            raise ValueError("Text query must be text or pending null")
        changed = any(row[k] != original[k] for k in EDITABLE)
        if changed and (
            row["reviewer"].strip() != value["reviewer"].strip()
            or row["review_origin"] != "human"
        ):
            raise ValueError("Every edited query needs the declared author identity")
        if row["approved"]:
            if not row["relevance_rule"].strip() or (
                row["kind"] == "text" and not str(row["text"] or "").strip()
            ):
                raise ValueError(
                    "Approved queries need actual text and a relevance rule"
                )
            approved += 1
    return {
        "query_slots": len(rows),
        "human_approved_queries": approved,
        "pending_queries": len(rows) - approved,
        "all_queries_approved": approved == len(rows),
        "human_relevance_labels": 0,
        "model_inference_allowed": False,
        "formal_acceptance": False,
        "formal_metrics": None,
        "next_step": "Independent human catalog/family QA and full-catalog relevance review; freeze final model before any acceptance inference",
    }


def prepare_query_authoring(root: Path = ROOT) -> dict:
    """Prepare query authoring.

    Args:
        root: Root directory used to resolve the operation artifacts.

    Returns:
        Mapping with fields: query_slots, image_queries, human_approved_queries,
        formal_ready.

    Raises:
        FileExistsError: Preserve the authoring packet.

    Notes:
        Writes derived artifacts through (target / 'images').mkdir, (target /
        'index.html').write_text, target.mkdir, write_json, write_rows.
    """
    root = ensure_output_path(root)
    target = root / "query_author"
    if (target / "manifest.json").exists():
        raise FileExistsError("Preserve the authoring packet")
    dataset = root / "dataset"
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    verify_files(dataset, manifest["files"])
    release_id = sha256_file(dataset / "manifest.json")
    rows = read_rows(dataset / "evaluator/queries.draft.jsonl")
    target.mkdir(parents=True, exist_ok=True)
    images = {}
    for row in rows:
        if row["image_path"]:
            source = dataset / row["image_path"]
            relative = "images/" + source.name
            (target / "images").mkdir(exist_ok=True)
            shutil.copyfile(source, target / relative)
            images[row["id"]] = relative
    payload = json.dumps(
        dict(release_id=release_id, rows=rows, images=images), ensure_ascii=False
    ).replace("</", "<\\/")
    content = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>独立验收查询确认</title><style>body{font:16px system-ui;margin:24px;max-width:1100px;background:#f5f8fb;color:#15283e}header{position:sticky;top:0;background:white;padding:16px;z-index:1}article{padding:16px;background:white;border:1px solid #bbc9d8;margin:12px 0}img{width:230px;height:200px;object-fit:contain}input,textarea,button,select{font:inherit;padding:8px;margin:5px}textarea{width:90%;min-height:55px}.warning{color:#9b3024}</style><header><h1>独立验收查询 · 真人编写与确认</h1><p>只展示预先固定的需求范围及查询图片，图库隐藏。请按真实购物需求编写中文查询和明确的相关性标准，不看模型排序，不根据正例数量修改需求。图片主体不符或看不清时保持未批准并写原因，不删除查询。</p><input id="reviewer" placeholder="实际编写/确认人代号"><label><input type="checkbox" id="human">本人逐项编写和确认</label><button id="export">导出进度</button><input type="file" id="restore" accept=".json"><div id="progress"></div><p id="status" class="warning"></p><select id="filter"><option value="all">全部查询</option><option value="text">文本查询</option><option value="image">图片查询</option></select></header><main id="grid"></main><script>window.QUERY_JOB=__PAYLOAD__;</script><script src="author.js"></script></html>""".replace(
        "__PAYLOAD__", payload
    )
    (target / "index.html").write_text(content, encoding="utf-8")
    shutil.copyfile(
        PROJECT_ROOT / "assets/business_query_author.js", target / "author.js"
    )
    write_rows(target / "queries.draft.jsonl", rows)
    protocol = {
        "release_id": release_id,
        "dataset_manifest_sha256": release_id,
        "query_slots": len(rows),
        "human_approved_queries": 0,
        "human_relevance_labels": 0,
        "model_inference_allowed": False,
        "formal_acceptance": False,
        "scope": "30 predefined fine types; not traffic representative",
        "implementation": {
            name: sha256_file(PROJECT_ROOT / name)
            for name in [
                "src/multimodal_ecommerce/business_query_review.py",
                "scripts/business_query_review.py",
                "assets/business_query_author.js",
            ]
        },
        "files": {
            p.relative_to(target).as_posix(): sha256_file(p)
            for p in target.rglob("*")
            if p.is_file()
        },
    }
    write_json(target / "manifest.json", protocol)
    with zipfile.ZipFile(
        root / "query_author.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root))
        archive.writestr(
            "START.html",
            '<meta charset="utf-8"><a href="query_author/index.html">进入查询编写和确认</a>',
        )
    return {
        "query_slots": len(rows),
        "image_queries": len(images),
        "human_approved_queries": 0,
        "formal_ready": False,
    }


def import_query_authoring(path: Path, root: Path = ROOT) -> dict:
    """Import query authoring.

    Args:
        path: Path of the input or output artifact used by this operation.
        root: Root directory used to resolve the operation artifacts.

    Returns:
        Result mapping produced by the workflow.

    Raises:
        ValueError: Acceptance dataset manifest changed.

    Notes:
        Writes derived artifacts through target.mkdir, write_json, write_rows.
    """
    packet = root / "query_author"
    protocol = json.loads((packet / "manifest.json").read_text(encoding="utf-8"))
    verify_files(PROJECT_ROOT, protocol["implementation"])
    verify_files(packet, protocol["files"])
    if (
        sha256_file(root / "dataset/manifest.json")
        != protocol["dataset_manifest_sha256"]
    ):
        raise ValueError("Acceptance dataset manifest changed")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    status = validate_query_authoring(
        read_rows(packet / "queries.draft.jsonl"), value, protocol["release_id"]
    )
    target = ensure_output_path(root / "query_revisions" / sha256_file(path)[:24])
    if (target / "status.json").exists():
        return json.loads((target / "status.json").read_text(encoding="utf-8"))
    target.mkdir(parents=True, exist_ok=True)
    write_rows(target / "queries.jsonl", value["rows"])
    write_json(target / "submission.json", value)
    write_json(
        target / "status.json",
        {
            **status,
            "input_sha256": sha256_file(path),
            "output": target.relative_to(PROJECT_ROOT).as_posix(),
        },
    )
    return status
