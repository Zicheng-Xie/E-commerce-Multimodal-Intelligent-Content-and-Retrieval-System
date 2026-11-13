"""Portable independent human development review jobs and strict consensus import."""

from __future__ import annotations

import hashlib
import html
import json
import shutil
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.precision_labels import validate_precision_labels
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)

SOURCE = PROJECT_ROOT / "artifacts/module2_precision60/business/review"
ROOT = PROJECT_ROOT / "artifacts/module2_business_review/development"


def identity(value: str) -> str:
    """Return identity for this workflow.

    Args:
        value: Input value to validate or transform.

    Returns:
        Text produced by this operation.
    """
    return unicodedata.normalize("NFKC", value).strip().casefold()


def canonical_hash(value: Any) -> str:
    """Return canonical hash.

    Args:
        value: Input value to validate or transform.

    Returns:
        Text produced by this operation.
    """
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def page(payload: dict, script: str) -> str:
    """Return page for this workflow.

    Args:
        payload: Payload used by this operation.
        script: Script used by this operation.

    Returns:
        Text produced by this operation.
    """
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return (
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>商品相关性人工评审</title><style>
body{font:16px system-ui;margin:24px;max-width:1500px;background:#f6f8fb;color:#182539}header{position:sticky;top:0;background:#fff;padding:12px;z-index:2;border:1px solid #ccd6e4}button,input,textarea{font:inherit;margin:4px;padding:8px}button{cursor:pointer}#query img{max-height:210px;max-width:260px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:16px}.card{background:white;padding:12px;border:3px solid #d1dae5}.positive{border-color:#188450}.negative{border-color:#c64949}.card img{height:180px;width:100%;object-fit:contain}.card p{overflow-wrap:anywhere}.card input{max-width:90%}.chosen{background:#c5d9ff}.warn{color:#a32828}#status{white-space:pre-wrap}textarea{width:90%;min-height:60px}a{color:#194eb2}</style>
<header><strong id="heading"></strong><div>先填写本人代号并确认。相关=1，不相关=0；无法判断保持待定并写原因。图片可点击放大。不要查看模型排名或与另一位评审对答案。</div><input id="reviewer" placeholder="实际评审人姓名或唯一代号"><label><input id="human" type="checkbox">本人逐项判断</label><button id="export">导出当前进度</button><label>恢复文件<input id="restore" type="file" accept=".json"></label><div id="progress"></div><div id="status" class="warn"></div></header><section id="query"></section><nav><button id="prev">上一页</button><button id="next">下一页</button></nav><main id="grid" class="grid"></main><p>浏览器缓存仅作辅助，关闭前请导出JSON；恢复时选择同一批、同一席位的导出文件。待定项不会自动作为负例。</p>
<script>window.JOB="""
        + data
        + """;</script><script src="""
        + json.dumps(script)
        + "></script></html>"
    )


def build_jobs(root: Path = ROOT) -> dict:
    """Build jobs.

    Args:
        root: Root directory used to resolve the operation artifacts.

    Returns:
        Mapping with fields: pairs, batches, independent_work_items, media,
        human_labels.

    Raises:
        FileExistsError: Preserve the existing review release.
        ValueError: Start from blank templates, never prior human/model judgments.

    Notes:
        Writes derived artifacts through (folder / 'batches' / (batch['id'] +
        '.html')).write_text, (folder / 'batches').mkdir, (folder /
        'index.html').write_text, (root / 'index.html').write_text, root.mkdir,
        target.parent.mkdir, write_json, write_rows.
    """
    root = ensure_output_path(root)
    if (root / "manifest.json").exists():
        raise FileExistsError("Preserve the existing review release")
    source_manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    verify_files(SOURCE, source_manifest["files"])
    rows = read_rows(SOURCE / "judgments.blank.jsonl")
    validate_precision_labels(rows, rows)
    if any(r["grade"] is not None for r in rows):
        raise ValueError(
            "Start from blank templates, never prior human/model judgments"
        )
    script_source = PROJECT_ROOT / "assets/business_review_jobs.js"
    release_id = canonical_hash(
        {
            "source": sha256_file(SOURCE / "manifest.json"),
            "batch_size": 96,
            "script": sha256_file(script_source),
        }
    )
    root.mkdir(parents=True, exist_ok=True)
    media: dict[str, str] = {}
    presentation = {}
    for row in rows:
        entry = {}
        for field in ("query_image", "candidate_image"):
            value = row[field]
            if not value:
                entry[field] = ""
                continue
            original = (SOURCE / value).resolve()
            original.relative_to(PROJECT_ROOT)
            if value not in media:
                name = "media/" + sha256_file(original) + original.suffix.lower()
                target = root / name
                target.parent.mkdir(exist_ok=True)
                if not target.exists():
                    shutil.copyfile(original, target)
                media[value] = name
            entry[field] = media[value]
        presentation[row["id"]] = entry
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["query_id"], []).append(row)
    batches: list[dict] = []
    for query_id, group in grouped.items():
        for offset in range(0, len(group), 96):
            chunk = group[offset : offset + 96]
            batches.append(
                {
                    "id": f"batch_{len(batches)+1:03}",
                    "query_id": query_id,
                    "row_ids": [r["id"] for r in chunk],
                }
            )
    manifest = {
        "release_id": release_id,
        "role": "development_training_only",
        "source_manifest_sha256": sha256_file(SOURCE / "manifest.json"),
        "source_rows_sha256": sha256_file(SOURCE / "judgments.blank.jsonl"),
        "pairs": len(rows),
        "queries": len(grouped),
        "batches": batches,
        "slots": ["A", "B"],
        "owners": {"A": None, "B": None, "adjudicator": None},
        "human_labels": 0,
        "quality_metrics": None,
        "formal_acceptance": False,
        "identity_limit": "Names and declarations are checked; actual identities and physical independence are not authenticated",
        "presentation": presentation,
    }
    by_id = {r["id"]: r for r in rows}
    for slot in ("A", "B"):
        folder = root / ("rater_" + slot)
        (folder / "batches").mkdir(parents=True)
        shutil.copyfile(script_source, folder / "workbench.js")
        links = []
        for batch in batches:
            selected = [by_id[rid] for rid in batch["row_ids"]]
            payload = {
                "release_id": release_id,
                "batch_id": batch["id"],
                "review_slot": slot,
                "rows": selected,
                "presentation": {
                    r["id"]: {
                        k: "../../" + v if v else ""
                        for k, v in presentation[r["id"]].items()
                    }
                    for r in selected
                },
                "adjudication_of": None,
            }
            (folder / "batches" / (batch["id"] + ".html")).write_text(
                page(payload, "../workbench.js"), encoding="utf-8"
            )
            links.append(
                f'<li><a href="batches/{batch["id"]}.html">{batch["id"]} · {html.escape(batch["query_id"])} · {len(selected)}项</a></li>'
            )
        (folder / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>评审任务</title><h1>独立评审席位 '
            + slot
            + "</h1><p>领取人待填写。只发本席位的ZIP给本人；不要交换标签。每批最多96项，可分次导出；同一席位全程使用同一代号。交回每批的最新一个JSON，旧版本请移出收件文件夹。</p><ol>"
            + "".join(links)
            + "</ol>",
            encoding="utf-8",
        )
    (root / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>开发网格评审协调页</title><h1>开发网格评审</h1><p>5,143对开发关系，54批，A/B各54项领取任务；真实人工标签0。席位均待真人领取，开发成绩不作验收成绩。</p><p><a href="rater_A/index.html">席位A</a> · <a href="rater_B/index.html">席位B</a></p><p>协调人分别分发rater_A.zip、rater_B.zip，两人独立评审。全部商品图片已随包携带，解压后离线使用。</p>',
        encoding="utf-8",
    )
    write_rows(root / "judgments.blank.jsonl", rows)
    write_json(root / "manifest.json", manifest)
    for slot in ("A", "B"):
        with zipfile.ZipFile(
            root / ("rater_" + slot + ".zip"), "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for folder in (root / "media", root / ("rater_" + slot)):
                for path in sorted(folder.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(root))
            archive.writestr(
                "START.html",
                '<meta charset="utf-8"><a href="rater_'
                + slot
                + '/index.html">点击进入人工评审任务</a>',
            )
    receipt = {
        "files": {
            p.relative_to(root).as_posix(): sha256_file(p)
            for p in root.rglob("*")
            if p.is_file()
        },
        "role": "development_review_materials",
        "human_labels": 0,
        "formal_acceptance": False,
    }
    write_json(root / "release_receipt.json", receipt)
    return {
        "pairs": len(rows),
        "batches": len(batches),
        "independent_work_items": 2 * len(batches),
        "media": len(set(media.values())),
        "human_labels": 0,
    }


def validate_submission(
    manifest: dict,
    templates: list[dict],
    value: dict,
    adjudication_of: str | None = None,
) -> list[dict]:
    """Validate submission.

    Args:
        manifest: Manifest used by this operation.
        templates: Templates used by this operation.
        value: Input value to validate or transform.
        adjudication_of: Adjudication of used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Submission schema or release mismatch; Unknown review slot; Stale or
            unrelated adjudication; Primary review cannot declare adjudication; A named
            real human declaration is required; Unknown batch; Mixed or changed reviewer
            identity; Adjudication needs a written reason.
    """
    keys = {
        "release_id",
        "batch_id",
        "review_slot",
        "reviewer",
        "declared_human",
        "adjudication_of",
        "rows",
    }
    if set(value) != keys or value["release_id"] != manifest["release_id"]:
        raise ValueError("Submission schema or release mismatch")
    slot = value["review_slot"]
    if slot not in ("A", "B", "adjudicator"):
        raise ValueError("Unknown review slot")
    if slot == "adjudicator":
        if not adjudication_of or value["adjudication_of"] != adjudication_of:
            raise ValueError("Stale or unrelated adjudication")
    elif value["adjudication_of"] is not None:
        raise ValueError("Primary review cannot declare adjudication")
    if (
        value["declared_human"] is not True
        or not isinstance(value["reviewer"], str)
        or not identity(value["reviewer"])
    ):
        raise ValueError("A named real human declaration is required")
    by_id = {r["id"]: r for r in templates}
    batch = next((b for b in manifest["batches"] if b["id"] == value["batch_id"]), None)
    if batch is None:
        raise ValueError("Unknown batch")
    expected = [by_id[rid] for rid in batch["row_ids"]]
    validate_precision_labels(expected, value["rows"])
    for row in value["rows"]:
        if row["grade"] is not None or row["note"].strip():
            if identity(row["reviewer"]) != identity(value["reviewer"]):
                raise ValueError("Mixed or changed reviewer identity")
        if (
            slot == "adjudicator"
            and row["grade"] is not None
            and not row["note"].strip()
        ):
            raise ValueError("Adjudication needs a written reason")
    return value["rows"]


def merge_values(
    manifest: dict,
    templates: list[dict],
    submissions: list[dict],
    adjudications: list[dict] | None = None,
) -> dict:
    """Merge values.

    Args:
        manifest: Manifest used by this operation.
        templates: Templates used by this operation.
        submissions: Submissions used by this operation.
        adjudications: Adjudications used by this operation.

    Returns:
        Mapping with fields: rows, provenance, conflicts, adjudication_batches,
        primary_digest, status.

    Raises:
        ValueError: Primary inbox only accepts A/B exports; Conflicting batch revisions;
            keep only the chosen latest export; One consistent reviewer per independent
            slot is required; A and B must be different reviewers; Adjudication inbox
            requires adjudicator exports; Adjudicator must be a consistent third
            reviewer; Only actual disagreements may be adjudicated; Conflicting
            adjudication revisions.
    """
    accepted: dict[tuple[str, str], dict] = {}
    owners: dict[str, str] = {}
    primary_values = []
    for value in submissions:
        if value.get("review_slot") not in ("A", "B"):
            raise ValueError("Primary inbox only accepts A/B exports")
        validate_submission(manifest, templates, value)
        key = (value["review_slot"], value["batch_id"])
        if key in accepted:
            if value != accepted[key]:
                raise ValueError(
                    "Conflicting batch revisions; keep only the chosen latest export"
                )
            continue
        slot = value["review_slot"]
        owner = identity(value["reviewer"])
        if slot in owners and owners[slot] != owner:
            raise ValueError("One consistent reviewer per independent slot is required")
        owners[slot] = owner
        accepted[key] = value
        primary_values.append(value)
    if len(owners) == 2 and owners["A"] == owners["B"]:
        raise ValueError("A and B must be different reviewers")
    primary_digest = canonical_hash(
        sorted(primary_values, key=lambda x: (x["review_slot"], x["batch_id"]))
    )
    lookups = {
        slot: {
            r["id"]: r
            for (s, _), value in accepted.items()
            if s == slot
            for r in value["rows"]
        }
        for slot in ("A", "B")
    }
    output = []
    conflicts = []
    provenance = []
    counts: Counter[str] = Counter()
    for template in templates:
        pair = [lookups[slot].get(template["id"]) for slot in ("A", "B")]
        row = dict(template)
        for slot, judgment in zip(("A", "B"), pair):
            if judgment and judgment["grade"] is not None:
                counts[slot + "_completed"] += 1
        if all(r is not None and r["grade"] is not None for r in pair):
            left, right = pair
            assert left is not None and right is not None
            if left["grade"] == right["grade"]:
                counts["agreement"] += 1
                row.update(
                    grade=left["grade"],
                    reviewer=left["reviewer"] + " & " + right["reviewer"],
                    review_origin="human",
                    note="Two declared human judgments agree; see consensus provenance.",
                )
                provenance.append(
                    {
                        "id": row["id"],
                        "A": left,
                        "B": right,
                        "resolution": "human_agreement",
                    }
                )
            else:
                conflicts.append({"id": template["id"], "A": left, "B": right})
        output.append(row)
    conflict_ids = {r["id"] for r in conflicts}
    adjud_batches: list[dict] = []
    for offset in range(0, len(conflicts), 96):
        adjud_batches.append(
            {
                "id": f"adjud_{len(adjud_batches)+1:03}",
                "row_ids": [r["id"] for r in conflicts[offset : offset + 96]],
            }
        )
    adjud_manifest = {**manifest, "batches": adjud_batches}
    resolutions: dict[str, dict] = {}
    adjud_owner = None
    for value in adjudications or []:
        if value.get("review_slot") != "adjudicator":
            raise ValueError("Adjudication inbox requires adjudicator exports")
        reviewed = validate_submission(adjud_manifest, templates, value, primary_digest)
        owner = identity(value["reviewer"])
        if owner in owners.values() or (
            adjud_owner is not None and owner != adjud_owner
        ):
            raise ValueError("Adjudicator must be a consistent third reviewer")
        adjud_owner = owner
        for row in reviewed:
            if row["id"] not in conflict_ids:
                raise ValueError("Only actual disagreements may be adjudicated")
            if row["id"] in resolutions and row != resolutions[row["id"]]:
                raise ValueError("Conflicting adjudication revisions")
            resolutions[row["id"]] = row
    conflict_map = {r["id"]: r for r in conflicts}
    for i, row in enumerate(output):
        resolved = resolutions.get(row["id"])
        if resolved and resolved["grade"] is not None:
            output[i] = resolved
            counts["adjudicated"] += 1
            provenance.append(
                {
                    **conflict_map[row["id"]],
                    "adjudicator": resolved,
                    "resolution": "third_human_adjudication",
                }
            )
    progress = validate_precision_labels(templates, output)
    progress.pop("completed_rows")
    progress["full_gallery_recall_computable"] = progress["pending"] == 0
    return {
        "rows": output,
        "provenance": provenance,
        "conflicts": conflicts,
        "adjudication_batches": adjud_batches,
        "primary_digest": primary_digest,
        "status": {
            **progress,
            **{
                key: counts[key]
                for key in ("A_completed", "B_completed", "agreement", "adjudicated")
            },
            "conflicts": len(conflicts),
            "unresolved_conflicts": len(conflicts) - counts["adjudicated"],
            "owners": {
                "A": owners.get("A"),
                "B": owners.get("B"),
                "adjudicator": adjud_owner,
            },
            "received_primary_batches": len(accepted),
            "expected_primary_batches": 2 * len(manifest["batches"]),
            "quality_metrics": None,
            "formal_acceptance": False,
        },
    }


def merge_jobs(
    inbox: Path, adjudications: Path | None = None, root: Path = ROOT
) -> dict:
    """Merge jobs.

    Args:
        inbox: Inbox used by this operation.
        adjudications: Adjudications used by this operation.
        root: Root directory used to resolve the operation artifacts.

    Returns:
        Mapping with fields: primary_digest, output, source_inputs_digest,
        scoring_ready.

    Notes:
        Writes derived artifacts through (target / (batch['id'] +
        '.html')).write_text, target.mkdir, write_json, write_rows.
    """
    receipt = json.loads((root / "release_receipt.json").read_text(encoding="utf-8"))
    verify_files(root, receipt["files"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    templates = read_rows(root / "judgments.blank.jsonl")

    def read_inbox(folder: Path | None) -> list[dict]:
        """Read inbox.

        Args:
            folder: Directory containing the operation artifacts.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        return (
            [
                json.loads(p.read_text(encoding="utf-8-sig"))
                for p in sorted(folder.rglob("*.json"))
            ]
            if folder
            else []
        )

    primary = read_inbox(inbox)
    adjud = read_inbox(adjudications)
    result = merge_values(manifest, templates, primary, adjud)
    revision = canonical_hash({"primary": primary, "adjudications": adjud})[:24]
    target = ensure_output_path(root / "merges" / revision)
    if (target / "status.json").exists():
        return json.loads((target / "status.json").read_text(encoding="utf-8"))
    target.mkdir(parents=True, exist_ok=True)
    write_rows(target / "judgments.jsonl", result["rows"])
    write_rows(target / "consensus_provenance.jsonl", result["provenance"])
    write_rows(target / "conflicts.jsonl", result["conflicts"])
    by_id = {r["id"]: r for r in templates}
    for batch in result["adjudication_batches"]:
        selected = [by_id[rid] for rid in batch["row_ids"]]
        conflict_map = {r["id"]: r for r in result["conflicts"]}
        payload = {
            "release_id": manifest["release_id"],
            "batch_id": batch["id"],
            "review_slot": "adjudicator",
            "adjudication_of": result["primary_digest"],
            "rows": selected,
            "presentation": {
                r["id"]: {
                    k: "../../" + v if v else ""
                    for k, v in manifest["presentation"][r["id"]].items()
                }
                for r in selected
            },
            "conflicts": {r["id"]: conflict_map[r["id"]] for r in selected},
        }
        (target / (batch["id"] + ".html")).write_text(
            page(payload, "../../rater_A/workbench.js"), encoding="utf-8"
        )
    status = {
        **result["status"],
        "primary_digest": result["primary_digest"],
        "output": target.relative_to(PROJECT_ROOT).as_posix(),
        "source_inputs_digest": revision,
        "scoring_ready": result["status"]["pending"] == 0,
    }
    write_json(target / "status.json", status)
    write_json(
        target / "input_provenance.json", {"primary": primary, "adjudications": adjud}
    )
    return status
