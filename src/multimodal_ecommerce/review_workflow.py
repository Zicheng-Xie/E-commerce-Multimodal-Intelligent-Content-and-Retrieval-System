"""Small human review batches, strict imports and complete-only final scoring."""

from __future__ import annotations

import html
import json
import os
from collections import defaultdict
from pathlib import Path

from multimodal_ecommerce.copy_experiment import score_copy_run
from multimodal_ecommerce.dataset_release import order_key, read_rows, write_rows
from multimodal_ecommerce.evaluation import validate_qrels, verify_files
from multimodal_ecommerce.frozen_retrieval import load_kit, score_predictions
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)

SCORES = ("relevance_pass", "fluency_pass", "factual_error")
EDITABLE = {"reviewer", "note", "review_origin", "grade", *SCORES}


def label_key(row: dict) -> tuple:
    """Identify a fixed task without using the mutable reviewer identity.

    Args:
        row: Row used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Unknown review task; Missing retrieval IDs; Missing copy ID.
    """
    if not isinstance(row, dict) or row.get("task") not in ("retrieval", "copy"):
        raise ValueError("Unknown review task")
    if row["task"] == "retrieval":
        if not isinstance(row.get("query_id"), str) or not isinstance(
            row.get("product_id"), str
        ):
            raise ValueError("Missing retrieval IDs")
        return ("retrieval", row["query_id"], row["product_id"])
    if not isinstance(row.get("id"), str):
        raise ValueError("Missing copy ID")
    return ("copy", row["id"])


def validate_batch(templates: list[dict], rows: list[dict]) -> dict:
    """Allow incomplete work, reject mismatched IDs, fake types and provenance.

    Args:
        templates: Templates used by this operation.
        rows: Records processed by this operation.

    Returns:
        Mapping with fields: total, completed, pending, ready.

    Raises:
        ValueError: Review batch cardinality mismatch; Unknown or duplicate review ID;
            Immutable review fields changed; Reviewer and note must be text; Formal
            review packets accept declared human judgments only; Relevance grade must be
            null, integer 0 or 1; A judgment requires its human reviewer declaration;
            Copy quality labels must be boolean or null; Absent output cannot receive
            quality scores; Factual errors cannot pass the frozen relevance rubric.
    """
    expected = {label_key(row): row for row in templates}
    if len(expected) != len(templates) or len(rows) != len(templates):
        raise ValueError("Review batch cardinality mismatch")
    seen = set()
    completed = 0
    for row in rows:
        key = label_key(row)
        if key not in expected or key in seen:
            raise ValueError("Unknown or duplicate review ID")
        seen.add(key)
        template = expected[key]
        if set(row) != set(template) or any(
            row[field] != template[field] for field in template if field not in EDITABLE
        ):
            raise ValueError("Immutable review fields changed")
        if not isinstance(row["reviewer"], str) or not isinstance(row["note"], str):
            raise ValueError("Reviewer and note must be text")
        if row["review_origin"] not in (None, "human"):
            raise ValueError(
                "Formal review packets accept declared human judgments only"
            )
        signed = bool(row["reviewer"].strip()) and row["review_origin"] == "human"
        if row["task"] == "retrieval":
            grade = row["grade"]
            if grade is not None and (type(grade) is not int or grade not in (0, 1)):
                raise ValueError("Relevance grade must be null, integer 0 or 1")
            if grade is not None and not signed:
                raise ValueError("A judgment requires its human reviewer declaration")
            completed += int(grade is not None and signed)
            continue
        if any(
            row[field] is not None and type(row[field]) is not bool for field in SCORES
        ):
            raise ValueError("Copy quality labels must be boolean or null")
        has_output = row["status"] in (
            "draft_requires_human_review",
            "blocked_by_rules",
        )
        if not has_output and any(row[field] is not None for field in SCORES):
            raise ValueError("Absent output cannot receive quality scores")
        if any(row[field] is not None for field in SCORES) and not signed:
            raise ValueError("A judgment requires its human reviewer declaration")
        if row["factual_error"] is True and row["relevance_pass"] is True:
            raise ValueError("Factual errors cannot pass the frozen relevance rubric")
        if not has_output:
            completed += int(signed and bool(row["note"].strip()))
        else:
            filled = all(type(row[field]) is bool for field in SCORES)
            needs_note = (
                row["relevance_pass"] is False
                or row["fluency_pass"] is False
                or row["factual_error"] is True
            )
            completed += int(
                signed and filled and (not needs_note or bool(row["note"].strip()))
            )
    return {
        "total": len(rows),
        "completed": completed,
        "pending": len(rows) - completed,
        "ready": completed == len(rows),
    }


def _display_path(path: str | None, output: Path) -> str | None:
    return os.path.relpath(path, output).replace("\\", "/") if path else None


def prepare_packet(kit: Path, copy_run: Path, output: Path) -> dict:
    """Create 18 independent query batches plus three balanced copy batches.

    Args:
        kit: Kit used by this operation.
        copy_run: Copy run used by this operation.
        output: Output used by this operation.

    Returns:
        Mapping with fields: kit, copy_run, kit_manifest_sha256,
        copy_run_manifest_sha256, expected_retrieval_judgments, expected_copy_judgments,
        batches, files, scope.

    Raises:
        FileExistsError: Use a new review packet directory.
        ValueError: Preparation must start from blank frozen templates.
    """
    output = ensure_output_path(output)
    if output.exists():
        raise FileExistsError("Use a new review packet directory")
    kit_manifest, queries, catalog = load_kit(kit)
    verify_files(
        copy_run, json.loads((copy_run / "manifest.json").read_text(encoding="utf-8"))
    )
    copy_results = read_rows(copy_run / "results.jsonl")
    copy_labels = {
        row["id"]: row for row in read_rows(copy_run / "human_scores.template.jsonl")
    }
    qrels = read_rows(kit / "qrels.template.jsonl")
    if any(row["grade"] is not None for row in qrels) or any(
        row[field] is not None for row in copy_labels.values() for field in SCORES
    ):
        raise ValueError("Preparation must start from blank frozen templates")
    output.mkdir(parents=True)
    template = (PROJECT_ROOT / "assets/human_review.html").read_text(encoding="utf-8")
    javascript = (PROJECT_ROOT / "assets/human_review.js").read_text(encoding="utf-8")
    batches: list[dict] = []

    def save_batch(batch_id: str, task: str, items: list[dict]) -> None:
        """Persist batch.

        Args:
            batch_id: Batch id used by this operation.
            task: Task used by this operation.
            items: Items used by this operation.

        Notes:
            Writes derived artifacts through (output /
            f'{batch_id}.html').write_text, write_rows.
        """
        labels = []
        for item in items:
            label = {
                **item["label"],
                "task": task,
                "batch_id": batch_id,
                "review_origin": None,
            }
            item["label"] = label
            labels.append(label)
        validate_batch(labels, labels)
        data = {"batch_id": batch_id, "task": task, "items": items}
        page = template.replace("__REVIEW_SCRIPT__", javascript).replace(
            "__REVIEW_DATA__",
            json.dumps(data, ensure_ascii=False).replace("<", "\\u003c"),
        )
        (output / f"{batch_id}.html").write_text(page, encoding="utf-8")
        write_rows(output / f"{batch_id}.template.jsonl", labels)
        batches.append(
            {
                "id": batch_id,
                "task": task,
                "count": len(items),
                "template": f"{batch_id}.template.jsonl",
                "page": f"{batch_id}.html",
            }
        )

    by_product = {row["id"]: row for row in catalog}
    for query in queries:
        display_query = {
            **query,
            "image_path": _display_path(query["image_path"], output),
        }
        items = []
        selected = [row for row in qrels if row["query_id"] == query["id"]]
        # Order independent of model scores; every candidate remains reviewable.
        for label in sorted(
            selected, key=lambda row: order_key(row["product_id"], 211)
        ):
            product = by_product[label["product_id"]]
            items.append(
                {
                    "label": dict(label),
                    "query": display_query,
                    "product": {
                        **product,
                        "image_path": _display_path(product["image_path"], output),
                    },
                }
            )
        save_batch("retrieval_" + query["id"], "retrieval", items)
    categories: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(copy_results, key=lambda row: order_key(row["id"], 212)):
        categories[row["category"]].append(row)
    interleaved = [
        group[i]
        for i in range(max(map(len, categories.values())))
        for _, group in sorted(categories.items())
        if i < len(group)
    ]
    for offset in range(0, len(interleaved), 30):
        items = []
        for row in interleaved[offset : offset + 30]:
            items.append(
                {
                    "label": dict(copy_labels[row["id"]]),
                    "record": {
                        "id": row["id"],
                        "category": row["category"],
                        "source": row["source"],
                        "copy": {
                            key: value
                            for key, value in row.get("copy", {}).items()
                            if key != "_meta"
                        },
                        "source_policy": row["source_policy"],
                        "automatic_checks": row.get("copy", {})
                        .get("_meta", {})
                        .get("fact_check"),
                        "error": row.get("error"),
                    },
                }
            )
        save_batch(f"copy_{offset//30+1:02d}", "copy", items)
    links = "".join(
        f'<li><a href="{html.escape(row["page"], quote=True)}">{html.escape(row["id"])}</a> · {row["count"]}项</li>'
        for row in batches
    )
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>第二阶段评审包</title><style>body{font:17px/1.8 system-ui;max-width:850px;margin:40px auto;padding:20px}a{color:#145d79}</style><h1>第二阶段评审包</h1><p>18个检索批次，每批一个完整查询；3个文案批次，每批30件、三类各10件。所有判断初始为空。页面不显示检索排序，文案自动规则默认折叠。请填写真实评审人，逐项判断并导出，随后通过命令合并。</p><p>保存至本项目 artifacts/review_stage2_v1/reviewed/；文件名保留批次编号。不能把模型评分冒充人工评分。可先导出部分进度，再导入同一批次续填。</p><ul>'
        + links
        + "</ul></html>",
        encoding="utf-8",
    )
    (output / "reviewed").mkdir()
    manifest = {
        "kit": str(kit.resolve()),
        "copy_run": str(copy_run.resolve()),
        "kit_manifest_sha256": sha256_file(kit / "manifest.json"),
        "copy_run_manifest_sha256": sha256_file(copy_run / "manifest.json"),
        "expected_retrieval_judgments": kit_manifest["judgments_required"],
        "expected_copy_judgments": len(copy_results),
        "batches": batches,
        "files": {
            path.name: sha256_file(path) for path in output.iterdir() if path.is_file()
        },
        "scope": "Blank human review batches; no model ranking displayed or inferred labels. Original frozen inputs unchanged.",
    }
    write_json(output / "packet.json", manifest)
    return manifest


def merge_reviews(
    packet: Path, inputs: list[Path], output: Path, predictions: Path | None = None
) -> dict:
    """Merge disjoint batches; preserve nulls and score only complete task sets.

    Args:
        packet: Packet used by this operation.
        inputs: Inputs used by this operation.
        output: Output used by this operation.
        predictions: Predictions used by this operation.

    Returns:
        Mapping with fields: ready, progress, completed, received_batches,
        packet_sha256, input_sha256, scope.

    Raises:
        FileExistsError: Use a new merged review directory.
        ValueError: Frozen review source changed; Each imported file must contain one
            nonempty batch; Unknown or duplicate batch; choose one version explicitly.
    """
    output = ensure_output_path(output)
    if output.exists():
        raise FileExistsError("Use a new merged review directory")
    manifest = json.loads((packet / "packet.json").read_text(encoding="utf-8"))
    verify_files(packet, manifest)
    for name in ("kit", "copy_run"):
        if (
            sha256_file(Path(manifest[name]) / "manifest.json")
            != manifest[f"{name}_manifest_sha256"]
        ):
            raise ValueError("Frozen review source changed")
    batches = {row["id"]: row for row in manifest["batches"]}
    labels = {
        key: read_rows(packet / batch["template"]) for key, batch in batches.items()
    }
    received = set()
    for path in inputs:
        rows = read_rows(path)
        if not rows or len({row["batch_id"] for row in rows}) != 1:
            raise ValueError("Each imported file must contain one nonempty batch")
        batch_id = rows[0]["batch_id"]
        if batch_id not in batches or batch_id in received:
            raise ValueError(
                "Unknown or duplicate batch; choose one version explicitly"
            )
        validate_batch(labels[batch_id], rows)
        labels[batch_id] = rows
        received.add(batch_id)
    progress = {
        key: validate_batch(read_rows(packet / batches[key]["template"]), rows)
        for key, rows in labels.items()
    }
    retrieval = [
        row
        for key, rows in labels.items()
        if batches[key]["task"] == "retrieval"
        for row in rows
    ]
    copies = [
        row
        for key, rows in labels.items()
        if batches[key]["task"] == "copy"
        for row in rows
    ]
    ready = {
        task: all(
            progress[key]["ready"] for key in batches if batches[key]["task"] == task
        )
        for task in ("retrieval", "copy")
    }
    output.mkdir(parents=True)
    qrels_path = output / (
        "qrels.reviewed.jsonl" if ready["retrieval"] else "qrels.partial.jsonl"
    )
    copy_path = output / (
        "copy_scores.reviewed.jsonl" if ready["copy"] else "copy_scores.partial.jsonl"
    )
    write_rows(qrels_path, retrieval)
    write_rows(copy_path, copies)
    if ready["retrieval"]:
        _, queries, catalog = load_kit(Path(manifest["kit"]))
        validate_qrels(queries, {row["id"] for row in catalog}, retrieval)
        if predictions:
            write_json(
                output / "retrieval_scores.json",
                score_predictions(predictions, qrels_path),
            )
    if ready["copy"]:
        write_json(
            output / "copy_scores.json",
            score_copy_run(Path(manifest["copy_run"]), copy_path),
        )
    report = {
        "ready": ready,
        "progress": progress,
        "completed": {
            task: sum(
                progress[key]["completed"]
                for key in batches
                if batches[key]["task"] == task
            )
            for task in ready
        },
        "received_batches": sorted(received),
        "packet_sha256": sha256_file(packet / "packet.json"),
        "input_sha256": {str(path.resolve()): sha256_file(path) for path in inputs},
        "scope": "Unreviewed entries remain null. No formal score until the corresponding task is fully reviewed. Reviewer type is an explicit declaration, not external identity verification.",
    }
    write_json(output / "progress.json", report)
    return report
