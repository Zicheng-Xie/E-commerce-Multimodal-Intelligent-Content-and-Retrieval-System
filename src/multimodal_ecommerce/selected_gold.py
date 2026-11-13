"""Independent human authoring and complete labels kept outside model inputs."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from multimodal_ecommerce.human_audit_ui import render_review
from multimodal_ecommerce.safety import ensure_output_path, sha256_file
from multimodal_ecommerce.selected_release import reservation_role, verify_release
from multimodal_ecommerce.selected_runtime_io import atomic_json, digest, load
from multimodal_ecommerce.simple_baseline import human_accuracy

FIELDS = [
    "task_id",
    "reviewer_type",
    "reviewer",
    "reviewed_at",
    "human_grade",
    "reason",
]


def _annotation_binding(release: Path, holdout: Path) -> dict:
    if reservation_role(holdout).get("consumed"):
        raise PermissionError("Consumed acceptance data cannot be relabeled")
    model = verify_release(release)
    return dict(
        release_id=model["release_id"],
        dataset_seal_sha256=sha256_file(holdout / "seal.json"),
    )


def _output(output: Path, holdout: Path) -> Path:
    path = ensure_output_path(output)
    if path.is_relative_to(holdout.resolve()) or holdout.resolve().is_relative_to(path):
        raise PermissionError("Human answers must stay outside the sealed reservation")
    return path


def text_template(release: Path, holdout: Path, output: Path) -> None:
    """Allow independent query authors to see only the predefined fine-type slots.

    Args:
        release: Frozen model implementation, verified before authoring begins.
        holdout: Immutable reservation; only its blank text template is read.
        output: Separate private gold directory.
    """
    binding = _annotation_binding(release, holdout)
    output = _output(output, holdout)
    if (output / "binding.json").exists():
        if load(output / "binding.json") != binding:
            raise ValueError("Gold authoring belongs to another frozen model/dataset")
        return
    source = holdout / "text_queries.human_pending.csv"
    output.mkdir(parents=True, exist_ok=True)
    (output / "text_queries.csv").write_bytes(source.read_bytes())
    atomic_json(
        output / "binding.json",
        binding,
    )


def prepare_packet(release: Path, holdout: Path, output: Path) -> dict:
    """Build a blinded full-gallery packet after all text queries are independently authored.

    Args:
        release: Frozen model manifest.
        holdout: Sealed catalog/images; unchanged by annotation.
        output: Separate private gold directory.

    Returns:
        Public completion metadata containing no relevance answers.
    """
    expected_binding = _annotation_binding(release, holdout)
    output = _output(output, holdout)
    binding = load(output / "binding.json")
    if binding != expected_binding:
        raise ValueError("Gold authoring/model identity changed")
    with (output / "text_queries.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    seal = load(holdout / "seal.json")
    if len(rows) != seal["independently_authored_text_queries_pending"] or len(
        {r["fine_type"] for r in rows}
    ) != len(rows):
        raise ValueError("Preserve all predefined independent query slots")
    with (holdout / "text_queries.human_pending.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        expected_types = {r["fine_type"] for r in csv.DictReader(f)}
    if {r["fine_type"] for r in rows} != expected_types:
        raise ValueError("Independent query categories cannot be substituted")
    queries: list[dict] = []
    for i, row in enumerate(rows):
        if not row["human_author"].strip() or not row["independent_query_text"].strip():
            raise ValueError(
                "A named independent author and nonempty query are required"
            )
        datetime.fromisoformat(row["approved_at"].replace("Z", "+00:00"))
        queries.append(
            dict(
                id=f"text_{i:03}",
                mode="text",
                text=row["independent_query_text"].strip(),
                exclusions=[],
            )
        )
    for i, row in enumerate(load(holdout / "image_queries.reserved.json")):
        queries.append(
            dict(
                id=f"image_{i:03}",
                mode="image",
                image_path=str(holdout / row["image_path"]),
                image_sha256=row["fingerprints"]["sha256"],
                exclusions=[],
            )
        )
    gallery = [
        dict(
            id=r["id"],
            title=r["text"],
            image_path=str(holdout / r["image_path"]),
            image_sha256=r["fingerprints"]["sha256"],
        )
        for r in load(holdout / "gallery.reserved.json")
    ]
    tasks = []
    for q in queries:
        for doc in gallery:
            tasks.append(
                dict(
                    task_id=digest(
                        dict(
                            dataset=binding["dataset_seal_sha256"],
                            query=q["id"],
                            product=doc["id"],
                        )
                    )[:24],
                    query_id=q["id"],
                    candidate_id=doc["id"],
                    query=dict(text=q.get("text"), image_path=q.get("image_path")),
                    candidate=dict(title=doc["title"], image_path=doc["image_path"]),
                )
            )
    packet = dict(binding=binding, catalog=gallery, queries=queries, tasks=tasks)
    path = output / "packet.private.json"
    if path.exists() and load(path) != packet:
        raise ValueError("Preserve the original blinded packet identities")
    if path.exists() and (output / "public_status.json").exists():
        return load(output / "public_status.json")
    atomic_json(path, packet)
    render_review(
        tasks,
        output / "review",
        title="独立验收人工相关性标注",
        instructions=(
            "仅按查询原图／文字、商品原图与原标题判断相关、不相关或证据不足。"
            + "不得参考任何模型排序或建议答案。导出 CSV 到本页同目录 answers；保留全部任务。原相关性标准："
            + verify_release(release)["relevance_instruction"]
        ),
    )
    status = dict(
        **binding,
        expected=len(tasks),
        answered=0,
        binary_complete=False,
        all_answers_complete=False,
        text_queries_complete=True,
        packet_sha256=sha256_file(path),
        formal_metrics=None,
    )
    atomic_json(output / "public_status.json", status)
    return status


def freeze_gold(release: Path, holdout: Path, output: Path) -> dict:
    """Validate real human CSVs and seal answers separately from runtime inputs.

    Args:
        release: Existing frozen model release.
        holdout: Unchanged reservation.
        output: Private annotation packet and answers directory.

    Returns:
        Status counts and hashes only; no answers exposed to development callers.
    """
    binding = _annotation_binding(release, holdout)
    output = _output(output, holdout)
    packet_path = output / "packet.private.json"
    status_path = output / "public_status.json"
    if sha256_file(packet_path) != load(status_path)["packet_sha256"]:
        raise ValueError("Blinded annotation packet was altered")
    packet = load(packet_path)
    if packet["binding"] != binding:
        raise ValueError("Gold labels are bound to another frozen release")
    tasks = packet["tasks"]
    task_ids = {r["task_id"]: r for r in tasks}
    task_batches = {r["task_id"]: i // 30 for i, r in enumerate(tasks)}
    seen, answers, inputs = set(), [], {}
    for path in sorted((output / "review/answers").glob("*.csv")):
        inputs[str(path.resolve())] = sha256_file(path)
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != FIELDS:
                raise ValueError("Human CSV fields differ from the independent packet")
            batches = set()
            for row in reader:
                key = row.get("task_id")
                if (
                    key not in task_ids
                    or key in seen
                    or None in row
                    or any(v is None for v in row.values())
                ):
                    raise ValueError(
                        "Foreign, duplicated or malformed independent answer"
                    )
                seen.add(key)
                batches.add(task_batches[key])
                if row["human_grade"].strip():
                    datetime.fromisoformat(row["reviewed_at"].replace("Z", "+00:00"))
                answers.append(row)
            if len(batches) > 1:
                raise ValueError("An exported file contains mixed task batches")
        if sha256_file(path) != inputs[str(path.resolve())]:
            raise ValueError("An independent answer changed during import")
    checked = human_accuracy([dict(t, grade=None) for t in tasks], answers)
    graded = []
    for row in answers:
        if not row["human_grade"].strip():
            continue
        task = task_ids[row["task_id"]]
        graded.append(
            dict(
                query_id=task["query_id"],
                candidate_id=task["candidate_id"],
                grade={"1": 1, "0": 0, "uncertain": None}[row["human_grade"]],
                human=True,
            )
        )
    binary = checked["complete"] and all(row["grade"] is not None for row in graded)
    status = dict(
        **binding,
        expected=len(tasks),
        answered=checked["completed"],
        binary_complete=binary,
        all_answers_complete=checked["complete"],
        text_queries_complete=True,
        packet_sha256=sha256_file(packet_path),
        unresolved=sum(r["grade"] is None for r in graded),
        formal_metrics=None,
    )
    if checked["complete"]:
        gold = dict(
            binding=binding,
            packet_sha256=status["packet_sha256"],
            answers=graded,
            answer_csv_sha256=inputs,
        )
        seal_path = output / "gold.private.json"
        if seal_path.exists() and load(seal_path) != gold:
            raise ValueError("Preserve the original frozen human gold")
        atomic_json(seal_path, gold)
        status["gold_sha256"] = sha256_file(seal_path)
    atomic_json(output / "public_status.json", status)
    return status
