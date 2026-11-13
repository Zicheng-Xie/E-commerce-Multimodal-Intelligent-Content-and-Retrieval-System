"""Freeze and run bounded pixel review; never mutate the original weak labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.machine_teacher import MODEL, resolve_local_teacher
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.pixel_refinement import (
    PIXEL_OPTIONS,
    PIXEL_PROMPT,
    PIXEL_SCHEMA,
    merge_machine_refinement,
    review_pixels,
    select_uncertain,
)
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

BASE = PROJECT_ROOT / "artifacts/module2_machine_assist"
ROOT = PROJECT_ROOT / "artifacts/module2_continuation/pixel_refinement"


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        FileExistsError: Preserve the pixel-refinement protocol.
        ValueError: Reserved acceptance cannot enter pixel review.

    Notes:
        Writes derived artifacts through ROOT.mkdir, write_json.
    """
    if (ROOT / "protocol.json").exists():
        raise FileExistsError("Preserve the pixel-refinement protocol")
    verify_files(
        BASE,
        json.loads((BASE / "completion_receipt.json").read_text(encoding="utf-8"))[
            "files"
        ],
    )
    labels = read_rows(BASE / "weak_labels.jsonl")
    rankings = json.loads(
        (BASE / "semantic_ranker/rankings.json").read_text(encoding="utf-8")
    )
    selection = select_uncertain(labels, rankings, 12)
    context = json.loads((BASE / "development_inputs.json").read_text(encoding="utf-8"))
    queries = {r["id"]: r for r in context["queries"]}
    catalog = {r["id"]: r for r in context["catalog"]}
    reserved = read_rows(
        PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl"
    )
    forbidden = {r["image_sha256"] for r in reserved}
    tasks = []
    files = {}
    for row in selection:
        query = queries[row["query_id"]]
        doc = catalog[row["candidate_id"]]
        pictures = ([query["image_path"]] if query["image_path"] else []) + [
            doc["image_path"]
        ]
        for name in pictures:
            path = Path(name)
            path.resolve().relative_to(PROJECT_ROOT)
            digest = sha256_file(path)
            if digest in forbidden:
                raise ValueError("Reserved acceptance cannot enter pixel review")
            files[path.relative_to(PROJECT_ROOT).as_posix()] = digest
        tasks.append(
            dict(
                query_id=row["query_id"],
                candidate_id=row["candidate_id"],
                query_text=query["text"],
                query_rule=query["relevance_rule"],
                title=doc["title"],
                images=pictures,
                selection_min_top30_rank=row["selection_min_top30_rank"],
            )
        )
    client = OllamaClient(model=MODEL)
    try:
        teacher = resolve_local_teacher(client, Path("E:/LLM_for_Ollama"))
    finally:
        client.close()
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / "tasks.json", tasks)
    paths = [
        "src/multimodal_ecommerce/pixel_refinement.py",
        "scripts/refine_machine_labels.py",
        "src/multimodal_ecommerce/machine_teacher.py",
    ]
    write_json(
        ROOT / "protocol.json",
        dict(
            prompt=PIXEL_PROMPT,
            schema=PIXEL_SCHEMA,
            options=PIXEL_OPTIONS,
            max_unknown_per_query=12,
            selected_pairs=len(tasks),
            selection="Minimum rank within top30 union of eight frozen old development methods; seeded tie-break, then seeded remaining unknowns; no outcome-based stopping",
            teacher=teacher,
            tasks_sha256=sha256_file(ROOT / "tasks.json"),
            original_labels_sha256=sha256_file(BASE / "weak_labels.jsonl"),
            implementation={name: sha256_file(PROJECT_ROOT / name) for name in paths},
            input_files=files,
            human_labels=0,
            formal_acceptance=False,
            formal_metrics=None,
            goal_met=False,
            acceptance_model_predictions=0,
        ),
    )
    print(
        json.dumps(
            dict(
                selected_pairs=len(tasks),
                queries=len({r["query_id"] for r in tasks}),
                formal_acceptance=False,
            )
        )
    )


def run_reviews() -> None:
    """Apply the run reviews workflow step.

    Raises:
        FileExistsError: Preserve completed refinement.
        ValueError: Pixel review tasks changed; Teacher changed; Pixel review cache
            mismatch; Original weak labels changed.

    Notes:
        Writes derived artifacts through folder.mkdir, write_json, write_rows.
    """
    if (ROOT / "receipt.json").exists():
        raise FileExistsError("Preserve completed refinement")
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    verify_files(PROJECT_ROOT, protocol["implementation"])
    verify_files(PROJECT_ROOT, protocol["input_files"])
    if sha256_file(ROOT / "tasks.json") != protocol["tasks_sha256"]:
        raise ValueError("Pixel review tasks changed")
    client = OllamaClient(model=MODEL, timeout=240)
    teacher = resolve_local_teacher(client, Path("E:/LLM_for_Ollama"))
    if teacher["digest"] != protocol["teacher"]["digest"]:
        raise ValueError("Teacher changed")
    tasks = json.loads((ROOT / "tasks.json").read_text(encoding="utf-8"))
    folder = ROOT / "reviews"
    folder.mkdir(exist_ok=True)
    records = []
    try:
        for i, task in enumerate(tasks, 1):
            key = hashlib.sha256(
                (task["query_id"] + "|" + task["candidate_id"]).encode()
            ).hexdigest()[:24]
            path = folder / (key + ".json")
            public = {k: task[k] for k in ["query_text", "query_rule", "title"]}
            public["image_order"] = (
                "query then candidate" if len(task["images"]) == 2 else "candidate only"
            )
            prompt = PIXEL_PROMPT + json.dumps(public, ensure_ascii=False)
            if path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))
                if (
                    row["query_id"] != task["query_id"]
                    or row["candidate_id"] != task["candidate_id"]
                    or row["prompt_sha256"]
                    != hashlib.sha256(prompt.encode()).hexdigest()
                ):
                    raise ValueError("Pixel review cache mismatch")
            else:
                result = review_pixels(
                    client, prompt, [Path(p) for p in task["images"]]
                )
                row = dict(
                    query_id=task["query_id"],
                    candidate_id=task["candidate_id"],
                    review_origin="model",
                    human_label=False,
                    teacher_digest=teacher["digest"],
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    **result,
                )
                write_json(path, row)
                print(
                    json.dumps(
                        dict(
                            review=i,
                            total=len(tasks),
                            grade=row["grade"],
                            seconds=row["raw"]["wall_seconds"],
                        )
                    ),
                    flush=True,
                )
            records.append(row)
    finally:
        client._request("/api/generate", dict(model=MODEL, keep_alive=0))
        client.close()
    if sha256_file(BASE / "weak_labels.jsonl") != protocol["original_labels_sha256"]:
        raise ValueError("Original weak labels changed")
    original = read_rows(BASE / "weak_labels.jsonl")
    merged = merge_machine_refinement(original, records)
    write_rows(ROOT / "refined_weak_labels.jsonl", merged)
    status = dict(
        reviewed_pairs=len(records),
        newly_resolved=sum(r["grade"] is not None for r in records),
        still_unknown_in_review=sum(r["grade"] is None for r in records),
        total=len(merged),
        known=sum(r["grade"] is not None for r in merged),
        unknown=sum(r["grade"] is None for r in merged),
        positive=sum(r["grade"] == 1 for r in merged),
        negative=sum(r["grade"] == 0 for r in merged),
        human_labels=0,
        formal_acceptance=False,
        formal_metrics=None,
        goal_met=False,
        acceptance_model_predictions=0,
    )
    write_json(ROOT / "status.json", status)
    write_json(
        ROOT / "receipt.json",
        dict(
            files={
                p.relative_to(ROOT).as_posix(): sha256_file(p)
                for p in ROOT.rglob("*")
                if p.is_file()
            },
            formal_acceptance=False,
        ),
    )
    print(json.dumps(status))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run"])
    args = parser.parse_args()
    {"prepare": prepare, "run": run_reviews}[args.action]()
