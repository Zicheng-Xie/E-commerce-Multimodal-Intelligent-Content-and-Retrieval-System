"""Bounded direct-pixel machine re-review of contradictory weak positives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import run  # noqa: F401
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.machine_teacher import MODEL, resolve_local_teacher
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.pixel_refinement import (
    PIXEL_OPTIONS,
    PIXEL_PROMPT,
    review_pixels,
)
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

ROOT = PROJECT_ROOT / "artifacts/module2_precision_recall/machine_adjudication"
SOURCE = PROJECT_ROOT / "artifacts/module2_precision_recall"
WEAK = (
    PROJECT_ROOT
    / "artifacts/module2_continuation/pixel_refinement/refined_weak_labels.jsonl"
)


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve completed machine adjudication.

    Notes:
        Writes derived artifacts through ROOT.mkdir, folder.mkdir, write_json,
        write_rows.
    """
    if (ROOT / "receipt.json").exists():
        raise FileExistsError("Preserve completed machine adjudication")
    context = load(
        PROJECT_ROOT / "artifacts/module2_machine_assist/development_inputs.json"
    )
    queries = {q["id"]: q for q in context["queries"]}
    catalog = {r["id"]: r for r in context["catalog"]}
    conflicts = load(SOURCE / "machine_conflicts_for_review.json")["rows"]
    client = OllamaClient(model=MODEL, timeout=240)
    try:
        teacher = resolve_local_teacher(client, Path("E:/LLM_for_Ollama"))
        protocol = dict(
            selection="All 59 known machine-positive/public-evidence contradictions found in the completed development comparison; selection is adaptive development diagnosis, not unbiased evaluation",
            selected_pairs=[
                dict(query_id=r["query_id"], candidate_id=r["candidate_id"])
                for r in conflicts
            ],
            prompt=PIXEL_PROMPT,
            options=PIXEL_OPTIONS,
            teacher=teacher,
            prior_labels_sha256=sha256_file(WEAK),
            conflicts_sha256=sha256_file(SOURCE / "machine_conflicts_for_review.json"),
            source_sha256=sha256_file(Path(__file__)),
            human_labels=0,
            formal_metrics=None,
            goal_met=False,
        )
        ROOT.mkdir(parents=True, exist_ok=True)
        if (ROOT / "protocol.json").exists():
            assert load(ROOT / "protocol.json") == protocol
        else:
            write_json(ROOT / "protocol.json", protocol)
        reviews = {}
        folder = ROOT / "reviews"
        folder.mkdir(exist_ok=True)
        for i, row in enumerate(conflicts, 1):
            q, d = queries[row["query_id"]], catalog[row["candidate_id"]]
            images = ([Path(q["image_path"])] if q["image_path"] else []) + [
                Path(d["image_path"])
            ]
            public = dict(
                query_text=q["text"],
                query_rule=q["relevance_rule"],
                title=d["title"],
                image_order=(
                    "query then candidate" if len(images) == 2 else "candidate only"
                ),
            )
            prompt = PIXEL_PROMPT + json.dumps(public, ensure_ascii=False)
            key = hashlib.sha256(
                (row["query_id"] + "|" + row["candidate_id"]).encode()
            ).hexdigest()[:24]
            path = folder / (key + ".json")
            if path.exists():
                result = load(path)
                assert (
                    result["prompt_sha256"]
                    == hashlib.sha256(prompt.encode()).hexdigest()
                )
            else:
                value = review_pixels(client, prompt, images)
                result = dict(
                    query_id=row["query_id"],
                    candidate_id=row["candidate_id"],
                    review_origin="model",
                    human_label=False,
                    teacher_digest=teacher["digest"],
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    image_files={
                        str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in images
                    },
                    **value,
                )
                write_json(path, result)
                print(
                    json.dumps(
                        dict(review=i, total=len(conflicts), grade=result["grade"])
                    ),
                    flush=True,
                )
            reviews[row["query_id"], row["candidate_id"]] = result
    finally:
        client._request("/api/generate", dict(model=MODEL, keep_alive=0))
        client.close()
    rows = read_rows(WEAK)
    revised = []
    for row in rows:
        review = reviews.get((row["query_id"], row["candidate_id"]))
        if review:
            assert row["grade"] == 1
            row = {
                **row,
                "previous_machine_grade": row["grade"],
                "grade": review["grade"],
                "label_origin": "machine_conflict_direct_pixel_review",
                "machine_reason": review["reason"],
            }
        revised.append(row)
    write_rows(ROOT / "revised_weak_labels.jsonl", revised)
    write_json(
        ROOT / "status.json",
        dict(
            reviewed=len(reviews),
            retained_positive=sum(r["grade"] == 1 for r in reviews.values()),
            revised_negative=sum(r["grade"] == 0 for r in reviews.values()),
            revised_unknown=sum(r["grade"] is None for r in reviews.values()),
            total_positive=sum(r["grade"] == 1 for r in revised),
            total_negative=sum(r["grade"] == 0 for r in revised),
            total_unknown=sum(r["grade"] is None for r in revised),
            human_labels=0,
            formal_metrics=None,
            goal_met=False,
            automatic_ontology_relabeling=False,
        ),
    )
    write_json(
        ROOT / "receipt.json",
        dict(
            files={
                p.relative_to(ROOT).as_posix(): sha256_file(p)
                for p in ROOT.rglob("*")
                if p.is_file()
            }
        ),
    )
    print(json.dumps(load(ROOT / "status.json")))


if __name__ == "__main__":
    main()
