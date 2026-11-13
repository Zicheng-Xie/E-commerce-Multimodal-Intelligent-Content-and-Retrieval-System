"""Run complete, resumable AI review without filling any human-review stream."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run  # noqa: F401
from multimodal_ecommerce.ai_review import (
    IMAGE_PROMPT,
    IMAGE_SCHEMA,
    OPTIONS,
    PROFILE_PROMPT,
    PROFILE_SCHEMA,
    diagnostic_intervals,
    infer_images,
    validate_profile,
)
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.machine_teacher import (
    MODEL,
    resolve_local_teacher,
    strict_grades,
)
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.pixel_refinement import PIXEL_PROMPT, review_pixels
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

ROOT = PROJECT_ROOT / "artifacts/module2_ai_review"
OLD = PROJECT_ROOT / "artifacts/module2_machine_assist"
CAT = PROJECT_ROOT / "artifacts/module2_catalog_review"
RANK = (
    PROJECT_ROOT / "artifacts/module2_precision_recall/adjudicated_ranker/rankings.json"
)
IMAGE_TYPES = dict(
    zip(
        [f"image_{i:02}" for i in range(1, 7)],
        [
            "cabinet",
            "phone_holder",
            "wall_spice_rack",
            "phone_case",
            "women_pantsuit",
            "tire_valve_cap",
        ],
    )
)
BUCKET_TYPES = dict(
    zip(
        [
            "wireless_headphones",
            "computer_keyboards",
            "phone_cases",
            "power_banks",
            "short_tshirts",
            "women_dresses",
            "sleep_pillows",
            "home_chairs",
        ],
        [
            "headphone",
            "keyboard",
            "phone_case",
            "power_bank",
            "tshirt",
            "dress",
            "pillow",
            "chair",
        ],
    )
)


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def dump_rows(path: Path, rows: list[dict]) -> None:
    """Apply the dump rows workflow step.

    Args:
        path: Path of the input or output artifact used by this operation.
        rows: Records processed by this operation.

    Notes:
        Writes derived artifacts through path.write_text.
    """
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def key(value: str) -> str:
    """Return key for this workflow.

    Args:
        value: Input value to validate or transform.

    Returns:
        Text produced by this operation.
    """
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def sources() -> tuple[list[dict], list[dict]]:
    """Return sources for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    base = load(OLD / "development_inputs.json")
    products = [dict(r, catalog_source="original286") for r in base["catalog"]]
    for r in read_rows(CAT / "dataset/products.jsonl"):
        products.append(
            dict(
                id=r["id"],
                title=r["text"],
                image_path=str((CAT / "dataset" / r["image_path"]).resolve()),
                bucket=r["bucket"],
                catalog_source="expansion64",
            )
        )
    return base["queries"], products


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        ValueError: AI protocol already frozen; resume with run.

    Notes:
        Writes derived artifacts through ROOT.mkdir, write_json.
    """
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "protocol.json").exists():
        raise ValueError("AI protocol already frozen; resume with run")
    queries, products = sources()
    inputs = dict(queries=queries, catalog=products)
    write_json(ROOT / "inputs.json", inputs)
    client = OllamaClient(model=MODEL)
    try:
        teacher = resolve_local_teacher(
            client, PROJECT_ROOT.parent.parent / "LLM_for_Ollama"
        )
    finally:
        client.close()
    implementation = {
        p: sha256_file(PROJECT_ROOT / p)
        for p in [
            "scripts/review_module2_ai.py",
            "src/multimodal_ecommerce/ai_review.py",
            "src/multimodal_ecommerce/machine_teacher.py",
            "src/multimodal_ecommerce/pixel_refinement.py",
            "src/multimodal_ecommerce/ollama_client.py",
        ]
    }
    guards = {
        p: sha256_file(PROJECT_ROOT / p)
        for p in [
            "data/module2_reserved_acceptance/records.jsonl",
            "data/module2_development_expansion/records.jsonl",
            "artifacts/module2_catalog_review/post_preparation_exclusion_snapshot.json",
            "artifacts/module2_precision_recall/adjudicated_ranker/rankings.json",
            "artifacts/module2_business_review/development/judgments.blank.jsonl",
        ]
    }
    image_files = {
        str(Path(p).resolve().relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        ): sha256_file(Path(p))
        for p in [
            *[r["image_path"] for r in products],
            *[q["image_path"] for q in queries if q["image_path"]],
        ]
    }
    protocol = dict(
        frozen_at=datetime.now(timezone.utc).isoformat(),
        authorization="User delegated all reviews to AI; no human assessors available.",
        role="AI_reviewed_development_not_independent_human_truth",
        review_origin="model",
        human_label=False,
        teacher=teacher,
        queries=len(queries),
        original_products=286,
        expansion_candidates=64,
        input_sha256=sha256_file(ROOT / "inputs.json"),
        implementation=implementation,
        guards=guards,
        image_files=image_files,
        profile_prompt=PROFILE_PROMPT,
        profile_schema=PROFILE_SCHEMA,
        image_prompt=IMAGE_PROMPT,
        image_schema=IMAGE_SCHEMA,
        options=OPTIONS,
        image_type_screen=IMAGE_TYPES,
        screening_rule="Only readable, named mutually incompatible subject types yield image negatives; other/unclear get all six actual reference images. Matching types get a direct image pair review.",
        catalog_QA_rule="Readable observed subject must match acquisition bucket; an explicit title/image ambiguity remains uncertain. Catalog usability is not a positive relevance label.",
        null_policy="Unknowns remain null; never negative-filled. Intervals are conditional on AI judgments, not truth-confidence intervals.",
        fixed_k=10,
        precision_required=0.6,
        recall_preferred=0.85,
        acceptance_predictions_allowed=False,
        human_submissions_written=False,
        retrieval_strategy_training_in_this_review=False,
    )
    write_json(ROOT / "protocol.json", protocol)
    assignments = load(CAT / "work_assignments.json")
    assignments.update(
        actual_assignment_pending=False,
        independent_actual_people_required=False,
        active_workflow="AI_review",
        active_owner="Codex with pinned local vision model",
        human_assessors_available=False,
        user_delegated_all_reviews_to_AI=True,
        AI_review_protocol="artifacts/module2_ai_review/protocol.json",
        historical_human_slots_retained_blank=True,
    )
    write_json(CAT / "work_assignments.json", assignments)
    status = load(CAT / "public_status.json")
    status.update(
        state="AI_review_in_progress_user_delegated_no_human_wait",
        human_assignment_pending=False,
        active_review_status="artifacts/module2_ai_review/status.json",
    )
    write_json(CAT / "public_status.json", status)
    write_json(
        ROOT / "status.json",
        dict(
            state="prepared",
            profile_reviews=0,
            human_labels=0,
            formal_metrics=None,
            goal_met=False,
        ),
    )
    print(
        json.dumps(
            dict(prepared=True, products=len(products), teacher=teacher["model"])
        )
    )


def run_reviews() -> None:
    """Apply the run reviews workflow step.

    Raises:
        ValueError: Review input changed; Local teacher identity changed;
            Multi-reference mapping invalid.

    Notes:
        Writes derived artifacts through (ROOT / name).mkdir, write_json.
    """
    protocol = load(ROOT / "protocol.json")
    verify_files(PROJECT_ROOT, protocol["implementation"])
    verify_files(PROJECT_ROOT, protocol["guards"])
    verify_files(PROJECT_ROOT, protocol["image_files"])
    if sha256_file(ROOT / "inputs.json") != protocol["input_sha256"]:
        raise ValueError("Review input changed")
    inputs = load(ROOT / "inputs.json")
    queries, products = inputs["queries"], inputs["catalog"]
    client = OllamaClient(model=MODEL, timeout=360)
    current = resolve_local_teacher(
        client, PROJECT_ROOT.parent.parent / "LLM_for_Ollama"
    )
    if current != protocol["teacher"]:
        raise ValueError("Local teacher identity changed")
    for name in ["profiles", "image_reviews", "failures"]:
        (ROOT / name).mkdir(exist_ok=True)
    try:
        text_queries = [
            dict(id=q["id"], text=q["text"], relevance_rule=q["relevance_rule"])
            for q in queries
            if q["text"]
        ]
        for n, product in enumerate(products, 1):
            path = ROOT / "profiles" / (key(product["id"]) + ".json")
            if not path.exists():
                prompt = PROFILE_PROMPT + json.dumps(
                    dict(title=product["title"], queries=text_queries),
                    ensure_ascii=False,
                )
                result = infer_images(
                    client, prompt, PROFILE_SCHEMA, [(Path(product["image_path"]), 512)]
                )
                validated = validate_profile(result["value"])
                write_json(
                    path,
                    dict(
                        candidate_id=product["id"],
                        review_origin="model",
                        human_label=False,
                        prompt=prompt,
                        image_sha256=sha256_file(Path(product["image_path"])),
                        result=validated,
                        raw=result["raw"],
                    ),
                )
            if n % 10 == 0 or n == len(products):
                write_json(
                    ROOT / "status.json",
                    dict(
                        state="reviewing_profiles",
                        profile_reviews=n,
                        total_products=len(products),
                        human_labels=0,
                        formal_metrics=None,
                        goal_met=False,
                    ),
                )
                print(f"profile {n}/{len(products)}", flush=True)
        image_queries = [q for q in queries if q["image_path"]]
        for n, product in enumerate(products, 1):
            path = ROOT / "image_reviews" / (key(product["id"]) + ".json")
            if path.exists():
                continue
            profile = load(ROOT / "profiles" / (key(product["id"]) + ".json"))["result"]
            observed = profile["observed_type"]
            results = {}
            if observed in {"other", "unclear"} or not profile["readable"]:
                prompt = IMAGE_PROMPT + product["title"]
                out = infer_images(
                    client,
                    prompt,
                    IMAGE_SCHEMA,
                    [(Path(product["image_path"]), 512)]
                    + [(Path(q["image_path"]), 256) for q in image_queries],
                )
                value = out["value"]
                if (
                    set(value) != {"grades", "reasons"}
                    or len(value["reasons"]) != 6
                    or any(
                        not isinstance(r, str) or not r.strip()
                        for r in value["reasons"]
                    )
                ):
                    raise ValueError("Multi-reference mapping invalid")
                grades = strict_grades(value["grades"], 6)
                for q, grade, reason in zip(image_queries, grades, value["reasons"]):
                    results[q["id"]] = dict(
                        grade=grade, reason=reason, origin="actual_seven_image_review"
                    )
                raw = out["raw"]
            else:
                raw = []
                for q in image_queries:
                    if q["exclude_id"] == product["id"]:
                        continue
                    if observed != IMAGE_TYPES[q["id"]]:
                        results[q["id"]] = dict(
                            grade=0,
                            reason=f"Observed sold subject {observed} is incompatible with query subject {IMAGE_TYPES[q['id']]}. "
                            + profile["evidence"],
                            origin="visible_subject_type_mismatch",
                        )
                    else:
                        prompt = PIXEL_PROMPT + json.dumps(
                            dict(query=q, candidate_title=product["title"]),
                            ensure_ascii=False,
                        )
                        reviewed = review_pixels(
                            client,
                            prompt,
                            [Path(q["image_path"]), Path(product["image_path"])],
                        )
                        results[q["id"]] = dict(
                            grade=reviewed["grade"],
                            reason=reviewed["reason"],
                            origin="actual_two_image_review",
                        )
                        raw.append(
                            dict(
                                query_id=q["id"],
                                prompt=prompt,
                                response=reviewed["raw"],
                            )
                        )
            write_json(
                path,
                dict(
                    candidate_id=product["id"],
                    review_origin="model",
                    human_label=False,
                    results=results,
                    raw=raw,
                ),
            )
            if n % 10 == 0 or n == len(products):
                write_json(
                    ROOT / "status.json",
                    dict(
                        state="reviewing_image_relations",
                        profile_reviews=len(products),
                        image_relation_products=n,
                        total_products=len(products),
                        human_labels=0,
                        formal_metrics=None,
                        goal_met=False,
                    ),
                )
                print(f"image relations {n}/{len(products)}", flush=True)
    except Exception as error:
        write_json(
            ROOT
            / "failures"
            / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + ".json"),
            dict(error_type=type(error).__name__, error=str(error)),
        )
        raise
    finally:
        client._request("/api/generate", dict(model=MODEL, keep_alive=0))
        client.close()
    summarize()


def summarize() -> None:
    """Apply the summarize workflow step.

    Notes:
        Writes derived artifacts through write_json.
    """
    inputs = load(ROOT / "inputs.json")
    labels, qa = [], []
    for product in inputs["catalog"]:
        profile = load(ROOT / "profiles" / (key(product["id"]) + ".json"))["result"]
        images = load(ROOT / "image_reviews" / (key(product["id"]) + ".json"))[
            "results"
        ]
        for i, q in enumerate(inputs["queries"]):
            if q.get("exclude_id") == product["id"]:
                continue
            if q["text"]:
                grade, reason, origin = (
                    profile["text_grades"][i],
                    profile["evidence"],
                    "actual_candidate_image_and_title_against_text_rules",
                )
            else:
                r = images[q["id"]]
                grade, reason, origin = r["grade"], r["reason"], r["origin"]
            labels.append(
                dict(
                    query_id=q["id"],
                    candidate_id=product["id"],
                    grade=grade,
                    reason=reason,
                    evidence_origin=origin,
                    review_origin="model",
                    human_label=False,
                )
            )
        if product["catalog_source"] == "expansion64":
            if (
                profile["observed_type"] in {"other", "unclear"}
                or not profile["readable"]
            ):
                verdict = "uncertain"
            elif profile["observed_type"] != BUCKET_TYPES[product["bucket"]]:
                verdict = "reject"
            elif profile["uncertainty"].strip():
                verdict = "uncertain"
            else:
                verdict = "usable"
            qa.append(
                dict(
                    candidate_id=product["id"],
                    bucket=product["bucket"],
                    verdict=verdict,
                    observed_type=profile["observed_type"],
                    reason=profile["evidence"],
                    uncertainty=profile["uncertainty"],
                    review_origin="model",
                    human_label=False,
                    human_approved=False,
                )
            )
    dump_rows(ROOT / "judgments.ai.jsonl", labels)
    dump_rows(ROOT / "catalog_QA.ai.jsonl", qa)
    ranks = load(RANK)
    old_ids = {
        r["id"] for r in inputs["catalog"] if r["catalog_source"] == "original286"
    }
    original_labels = [r for r in labels if r["candidate_id"] in old_ids]
    modes = {q["id"]: "text" if q["text"] else "image" for q in inputs["queries"]}
    intervals = diagnostic_intervals(ranks, original_labels, modes)
    write_json(ROOT / "original_catalog_diagnostic.json", intervals)
    conflicts = load(
        PROJECT_ROOT
        / "artifacts/module2_precision_recall/machine_conflicts_for_review.json"
    )["rows"]
    keys = {(r["query_id"], r["candidate_id"]) for r in conflicts}
    dump_rows(
        ROOT / "priority59.ai.jsonl",
        [r for r in labels if (r["query_id"], r["candidate_id"]) in keys],
    )
    status = dict(
        state="complete_AI_review_pending_native_visual_audit",
        reviewed_products=len(inputs["catalog"]),
        reviewed_relations=len(labels),
        original_relations=len(original_labels),
        priority_pairs_reviewed=len(keys),
        catalog_QA={
            v: sum(r["verdict"] == v for r in qa)
            for v in ["usable", "reject", "uncertain"]
        },
        positive=intervals["positive"],
        negative=intervals["negative"],
        unknown=intervals["unknown"],
        review_origin="model",
        human_labels=0,
        human_assignment_pending=False,
        formal_metrics=None,
        formal_acceptance=False,
        goal_met=False,
        acceptance_prediction_count=0,
        default_strategy_changed=False,
    )
    write_json(ROOT / "status.json", status)
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    if (ROOT / "native_audit/full_review_protocol.json").exists():
        raise SystemExit(
            "Automatic review superseded; use finalize/verify_module2_ai_review."
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "summarize"])
    action = parser.parse_args().action
    {"prepare": prepare, "run": run_reviews, "summarize": summarize}[action]()
