"""Compare a fully scored wider pool and corroborated subject penalties."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run  # noqa: F401
from module2_expand_pool import ROOT, TARGET
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.interval_retrieval_metrics import (
    evaluate_intervals,
    query_bounds,
)
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.simple_baseline import stable_order
from paper_fusion_experiment import COHORTS, DATA, GROUND, ROLE, SIMPLE, load
from primary_subject_ranker import penalty, signature
from querywise_paired_bounds import paired_querywise

BASE = "previous_blend50"
WIDE = "top30_complete_blend50"
SUBJECT = "top30_corroborated_subject"


def corroborated_profiles() -> dict:
    """Map same-source original-image observations to conservative subject types.

    Returns:
        Subject signatures with exact original image hashes. A disagreeing subtype
        stays unknown even if both observations agree on the broad family.
    """
    role_inputs = {
        r["key"]: r["source_sha256"] for r in load(ROLE / "profile_inputs.public.json")
    }
    ground_inputs = {
        r["key"]: r["source_sha256"] for r in load(GROUND / "views_inputs.public.json")
    }
    first = {r["key"]: r["value"] for r in load(ROLE / "profiles.json")}
    result = {}
    for row in load(GROUND / "semantic_results.json"):
        key = row["key"]
        left, right = first.get(key), row.get("native_value")
        source = role_inputs.get(key)
        if not source or source != ground_inputs.get(key) or not left or not right:
            continue
        if (
            left.get("primary_unclear") is not False
            or right.get("box_correct_sale_item") is not True
        ):
            continue
        a, b = signature(left["sale_item"]), signature(right["sale_item"])
        if a["family"] is None or a["family"] != b["family"]:
            continue
        result[key] = dict(
            source_sha256=source,
            family=a["family"],
            subtype=a["subtype"] if a["subtype"] == b["subtype"] else None,
            structures=[],
        )
    return result


def evidence(registry: dict) -> None:
    """Freeze the label-free subject evidence for every planned scored candidate.

    Args:
        registry: Development role registry checked before reading data inputs.
    """
    files = [
        ROOT / "sid_protocol.json",
        SIMPLE / "inputs.json",
        ROLE / "profile_inputs.public.json",
        ROLE / "profiles.json",
        GROUND / "views_inputs.public.json",
        GROUND / "semantic_results.json",
    ]
    require_data_access(files, "development", registry)
    protocol = load(files[0])
    original = load(files[1])
    profiles = corroborated_profiles()
    output = {}
    for job in protocol["jobs"]:
        query = job["query"]
        key = f"{job['cohort']}/{query['id']}"
        signature_q = profiles.get(key)
        if signature_q and signature_q["source_sha256"] != query["source_sha256"]:
            raise ValueError("Query observation source mismatch")
        scored = {}
        for cid in job["head"] + job["extra"]:
            item = original["documents"][cid]
            signature_d = profiles.get(cid)
            if signature_d and signature_d["source_sha256"] != item["source_sha256"]:
                raise ValueError("Candidate observation source mismatch")
            if signature_q and signature_d:
                scored[cid] = penalty(signature_q, signature_d)
            else:
                scored[cid] = dict(
                    value=0.0, reason="missing corroborated subject evidence"
                )
        output[key] = dict(query=signature_q, candidates=scored)
    files.extend(
        [Path(__file__).resolve(), PROJECT_ROOT / "scripts/primary_subject_ranker.py"]
    )
    manifest = dict(
        files={
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in files
        },
        policy="Two same-image profiles agree on family. Subtype requires agreement. Existing single-conflict penalty.",
        cases=output,
        formal_metrics=None,
    )
    target = ROOT / "subject_evidence.json"
    if target.exists() and load(target) != manifest:
        raise ValueError("Frozen subject evidence changed")
    write_json(target, manifest)
    print(f"Frozen subject evidence for {len(output)} image queries")


def blend(
    base: list[str], cross: list[str], scores: dict[str, float], offsets: dict
) -> list[str]:
    """Rank the complete scored head using the previous 50/50 rank blend.

    Args:
        base: Previous full gallery order used as the tail.
        cross: Previous full cross20 order.
        scores: Real same-input Qwen logits for all scored items.
        offsets: Optional existing conflict penalties in normalized rank units.

    Returns:
        The reranked head followed by all remaining original gallery entries.
    """
    head = stable_order(list(scores), np.array(list(scores.values())))
    cross_positions = {
        cid: i for i, cid in enumerate(cid for cid in cross if cid in scores)
    }
    values = np.array(
        [
            -(i + cross_positions[cid]) / (2 * (len(head) - 1))
            - offsets.get(cid, {}).get("value", 0.0)
            for i, cid in enumerate(head)
        ]
    )
    return stable_order(head, values) + [cid for cid in base if cid not in scores]


def rankings(protocol: dict, evidence_bank: dict, baseline: dict) -> dict:
    """Replay all baseline rankings and reject incomplete or mismatched model scores.

    Args:
        protocol: Frozen old and expanded scored-pool plan.
        evidence_bank: Frozen corroborated subject offsets for the exact head.
        baseline: Full image and text development rankings.

    Returns:
        Full rankings for control, wider pool and wider pool with subject evidence.
    """
    selected = protocol["selected_baseline"]
    output = {
        cohort: {
            qid: {name: row[selected] for name in (BASE, WIDE, SUBJECT)}
            for qid, row in group.items()
        }
        for cohort, group in baseline.items()
    }
    digest = sha256_file(ROOT / "sid_protocol.json")
    for job in protocol["jobs"]:
        cohort, query = job["cohort"], job["query"]
        qid = query["id"]
        cache = load(ROOT / "sid_pairs" / cohort / f"{qid}.json")
        if cache["identity"] != dict(
            protocol_sha256=digest,
            query_sha256=query["source_sha256"],
            extra=job["extra"],
        ) or set(cache["scores"]) != set(job["extra"]):
            raise ValueError("New real scores do not cover the exact frozen head")
        row = baseline[cohort][qid]
        original, cross = row[selected], row["cross20"]
        old = job["original_scores"]
        if blend(original, cross, old, {}) != original:
            raise ValueError("Previous blend50 replay differed")
        combined = {**old, **cache["scores"]}
        offsets = evidence_bank["cases"][f"{cohort}/{qid}"]["candidates"]
        if set(combined) != set(offsets):
            raise ValueError("Subject penalties are not aligned with the complete head")
        output[cohort][qid] = {
            BASE: original,
            WIDE: blend(original, cross, combined, {}),
            SUBJECT: blend(original, cross, combined, offsets),
        }
    return output


def evaluate(registry: dict) -> None:
    """Measure development effects and the user's targeted judgments separately.

    Args:
        registry: Data role registry used to restrict every opened input.
    """
    protocol_path = ROOT / "sid_protocol.json"
    bank_path = ROOT / "subject_evidence.json"
    answer_path = TARGET / "human_review/adjudications.json"
    baseline_path = TARGET / "rankings.json"
    paths = [protocol_path, bank_path, answer_path, baseline_path]
    paths.extend(DATA / cohort / "judgments.ai.jsonl" for cohort in COHORTS)
    paths.extend(sorted((ROOT / "sid_pairs").glob("*/*.json")))
    require_data_access(paths, "development", registry)
    protocol = load(protocol_path)
    bank = load(bank_path)
    for name, digest in {**protocol["files"], **bank["files"]}.items():
        if sha256_file(PROJECT_ROOT / name) != digest:
            raise ValueError(f"Frozen experiment input changed: {name}")
    result = rankings(protocol, bank, load(baseline_path))
    full = {
        f"{cohort}/{qid}": row
        for cohort, group in result.items()
        for qid, row in group.items()
    }
    labels = []
    for cohort in COHORTS:
        for line in (
            (DATA / cohort / "judgments.ai.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ):
            row = json.loads(line)
            labels.append(dict(row, query_id=f"{cohort}/{row['query_id']}"))
    modes = {qid: "text" if "/text_" in qid else "image" for qid in full}
    absolute = evaluate_intervals(full, labels, modes)
    paired = paired_querywise(
        full, labels, modes, [(WIDE, BASE), (SUBJECT, WIDE), (SUBJECT, BASE)]
    )
    grouped: dict[str, dict] = {}
    for row in load(answer_path):
        key = f"{row['cohort']}/{row['query_id']}"
        grouped.setdefault(key, {})[row["candidate_id"]] = row
    human: dict[str, dict] = {}
    for key, reviews in grouped.items():
        original = {
            row["candidate_id"]: row["grade"]
            for row in labels
            if row["query_id"] == key
        }
        for cid, row in reviews.items():
            if original[cid] != row["original_ai_grade"]:
                raise ValueError("Human review original label identity changed")
            original[cid] = (
                None if row["human_grade"] == "uncertain" else int(row["human_grade"])
            )
        human[key] = {}
        for method, order in full[key].items():
            top = [reviews[cid]["human_grade"] for cid in order[:10] if cid in reviews]
            human[key][method] = dict(
                human_confirmed_relevant_top10=top.count("1"),
                human_confirmed_irrelevant_top10=top.count("0"),
                mixed_label_ranges=query_bounds(order, original)["ranges"],
            )
    write_json(ROOT / "rankings.json", result)
    write_json(
        ROOT / "results.json",
        dict(
            AI_development=absolute,
            paired_AI_development=paired,
            targeted_review_diagnostic=human,
            human_review_sha256=sha256_file(answer_path),
            full_real_new_scores=protocol["new_pairs"],
            paired_pool_scores_reused=protocol["original_pairs"],
            subject_penalty_pairs=sum(
                row["value"] > 0
                for q in bank["cases"].values()
                for row in q["candidates"].values()
            ),
            frozen_acceptance_model_calls=0,
            formal_metrics=None,
        ),
    )
    print("Replayed 120 full rankings with exact original and additional Qwen scores.")


def main() -> None:
    """Freeze subject evidence before scoring, or evaluate the complete development run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "evaluate"])
    registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
    {"prepare": evidence, "evaluate": evaluate}[parser.parse_args().action](registry)


if __name__ == "__main__":
    main()
