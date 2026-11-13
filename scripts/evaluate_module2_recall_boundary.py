"""Separate image candidate coverage from reranking at the Top10 boundary."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import run  # noqa: F401
from module2_poster_text_replay import (
    SWIFT,
    candidate_signature,
    query_signature,
    subject_hints,
)
from module2_recall_boundary import EXPANDED, POSTER, ROOT, TARGET, TITLE, WORKER
from module2_subject_pool_evaluation import blend, corroborated_profiles
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from paper_fusion_experiment import COHORTS, DATA, SIMPLE, load
from primary_subject_ranker import penalty, signature
from querywise_paired_bounds import paired_querywise

CONSENSUS = "top30_consensus_support"
EXPANSION = "top33_cross_support"
COMBINED = "top33_consensus_support"
TIE = "top30_consensus_boundary_tie"
METHODS = (TITLE, CONSENSUS, EXPANSION, COMBINED, TIE)


def read_scores(root: Path, job: dict) -> dict[str, float]:
    """Load scores for exactly the frozen query and candidates.

    Args:
        root: Experiment owning the score protocol and cache.
        job: Frozen query source and new candidate IDs.

    Returns:
        The finite, complete same-model logits.
    """
    query = job["query"]
    path = root / "sid_pairs" / job["cohort"] / f"{query['id']}.json"
    cached = load(path)
    scores = cached["scores"]
    if (
        cached["identity"]
        != dict(
            protocol_sha256=sha256_file(root / "sid_protocol.json"),
            query_sha256=query["source_sha256"],
            extra=job["extra"],
        )
        or set(scores) != set(job["extra"])
        or not all(map(math.isfinite, scores.values()))
    ):
        raise ValueError(f"Score cache does not match the frozen experiment: {path}")
    return scores


def break_boundary_tie(
    base: list[str],
    cross: list[str],
    consensus: list[str],
    scores: dict[str, float],
    offsets: dict,
) -> list[str]:
    """Use consensus as a tiebreaker for exactly the tenth and eleventh ranks.

    Args:
        base: Verified baseline order, including the full gallery.
        cross: Original auxiliary rank view used by the baseline blend.
        consensus: Full-gallery median rank across the three retrieval views.
        scores: Same-model logits for the original Top30.
        offsets: Corroborated subject penalties used by the baseline.

    Returns:
        Baseline order with at most the adjacent boundary positions swapped.
    """
    incumbent, challenger = base[9:11]
    if consensus.index(incumbent) < consensus.index(challenger):
        return base
    model = sorted(scores, key=lambda cid: (-scores[cid], cid))
    auxiliary = [cid for cid in cross if cid in scores]
    values = [
        -(model.index(cid) + auxiliary.index(cid)) / (2 * (len(scores) - 1))
        - offsets[cid]["value"]
        for cid in (incumbent, challenger)
    ]
    if not math.isclose(*values, rel_tol=0.0, abs_tol=1e-12):
        return base
    return [*base[:9], challenger, incumbent, *base[11:]]


def replay(protocol: dict) -> dict:
    """Reproduce the baseline and vary candidate coverage and support independently.

    Args:
        protocol: Frozen additions and consensus orders made without judgments.

    Returns:
        Full-gallery rankings for each original query and method.
    """
    previous = load(POSTER / "rankings.json")
    originals = load(TARGET / "rankings.json")
    inputs = load(SIMPLE / "inputs.json")["documents"]
    lines = load(POSTER / "ocr.json")
    profiles = corroborated_profiles()
    role, ground = subject_hints()
    unknown = signature("")
    old_jobs = {
        f"{job['cohort']}/{job['query']['id']}": job
        for job in load(EXPANDED / "sid_protocol.json")["jobs"]
    }
    rankings = {
        key: dict.fromkeys(METHODS, row[TITLE]) for key, row in previous.items()
    }
    for job in protocol["jobs"]:
        cohort, query = job["cohort"], job["query"]
        qid = query["id"]
        key = f"{cohort}/{qid}"
        old_job = old_jobs[key]
        if query != old_job["query"]:
            raise ValueError("Original query identity differs")
        old = {**old_job["original_scores"], **read_scores(EXPANDED, old_job)}
        added = read_scores(ROOT, job)
        scores = {**old, **added}
        base = previous[key][TITLE]
        cross = originals[cohort][qid]["cross20"]
        if (
            job["head"] != base[:30]
            or set(old) != set(job["head"])
            or set(old) & set(added)
            or not set(added) <= set(base)
        ):
            raise ValueError("New scores must add distinct in-gallery candidates")
        subject = profiles.get(key, unknown)
        source = ground.get(key)
        if source and source[0] == query["source_sha256"]:
            subject = query_signature(lines[key], subject, source[1])
        offsets = {}
        for cid in scores:
            doc = inputs[cid]
            hints = [
                row[1]
                for row in (role.get(cid), ground.get(cid))
                if row and row[0] == doc["source_sha256"]
            ]
            candidate = candidate_signature(doc, profiles.get(cid, unknown), hints)
            offsets[cid] = penalty(subject, candidate)
        if blend(base, cross, old, offsets) != base:
            raise ValueError(f"The exact Top30 control could not be replayed: {key}")
        rankings[key] = {
            TITLE: base,
            CONSENSUS: blend(base, job["consensus"], old, offsets),
            EXPANSION: blend(base, cross, scores, offsets),
            COMBINED: blend(base, job["consensus"], scores, offsets),
            TIE: break_boundary_tie(base, cross, job["consensus"], old, offsets),
        }
    return rankings


def human_movements(rankings: dict, jobs: dict, reviewed: dict) -> dict:
    """Track actual reviewed image pairs by source, pool stage and Top10 crossing.

    Args:
        rankings: Complete frozen rankings from label-free replay.
        jobs: Label-free expansion jobs keyed by cohort and image query.
        reviewed: Existing targeted and incomplete-random human review snapshot.

    Returns:
        Observed movements and confirmed grades, separate from AI metrics.
    """
    output = {}
    for design, group in reviewed.items():
        rows = []
        for record in group["records"]:
            key, cid = record["query"], record["candidate_id"]
            if key not in jobs:
                continue
            positions = {m: rankings[key][m].index(cid) + 1 for m in METHODS}
            rows.append(
                dict(
                    query=key,
                    candidate_id=cid,
                    grade=record["human_grade"],
                    base_stage=(
                        "top10"
                        if positions[TITLE] <= 10
                        else "scored_11_to_30" if positions[TITLE] <= 30 else "unscored"
                    ),
                    newly_scored=cid in jobs[key]["extra"],
                    ranks=positions,
                )
            )
        output[design] = dict(
            reviewed=len(rows),
            confirmed_relevant_by_base_stage=dict(
                Counter(row["base_stage"] for row in rows if row["grade"] == "1")
            ),
            confirmed_top10={
                method: {
                    grade: sum(
                        row["grade"] == grade and row["ranks"][method] <= 10
                        for row in rows
                    )
                    for grade in ("0", "1")
                }
                for method in METHODS
            },
            records=rows,
        )
    return output


def boundary_changes(rankings: dict, jobs: dict, grades: dict, reviewed: dict) -> dict:
    """Record each actual entrant and exit, with available human and AI evidence.

    Args:
        rankings: Label-free full rankings.
        jobs: Image queries and their added candidates.
        grades: Full-gallery AI labels keyed by query and candidate.
        reviewed: Human movement diagnostic split by review design.

    Returns:
        Per-method lists of changed Top10 memberships on image queries.
    """
    human: dict[tuple[str, str], str] = {}
    for group in reviewed.values():
        for row in group["records"]:
            key = (row["query"], row["candidate_id"])
            if key in human and human[key] != row["grade"]:
                raise ValueError("Existing human judgments disagree on the same pair")
            human[key] = row["grade"]
    changes: dict[str, list[dict]] = {method: [] for method in METHODS[1:]}
    for key, job in jobs.items():
        order = rankings[key]
        baseline = set(order[TITLE][:10])
        for method in changes:
            top = set(order[method][:10])
            if top == baseline:
                continue
            moves = {}
            for name, diff in (("entered", top - baseline), ("exited", baseline - top)):
                moves[name] = [
                    dict(
                        candidate_id=cid,
                        ai_grade=grades[key, cid],
                        human_grade=human.get((key, cid)),
                        newly_scored=cid in job["extra"],
                        baseline_rank=order[TITLE].index(cid) + 1,
                        new_rank=order[method].index(cid) + 1,
                    )
                    for cid in sorted(diff)
                ]
            changes[method].append(dict(query=key, **moves))
    return changes


def main() -> None:
    """Verify previous evidence, score receipts and full-gallery development results."""
    registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
    protocol_path = ROOT / "sid_protocol.json"
    source_runtime = EXPANDED / "runtime.json"
    runtime_path = ROOT / "runtime.json"
    prior = POSTER / "results.json"
    receipt = POSTER / "summary.json"
    human_path = TARGET / "subject_challengers/results.json"
    paths = [protocol_path, prior, receipt, human_path, source_runtime, runtime_path]
    require_data_access(paths, "development", registry)
    protocol = load(protocol_path)
    expected = {
        **load(prior)["inputs"],
        **protocol["files"],
        **{
            str((TARGET / p).relative_to(PROJECT_ROOT)): h
            for p, h in load(receipt)["files"].items()
        },
    }
    jobs = {f"{job['cohort']}/{job['query']['id']}": job for job in protocol["jobs"]}
    cache_paths = [
        ROOT / "sid_pairs" / job["cohort"] / f"{job['query']['id']}.json"
        for job in jobs.values()
    ]
    paths.extend([*(PROJECT_ROOT / name for name in expected), *cache_paths])
    sources = {
        SWIFT,
        WORKER,
        PROJECT_ROOT / "scripts/module2_poster_text_replay.py",
        PROJECT_ROOT / "scripts/module2_recall_boundary.py",
    }
    require_data_access(
        [path for path in paths if path.resolve() not in sources],
        "development",
        registry,
    )
    for name, digest in expected.items():
        if sha256_file(PROJECT_ROOT / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    old_runtime, runtime = load(source_runtime), load(runtime_path)
    if any(runtime[key] != old_runtime[key] for key in ("model", "score")):
        raise ValueError("New and previous Qwen scores use different scoring settings")
    if len(jobs) != 24 or sum(len(job["extra"]) for job in jobs.values()) != 72:
        raise ValueError(
            "The image and new-pair counts differ from the frozen experiment"
        )
    rankings = replay(protocol)
    labels = []
    for cohort in COHORTS:
        label_path = DATA / cohort / "judgments.ai.jsonl"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            labels.append(dict(row, query_id=f"{cohort}/{row['query_id']}"))
    grades = {(row["query_id"], row["candidate_id"]): row["grade"] for row in labels}
    modes = {key: "image" if key in jobs else "text" for key in rankings}
    metrics = evaluate_intervals(rankings, labels, modes)
    paired = paired_querywise(
        rankings,
        labels,
        modes,
        [(method, TITLE) for method in METHODS[1:]]
        + [(COMBINED, CONSENSUS), (COMBINED, EXPANSION)],
    )
    human = human_movements(rankings, jobs, load(human_path)["human_review_diagnostic"])
    changes = boundary_changes(rankings, jobs, grades, human)
    additions = Counter(
        str(grades[key, cid]) for key, job in jobs.items() for cid in job["extra"]
    )
    write_json(ROOT / "rankings.json", rankings)
    write_json(
        ROOT / "results.json",
        dict(
            role="exposed_development",
            scored_additions_by_ai_grade=dict(additions),
            added_pairs=protocol["added_pairs"],
            original_pairs=30 * len(jobs),
            AI_development=metrics,
            paired_AI_development=paired,
            human_review_diagnostic=human,
            top10_membership_changes=changes,
            formal_metrics=None,
            frozen_acceptance_model_calls=0,
            inputs={
                str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
                for path in sorted(set(paths))
            },
            evaluation_source_sha256=sha256_file(Path(__file__)),
        ),
    )
    print(
        f"Verified {len(rankings)} full rankings and {protocol['added_pairs']} new pairs."
    )


if __name__ == "__main__":
    main()
