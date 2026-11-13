"""Small controlled retrieval experiment and honest human-label audit helpers."""

from __future__ import annotations

import hashlib
import math
from collections import Counter

import numpy as np


def stable_order(ids: list[str], scores: np.ndarray) -> list[str]:
    """Sort a full gallery without deleting ties or candidates.

    Args:
        ids: Unique candidate identities aligned with scores.
        scores: Finite one-dimensional similarities or logits.

    Returns:
        Complete ranking, breaking ties by candidate identity.

    Raises:
        ValueError: Candidate identities or real scores are incomplete.
    """
    if (
        len(set(ids)) != len(ids)
        or scores.shape != (len(ids),)
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Complete unique gallery and finite scores required")
    return [
        ids[i]
        for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))
    ]


def rerank_head(
    order: list[str], scores: dict[str, float], pool_size: int = 20
) -> list[str]:
    """Replace only the scored head and preserve the full original tail.

    Args:
        order: Complete cosine ranking.
        scores: One actual model score for every head candidate.
        pool_size: Fixed head size; twenty for the frozen experiment.

    Returns:
        Completely preserved candidate set with the head reranked.

    Raises:
        ValueError: Scores are missing, invented for other candidates or nonfinite.
    """
    head = order[:pool_size]
    if len(order) != len(set(order)) or set(scores) != set(head):
        raise ValueError("Exact complete Top20 model scores required")
    ranked = stable_order(head, np.array([scores[c] for c in head]))
    return ranked + order[len(head) :]


def sample_audit(
    rows: list[dict], fraction: float = 0.2, seed: int = 2026110501
) -> list[dict]:
    """Sample each query proportionally by AI grade with deterministic allocation.

    Args:
        rows: Complete development AI judgments, including nulls.
        fraction: Fraction sampled within every query.
        seed: Frozen sampling seed independent of retrieval outputs.

    Returns:
        Original rows sampled without using any model rank or score.

    Raises:
        ValueError: The fraction, provenance or relation identities are invalid.
    """
    if not 0 < fraction <= 1:
        raise ValueError("Sampling fraction must be in (0,1]")
    groups: dict = {}
    identities = set()
    for row in rows:
        key = row["query_id"], row["candidate_id"]
        if key in identities or row["human_label"] is not False:
            raise ValueError("Unique AI development relations required")
        identities.add(key)
        groups.setdefault(row["query_id"], []).append(row)
    result = []
    for query, group in sorted(groups.items()):
        target = math.ceil(len(group) * fraction)
        strata = {
            grade: [r for r in group if r["grade"] == grade] for grade in (0, 1, None)
        }
        quota = {g: len(values) * target / len(group) for g, values in strata.items()}
        counts = {g: int(v) for g, v in quota.items()}
        remaining = target - sum(counts.values())
        for grade in sorted(strata, key=lambda g: (-(quota[g] - counts[g]), str(g)))[
            :remaining
        ]:
            counts[grade] += 1
        for grade, values in strata.items():

            def priority(row: dict) -> str:
                """Return a rank-independent deterministic relation priority.

                Args:
                    row: One relation in this frozen sampling stratum.

                Returns:
                    SHA256 sampling key.
                """
                return hashlib.sha256(
                    f"{seed}/{query}/{row['candidate_id']}".encode()
                ).hexdigest()

            result.extend(sorted(values, key=priority)[: counts[grade]])
    return result


def human_accuracy(tasks: list[dict], answers: list[dict]) -> dict:
    """Score only declared human answers and keep AI abstentions separate.

    Args:
        tasks: Frozen sampled relations with task IDs and original AI grades.
        answers: Human CSV rows containing identity, decision and explanation.

    Returns:
        Completion, sample agreement, confusion counts and an 80% quality gate.
        No retrieval metric is multiplied by the agreement rate.

    Raises:
        ValueError: Answers are duplicated, undeclared as human or malformed.
    """
    expected = {r["task_id"]: r for r in tasks}
    seen = set()
    counts: Counter[str] = Counter()
    for answer in answers:
        key = answer["task_id"]
        if not answer.get("human_grade", "").strip():
            continue
        if key not in expected or key in seen:
            raise ValueError("Unknown or repeated human task")
        if (
            answer.get("reviewer_type") != "human"
            or not answer.get("reviewer", "").strip()
            or not answer.get("reviewed_at", "").strip()
        ):
            raise ValueError("Actual named human review and timestamp required")
        grade = answer["human_grade"]
        if grade not in {"0", "1", "uncertain"} or not answer.get("reason", "").strip():
            raise ValueError("Human decision and reasoning required")
        seen.add(key)
        original = expected[key]["grade"]
        if original is None:
            counts["AI_abstentions"] += 1
        elif grade == "uncertain":
            counts["human_uncertain_on_AI_binary"] += 1
        else:
            counts["binary_compared"] += 1
            counts["agreements"] += int(int(grade) == original)
            counts[f"AI_{original}_human_{grade}"] += 1
    complete = len(seen) == len(tasks) and bool(tasks)
    known = sum(r["grade"] is not None for r in tasks)
    lower = counts["agreements"] / known if complete and known else None
    upper = (
        (counts["agreements"] + counts["human_uncertain_on_AI_binary"]) / known
        if complete and known
        else None
    )
    return dict(
        expected=len(tasks),
        completed=len(seen),
        complete=complete,
        AI_known_sample=known,
        counts=dict(counts),
        binary_sample_agreement=(
            counts["agreements"] / counts["binary_compared"]
            if complete and counts["binary_compared"]
            else None
        ),
        sample_agreement_bounds=[lower, upper],
        annotation_quality=(
            "pending_human_review"
            if not complete
            else (
                "below_80"
                if upper is not None and upper < 0.8
                else (
                    "at_least_80_on_sample"
                    if lower is not None and lower >= 0.8
                    else "requires_adjudication"
                )
            )
        ),
        accuracy_is="sample agreement against human review, not guaranteed population truth",
        metric_correction="No automatic accuracy multiplier; expand human review if unreliable; AI-bound intervals omit AI labeling errors",
        formal_metrics=None,
    )
