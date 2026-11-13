"""Human-supervised pairwise calibration using public retrieval observations only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multimodal_ecommerce.business_retrieval import lexical_terms
from multimodal_ecommerce.intent_retrieval import attributes
from multimodal_ecommerce.public_conflict_guard import (
    COMPATIBLE,
    reorder_public_conflicts,
)

FEATURES = (
    "cross_z",
    "cross_percentile",
    "visual_log_rank",
    "selected_log_rank",
    "coarse_top10",
    "protected_top2",
    "type_match",
    "type_conflict",
    "explicit_conflict",
    "attribute_conflicts",
    "attribute_supported",
    "lexical_overlap",
    "text_cross_z",
    "text_type_conflict",
    "text_attribute_conflict",
)


@dataclass
class CalibrationContext:
    """Public candidate observations; annotations and arbitrary IDs are not features.

    Attributes:
        order: Complete selected baseline ranking.
        original: Complete original coarse order.
        scores: Actual cross-encoder logits for exactly the current Top20.
        titles: Original public titles for the eligible gallery.
        text: Optional text query.
        caption: Optional query-only image caption.
    """

    order: list[str]
    original: list[str]
    scores: dict[str, float]
    titles: dict[str, str]
    text: str | None = None
    caption: dict | None = None


def public_features(context: CalibrationContext) -> tuple[np.ndarray, str]:
    """Extract ID-invariant features and a public semantic family for grouped validation.

    Args:
        context: Model observations and public inputs without annotation fields.

    Returns:
        Top20 feature matrix and query-family key derived only from public words.
    """
    head = context.order[:20]
    if (
        len(head) != 20
        or len(context.order) != len(set(context.order))
        or set(context.order) != set(context.original)
        or set(context.titles) != set(context.order)
        or set(context.scores) != set(head)
    ):
        raise ValueError(
            "Unique complete equal galleries and real scored Top20 required"
        )
    logits = np.array([context.scores[c] for c in head], dtype=np.float64)
    if not np.isfinite(logits).all():
        raise ValueError("Actual finite cross-encoder scores are required")
    z = np.clip((logits - logits.mean()) / max(float(logits.std()), 1e-6), -5, 5)
    percentile = np.array([np.mean(logits < v) for v in logits])
    _, trace = reorder_public_conflicts(
        context.order,
        context.titles,
        text=context.text,
        caption=context.caption,
        options=dict(protect=0, broad=False, matched_only=True),
    )
    evidence = trace["evidence"]
    kind = evidence[head[0]]["query_type"]
    family = next(
        ("compatible_" + "_".join(sorted(f)) for f in COMPATIBLE if kind in f),
        kind or "unresolved",
    )
    words = context.text or " ".join(
        str((context.caption or {}).get(k, "")) for k in ("product", "appearance")
    )
    terms, original = lexical_terms(words), {
        c: i + 1 for i, c in enumerate(context.original)
    }
    denominator = np.log1p(len(context.order))
    rows = []
    for i, cid in enumerate(head):
        obs, title = evidence[cid], context.titles[cid]
        attr = attributes(words, title)
        overlap = len(terms & lexical_terms(title)) / max(1, len(terms))
        match = bool(kind and obs["title_type"] and not obs["type_conflict"])
        text_mode = bool(context.text)
        rows.append(
            [
                z[i],
                percentile[i],
                -np.log1p(original[cid]) / denominator,
                -np.log1p(i + 1) / np.log(21),
                float(original[cid] <= 10),
                float(original[cid] <= 2),
                float(match),
                float(obs["type_conflict"]),
                float(obs["explicit_conflict"]),
                float(len(attr["conflicts"])),
                float(len(attr["supported"])),
                overlap,
                z[i] * text_mode,
                float(obs["type_conflict"] and text_mode),
                float(len(attr["conflicts"]) * text_mode),
            ]
        )
    return np.asarray(rows, dtype=np.float64), family


def fit_pairwise(batches: list[tuple[np.ndarray, list[int | None]]]) -> dict:
    """Fit a regularized logistic preference head from explicit human pairs only.

    Args:
        batches: Feature rows and human binary or unresolved grades, separated by query.

    Returns:
        Numeric weights and normalization; no query IDs, item IDs or answers retained.
    """
    if not batches:
        raise ValueError("Training queries are required")
    for matrix, grades in batches:
        if (
            matrix.shape != (len(grades), len(FEATURES))
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("Finite aligned training features required")
        if any(
            g is not None and (type(g) is not int or g not in (0, 1)) for g in grades
        ):
            raise ValueError("Only explicit human binary or unresolved grades allowed")
    training = np.concatenate([x for x, _ in batches])
    mean, scale = training.mean(axis=0), training.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    differences: list[np.ndarray] = []
    weights: list[float] = []
    for matrix, grades in batches:
        positives = [i for i, g in enumerate(grades) if g == 1]
        negatives = [i for i, g in enumerate(grades) if g == 0]
        n = len(positives) * len(negatives)
        if not n:
            continue
        x = (matrix - mean) / scale
        differences.extend(x[p] - x[n] for p in positives for n in negatives)
        weights.extend([1 / n] * n)
    if not differences:
        raise ValueError(
            "At least one query with human positives and negatives required"
        )
    design, mass = np.asarray(differences), np.asarray(weights)
    mass /= mass.sum()
    coefficients = np.zeros(len(FEATURES))
    penalty, iterations = 0.03, 0

    def objective(value: np.ndarray) -> float:
        """Compute the fixed regularized training objective.

        Args:
            value: Candidate preference coefficients.

        Returns:
            Query-balanced logistic loss plus L2 regularization.
        """
        return float(
            mass @ np.logaddexp(0, -(design @ value)) + penalty * (value @ value) / 2
        )

    initial = objective(coefficients)
    for iterations in range(1, 61):
        probability = 1 / (1 + np.exp(np.clip(design @ coefficients, -40, 40)))
        gradient = -(design.T @ (mass * probability)) + penalty * coefficients
        hessian = design.T @ (
            (mass * probability * (1 - probability))[:, None] * design
        )
        hessian += penalty * np.eye(len(FEATURES))
        step = np.linalg.solve(hessian, gradient)
        if np.linalg.norm(gradient) < 1e-8:
            break
        size, current = 1.0, objective(coefficients)
        while size > 1e-8 and objective(coefficients - size * step) > current:
            size /= 2
        coefficients -= size * step
    return dict(
        format="public_pairwise_calibration_v1",
        features=list(FEATURES),
        mean=mean.tolist(),
        scale=scale.tolist(),
        weights=coefficients.tolist(),
        l2=penalty,
        pairs=len(design),
        training_queries=len(batches),
        iterations=iterations,
        initial_loss=initial,
        final_loss=objective(coefficients),
        features_contain_annotation_or_ids=False,
    )


def preference_scores(matrix: np.ndarray, model: dict) -> np.ndarray:
    """Apply numeric parameters without accessing annotations or historical identities.

    Args:
        matrix: Public feature rows.
        model: Frozen numeric preference parameters.

    Returns:
        Finite, uncalibrated pairwise preference scores, not relevance probabilities.
    """
    if model["features"] != list(FEATURES):
        raise ValueError("Calibration feature contract changed")
    mean, scale, weights = (
        np.asarray(model[k], dtype=np.float64) for k in ("mean", "scale", "weights")
    )
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(FEATURES)
        or any(v.shape != (len(FEATURES),) for v in (mean, scale, weights))
        or not all(np.isfinite(v).all() for v in (matrix, mean, scale, weights))
        or not (scale > 0).all()
    ):
        raise ValueError("Finite aligned model parameters and positive scales required")
    return ((matrix - mean) / scale) @ weights


def calibrated_order(
    order: list[str], scores: np.ndarray, guarded: bool = True
) -> list[str]:
    """Compare one conservative position swap against an unrestricted Top20 control.

    Args:
        order: Complete baseline order.
        scores: Twenty numeric preference scores.
        guarded: Keep the first two and require a fixed log(4) preference margin for one swap.

    Returns:
        Complete unchanged candidate universe with a model-only ranking proposal.
    """
    if (
        len(order) < 20
        or len(set(order)) != len(order)
        or scores.shape != (20,)
        or not np.isfinite(scores).all()
    ):
        raise ValueError(
            "Unique full gallery and twenty finite preference scores required"
        )
    if not guarded:
        indices = sorted(range(20), key=lambda i: (-scores[i], i))
        return [order[i] for i in indices] + order[20:]
    outgoing = min(range(2, 10), key=lambda i: (scores[i], -i))
    incoming = max(range(10, 20), key=lambda i: (scores[i], -i))
    result = list(order)
    if scores[incoming] - scores[outgoing] >= np.log(4):
        result[outgoing], result[incoming] = result[incoming], result[outgoing]
    return result
