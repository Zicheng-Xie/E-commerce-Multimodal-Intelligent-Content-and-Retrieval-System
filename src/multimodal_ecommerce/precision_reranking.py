"""Fixed top-pool precision refinement and group-disjoint hard-negative RankNet."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from multimodal_ecommerce.retrieval_strategies import ordered_indices

FEATURE_NAMES = [
    "global_cosine",
    "region_attention",
    "reciprocal_fusion",
    "color_similarity",
    "base_rank",
]


def color_descriptor(path: Path) -> np.ndarray:
    """Center-80% HSV Hellinger histogram, using pixels rather than source ROI.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Computed array in the order of the supplied records.
    """
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        width, height = image.size
        image = image.crop(
            (int(width * 0.1), int(height * 0.1), int(width * 0.9), int(height * 0.9))
        )
        pixels = np.asarray(
            image.resize((64, 64), Image.Resampling.LANCZOS).convert("HSV")
        ).reshape(-1, 3)
    hist, _ = np.histogramdd(pixels, bins=(12, 4, 4), range=((0, 256),) * 3)
    values = np.sqrt(hist.ravel() / hist.sum())
    return values


def refine_pool(
    base: list[str],
    candidate_ids: list[str],
    scores: np.ndarray,
    *,
    blend: float = 1.0,
    anchors: int = 0,
) -> list[str]:
    """Keep complete tail and optional first anchors; never shrink P@10 to P@k.

    Args:
        base: Base used by this operation.
        candidate_ids: Candidate ids used by this operation.
        scores: Scores used by this operation.
        blend: Blend used by this operation.
        anchors: Anchors used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Candidate pool must be a unique baseline prefix; Invalid blend or
            anchor count.
    """
    if len(base) != len(set(base)) or candidate_ids != base[: len(candidate_ids)]:
        raise ValueError("Candidate pool must be a unique baseline prefix")
    if not 0 <= blend <= 1 or not 0 <= anchors <= len(candidate_ids):
        raise ValueError("Invalid blend or anchor count")
    order = ordered_indices(scores, candidate_ids)
    n = len(candidate_ids)
    rank_score = np.zeros(n)
    rank_score[order] = np.linspace(1.0, 0.0, n)
    combined = (1 - blend) * np.linspace(1.0, 0.0, n) + blend * rank_score
    remaining = np.arange(anchors, n)
    order = ordered_indices(combined[remaining], candidate_ids[anchors:])
    front = candidate_ids[:anchors] + [candidate_ids[remaining[i]] for i in order]
    return front + base[n:]


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Return sigmoid for this workflow.

    Args:
        values: Values used by this operation.

    Returns:
        Computed array in the order of the supplied records.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50, 50)))


def matrix(entry: dict, include_teacher: bool) -> np.ndarray:
    """Return matrix for this workflow.

    Args:
        entry: Entry used by this operation.
        include_teacher: Include teacher used by this operation.

    Returns:
        Computed array in the order of the supplied records.
    """
    base = np.asarray(entry["features"], dtype=float)
    if include_teacher:
        return np.column_stack([base, np.asarray(entry["teacher"], dtype=float)])
    return base


def fit_pairwise(
    entries: list[dict],
    policy: Mapping,
    *,
    excluded_groups: set[str],
    include_teacher: bool = False,
    distill: bool = False,
) -> dict:
    """Fit only training-query groups and nonvalidation gallery groups.

    Positives pair with the first eight baseline hard negatives per query. The
    teacher can soften training targets; no validation labels enter scaling,
    pair mining, gradients, or teacher target construction.

    Args:
        entries: Entries used by this operation.
        policy: Policy used by this operation.
        excluded_groups: Excluded groups used by this operation.
        include_teacher: Include teacher used by this operation.
        distill: Distill used by this operation.

    Returns:
        Mapping with fields: mean, scale, weights, feature_names, include_teacher,
        distill, training_query_groups, training_gallery_groups, excluded_groups,
        pair_count, loss.

    Raises:
        ValueError: No group-disjoint hard-negative training pairs.
    """
    usable = []
    for entry in entries:
        if entry["group"] in excluded_groups:
            continue
        allowed = np.array(
            [g not in excluded_groups for g in entry["candidate_groups"]]
        )
        labels = np.asarray(entry["labels"], dtype=bool)
        positive_indices = np.flatnonzero(allowed & labels)
        negative = np.flatnonzero(allowed & ~labels)[: policy["hard_negative_limit"]]
        if len(positive_indices) and len(negative):
            usable.append((entry, allowed, positive_indices, negative))
    if not usable:
        raise ValueError("No group-disjoint hard-negative training pairs")
    bank = np.concatenate(
        [matrix(e, include_teacher)[allowed] for e, allowed, _, _ in usable]
    )
    mean, scale = bank.mean(axis=0), bank.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    differences, targets, weights = [], [], []
    gallery_groups: set[str] = set()
    query_groups = set()
    for entry, allowed, positives, negatives in usable:
        query_groups.add(entry["group"])
        gallery_groups.update(
            g for g, keep in zip(entry["candidate_groups"], allowed) if keep
        )
        features = (matrix(entry, include_teacher) - mean) / scale
        teacher = np.asarray(entry["teacher"], dtype=float)
        for positive in positives:
            for negative in negatives:
                differences.append(features[positive] - features[negative])
                target = 1.0
                if distill:
                    soft = float(
                        sigmoid(
                            np.array(
                                [
                                    (teacher[positive] - teacher[negative])
                                    / policy["distillation_temperature"]
                                ]
                            )
                        )[0]
                    )
                    target = (1 - policy["distillation_mix"]) + policy[
                        "distillation_mix"
                    ] * soft
                targets.append(target)
                weights.append(1.0 / (len(usable) * len(positives) * len(negatives)))
    x, y, importance = np.asarray(differences), np.asarray(targets), np.asarray(weights)
    learned = np.zeros(x.shape[1])
    for _ in range(policy["pairwise_steps"]):
        gradient = (
            x.T @ ((sigmoid(x @ learned) - y) * importance)
            + policy["pairwise_l2"] * learned
        )
        learned -= policy["pairwise_lr"] * gradient
    assert not query_groups & excluded_groups
    assert not gallery_groups & excluded_groups
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": learned.tolist(),
        "feature_names": FEATURE_NAMES
        + (["cross_encoder_logit"] if include_teacher else []),
        "include_teacher": include_teacher,
        "distill": distill,
        "training_query_groups": sorted(query_groups),
        "training_gallery_groups": sorted(gallery_groups),
        "excluded_groups": sorted(excluded_groups),
        "pair_count": len(y),
        "loss": float(
            np.sum((np.logaddexp(0, x @ learned) - y * (x @ learned)) * importance)
            + policy["pairwise_l2"] * np.dot(learned, learned) / 2
        ),
    }


def pairwise_scores(entry: dict, model: dict) -> np.ndarray:
    """Return pairwise scores for this workflow.

    Args:
        entry: Entry used by this operation.
        model: Model used by this operation.

    Returns:
        Computed array in the order of the supplied records.
    """
    values = matrix(entry, model["include_teacher"])
    return (
        (values - np.asarray(model["mean"])) / np.asarray(model["scale"])
    ) @ np.asarray(model["weights"])


def pareto_gate(
    summary: dict, baseline: dict, min_precision_gain: float = 0.02
) -> dict:
    """Return pareto gate for this workflow.

    Args:
        summary: Summary used by this operation.
        baseline: Baseline used by this operation.
        min_precision_gain: Min precision gain used by this operation.

    Returns:
        Mapping with fields: passes_development_metric_gate, precision_at_10_delta,
        regressed_metrics, formal_acceptance, independent_evaluation_required.
    """
    keys = ("recall@10", "ap", "ndcg@10", "mrr", "hit@1")
    regressions = [key for key in keys if summary[key] < baseline[key] - 1e-12]
    gain = summary["precision@10"] - baseline["precision@10"]
    return {
        "passes_development_metric_gate": not regressions
        and gain >= min_precision_gain - 1e-12,
        "precision_at_10_delta": gain,
        "regressed_metrics": regressions,
        "formal_acceptance": False,
        "independent_evaluation_required": True,
    }
