"""Additional supervised fusion over complete frozen model signals."""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from multimodal_ecommerce.semantic_ranker import FEATURE_NAMES as PUBLIC_FEATURES

FEATURE_NAMES = list(PUBLIC_FEATURES) + [
    "old_whole_CE_rank",
    "new_whole_CE_rank",
    "actual_crop_CE_rank",
    "old_full_gallery_base_rank",
    "new_full_gallery_base_rank",
    "whole_FG_fusion_rank",
    "ROI_FG_fusion_rank",
]


class TrainingPolicy(TypedDict):
    """Frozen optimizer, representation and provenance fields for this model."""

    query_folds: int
    gallery_family_folds: int
    seed: int
    l2: float
    steps: int
    learning_rate: float
    hard_negatives: int
    label_origin: str
    null_labels: str
    calibrated_output: str
    auxiliary_to_full_models: bool
    full_neural_scores_replaced_or_imputed: bool


POLICY: TrainingPolicy = dict(
    query_folds=3,
    gallery_family_folds=3,
    seed=2026102205,
    l2=0.05,
    steps=400,
    learning_rate=0.1,
    hard_negatives=8,
    label_origin="frozen_native_AI_business_development_not_human",
    null_labels="excluded from scalers, gradients and hard-negative mining",
    calibrated_output="training-only location/scale; not business probability",
    auxiliary_to_full_models=True,
    full_neural_scores_replaced_or_imputed=False,
)


def fit(
    features: np.ndarray,
    labels: np.ndarray,
    allowed: np.ndarray,
    base: np.ndarray,
    *,
    pairwise: bool,
) -> dict:
    """Return fit for this workflow.

    Args:
        features: Feature array aligned with the candidate or query records.
        labels: Judgments used for the stated development or evaluation protocol.
        allowed: Allowed used by this operation.
        base: Base used by this operation.
        pairwise: Pairwise used by this operation.

    Returns:
        Mapping with fields: feature_names, mean, scale, weights, pairwise,
        score_location, score_scale, training_cells, positive, negative,
        optimization_rows, loss, policy, human_labels, formal_metrics.

    Raises:
        ValueError: Feature configuration mismatch; Both classes required within the
            allowed training cells; Missing real full-model training signals; No
            group-diverse supervised training pairs.
    """
    known = allowed & np.isfinite(labels)
    if features.shape[-1] != len(FEATURE_NAMES):
        raise ValueError("Feature configuration mismatch")
    if not known.any() or set(np.unique(labels[known])) != {0.0, 1.0}:
        raise ValueError("Both classes required within the allowed training cells")
    if not np.isfinite(features[known]).all():
        raise ValueError("Missing real full-model training signals")
    mean, scale = features[known].mean(0), features[known].std(0)
    scale = np.where(scale < 1e-6, 1, scale)
    standardized = (features - mean) / scale
    xs, ys, weights = [], [], []
    for q in range(len(labels)):
        pos = np.flatnonzero(known[q] & (labels[q] == 1))
        neg = np.flatnonzero(known[q] & (labels[q] == 0))
        if pairwise:
            if not len(pos) or not len(neg):
                continue
            neg = neg[
                np.argsort(-base[q, neg], kind="stable")[: POLICY["hard_negatives"]]
            ]
            for p in pos:
                xs.extend(standardized[q, p] - standardized[q, neg])
                ys.extend([1] * len(neg))
                weights.extend([1 / (len(pos) * len(neg))] * len(neg))
        else:
            classes = [(idx, grade) for idx, grade in ((neg, 0), (pos, 1)) if len(idx)]
            for idx, grade in classes:
                xs.extend(standardized[q, idx])
                ys.extend([grade] * len(idx))
                weights.extend([1 / (len(classes) * len(idx))] * len(idx))
    if not xs:
        raise ValueError("No group-diverse supervised training pairs")
    x, y, sample = np.asarray(xs), np.asarray(ys), np.asarray(weights)
    if not pairwise:
        x = np.column_stack([x, np.ones(len(x))])
    sample /= sample.sum()
    w = np.zeros(x.shape[-1])
    penalty = np.ones_like(w)
    if not pairwise:
        penalty[-1] = 0
    for _ in range(POLICY["steps"]):
        probability = 1 / (1 + np.exp(-np.clip(x @ w, -40, 40)))
        gradient = x.T @ ((probability - y) * sample) + POLICY["l2"] * w * penalty
        w -= POLICY["learning_rate"] * gradient
    raw = (
        standardized[known] @ w
        if pairwise
        else np.column_stack([standardized[known], np.ones(known.sum())]) @ w
    )
    location, spread = float(raw.mean()), float(raw.std())
    return dict(
        feature_names=FEATURE_NAMES,
        mean=mean.tolist(),
        scale=scale.tolist(),
        weights=w.tolist(),
        pairwise=pairwise,
        score_location=location,
        score_scale=spread if spread > 1e-6 else 1.0,
        training_cells=int(known.sum()),
        positive=int((labels[known] == 1).sum()),
        negative=int((labels[known] == 0).sum()),
        optimization_rows=len(x),
        loss=float(
            np.sum((np.logaddexp(0, x @ w) - y * (x @ w)) * sample)
            + POLICY["l2"] * np.sum(w * w * penalty) / 2
        ),
        policy=POLICY,
        human_labels=0,
        formal_metrics=None,
    )


def predict(features: np.ndarray, model: dict) -> np.ndarray:
    """Return predict for this workflow.

    Args:
        features: Feature array aligned with the candidate or query records.
        model: Model used by this operation.

    Returns:
        Computed array in the order of the supplied records.

    Raises:
        ValueError: Feature version changed.
    """
    if model["feature_names"] != FEATURE_NAMES:
        raise ValueError("Feature version changed")
    values = (features - model["mean"]) / model["scale"]
    if not model["pairwise"]:
        values = np.concatenate([values, np.ones((*values.shape[:-1], 1))], axis=-1)
    return (values @ model["weights"] - model["score_location"]) / model["score_scale"]
