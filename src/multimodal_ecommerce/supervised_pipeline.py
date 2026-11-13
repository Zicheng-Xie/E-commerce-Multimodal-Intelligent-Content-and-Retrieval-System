"""Group-masked refitting with explicit supervision origin and pending-label states."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from multimodal_ecommerce.semantic_ranker import (
    SemanticRankerFitRankerInputs,
    SemanticRankerTrainingMaskInputs,
    fit_ranker,
    predict_ranker,
    training_mask,
)

MODEL_NAMES = ("balanced_logistic", "hard_negative_ranknet")
LABEL_ORIGINS = {
    "machine_pixel_refined_development",
    "declared_human_business_development",
}


@dataclass(frozen=True)
class SupervisedPipelineFitSupervisedBundleInputs:
    """Input values for fit supervised bundle.

    Attributes:
        features: Feature array aligned with the candidate or query records.
        labels: Judgments used for the stated development or evaluation protocol.
        valid: Valid used by this operation.
        query_folds: Query folds used by this operation.
        gallery_folds: Gallery folds used by this operation.
        base_scores: Base scores used by this operation.
        metadata: Metadata used by this operation.
        policy: Policy used by this operation.
        label_origin: Label origin used by this operation.
    """

    features: np.ndarray
    labels: np.ndarray
    valid: np.ndarray
    query_folds: np.ndarray
    gallery_folds: np.ndarray
    base_scores: np.ndarray
    metadata: dict
    policy: dict
    label_origin: str


def fit_supervised_bundle(options: SupervisedPipelineFitSupervisedBundleInputs) -> dict:
    """Fit supervised bundle.

    Args:
        options: Validated input fields described by
            SupervisedPipelineFitSupervisedBundleInputs.

    Returns:
        Mapping with fields: trained, state, known_labels, failed_fold, reason,
        models, audits, predictions, candidates, formal_acceptance,
        human_quality_validated.

    Raises:
        ValueError: Unknown supervision origin; Training labels must be binary or
            unknown; Incomplete cross-fitted prediction coverage.
    """
    features = options.features
    labels = options.labels
    valid = options.valid
    query_folds = options.query_folds
    gallery_folds = options.gallery_folds
    base_scores = options.base_scores
    metadata = options.metadata
    policy = options.policy
    label_origin = options.label_origin
    if label_origin not in LABEL_ORIGINS:
        raise ValueError("Unknown supervision origin")
    known = valid & np.isfinite(labels)
    if not known.any():
        return dict(trained=False, state="awaiting_real_labels", known_labels=0)
    if not np.isin(labels[known], [0, 1]).all():
        raise ValueError("Training labels must be binary or unknown")
    models = {}
    audits = []
    predictions = {name: np.full(valid.shape, np.nan) for name in MODEL_NAMES}
    for qfold in range(3):
        for gfold in range(3):
            mask, evidence = training_mask(
                SemanticRankerTrainingMaskInputs(
                    metadata["query_ids"],
                    query_folds,
                    metadata["query_families"],
                    metadata["gallery_families"],
                    gallery_folds,
                    qfold,
                    gfold,
                )
            )
            mask &= valid
            validation = (
                (query_folds[:, None] == qfold)
                & (gallery_folds[None, :] == gfold)
                & valid
            )
            for name in MODEL_NAMES:
                try:
                    model = fit_ranker(
                        SemanticRankerFitRankerInputs(
                            features,
                            labels,
                            mask,
                            base_scores,
                            policy,
                            pairwise=name == "hard_negative_ranknet",
                        )
                    )
                except ValueError as error:
                    logging.getLogger(__name__).debug(
                        "Handled exception in fit_supervised_bundle", exc_info=True
                    )
                    return dict(
                        trained=False,
                        state="awaiting_more_group_diverse_labels",
                        known_labels=int(known.sum()),
                        failed_fold=dict(query=qfold, gallery=gfold),
                        reason=str(error),
                    )
                model["label_origin"] = label_origin
                values = predict_ranker(features, model)
                predictions[name][validation] = values[validation]
                key = f"{name}_q{qfold}_g{gfold}"
                models[key] = model
                audits.append(
                    dict(
                        model=key,
                        **evidence,
                        known_training_cells=model["training_cells"],
                        validation_cells=int(validation.sum()),
                        label_origin=label_origin,
                    )
                )
    if any(not np.isfinite(scores[valid]).all() for scores in predictions.values()):
        raise ValueError("Incomplete cross-fitted prediction coverage")
    candidates = {}
    for name in MODEL_NAMES:
        model = fit_ranker(
            SemanticRankerFitRankerInputs(
                features,
                labels,
                valid,
                base_scores,
                policy,
                pairwise=name == "hard_negative_ranknet",
            )
        )
        model["label_origin"] = label_origin
        candidates[name] = model
    return dict(
        trained=True,
        models=models,
        audits=audits,
        predictions=predictions,
        candidates=candidates,
        known_labels=int(known.sum()),
        formal_acceptance=False,
        human_quality_validated=False,
    )
