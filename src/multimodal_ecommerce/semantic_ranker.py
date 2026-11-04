"""Public-feature semantic fusion and group-masked training on machine weak labels."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import KW_ONLY, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from multimodal_ecommerce.business_retrieval import lexical_scores
from multimodal_ecommerce.model_interfaces import FeatureIndex
from multimodal_ecommerce.precision_reranking import color_descriptor
from multimodal_ecommerce.regional_retrieval import RegionalSearchRequest
from multimodal_ecommerce.retrieval_strategies import ordered_indices

FEATURE_NAMES = [
    "global_cosine",
    "attention_cosine",
    "best_region_cosine",
    "title_cosine",
    "semantic_profile_cosine",
    "title_literal",
    "profile_literal",
    "profile_title_agreement",
    "color_similarity",
    "image_query",
    "image_global",
    "image_title",
    "image_profile",
]


@dataclass(frozen=True)
class SemanticRankerFeatureMatrixInputs:
    """Input values for feature matrix.

    Attributes:
        index: Index used by this operation.
        title_bank: Title bank used by this operation.
        profile_bank: Profile bank used by this operation.
        profile_texts: Profile texts used by this operation.
        query: Query record or representation used by this operation.
        text: Text input used by the operation.
        image: Image input used by the operation.
        color_bank: Color bank used by this operation.
    """

    index: FeatureIndex
    title_bank: np.ndarray
    profile_bank: np.ndarray
    profile_texts: list[str]
    query: np.ndarray
    _: KW_ONLY
    text: str | None = None
    image: Path | None = None
    color_bank: np.ndarray | None = None


def feature_matrix(options: SemanticRankerFeatureMatrixInputs) -> np.ndarray:
    """Return feature matrix for this workflow.

    Args:
        options: Validated input fields described by SemanticRankerFeatureMatrixInputs.

    Returns:
        Computed array in the order of the supplied records.

    Raises:
        ValueError: Invalid query vector; Image query needs cached public pixel
            histograms; Malformed semantic public features.
    """
    index = options.index
    title_bank = options.title_bank
    profile_bank = options.profile_bank
    profile_texts = options.profile_texts
    query = options.query
    text = options.text
    image = options.image
    color_bank = options.color_bank
    q = np.asarray(query, dtype=float)
    if q.shape != (index.vectors.shape[-1],) or not np.isfinite(q).all():
        raise ValueError("Invalid query vector")
    cosine = np.einsum("nvd,d->nv", index.vectors, q)
    maximum = np.where(index.mask, cosine, -np.inf).max(axis=1)
    attended = {
        r["id"]: r["score"]
        for r in index.search(
            RegionalSearchRequest(q, top_k=len(index.records), strategy="attention")
        )
    }
    attention = np.array([attended[r["id"]] for r in index.records])
    title = title_bank @ q
    profile = profile_bank @ q

    def literal(values: list[str]) -> np.ndarray:
        """Return literal for this workflow.

        Args:
            values: Values used by this operation.

        Returns:
            Computed array in the order of the supplied records.
        """
        result = lexical_scores(text, values) if text else np.zeros(len(index.records))
        return result / max(1.0, float(result.max()))

    if image is not None:
        if color_bank is None:
            raise ValueError("Image query needs cached public pixel histograms")
        colors = color_bank @ color_descriptor(image)
    else:
        colors = np.zeros(len(index.records))
    mode = float(image is not None)
    features = np.column_stack(
        [
            cosine[:, 0],
            attention,
            maximum,
            title,
            profile,
            literal([r["title"] for r in index.records]),
            literal(profile_texts),
            np.einsum("nd,nd->n", title_bank, profile_bank),
            colors,
            np.full(len(index.records), mode),
            mode * cosine[:, 0],
            mode * title,
            mode * profile,
        ]
    )
    if (
        features.shape != (len(index.records), len(FEATURE_NAMES))
        or not np.isfinite(features).all()
    ):
        raise ValueError("Malformed semantic public features")
    return features


def assign_folds(groups: list[str], folds: int, seed: int) -> dict[str, int]:
    """Return assign folds for this workflow.

    Args:
        groups: Groups used by this operation.
        folds: Folds used by this operation.
        seed: Seed used to reproduce randomized ordering.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    ordered = sorted(
        set(groups), key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest()
    )
    return {group: i % folds for i, group in enumerate(ordered)}


@dataclass(frozen=True)
class SemanticRankerTrainingMaskInputs:
    """Input values for training mask.

    Attributes:
        query_ids: Query ids used by this operation.
        query_folds: Query folds used by this operation.
        query_families: Query families used by this operation.
        gallery_families: Gallery families used by this operation.
        gallery_folds: Gallery folds used by this operation.
        held_query_fold: Held query fold used by this operation.
        held_gallery_fold: Held gallery fold used by this operation.
    """

    query_ids: list[str]
    query_folds: np.ndarray
    query_families: list[str | None]
    gallery_families: list[str]
    gallery_folds: np.ndarray
    held_query_fold: int
    held_gallery_fold: int


def training_mask(options: SemanticRankerTrainingMaskInputs) -> tuple[np.ndarray, dict]:
    """Return training mask for this workflow.

    Args:
        options: Validated input fields described by SemanticRankerTrainingMaskInputs.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Cross-fit source-family leakage.
    """
    query_ids = options.query_ids
    query_folds = options.query_folds
    query_families = options.query_families
    gallery_families = options.gallery_families
    gallery_folds = options.gallery_folds
    held_query_fold = options.held_query_fold
    held_gallery_fold = options.held_gallery_fold
    excluded_families = {
        family
        for family, fold in zip(gallery_families, gallery_folds)
        if fold == held_gallery_fold
    }
    excluded_families.update(
        family
        for family, fold in zip(query_families, query_folds)
        if fold == held_query_fold and family is not None
    )
    q_allowed = np.array(
        [
            fold != held_query_fold and family not in excluded_families
            for family, fold in zip(query_families, query_folds)
        ]
    )
    g_allowed = np.array(
        [family not in excluded_families for family in gallery_families]
    )
    mask = q_allowed[:, None] & g_allowed[None, :]
    evidence: dict = dict(
        training_query_ids=[qid for qid, keep in zip(query_ids, q_allowed) if keep],
        excluded_query_ids=[
            qid for qid, fold in zip(query_ids, query_folds) if fold == held_query_fold
        ],
        training_gallery_families=sorted(
            {f for f, keep in zip(gallery_families, g_allowed) if keep}
        ),
        excluded_gallery_families=sorted(excluded_families),
        held_query_fold=held_query_fold,
        held_gallery_fold=held_gallery_fold,
    )
    if (
        set(evidence["training_query_ids"]) & set(evidence["excluded_query_ids"])
        or set(evidence["training_gallery_families"]) & excluded_families
    ):
        raise ValueError("Cross-fit source-family leakage")
    return mask, evidence


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Return sigmoid for this workflow.

    Args:
        x: X used by this operation.

    Returns:
        Computed array in the order of the supplied records.
    """
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


@dataclass(frozen=True)
class SemanticRankerFitRankerInputs:
    """Input values for fit ranker.

    Attributes:
        features: Feature array aligned with the candidate or query records.
        labels: Judgments used for the stated development or evaluation protocol.
        allowed: Allowed used by this operation.
        base_scores: Base scores used by this operation.
        policy: Policy used by this operation.
        pairwise: Pairwise used by this operation.
    """

    features: np.ndarray
    labels: np.ndarray
    allowed: np.ndarray
    base_scores: np.ndarray
    policy: dict
    _: KW_ONLY
    pairwise: bool


def fit_ranker(options: SemanticRankerFitRankerInputs) -> dict:
    """Ignore every unknown; fit scalers, mining and gradients on training cells
    only.

    Args:
        options: Validated input fields described by SemanticRankerFitRankerInputs.

    Returns:
        Mapping with fields: mean, scale, weights, bias, pairwise, feature_names,
        training_cells, training_positive, training_negative, training_pairs, loss,
        label_origin, formal_acceptance, human_quality_validated.

    Raises:
        ValueError: Need both classes of known machine labels in training only; No
            positive/negative machine training pairs.
    """
    features = options.features
    labels = options.labels
    allowed = options.allowed
    base_scores = options.base_scores
    policy = options.policy
    pairwise = options.pairwise
    known = allowed & np.isfinite(labels)
    if not known.any() or len(np.unique(labels[known])) != 2:
        raise ValueError("Need both classes of known machine labels in training only")
    values = features[known]
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    standardized = (features - mean) / scale
    if pairwise:
        differences = []
        importance = []
        for q in range(len(features)):
            pos = np.flatnonzero(known[q] & (labels[q] == 1))
            neg = np.flatnonzero(known[q] & (labels[q] == 0))
            if not len(pos) or not len(neg):
                continue
            neg = neg[np.argsort(-base_scores[q, neg], kind="stable")[:8]]
            for p in pos:
                differences.extend(standardized[q, p] - standardized[q, neg])
                importance.extend([1 / (len(pos) * len(neg))] * len(neg))
        if not differences:
            raise ValueError("No positive/negative machine training pairs")
        x = np.asarray(differences)
        y = np.ones(len(x))
        sample = np.asarray(importance)
        bias = False
    else:
        x = standardized[known]
        y = labels[known]
        sample = np.zeros(labels.shape)
        for q in range(len(features)):
            counts = Counter(labels[q, known[q]])
            for grade, count in counts.items():
                sample[q, known[q] & (labels[q] == grade)] = 1 / (len(counts) * count)
        sample = sample[known]
        x = np.column_stack([x, np.ones(len(x))])
        bias = True
    sample /= sample.sum()
    weights = np.zeros(x.shape[-1])
    penalty = np.ones_like(weights)
    if bias:
        penalty[-1] = 0
    for _ in range(policy["steps"]):
        gradient = (
            x.T @ ((sigmoid(x @ weights) - y) * sample)
            + policy["l2"] * weights * penalty
        )
        weights -= policy["learning_rate"] * gradient
    return dict(
        mean=mean.tolist(),
        scale=scale.tolist(),
        weights=weights.tolist(),
        bias=bias,
        pairwise=pairwise,
        feature_names=FEATURE_NAMES,
        training_cells=int(known.sum()),
        training_positive=int((labels[known] == 1).sum()),
        training_negative=int((labels[known] == 0).sum()),
        training_pairs=len(x) if pairwise else None,
        loss=float(
            np.sum((np.logaddexp(0, x @ weights) - y * (x @ weights)) * sample)
            + policy["l2"] * np.sum(weights**2 * penalty) / 2
        ),
        label_origin="machine_weak_development_only",
        formal_acceptance=False,
        human_quality_validated=False,
    )


def predict_ranker(features: np.ndarray, model: dict) -> np.ndarray:
    """Predict ranker.

    Args:
        features: Feature array aligned with the candidate or query records.
        model: Model used by this operation.

    Returns:
        Computed array in the order of the supplied records.

    Raises:
        ValueError: Feature version mismatch.
    """
    if model["feature_names"] != FEATURE_NAMES:
        raise ValueError("Feature version mismatch")
    values = (features - np.asarray(model["mean"])) / np.asarray(model["scale"])
    if model["bias"]:
        values = np.concatenate([values, np.ones((*values.shape[:-1], 1))], axis=-1)
    raw = values @ np.asarray(model["weights"])
    return sigmoid(raw) if model["bias"] else raw


def machine_topk_bounds(rankings: dict, labels: dict, k: int = 10) -> dict:
    """Report teacher-agreement bounds, never human precision or acceptance scores.

    Args:
        rankings: Candidate rankings keyed by query and retrieval method.
        labels: Judgments used for the stated development or evaluation protocol.
        k: K used by this operation.

    Returns:
        Mapping with fields: top10_machine_positive_lower,
        top10_machine_positive_upper, top10_machine_known_relation_fraction, queries, k,
        human_quality_metrics, formal_metrics, formal_acceptance,
        not_a_measurement_of_real_business_precision.

    Raises:
        ValueError: Unchanged unique gallery and fixed K required.
    """
    known_positive = unknown = total = 0
    for query, ranked in rankings.items():
        if len(ranked) < k or len(ranked) != len(set(ranked)):
            raise ValueError("Unchanged unique gallery and fixed K required")
        total += k
        for cid in ranked[:k]:
            grade = labels[(query, cid)]
            known_positive += grade == 1
            unknown += grade is None
    return dict(
        top10_machine_positive_lower=known_positive / total,
        top10_machine_positive_upper=(known_positive + unknown) / total,
        top10_machine_known_relation_fraction=1 - unknown / total,
        queries=len(rankings),
        k=k,
        human_quality_metrics=None,
        formal_metrics=None,
        formal_acceptance=False,
        not_a_measurement_of_real_business_precision=True,
    )


def ranked_ids(ids: list[str], scores: npt.ArrayLike) -> list[str]:
    """Return ranked ids for this workflow.

    Args:
        ids: Identifiers in the order expected by the associated arrays.
        scores: Scores used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return [ids[i] for i in ordered_indices(scores, ids)]
