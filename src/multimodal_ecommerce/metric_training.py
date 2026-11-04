"""Bounded diagonal metric learning over frozen real multimodal embeddings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

import numpy as np


class _MetricPolicySpec(TypedDict):
    """Exact fields of the METRIC_POLICY configuration and provenance record."""

    seed: int
    steps: int
    learning_rate: float
    temperature: float
    hard_negatives: int
    weight_min: float
    weight_max: float
    regularization: float
    checkpoints_selected_by_validation: bool
    encoder_trainable: bool
    methods: list[str]
    minimum_precision_gain: float


METRIC_POLICY: _MetricPolicySpec = {
    "seed": 2026102202,
    "steps": 300,
    "learning_rate": 0.03,
    "temperature": 0.05,
    "hard_negatives": 20,
    "weight_min": 0.5,
    "weight_max": 2.0,
    "regularization": 0.05,
    "checkpoints_selected_by_validation": False,
    "encoder_trainable": False,
    "methods": [
        "original_global",
        "learned_global",
        "fusion_metric25",
        "fusion_metric50",
        "fusion_metric50_anchor2",
    ],
    "minimum_precision_gain": 0.02,
}


def metric_transform(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply the same learned positive metric to query and document vectors.

    Args:
        vectors: Vectors used by this operation.
        weights: Weights used by this operation.

    Returns:
        Computed array in the order of the supplied records.

    Raises:
        ValueError: Metric dimension mismatch; Metric requires finite vectors and
            positive weights; Cannot transform a zero vector.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if vectors.ndim != 2 or weights.shape != (vectors.shape[1],):
        raise ValueError("Metric dimension mismatch")
    if (
        not np.isfinite(vectors).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise ValueError("Metric requires finite vectors and positive weights")
    result = vectors * np.sqrt(weights)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Cannot transform a zero vector")
    return result / norms


def train_metric(vectors: np.ndarray, records: list[dict], policy: Mapping) -> dict:
    """Mine negatives and optimize exclusively within declared training groups.

    Args:
        vectors: Vectors used by this operation.
        records: Records used by this operation.
        policy: Policy used by this operation.

    Returns:
        Mapping with fields: algorithm, policy, weights, loss_history,
        training_query_groups, training_gallery_groups, excluded_development_groups,
        training_queries, training_candidates, positive_relations, formal_acceptance,
        label_origin.

    Raises:
        ValueError: Record/feature alignment mismatch; A source pair group crosses
            train/development; Need at least three groups and positive/negative pairs
            per query; Nonfinite training embedding; Log-symmetric bounds required.
    """
    import torch
    import torch.nn.functional as functional

    if len(vectors) != len(records):
        raise ValueError("Record/feature alignment mismatch")
    training = [i for i, r in enumerate(records) if r["project_role"] == "training"]
    development = [r for r in records if r["project_role"] == "development_validation"]
    train_groups = {records[i]["pair_id"] for i in training}
    dev_groups = {r["pair_id"] for r in development}
    if train_groups & dev_groups:
        raise ValueError("A source pair group crosses train/development")
    queries = [i for i in training if records[i]["source"] == "user"]
    gallery = [i for i in training if records[i]["source"] == "shop"]
    labels = np.array(
        [
            [records[q]["identity"] == records[g]["identity"] for g in gallery]
            for q in queries
        ]
    )
    if len(queries) < 3 or not labels.any(axis=1).all() or labels.all(axis=1).any():
        raise ValueError(
            "Need at least three groups and positive/negative pairs per query"
        )
    # CPU training is small and deterministic; the 2B image encoder remains frozen.
    torch.set_num_threads(1)
    torch.manual_seed(policy["seed"])
    q = torch.tensor(vectors[queries], dtype=torch.float32)
    g = torch.tensor(vectors[gallery], dtype=torch.float32)
    if not torch.isfinite(q).all() or not torch.isfinite(g).all():
        raise ValueError("Nonfinite training embedding")
    positives = torch.tensor(labels)
    raw = torch.nn.Parameter(torch.zeros(vectors.shape[1]))
    optimizer = torch.optim.Adam([raw], lr=policy["learning_rate"])
    history = []
    limit = float(np.log(policy["weight_max"]))
    if not np.isclose(policy["weight_min"], 1 / policy["weight_max"]):
        raise ValueError("Log-symmetric bounds required")
    for step in range(policy["steps"]):
        optimizer.zero_grad()
        log_weights = limit * torch.tanh(raw)
        transform = torch.exp(log_weights / 2)
        scores = (
            functional.normalize(q * transform, dim=1)
            @ functional.normalize(g * transform, dim=1).T
        )
        negative_scores = scores.masked_fill(positives, -float("inf"))
        count = min(policy["hard_negatives"], min((~positives).sum(dim=1)).item())
        negative_indices = negative_scores.topk(int(count), dim=1).indices
        allowed = positives.clone()
        allowed.scatter_(1, negative_indices, True)
        logits = scores / policy["temperature"]
        loss = (
            torch.logsumexp(logits.masked_fill(~allowed, -float("inf")), dim=1)
            - torch.logsumexp(logits.masked_fill(~positives, -float("inf")), dim=1)
        ).mean()
        objective = loss + policy["regularization"] * log_weights.square().mean()
        objective.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0:
            history.append(
                {
                    "step": step + 1,
                    "training_loss": float(loss.detach()),
                    "regularized_loss": float(objective.detach()),
                }
            )
    weights = torch.exp(limit * torch.tanh(raw)).detach().numpy()
    return {
        "algorithm": "bounded_diagonal_metric_multi_positive_hard_negative_softmax",
        "policy": policy,
        "weights": weights.tolist(),
        "loss_history": history,
        "training_query_groups": sorted({records[i]["pair_id"] for i in queries}),
        "training_gallery_groups": sorted({records[i]["pair_id"] for i in gallery}),
        "excluded_development_groups": sorted(dev_groups),
        "training_queries": len(queries),
        "training_candidates": len(gallery),
        "positive_relations": int(labels.sum()),
        "formal_acceptance": False,
        "label_origin": "official_same_item_identity_not_human_business_relevance",
    }
