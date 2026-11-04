"""Learn channel/view attention and a query-domain adapter from training identities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

import numpy as np


class _AttentionPolicySpec(TypedDict):
    """Exact fields of the ATTENTION_POLICY configuration and provenance record."""

    seed: int
    steps: int
    learning_rate: float
    hard_negatives: int
    loss_temperature: float
    regularization: float
    view_kinds: list[str]
    channel_weight_bounds: list[float]
    attention_temperature_bounds: list[float]
    global_weight_bounds: list[float]
    ridge: float
    query_residual_weight: float
    methods: list[str]
    selection: str
    new_development_gain_limit_note: str


ATTENTION_POLICY: _AttentionPolicySpec = {
    "seed": 2026102203,
    "steps": 200,
    "learning_rate": 0.01,
    "hard_negatives": 20,
    "loss_temperature": 0.05,
    "regularization": 0.05,
    "view_kinds": ["global", "contrast_roi", "center"],
    "channel_weight_bounds": [0.5, 2.0],
    "attention_temperature_bounds": [0.02, 0.15],
    "global_weight_bounds": [0.2, 0.8],
    "ridge": 1.0,
    "query_residual_weight": 0.5,
    "methods": [
        "frozen_three_view_attention",
        "learned_three_view_attention",
        "learned_attention_blend25_anchor2",
        "ridge_query_attention",
        "ridge_attention_blend25_anchor2",
    ],
    "selection": "Old development: P+2pp and no other mean regression. New development: all six means nondecreasing. No acceptance claim.",
    "new_development_gain_limit_note": "Original global P=.184 versus ceiling=.192; a +2pp requirement on this cohort is mathematically impossible",
}


def attention_scores(
    queries: np.ndarray,
    gallery: np.ndarray,
    mask: np.ndarray,
    model: dict | None = None,
) -> np.ndarray:
    """Score normalized global queries against pixel-derived document regions.

    Args:
        queries: Queries used by this operation.
        gallery: Gallery used by this operation.
        mask: Mask used by this operation.
        model: Model used by this operation.

    Returns:
        Computed array in the order of the supplied records.

    Raises:
        ValueError: Attention input shape mismatch; Invalid learned channel weights;
            Attention needs finite inputs and an original global view; Invalid attention
            controls.
    """
    model = model or {}
    weights = np.asarray(model.get("channel_weights", np.ones(queries.shape[-1])))
    if (
        queries.ndim != 2
        or gallery.ndim != 3
        or queries.shape[1] != gallery.shape[2]
        or mask.shape != gallery.shape[:2]
    ):
        raise ValueError("Attention input shape mismatch")
    if (
        weights.shape != (queries.shape[-1],)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise ValueError("Invalid learned channel weights")
    if (
        not np.isfinite(queries).all()
        or not np.isfinite(gallery).all()
        or not mask[:, 0].all()
    ):
        raise ValueError("Attention needs finite inputs and an original global view")
    q = queries * np.sqrt(weights)
    g = gallery * np.sqrt(weights)
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    g /= np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)
    sims = np.einsum("qd,gvd->qgv", q, g)
    temperature = float(model.get("temperature", 0.07))
    gate = float(model.get("global_weight", 0.6))
    if temperature <= 0 or not 0 <= gate <= 1:
        raise ValueError("Invalid attention controls")
    logits = np.where(mask[None], sims / temperature, -np.inf)
    probabilities = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return gate * sims[:, :, 0] + (1 - gate) * (probabilities * sims).sum(axis=-1)


def fit_attention(
    vectors: np.ndarray, mask: np.ndarray, records: list[dict], policy: Mapping
) -> dict:
    """Optimize a small attention head; never update the 2B embedding model.

    Args:
        vectors: Vectors used by this operation.
        mask: Mask used by this operation.
        records: Records used by this operation.
        policy: Policy used by this operation.

    Returns:
        Mapping with fields: policy, channel_weights, global_weight, temperature,
        loss_history, ridge_anchors, ridge_coefficients, training_query_groups,
        training_gallery_groups, excluded_development_groups, formal_acceptance,
        label_origin, encoder_trainable.

    Raises:
        ValueError: Attention training alignment mismatch; Source pair group crosses
            training/development; Need positive and negative identities for at least
            three queries.
    """
    import torch
    import torch.nn.functional as functional

    if (
        vectors.ndim != 3
        or mask.shape != vectors.shape[:2]
        or len(records) != len(vectors)
    ):
        raise ValueError("Attention training alignment mismatch")
    train = [i for i, r in enumerate(records) if r["project_role"] == "training"]
    development = {
        r["pair_id"] for r in records if r["project_role"] == "development_validation"
    }
    groups = {records[i]["pair_id"] for i in train}
    if groups & development:
        raise ValueError("Source pair group crosses training/development")
    qi = [i for i in train if records[i]["source"] == "user"]
    gi = [i for i in train if records[i]["source"] == "shop"]
    labels = np.array(
        [[records[q]["identity"] == records[g]["identity"] for g in gi] for q in qi]
    )
    if len(qi) < 3 or not labels.any(axis=1).all() or labels.all(axis=1).any():
        raise ValueError(
            "Need positive and negative identities for at least three queries"
        )
    torch.set_num_threads(1)
    torch.manual_seed(policy["seed"])
    q = torch.tensor(vectors[qi, 0], dtype=torch.float32)
    g = torch.tensor(vectors[gi], dtype=torch.float32)
    valid = torch.tensor(mask[gi])
    positive = torch.tensor(labels)
    raw = torch.nn.Parameter(torch.zeros(vectors.shape[-1]))
    controls = torch.nn.Parameter(
        torch.tensor([np.log(2.0), np.log(0.05 / 0.08)], dtype=torch.float32)
    )
    optimizer = torch.optim.Adam([raw, controls], lr=policy["learning_rate"])
    history = []
    for step in range(policy["steps"]):
        optimizer.zero_grad()
        log_weights = np.log(2.0) * raw.tanh()
        factor = (log_weights / 2).exp()
        qn = functional.normalize(q * factor, dim=-1)
        gn = functional.normalize(g * factor, dim=-1)
        sims = (qn @ gn.reshape(-1, qn.shape[-1]).T).reshape(
            len(qi), len(gi), g.shape[1]
        )
        gate = 0.2 + 0.6 * controls[0].sigmoid()
        temperature = 0.02 + 0.13 * controls[1].sigmoid()
        logits = (sims / temperature).masked_fill(~valid[None], -float("inf"))
        scores = gate * sims[:, :, 0] + (1 - gate) * (
            logits.softmax(dim=-1) * sims
        ).sum(dim=-1)
        count = min(policy["hard_negatives"], int((~positive).sum(dim=1).min()))
        negatives = (
            scores.masked_fill(positive, -float("inf")).topk(count, dim=1).indices
        )
        allowed = positive.clone()
        allowed.scatter_(1, negatives, True)
        logits = scores / policy["loss_temperature"]
        loss = (
            torch.logsumexp(logits.masked_fill(~allowed, -float("inf")), dim=1)
            - torch.logsumexp(logits.masked_fill(~positive, -float("inf")), dim=1)
        ).mean()
        objective = (
            loss
            + policy["regularization"] * log_weights.square().mean()
            + 0.1 * (gate - 0.6).square()
        )
        objective.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0:
            history.append({"step": step + 1, "training_loss": float(loss.detach())})
    qtrain = vectors[qi, 0].astype(np.float64)
    targets = np.stack(
        [vectors[np.array(gi)[rel], 0].mean(axis=0) for rel in labels]
    ).astype(np.float64)
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    coefficients = np.linalg.solve(
        qtrain @ qtrain.T + policy["ridge"] * np.eye(len(qtrain)), targets - qtrain
    )
    return {
        "policy": policy,
        "channel_weights": torch.exp(np.log(2.0) * raw.tanh()).detach().tolist(),
        "global_weight": float((0.2 + 0.6 * controls[0].sigmoid()).detach()),
        "temperature": float((0.02 + 0.13 * controls[1].sigmoid()).detach()),
        "loss_history": history,
        "ridge_anchors": qtrain.tolist(),
        "ridge_coefficients": coefficients.tolist(),
        "training_query_groups": sorted({records[i]["pair_id"] for i in qi}),
        "training_gallery_groups": sorted({records[i]["pair_id"] for i in gi}),
        "excluded_development_groups": sorted(development),
        "formal_acceptance": False,
        "label_origin": "official_same_item_identity",
        "encoder_trainable": False,
    }


def adapt_queries(queries: np.ndarray, model: dict) -> np.ndarray:
    """Regularized train-only user-to-shop residual mapping with normalized output.

    Args:
        queries: Queries used by this operation.
        model: Model used by this operation.

    Returns:
        Computed array in the order of the supplied records.

    Raises:
        ValueError: Invalid adapted query.
    """
    anchors = np.asarray(model["ridge_anchors"])
    coefficients = np.asarray(model["ridge_coefficients"])
    result = queries + model["policy"]["query_residual_weight"] * (
        queries @ anchors.T @ coefficients
    )
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if not np.isfinite(result).all() or np.any(norms < 1e-8):
        raise ValueError("Invalid adapted query")
    return result / norms
