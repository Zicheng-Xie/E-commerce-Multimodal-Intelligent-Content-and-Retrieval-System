"""Label-free alternatives to regional attention for small exact retrieval banks.

All inference inputs are vectors and IDs. Relevance/identity labels are used only
by the experiment runner after rankings are saved.
"""

from __future__ import annotations

import math
from dataclasses import KW_ONLY, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from multimodal_ecommerce.regional_retrieval import RegionalSearchRequest
from multimodal_ecommerce.retrieval import unit_rows

EXPLORATION_POLICY: dict[str, Any] = {
    "version": "module2-postranking-exploration-1",
    "qe_neighbors": 5,
    "qe_alpha": 3.0,
    "dba_neighbors": 3,
    "density_neighbors": 10,
    "reciprocal_k1": 20,
    "reciprocal_k2": 6,
    "reciprocal_original_weight": 0.3,
    "reciprocal_region_fusion": 0.5,
    "rrf_constant": 60,
    "diffusion_neighbors": 10,
    "diffusion_seeds": 20,
    "diffusion_restart_weight": 0.15,
    "diffusion_iterations": 20,
    "global_weight": 0.6,
}

STRATEGY_NAMES = (
    "global",
    "gallery_attention",
    "alpha_qe",
    "alpha_qe_attention",
    "database_augmentation",
    "gallery_centering",
    "density_correction",
    "reciprocal_global",
    "reciprocal_attention_fusion",
    "rrf_global_attention",
    "rrf_reciprocal_attention",
    "symmetric_local",
    "graph_diffusion",
)


def ordered_indices(scores: npt.ArrayLike, ids: list[str]) -> np.ndarray:
    """Sort finite scores descending, with stable ID ties.

    Args:
        scores: Scores used by this operation.
        ids: Identifiers in the order expected by the associated arrays.

    Returns:
        Computed array in the order of the supplied records.

    Raises:
        ValueError: Scores must be finite and aligned with IDs; Duplicate retrieval IDs.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(ids),) or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite and aligned with IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate retrieval IDs")
    return np.lexsort((np.asarray(ids), -scores))


def reciprocal_rank_fusion(
    branches: list[np.ndarray], ids: list[str], constant: int = 60
) -> np.ndarray:
    """Combine ranking evidence without assuming compatible score scales.

    Args:
        branches: Branches used by this operation.
        ids: Identifiers in the order expected by the associated arrays.
        constant: Constant used by this operation.

    Returns:
        Computed array in the order of the supplied records.

    Raises:
        ValueError: Need branches and positive RRF constant.
    """
    if not branches or constant < 1 or isinstance(constant, bool):
        raise ValueError("Need branches and positive RRF constant")
    result = np.zeros(len(ids), dtype=float)
    for scores in branches:
        order = ordered_indices(scores, ids)
        ranks = np.empty(len(ids), dtype=int)
        ranks[order] = np.arange(1, len(ids) + 1)
        result += 1.0 / (constant + ranks)
    return result


def extended_metrics(ranked_ids: list[str], relevant_ids: set[str]) -> dict[str, float]:
    """Full-list AP/MRR and binary NDCG, with fixed-k precision denominators.

    Args:
        ranked_ids: Ranked ids used by this operation.
        relevant_ids: Relevant ids used by this operation.

    Returns:
        Mapping with fields: ap, mrr, hit@1.

    Raises:
        ValueError: Need a nonempty unique complete ranking; All known positives must be
            in the ranked candidate pool.
    """
    if not ranked_ids or len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("Need a nonempty unique complete ranking")
    if not relevant_ids or not relevant_ids <= set(ranked_ids):
        raise ValueError("All known positives must be in the ranked candidate pool")
    hits = np.array([key in relevant_ids for key in ranked_ids], dtype=float)
    positions = np.arange(1, len(hits) + 1)
    prefix = hits.cumsum()
    positives = len(relevant_ids)
    found = np.flatnonzero(hits)
    metrics = {
        "ap": float((prefix / positions * hits).sum() / positives),
        "mrr": float(1 / (found[0] + 1)),
        "hit@1": float(hits[0]),
    }
    for k in (5, 10, 20, 50):
        count = float(hits[:k].sum())
        metrics[f"precision@{k}"] = count / k
        metrics[f"recall@{k}"] = count / positives
        metrics[f"hit@{k}"] = float(count > 0)
    discounts = 1 / np.log2(np.arange(2, 12))
    actual = float(np.dot(hits[:10], discounts[: min(10, len(hits))]))
    ideal = float(discounts[: min(positives, 10)].sum())
    metrics["ndcg@10"] = actual / ideal
    return metrics


def alpha_query_expansion(
    query: np.ndarray,
    gallery: np.ndarray,
    ids: list[str],
    neighbors: int = 5,
    alpha: float = 3,
) -> tuple[np.ndarray, list[str]]:
    """Blend original query (weight one) and top positive-cosine neighbors^alpha.

    Args:
        query: Query record or representation used by this operation.
        gallery: Gallery used by this operation.
        ids: Identifiers in the order expected by the associated arrays.
        neighbors: Neighbors used by this operation.
        alpha: Alpha used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Invalid query expansion settings.
    """
    if neighbors < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Invalid query expansion settings")
    q = unit_rows(np.asarray(query).reshape(1, -1))[0]
    bank = unit_rows(gallery)
    similarities = bank @ q
    selected = ordered_indices(similarities, ids)[:neighbors]
    weights = np.maximum(similarities[selected], 0) ** alpha
    expanded = q + np.sum(weights[:, None] * bank[selected], axis=0)
    return unit_rows(expanded.reshape(1, -1))[0], [ids[i] for i in selected]


@dataclass(frozen=True)
class RetrievalStrategiesReciprocalScoresInputs:
    """Input values for reciprocal scores.

    Attributes:
        query: Query record or representation used by this operation.
        gallery: Gallery used by this operation.
        ids: Identifiers in the order expected by the associated arrays.
        k1: K1 used by this operation.
        k2: K2 used by this operation.
        original_weight: Original weight used by this operation.
    """

    query: np.ndarray
    gallery: np.ndarray
    ids: list[str]
    _: KW_ONLY
    k1: int = 20
    k2: int = 6
    original_weight: float = 0.3


def reciprocal_scores(options: RetrievalStrategiesReciprocalScoresInputs) -> np.ndarray:
    """k-reciprocal expansion, LQE and weighted Jaccard from CVPR 2017.

    Uses one current query plus the allowed gallery, never a batch of evaluation
    queries. Self comes first in neighbor lists; k1 counts nonself neighbors.
    Unit-vector squared Euclidean distances are row-max normalized.

    Args:
        options: Validated input fields described by
            RetrievalStrategiesReciprocalScoresInputs.

    Returns:
        Computed array in the order of the supplied records.

    Raises:
        ValueError: Invalid reciprocal re-ranking policy; Gallery IDs disagree.
    """
    query = options.query
    gallery = options.gallery
    ids = options.ids
    k1 = options.k1
    k2 = options.k2
    original_weight = options.original_weight
    if k1 < 1 or k2 < 1 or not 0 <= original_weight <= 1:
        raise ValueError("Invalid reciprocal re-ranking policy")
    q = unit_rows(np.asarray(query).reshape(1, -1))
    bank = unit_rows(gallery)
    if bank.shape[0] != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Gallery IDs disagree")
    values = np.concatenate([q, bank], axis=0).astype(np.float64)
    distance = np.maximum(2 - 2 * (values @ values.T), 0)
    np.fill_diagonal(distance, 0)
    denominator = np.maximum(distance.max(axis=1, keepdims=True), 1e-12)
    distance /= denominator
    # Sorting uses corpus IDs for ties; self must remain in its own encoding.
    keys = ["query"] + ids
    ranks = np.empty(distance.shape, dtype=int)
    for i in range(len(values)):
        row = -distance[i].copy()
        row[i] = 1
        ranks[i] = np.lexsort((np.asarray(keys), -row))
    k1 = min(k1, len(values) - 1)
    half = max(1, int(round(k1 / 2)))

    def reciprocal(i: int, k: int) -> np.ndarray:
        """Return reciprocal for this workflow.

        Args:
            i: I used by this operation.
            k: K used by this operation.

        Returns:
            Computed array in the order of the supplied records.
        """
        forward = ranks[i, : k + 1]
        backward = ranks[forward, : k + 1]
        return forward[np.any(backward == i, axis=1)]

    encoded = np.zeros(distance.shape, dtype=float)
    for i in range(len(values)):
        neighbors = reciprocal(i, k1)
        expanded = set(neighbors.tolist())
        for candidate in neighbors:
            local = reciprocal(int(candidate), half)
            if len(set(local.tolist()) & set(neighbors.tolist())) > 2 * len(local) / 3:
                expanded.update(local.tolist())
        selected = np.array(sorted(expanded), dtype=int)
        weights = np.exp(-distance[i, selected])
        encoded[i, selected] = weights / weights.sum()
    if k2 > 1:
        encoded = np.stack(
            [
                encoded[ranks[i, : min(k2, len(values))]].mean(axis=0)
                for i in range(len(values))
            ]
        )
    intersections = np.minimum(encoded[0], encoded[1:]).sum(axis=1)
    unions = encoded[0].sum() + encoded[1:].sum(axis=1) - intersections
    jaccard = 1 - intersections / np.maximum(unions, 1e-12)
    final = (1 - original_weight) * jaccard + original_weight * distance[0, 1:]
    return -final


def attention_scores(index: Any, query: np.ndarray) -> np.ndarray:
    """Extract the exact existing attention branch aligned to gallery order.

    Args:
        index: Index used by this operation.
        query: Query record or representation used by this operation.

    Returns:
        Computed array in the order of the supplied records.
    """
    results = index.search(
        RegionalSearchRequest(query, len(index.records), strategy="attention")
    )
    by_id = {row["id"]: row["score"] for row in results}
    return np.array([by_id[row["id"]] for row in index.records])


def subset_region_index(index: Any, include: np.ndarray) -> Any:
    """Remove excluded IDs before neighborhood statistics or pseudo-feedback.

    Args:
        index: Index used by this operation.
        include: Include used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return type(index)(
        [index.records[i] for i in include], index.vectors[include], index.config
    )


class StrategyExplorer:
    """Prepare unlabelled gallery statistics once and compare fixed strategies."""

    def __init__(self, index: Any, policy: dict[str, Any] | None = None) -> None:
        """Initialize the component and validate its configured dependencies.

        Args:
            index: Index used by this operation.
            policy: Policy used by this operation.

        Raises:
            ValueError: Need at least two gallery records for neighborhoods.
        """
        self.index = index
        self.policy = dict(EXPLORATION_POLICY if policy is None else policy)
        self.ids = [row["id"] for row in index.records]
        self.global_bank = index.vectors[:, 0]
        n = len(self.ids)
        if n < 2:
            raise ValueError("Need at least two gallery records for neighborhoods")
        self.similarity = np.clip(self.global_bank @ self.global_bank.T, -1, 1)
        without_self = self.similarity.copy()
        np.fill_diagonal(without_self, -np.inf)
        neighbor_count = min(int(self.policy["density_neighbors"]), n - 1)
        self.density = np.sort(without_self, axis=1)[:, -neighbor_count:].mean(axis=1)
        self.mean = self.global_bank.mean(axis=0)
        self.centered = unit_rows(self.global_bank - self.mean)
        augmented = []
        dba = min(int(self.policy["dba_neighbors"]), n - 1)
        for i in range(n):
            nearest = ordered_indices(
                np.where(np.arange(n) == i, -2, self.similarity[i]), self.ids
            )[:dba]
            weights = (
                np.maximum(self.similarity[i, nearest], 0) ** self.policy["qe_alpha"]
            )
            augmented.append(
                self.global_bank[i]
                + (weights[:, None] * self.global_bank[nearest]).sum(axis=0)
            )
        self.augmented = unit_rows(np.stack(augmented))
        k = min(int(self.policy["diffusion_neighbors"]), n - 1)
        edges = np.zeros((n, n), dtype=float)
        for i in range(n):
            nearest = ordered_indices(
                np.where(np.arange(n) == i, -2, self.similarity[i]), self.ids
            )[:k]
            edges[i, nearest] = np.maximum(self.similarity[i, nearest], 0) ** 3
        edges = np.minimum(edges, edges.T)
        degree = edges.sum(axis=1)
        isolated = degree < 1e-12
        edges[isolated, isolated] = 1
        degree = edges.sum(axis=1)
        self.transition = edges / degree[:, None]

    def evaluate(self, query_views: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        """Compute every fixed candidate policy without accessing relevance labels.

        Args:
            query_views: Query views used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        p = self.policy
        queries = unit_rows(query_views)
        q = queries[0]
        global_scores = self.global_bank @ q
        regional = attention_scores(self.index, queries[:1])
        expanded, feedback_ids = alpha_query_expansion(
            q, self.global_bank, self.ids, p["qe_neighbors"], p["qe_alpha"]
        )
        reciprocal = reciprocal_scores(
            RetrievalStrategiesReciprocalScoresInputs(
                q,
                self.global_bank,
                self.ids,
                k1=p["reciprocal_k1"],
                k2=p["reciprocal_k2"],
                original_weight=p["reciprocal_original_weight"],
            )
        )
        # Both terms become smaller-is-better distances in [0,1] before fusion.
        attention_distance = (1 - regional) / 2
        mix = p["reciprocal_region_fusion"]
        combined = mix * reciprocal - (1 - mix) * attention_distance
        q_centered = unit_rows((q - self.mean).reshape(1, -1))[0]
        sims = np.einsum("qd,nvd->nqv", queries, self.index.vectors)
        mask = self.index.mask[:, None, :]
        forward = np.where(mask, sims, -np.inf).max(axis=2).mean(axis=1)
        backward = np.where(mask[:, 0], sims.max(axis=1), 0).sum(
            axis=1
        ) / self.index.mask.sum(axis=1)
        local = (
            p["global_weight"] * global_scores
            + (1 - p["global_weight"]) * (forward + backward) / 2
        )
        seed = np.zeros(len(self.ids))
        nearest = ordered_indices(global_scores, self.ids)[: p["diffusion_seeds"]]
        seed[nearest] = np.maximum(global_scores[nearest], 0) ** 3
        if seed.sum() > 0:
            seed /= seed.sum()
        diffused = seed.copy()
        restart = p["diffusion_restart_weight"]
        for _ in range(p["diffusion_iterations"]):
            diffused = restart * seed + (1 - restart) * (self.transition.T @ diffused)
        scores = {
            "global": global_scores,
            "gallery_attention": regional,
            "alpha_qe": self.global_bank @ expanded,
            "alpha_qe_attention": attention_scores(self.index, expanded.reshape(1, -1)),
            "database_augmentation": self.augmented @ q,
            "gallery_centering": self.centered @ q_centered,
            "density_correction": 2 * global_scores - self.density,
            "reciprocal_global": reciprocal,
            "reciprocal_attention_fusion": combined,
            "rrf_global_attention": reciprocal_rank_fusion(
                [global_scores, regional], self.ids, p["rrf_constant"]
            ),
            "rrf_reciprocal_attention": reciprocal_rank_fusion(
                [reciprocal, regional], self.ids, p["rrf_constant"]
            ),
            "symmetric_local": local,
            "graph_diffusion": reciprocal_rank_fusion(
                [global_scores, diffused], self.ids, p["rrf_constant"]
            ),
        }
        for array in scores.values():
            ordered_indices(array, self.ids)
        return scores, {
            "feedback_ids": feedback_ids,
            "feedback_ground_truth_used": False,
            "graph_seeds": [self.ids[i] for i in nearest],
            "query_views": len(queries),
        }
