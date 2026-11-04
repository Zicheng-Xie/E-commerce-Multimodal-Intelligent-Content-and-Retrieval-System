"""Business-relevance retrieval using published titles and image evidence only."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np

from multimodal_ecommerce.pair_reranker import PairReranker
from multimodal_ecommerce.retrieval_strategies import ordered_indices

BUSINESS_INSTRUCTION = (
    "Rank products relevant to the user's query using the product image and its original title. "
    + "For a text query, match the requested product type, intended use and explicit attributes. "
    + "Distinguish a requested main product from accessories, cases, stands, replacement parts and packaging. "
    + "For an image query, compare the primary product's type, use and visible appearance. "
    + "Do not require the identical SKU or brand unless the query explicitly asks for it. "
    + "Background, person or broad category similarity alone is insufficient. "
    + "Do not invent unseen material, compatibility or technical specifications."
)
BUSINESS_POLICY: dict = {
    "role": "development_business_relevance_prototype",
    "rrf_constant": 60,
    "rerank_pool": 20,
    "text_weights": {"regional": 0.5, "title": 0.3, "lexical": 0.2},
    "image_weights": {"regional": 0.7, "title": 0.3},
    "methods": [
        "global",
        "region_title",
        "hybrid",
        "business_reranker",
        "business_reranker_blend50_anchor2",
    ],
    "qrels_or_query_category_used": False,
    "instruction": BUSINESS_INSTRUCTION,
    "quality_targets": {"precision@10": 0.6, "recall@10": 0.7},
}


def lexical_terms(text: str) -> set[str]:
    """Deterministic Chinese character bigrams and Latin/digit words.

    Args:
        text: Text input used by the operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    terms = set(re.findall(r"[a-z0-9]+", text))
    for segment in re.findall(r"[\u3400-\u9fff]+", text):
        terms.update(segment[i : i + 2] for i in range(len(segment) - 1))
        if len(segment) == 1:
            terms.add(segment)
    return terms


def lexical_scores(query: str, titles: list[str]) -> np.ndarray:
    """IDF-weighted literal title evidence; no automatic relevance judgment.

    Args:
        query: Query record or representation used by this operation.
        titles: Titles used by this operation.

    Returns:
        Computed array in the order of the supplied records.
    """
    q = lexical_terms(query)
    docs = [lexical_terms(t) for t in titles]
    if not q:
        return np.zeros(len(titles), dtype=float)
    df = {term: sum(term in doc for doc in docs) for term in q}
    idf = {term: np.log1p((len(docs) + 1) / (df[term] + 1)) for term in q}
    return np.array([sum(idf[t] for t in q & doc) for doc in docs])


def hybrid_scores(
    ids: list[str],
    regional: np.ndarray,
    title: np.ndarray,
    *,
    text: str | None = None,
    titles: list[str] | None = None,
) -> np.ndarray:
    """Fuse fixed ranks within the allowed gallery; zero lexical evidence abstains.

    Args:
        ids: Identifiers in the order expected by the associated arrays.
        regional: Regional used by this operation.
        title: Public product title used as input.
        text: Text input used by the operation.
        titles: Titles used by this operation.

    Returns:
        Computed array in the order of the supplied records.

    Raises:
        ValueError: Text fusion requires aligned published titles; Malformed hybrid
            score channel.
    """
    weights = BUSINESS_POLICY["text_weights" if text else "image_weights"]
    channels = {"regional": regional, "title": title}
    if text:
        if titles is None or len(titles) != len(ids):
            raise ValueError("Text fusion requires aligned published titles")
        literal = lexical_scores(text, titles)
        if np.any(literal > 0):
            channels["lexical"] = literal
    result = np.zeros(len(ids), dtype=float)
    for name, values in channels.items():
        if np.asarray(values).shape != (len(ids),) or not np.isfinite(values).all():
            raise ValueError("Malformed hybrid score channel")
        ranks = np.empty(len(ids), dtype=int)
        ranks[ordered_indices(values, ids)] = np.arange(1, len(ids) + 1)
        result += weights[name] / (BUSINESS_POLICY["rrf_constant"] + ranks)
    return result


class BusinessReranker:
    """Reuse verified full local weights with a separately recorded business
    instruction.
    """

    def __init__(self) -> None:
        """Initialize the component and validate its configured dependencies."""
        self.adapter = PairReranker()
        self.adapter.config = {
            **self.adapter.config,
            "instruction": BUSINESS_INSTRUCTION,
            "task_scope": "business_relevance_development_only",
        }

    def score(
        self,
        documents: list[dict],
        *,
        query_image: Path | None = None,
        query_text: str | None = None,
    ) -> np.ndarray:
        """Return score for this workflow.

        Args:
            documents: Documents used by this operation.
            query_image: Query image used by this operation.
            query_text: Query text used by this operation.

        Returns:
            Computed array in the order of the supplied records.
        """
        return self.adapter.score(
            documents, query_image=query_image, query_text=query_text
        )

    def diagnostics(self) -> dict:
        """Describe the active backend and recorded runtime measurements.

        Returns:
            Result mapping produced by the workflow.
        """
        return self.adapter.diagnostics()

    def close(self) -> None:
        """Release the resources held by this component."""
        self.adapter.close()
