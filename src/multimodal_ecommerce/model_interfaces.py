"""Minimal structural interfaces used by generation and checkpoint verification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from multimodal_ecommerce.regional_retrieval import RegionalSearchRequest


class ModelInspector(Protocol):
    """Inspect local model metadata without requiring a concrete HTTP implementation."""

    def health(self) -> dict:
        """Inspect the local model service.

        Returns:
            Service version and installed-model metadata.
        """
        ...

    def _request(self, endpoint: str, payload: dict | None = None) -> dict:
        """Read or post a local API operation and return its decoded mapping."""
        ...


class CopyGenerator(Protocol):
    """Generate a validated factual draft from one normalized product record."""

    def generate_copy(self, product: dict) -> dict:
        """Return generated fields and their provenance for the supplied product.

        Args:
            product: Normalized product attributes and title.

        Returns:
            Draft copy and its generation metadata.
        """
        ...


class BatchGenerator(CopyGenerator, Protocol):
    """Expose the model name alongside generation for traceable batch receipts."""

    @property
    def model(self) -> str:
        """Identify the model recorded in batch provenance.

        Returns:
            Local model label reported by the generator.
        """
        ...


class SearchableIndex(Protocol):
    """Read candidate metadata and obtain a complete regional ranking.

    Attributes:
        records: Candidate metadata aligned with the underlying feature bank.
    """

    records: list[dict]

    def search(self, options: RegionalSearchRequest, /) -> list[dict]:
        """Rank candidates using the supplied query and explicit retrieval options.

        Args:
            options: Query vectors, candidate filters and ranking policy.

        Returns:
            Candidate result records ordered from highest to lowest score.
        """
        ...


class FeatureIndex(SearchableIndex, Protocol):
    """Expose the masked regional arrays needed by public-feature fusion.

    Attributes:
        vectors: Candidate-by-view embedding bank.
        mask: Boolean mask selecting the real views in the padded bank.
    """

    vectors: np.ndarray
    mask: np.ndarray
