"""Shared structural contracts for verified public inputs and frozen feature banks."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, TypedDict

import numpy as np

if TYPE_CHECKING:
    from multimodal_ecommerce.regional_retrieval import RegionalIndex

FeatureBanks: TypeAlias = tuple[
    "RegionalIndex", np.ndarray, np.ndarray, list[str], np.ndarray
]
FileDigests: TypeAlias = dict[str, str]


class PublicProfile(TypedDict):
    """Query/candidate-only descriptions; relevance judgments are never included.

    Attributes:
        product: Visible sale-item description.
        appearance: Visible shape, color and construction details.
        visible_text: Text transcribed from the image or public product title.
        primary_unclear: Whether the main sale item cannot be identified.
    """

    product: str
    appearance: str
    visible_text: str
    primary_unclear: bool


class OllamaLayer(TypedDict):
    """On-disk OCI layer identity used to verify a local Ollama checkpoint."""

    mediaType: str
    digest: str
    size: int
