"""Launch retrieval using the same project-owned caches as other commands."""

from __future__ import annotations

import run  # noqa: F401
from multimodal_ecommerce.retrieval_cli import main

if __name__ == "__main__":
    main()
