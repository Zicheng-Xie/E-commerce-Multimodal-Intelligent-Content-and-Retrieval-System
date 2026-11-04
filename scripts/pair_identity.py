"""Identify exact real-model input pairs before reusing historical scores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file


def pair_identity(query: dict, description: str, doc: dict) -> dict:
    """Return pair identity for this workflow.

    Args:
        query: Query record or representation used by this operation.
        description: Description used by this operation.
        doc: Doc used by this operation.

    Returns:
        Mapping with fields: sha256, inputs.
    """
    identity = dict(
        query_text=query["text"],
        query_image_sha256=(
            sha256_file(Path(query["image_path"])) if query["image_path"] else None
        ),
        query_description=description,
        candidate_id=doc["id"],
        candidate_title=doc["title"],
        candidate_image_sha256=sha256_file(Path(doc["image_path"])),
        reranker_config_sha256=sha256_file(
            PROJECT_ROOT / "configs/reranker_model.json"
        ),
        instruction_source_sha256=sha256_file(
            PROJECT_ROOT / "src/multimodal_ecommerce/visual_query.py"
        ),
    )
    key = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return dict(sha256=key, inputs=identity)
