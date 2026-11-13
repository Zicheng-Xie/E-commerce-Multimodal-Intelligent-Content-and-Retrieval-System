"""Freeze and execute query-only VLM descriptions and full-model query views."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.machine_teacher import MODEL, resolve_local_teacher
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.regional_retrieval import (
    RegionalRetrievalEncodeViewsInputs,
    encode_views,
)
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.visual_query import (
    CAPTION_OPTIONS,
    CAPTION_PROMPT,
    CAPTION_SCHEMA,
    VISUAL_QUERY_POLICY,
    describe_query,
)
from prepare_focus_coverage import PARENT, ROOT
from prepare_focus_coverage import check as check_acquisition

DESTINATION = ROOT / "query_enrichment"


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        FileExistsError: Preserve query-enrichment protocol.

    Notes:
        Writes derived artifacts through DESTINATION.mkdir, write_json.
    """
    check_acquisition()
    if DESTINATION.exists():
        raise FileExistsError("Preserve query-enrichment protocol")
    queries = [
        q for q in load(PARENT / "inputs.public.json")["queries"] if q["image_path"]
    ]
    reserved = read_rows(
        PROJECT_ROOT
        / "artifacts/module2_business_acceptance/dataset/evaluator/lineage.jsonl"
    )
    for q in queries:
        path = Path(q["image_path"])
        assert "module2_business_acceptance" not in path.parts
        assert not is_exposed(
            fingerprints(path), [r["fingerprints"] for r in reserved], 4
        )
    client = OllamaClient(model=MODEL)
    try:
        teacher = resolve_local_teacher(client, Path("E:/LLM_for_Ollama"))
    finally:
        client.close()
    DESTINATION.mkdir()
    sources = [
        "scripts/enrich_visual_queries.py",
        "src/multimodal_ecommerce/visual_query.py",
        "src/multimodal_ecommerce/ollama_client.py",
        "src/multimodal_ecommerce/machine_teacher.py",
    ]
    write_json(
        DESTINATION / "protocol.json",
        dict(
            teacher=teacher,
            prompt=CAPTION_PROMPT,
            schema=CAPTION_SCHEMA,
            options=CAPTION_OPTIONS,
            policy=VISUAL_QUERY_POLICY,
            queries=queries,
            source_files={p: sha256_file(PROJECT_ROOT / p) for p in sources},
            query_image_files={
                Path(q["image_path"])
                .relative_to(PROJECT_ROOT)
                .as_posix(): sha256_file(Path(q["image_path"]))
                for q in queries
            },
            no_candidates_or_labels_supplied=True,
            reserved_images_opened=0,
            acceptance_predictions=0,
            frozen_before_new_retrieval_comparison=True,
            methods=[
                "query_multi_view",
                "query_caption_view_fusion",
                "described_cross_pool20",
                "described_cross_pool60",
            ],
            text_queries_use_fixed_anchor="cross_evidence_pool60",
            caption_only_for_additional_channels_not_relevance_labels=True,
        ),
    )
    print(
        dict(
            frozen=True,
            image_queries=len(queries),
            description_model=teacher["model"],
            digest=teacher["digest"],
        )
    )


def check() -> Any:
    """Verify the prerequisites and integrity of the current workflow.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    protocol = load(DESTINATION / "protocol.json")
    verify_files(PROJECT_ROOT, protocol["source_files"])
    verify_files(PROJECT_ROOT, protocol["query_image_files"])
    for name, digest in protocol["teacher"]["files"].items():
        assert sha256_file(Path(name)) == digest
    assert (
        protocol["policy"] == VISUAL_QUERY_POLICY
        and protocol["prompt"] == CAPTION_PROMPT
    )
    return protocol


def captions() -> None:
    """Apply the captions workflow step.

    Raises:
        FileExistsError: Preserve completed descriptions.

    Notes:
        Writes derived artifacts through write_json.
    """
    protocol = check()
    if (DESTINATION / "caption_receipt.json").exists():
        raise FileExistsError("Preserve completed descriptions")
    client = OllamaClient(model=MODEL)
    try:
        for i, q in enumerate(protocol["queries"]):
            path = DESTINATION / "captions" / (q["id"] + ".json")
            if not path.exists():
                result = describe_query(
                    client,
                    Path(q["image_path"]),
                    last=i == len(protocol["queries"]) - 1,
                )
                write_json(
                    path,
                    dict(
                        query_id=q["id"],
                        image_sha256=sha256_file(Path(q["image_path"])),
                        **result,
                    ),
                )
            print(
                f"Query-only descriptions {i+1}/{len(protocol['queries'])}", flush=True
            )
    finally:
        client.close()
    write_json(
        DESTINATION / "caption_receipt.json",
        dict(
            products=len(protocol["queries"]),
            actual_query_only_VLM_requests=len(protocol["queries"]),
            model=MODEL,
            files={
                p.relative_to(DESTINATION).as_posix(): sha256_file(p)
                for p in sorted((DESTINATION / "captions").glob("*.json"))
            },
            gallery_candidates_supplied=0,
            relevance_labels_supplied=0,
            heldout_images_supplied=0,
        ),
    )


def encode_queries(
    encoder: QwenEmbedder, queries: list[dict], configuration: dict
) -> None:
    """Called during the catalog's single full embedding-model session.

    Args:
        encoder: Encoder used by this operation.
        queries: Queries used by this operation.
        configuration: Configuration used by this operation.
    """
    check()
    verify_files(DESTINATION, load(DESTINATION / "caption_receipt.json")["files"])
    receipt = {}
    for query in queries:
        if not query["image_path"]:
            continue
        qid = query["id"]
        folder = DESTINATION / "encoded" / qid
        if (folder / "receipt.json").exists():
            receipt[qid] = load(folder / "receipt.json")
            continue
        folder.mkdir(parents=True, exist_ok=True)
        attempt = 0
        while (folder / f"views_{attempt}").exists():
            attempt += 1
        vectors, views = encode_views(
            RegionalRetrievalEncodeViewsInputs(
                encoder,
                Path(query["image_path"]),
                folder / f"views_{attempt}",
                configuration,
                query=True,
            )
        )
        description = load(DESTINATION / "captions" / (qid + ".json"))["text"]
        caption = (
            encoder.encode(text=description, query=True)
            if description
            else np.zeros(2048, dtype=np.float32)
        )
        np.savez_compressed(folder / "vectors.npz", views=vectors, caption=caption)
        row = dict(
            query_id=qid,
            views=views,
            description=description,
            description_encoded=bool(description),
            vector_asset=(folder / "vectors.npz").relative_to(DESTINATION).as_posix(),
            vector_sha256=sha256_file(folder / "vectors.npz"),
            actual_embedding_calls=len(views) + int(bool(description)),
        )
        write_json(folder / "receipt.json", row)
        receipt[qid] = row
        print("Query enrichment encoded " + qid, flush=True)
    write_json(
        DESTINATION / "embedding_receipt.json",
        dict(
            queries=receipt,
            actual_embedding_calls=sum(
                r["actual_embedding_calls"] for r in receipt.values()
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "captions"])
    globals()[parser.parse_args().action]()
