"""Replay all business-development rankings without assigning relevance labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run  # noqa: F401
from business_precision import INDEX, OLD_VECTORS, OUT, QUERIES
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_retrieval import hybrid_scores
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.holdout_data import fingerprints
from multimodal_ecommerce.precision_labels import validate_precision_labels
from multimodal_ecommerce.precision_reranking import refine_pool
from multimodal_ecommerce.regional_retrieval import RegionalIndex, RegionalSearchRequest
from multimodal_ecommerce.retrieval_strategies import ordered_indices
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve the verified prototype evidence.

    Notes:
        Writes derived artifacts through write_json.
    """
    target = OUT / "verification.json"
    if target.exists():
        raise FileExistsError("Preserve the verified prototype evidence")
    receipt = json.loads((OUT / "prediction_receipt.json").read_text(encoding="utf-8"))
    verify_files(OUT, receipt["files"])
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    verify_files(PROJECT_ROOT, protocol["parents"])
    verify_files(OUT / "source_executed", protocol["implementation"])
    amendment = json.loads(
        (OUT / "source_typing_amendment.json").read_text(encoding="utf-8")
    )
    assert sha256_file(PROJECT_ROOT / amendment["file"]) == amendment["current_sha256"]
    index, texts, _ = RegionalIndex.load(INDEX)
    assert texts is not None
    queries = read_rows(QUERIES)
    public = json.loads((OUT / "model_inputs.json").read_text(encoding="utf-8"))
    assert all(
        set(q) == {"query_id", "query_text", "query_image", "documents"} for q in public
    )
    assert all(
        set(d) == {"id", "image_path", "text"} for q in public for d in q["documents"]
    )
    allowed = json.loads((OUT / "allowed_candidates.json").read_text(encoding="utf-8"))
    rankings = json.loads((OUT / "rankings.json").read_text(encoding="utf-8"))
    cached = {r["query_id"]: r for r in read_rows(OUT / "pair_scores.jsonl")}
    signatures = {r["id"]: fingerprints(Path(r["image_path"])) for r in index.records}
    checks = 0
    with np.load(OLD_VECTORS / "query_vectors.npz", allow_pickle=False) as bank:
        for number, query in enumerate(queries):
            q = bank["vectors"][number]
            excluded = {query["exclude_id"]} if query["exclude_id"] else set()
            if query.get("image_path"):
                signature = fingerprints(Path(query["image_path"]))
                excluded.update(
                    k
                    for k, v in signatures.items()
                    if v["sha256"] == signature["sha256"]
                    or v["pixels"] == signature["pixels"]
                )
            selected = [
                i for i, r in enumerate(index.records) if r["id"] not in excluded
            ]
            ids = [index.records[i]["id"] for i in selected]
            assert ids == allowed[query["id"]]
            visual = {
                r["id"]: r["score"]
                for r in index.search(
                    RegionalSearchRequest(
                        q,
                        top_k=len(index.records),
                        strategy="attention",
                        exclude_ids=excluded,
                    )
                )
            }
            regional = np.array([visual[k] for k in ids])
            title = texts[selected] @ q
            global_scores = index.vectors[selected, 0] @ q
            hybrid = hybrid_scores(
                ids,
                regional,
                title,
                text=query.get("text"),
                titles=[index.records[i]["title"] for i in selected],
            )
            bases = {
                "global": [ids[i] for i in ordered_indices(global_scores, ids)],
                "region_title": [
                    ids[i] for i in ordered_indices(0.8 * regional + 0.2 * title, ids)
                ],
                "hybrid": [ids[i] for i in ordered_indices(hybrid, ids)],
            }
            values = cached[query["id"]]
            pool = values["candidate_ids"]
            assert pool == bases["hybrid"][:20]
            scores = np.asarray(values["logits"])
            assert np.isfinite(scores).all() and scores.shape == (20,)
            bases["business_reranker"] = refine_pool(bases["hybrid"], pool, scores)
            bases["business_reranker_blend50_anchor2"] = refine_pool(
                bases["hybrid"], pool, scores, blend=0.5, anchors=2
            )
            assert rankings[query["id"]] == bases
            assert all(
                len(r) == len(ids) and set(r) == set(ids) for r in bases.values()
            )
            checks += len(bases)
    packet = OUT / "review"
    labels = read_rows(packet / "judgments.blank.jsonl")
    label_status = validate_precision_labels(labels, labels)
    assert label_status["completed"] == 0
    assert {(r["query_id"], r["candidate_id"]) for r in labels} == {
        (q, c) for q, ids in allowed.items() for c in ids
    }
    for path in [PROJECT_ROOT / "artifacts/module2_precision60/review", packet]:
        m = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        verify_files(path, m["files"])
    for folder in ["module2_blind", "module2_precision", "module2_training"]:
        root = PROJECT_ROOT / "artifacts" / folder
        old = json.loads((root / "completion_receipt.json").read_text(encoding="utf-8"))
        verify_files(root, old["files"])
        if "implementation" in old:
            verify_files(PROJECT_ROOT, old["implementation"])
    write_json(
        target,
        {
            "complete_rankings_replayed": checks,
            "queries_retained": len(queries),
            "reranker_pairs": sum(len(r["documents"]) for r in public),
            "full_gallery_review_relations": len(labels),
            "human_judgments": 0,
            "quality_metrics": None,
            "source_typing_only_amendment_preserves_executed_sources": True,
            "model_inputs_exclude_qrels_and_query_category": True,
            "query_self_or_pixel_duplicate_candidates_excluded": True,
            "previous_sealed_experiments_unchanged": True,
            "formal_acceptance": False,
        },
    )
    print(
        json.dumps(
            {
                "rankings_replayed": checks,
                "blank_business_pairs": len(labels),
                "human_judgments": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
