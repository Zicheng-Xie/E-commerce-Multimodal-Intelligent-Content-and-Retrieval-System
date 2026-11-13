"""Evaluate archived paper mechanisms on the complete exposed module-two gallery."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.business_retrieval import lexical_terms
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.paper_fusion import fuse_paper_candidates
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from querywise_paired_bounds import paired_querywise

ARCHIVE = PROJECT_ROOT / "ccfa-workfiles/literature/module2-paper-reproductions"
ROOT = PROJECT_ROOT / "artifacts/module2_paper_fusion_20261106"
DATA = PROJECT_ROOT / "artifacts/module2_balanced_gallery"
SIMPLE = PROJECT_ROOT / "artifacts/module2_simple_baseline_20261105"
ROLE = PROJECT_ROOT / "artifacts/module2_role_pool_development"
GROUND = PROJECT_ROOT / "artifacts/module2_business_grounding_development"
COHORTS = ("tune", "regression", "fresh")
PRIMARY = "paper_evidence_fusion"
BASELINES = ("global", "region_title", "cross20", "whole_crop_context_union")
SOURCES = {
    "sid": (
        "01-Pailitao-MMSearch/hybsid.py",
        "fa0748a06a55a509c5060836e9c8058d4ff218d7a00ac67947530d0a79e58755",
    ),
    "attributes": (
        "02-AFMRL/afmrl.py",
        "6d52e4330056df32a4566189907e5dfa2813a156946dbc66ed7ce5c1b33f4582",
    ),
    "expansion": (
        "05-SAM-D2Q/sam_d2q.py",
        "0cbcdfe757e85d1871196b4d8d257178042aa6a38e52b3bb97cbef5fc04bfcaf",
    ),
}
BM25_WEIGHT = 0.15


def load(path: Path) -> Any:
    """Read local JSON.

    Args:
        path: Existing project data or evidence file.

    Returns:
        Decoded data at the JSON boundary.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def mechanisms() -> SimpleNamespace:
    """Load the three inspected and checksum-pinned local reproductions.

    Returns:
        Isolated modules exposing HybSID, attribute BM25 and evidence postings.
    """
    modules = {}
    for name, (relative, digest) in SOURCES.items():
        path = ARCHIVE / relative
        if sha256_file(path) != digest:
            raise ValueError(f"Reproduction changed: {relative}")
        module_name = f"module2_paper_{name}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load reproduction: {relative}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return SimpleNamespace(**modules)


def source_inputs() -> list[Path]:
    """List every data input covered by the development role check.

    Returns:
        Explicit file paths for cohorts, vectors, and independent item observations.
    """
    inputs = [
        SIMPLE / "inputs.json",
        SIMPLE / "encoding_receipt.json",
        ROLE / "profiles.json",
        ROLE / "profile_inputs.public.json",
        GROUND / "semantic_results.json",
        GROUND / "views_inputs.public.json",
    ]
    for cohort in COHORTS:
        inputs.extend(
            DATA / cohort / name
            for name in (
                "catalog.public.json",
                "queries.public.json",
                "predictions.json",
                "judgments.ai.jsonl",
            )
        )
        inputs.extend(
            [
                SIMPLE / cohort / "predictions.json",
                SIMPLE / cohort / "index/whole.npz",
            ]
        )
    return inputs


def observed_terms() -> dict[str, tuple[str, set[str]]]:
    """Recover only sale-item terms corroborated by two own-image observations.

    Returns:
        Mapping from item key to original image hash and corroborated terms.
    """
    profiles = {r["key"]: r["value"] for r in load(ROLE / "profiles.json")}
    sources = {
        r["key"]: r["source_sha256"] for r in load(ROLE / "profile_inputs.public.json")
    }
    ground_sources = {
        r["key"]: r["source_sha256"] for r in load(GROUND / "views_inputs.public.json")
    }
    terms = {}
    for row in load(GROUND / "semantic_results.json"):
        key = row["key"]
        left = profiles.get(key)
        right = row.get("native_value")
        if (
            not left
            or not right
            or key not in sources
            or sources[key] != ground_sources.get(key)
        ):
            continue
        if (
            left.get("primary_unclear") is not False
            or right.get("box_correct_sale_item") is not True
        ):
            continue
        words = lexical_terms(left["sale_item"]) & lexical_terms(right["sale_item"])
        if words:
            terms[key] = (sources[key], words)
    return terms


def evidence_terms(
    observations: dict[str, tuple[str, set[str]]], key: str, source: str
) -> list[str]:
    """Look up corroborated terms from the exact same original image.

    Args:
        observations: Item keys, original hashes and corroborated tokens.
        key: Gallery item or cohort/query identity.
        source: Original image SHA256 for the active input.

    Returns:
        Sorted tokens, or no evidence for unmatched or uncertain observations.
    """
    digest, terms = observations.get(key, ("", set()))
    return sorted(terms) if digest == source else []


def catalog_tokens(catalog: list[dict], context: SimpleNamespace) -> tuple[dict, dict]:
    """Build title-only and corroborated expansion documents using SAM-D2Q postings.

    Args:
        catalog: Original published titles and IDs in the fixed gallery.
        context: Frozen source identities, observations, and reproduction modules.

    Returns:
        Token lists by ID for title-only and evidence-expanded documents.
    """
    products = []
    for row in catalog:
        cid = row["id"]
        verified = evidence_terms(
            context.observations, cid, context.simple["documents"][cid]["source_sha256"]
        )
        products.append(
            dict(
                id=cid,
                title_tokens=sorted(lexical_terms(row["title"])),
                verified_visual_tokens=verified,
                generated_queries=[verified],
            )
        )
    documents = []
    for expand in [False, True]:
        index = context.modules.expansion.postings(products, expand=expand)
        tokens: dict[str, list[str]] = {r["id"]: [] for r in catalog}
        for token, members in sorted(index.items()):
            for cid in members:
                tokens[cid].append(token)
        documents.append(tokens)
    return documents[0], documents[1]


def sid_pools(cohort: str, context: SimpleNamespace) -> dict[str, dict[str, list[str]]]:
    """Evaluate semantic ID prefix coverage alongside exact dense candidates.

    Args:
        cohort: One of the three fixed exposed cohorts.
        context: Frozen vector receipts, reproduction modules and query identities.

    Returns:
        Candidate IDs per image query for dense Top20 and semantic ID prefixes.
    """
    with np.load(SIMPLE / cohort / "index/whole.npz", allow_pickle=False) as bank:
        ids = bank["ids"].tolist()
        matrix = bank["vectors"]
    if ids != context.simple["cohorts"][cohort]["gallery"]:
        raise ValueError("HybSID vectors are not aligned to the fixed gallery")
    index = context.modules.sid.HybridIndex.fit(matrix, matrix, size=8, seed=0)
    queries = context.simple["cohorts"][cohort]["queries"]
    vectors = []
    for query in queries:
        name = f"vectors/{cohort}_{query['id']}.npy"
        path = SIMPLE / name
        require_data_access([path], "development", context.registry)
        if sha256_file(path) != context.vectors[name]:
            raise ValueError(f"Query vector changed: {name}")
        vectors.append(np.load(path, allow_pickle=False))
    query_bank = np.stack(vectors)
    codes = index.encode(query_bank)
    prefix = index.search(codes, query_bank, top_k=len(ids), prefix=1)
    result = {}
    for query, (rows, _), dense in zip(
        queries, prefix, query_bank @ matrix.T, strict=True
    ):
        top = np.argsort(-dense, kind="stable")[:20]
        result[query["id"]] = {
            "dense_top20": [ids[i] for i in top],
            "sid_top20": [ids[i] for i in rows[:20]],
            "sid_full_bucket": [ids[i] for i in rows],
        }
    return result


def predictions(cohort: str, context: SimpleNamespace) -> dict:
    """Create complete rank lists before reading relevance labels.

    Args:
        cohort: Fixed exposed data cohort.
        context: Verified inputs and paper-mechanism modules.

    Returns:
        Complete per-query rankings for controls and fusion ablations.
    """
    folder = DATA / cohort
    original = load(folder / "predictions.json")
    simple = load(SIMPLE / cohort / "predictions.json")
    plain, expanded = catalog_tokens(load(folder / "catalog.public.json"), context)
    query_inputs = {q["id"]: q for q in context.simple["cohorts"][cohort]["queries"]}
    results = {}
    for query in load(folder / "queries.public.json"):
        qid = query["id"]
        previous = original[qid]
        cross, dual, regional = (previous[k] for k in BASELINES[2:] + (BASELINES[1],))
        terms = (
            sorted(lexical_terms(query["text"]))
            if query["text"]
            else evidence_terms(
                context.observations,
                f"{cohort}/{qid}",
                query_inputs[qid]["source_sha256"],
            )
        )
        variants = {name: previous[name] for name in BASELINES}
        variants["simple_previous"] = (
            simple[qid]["crop_top20_complete_reranker"] if qid in simple else cross
        )
        variants["paper_rank_consensus"] = fuse_paper_candidates(
            cross, dual, regional, np.zeros(len(cross)), 0.0
        )
        for method, tokens in [("paper_title_bm25", plain), (PRIMARY, expanded)]:
            bm25 = context.modules.attributes.attribute_bm25(
                [terms], [tokens[cid] for cid in cross]
            ).numpy()[0]
            variants[method] = fuse_paper_candidates(
                cross, dual, regional, bm25, BM25_WEIGHT
            )
        results[qid] = variants
    return results


def evaluate(cohort: str, ranks: dict, pools: dict) -> dict:
    """Report fixed-K metrics, component gains and label-conditioned ceilings.

    Args:
        cohort: Frozen exposed development cohort.
        ranks: Already persisted full rankings.
        pools: Image-only semantic ID and dense candidate lists.

    Returns:
        Absolute and paired results plus diagnostic pool coverage and oracle ceilings.
    """
    labels = [
        json.loads(line)
        for line in (DATA / cohort / "judgments.ai.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    modes = {qid: "text" if qid.startswith("text_") else "image" for qid in ranks}
    absolute = evaluate_intervals(ranks, labels, modes)
    paired = paired_querywise(
        ranks,
        labels,
        modes,
        [
            (PRIMARY, method)
            for method in (*BASELINES, "simple_previous", "paper_title_bm25")
        ],
    )
    ceilings = {}
    for scope in ("all", *sorted(set(modes.values()))):
        counts = [
            (row["known_positive"], row["unknown"])
            for row in absolute["query_capacity"]
            if scope == "all" or modes[row["query_id"]] == scope
        ]
        possible = [min(10, p) / p if p else 1.0 for p, u in counts if p or u]
        ceilings[scope] = dict(
            optimistic_recall_at_10_ceiling=(
                sum(possible) / len(possible) if possible else None
            ),
            known_only_precision_at_10_ceiling=sum(min(10, p) for p, _ in counts)
            / (10 * len(counts)),
        )
    positives = {
        (row["query_id"], row["candidate_id"]) for row in labels if row["grade"] == 1
    }
    coverage = {
        name: {
            "mean_candidates": sum(len(pool[name]) for pool in pools.values())
            / len(pools),
            "known_positives_captured": sum(
                (qid, cid) in positives
                for qid, pool in pools.items()
                for cid in pool[name]
            ),
        }
        for name in ("dense_top20", "sid_top20", "sid_full_bucket")
    }
    return dict(
        absolute=absolute, paired=paired, capacity=ceilings, sid_coverage=coverage
    )


def main() -> None:
    """Run the frozen paper fusion comparison and retain development evidence."""
    registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
    paths = source_inputs()
    require_data_access(paths, "development", registry)
    vectors = load(SIMPLE / "encoding_receipt.json")["files"]
    for cohort in COHORTS:
        paths.extend(
            SIMPLE / name for name in vectors if name.startswith(f"vectors/{cohort}_")
        )
    require_data_access(paths, "development", registry)
    files = paths + [
        Path(__file__),
        PROJECT_ROOT / "src/multimodal_ecommerce/paper_fusion.py",
        PROJECT_ROOT / "configs/module2_current_target.json",
        *(ARCHIVE / source for source, _ in SOURCES.values()),
    ]
    protocol = dict(
        files={
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in files
        },
        methods=dict(
            primary=PRIMARY,
            rank_votes=[3, 2, 1],
            reciprocal_rank_offset=20,
            bm25_weight=BM25_WEIGHT,
            top1_protection="original cross20",
            scored_head="union of cross20 and whole_crop_context_union Top20",
            sid="image-only, original whole embeddings, 8 entries, prefix 1, seed 0",
        ),
        k=10,
        tuning=False,
        evaluation_role="exposed_development",
        formal_metrics=None,
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    receipt = ROOT / "protocol.json"
    if receipt.exists() and load(receipt) != protocol:
        raise ValueError("Frozen paper comparison inputs or code changed")
    write_json(receipt, protocol)
    context = SimpleNamespace(
        modules=mechanisms(),
        observations=observed_terms(),
        simple=load(SIMPLE / "inputs.json"),
        registry=registry,
        vectors=vectors,
    )
    ranks = {cohort: predictions(cohort, context) for cohort in COHORTS}
    pools = {cohort: sid_pools(cohort, context) for cohort in COHORTS}
    write_json(ROOT / "rankings.json", ranks)
    write_json(ROOT / "sid_candidates.json", pools)
    results = {
        cohort: evaluate(cohort, ranks[cohort], pools[cohort]) for cohort in COHORTS
    }
    write_json(ROOT / "results.json", results)
    print(
        json.dumps(
            {
                "queries": sum(map(len, ranks.values())),
                "methods": len(next(iter(ranks["tune"].values()))),
                "corroborated_items_and_queries": len(context.observations),
                "formal_metrics": None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
