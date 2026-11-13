"""Replay module-two development baselines and inventory a shared gallery."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any
from unicodedata import normalize

import run  # noqa: F401
from module2_p80_r90 import oracle_capacity
from multimodal_ecommerce.ai_review_metrics import optional_mean_bounds
from multimodal_ecommerce.business_evaluation import METRICS
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.interval_retrieval_metrics import (
    evaluate_intervals,
    query_bounds,
)
from multimodal_ecommerce.runtime_device import project_path
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from paper_fusion_experiment import COHORTS, load
from querywise_paired_bounds import paired_querywise

ARTIFACTS = PROJECT_ROOT / "artifacts"
TARGET = ARTIFACTS / "module2_p80_r90_20261106"
BALANCED = ARTIFACTS / "module2_balanced_gallery"
FOCUS = ARTIFACTS / "module2_focus_coverage"
STRESS = ARTIFACTS / "module2_rank_stability/fresh_query_check"
FOLLOWUP = ARTIFACTS / "module2_focus_followup_20261107"
DUAL = FOLLOWUP / "caption_model_comparison/poster_usage/dual_use_catalog"
OUTPUT = TARGET / "development_benchmark"
BASE = "poster_and_candidate_title"
TIE = "top30_consensus_boundary_tie"
SUPPORT = "dual_view_top10_support"
HISTORICAL = "historical_text60"
BLIND_POLICY = "blind_policy_text20"
CONTROL = "described_cross_pool20"
K = 10
REVIEW_GRADES = {"0": 0, "1": 1, "uncertain": None}


class Evidence:
    """Read only registered development data and record verified input hashes."""

    def __init__(self, registry: dict) -> None:
        """Initialize a reader with the project evaluation roles.

        Args:
            registry: Current evaluation access registry.
        """
        self.registry = registry
        self.files: dict[str, str] = {}

    def digest(self, path: Path, expected: str | None = None) -> str:
        """Record the digest of one registered input.

        Args:
            path: Workspace input path.
            expected: Prior receipt digest when available.

        Returns:
            SHA256 digest of the registered file.
        """
        require_data_access([path], "development", self.registry)
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        actual = self.files.get(relative)
        if actual is None:
            actual = sha256_file(path)
            self.files[relative] = actual
        if expected is not None and expected != actual:
            raise ValueError(f"Development input differs from its receipt: {relative}")
        return actual

    def json(self, path: Path, expected: str | None = None) -> Any:
        """Read registered JSON after checking its receipt.

        Args:
            path: Workspace JSON file.
            expected: Prior receipt digest when available.

        Returns:
            Decoded JSON content.
        """
        self.digest(path, expected)
        return load(path)

    def labels(self, path: Path, expected: str) -> list[dict]:
        """Read the original AI rows without changing their provenance.

        Args:
            path: AI judgment JSONL file.
            expected: Prior receipt digest.

        Returns:
            Decoded judgment rows.
        """
        self.digest(path, expected)
        return [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]


def _support_order(base: list[str], whole: list[str], region: list[str]) -> list[str]:
    if len(base) < 2 * K or any(
        len(view) != len(base) or set(view) != set(base) for view in (whole, region)
    ):
        raise ValueError("Ranking views must cover the same full gallery")
    incumbent = base[K - 1]
    whole_head, region_head = set(whole[:K]), set(region[:K])
    if incumbent in whole_head or incumbent in region_head:
        return base
    shared = whole_head & region_head
    for rank, cid in enumerate(base[K : 2 * K], K):
        if cid in shared:
            revised = base.copy()
            revised[K - 1], revised[rank] = revised[rank], incumbent
            return revised
    return base


def _balanced(evidence: Evidence) -> tuple[dict, list[dict], list[dict], list[dict]]:
    selection = evidence.json(TARGET / "selection.json")["inputs"]
    predictions = evidence.json(BALANCED / "prediction_receipt.json")["files"]
    boundary = evidence.json(TARGET / "recall_boundary/rankings.json")
    summary = evidence.json(TARGET / "poster_text_replay/summary.json")["files"]
    prior = evidence.json(
        TARGET / "poster_text_replay/rankings.json",
        summary["poster_text_replay/rankings.json"],
    )
    rankings, queries, catalog = {}, [], []
    labels: list[dict] = []
    for cohort in COHORTS:
        root = BALANCED / cohort
        prefix = root.relative_to(PROJECT_ROOT).as_posix()
        docs = evidence.json(
            root / "catalog.public.json", selection[f"{prefix}/catalog.public.json"]
        )
        catalog.extend(docs)
        ids = {item["id"] for item in docs}
        source = evidence.json(
            root / "queries.public.json", selection[f"{prefix}/queries.public.json"]
        )
        views = evidence.json(
            root / "predictions.json", predictions[f"{prefix}/predictions.json"]
        )
        for query in source:
            qid = query["id"]
            key = f"{cohort}/{qid}"
            queries.append(dict(query, id=key, cohort=cohort))
            base = prior[key][BASE]
            if set(base) != ids or base != boundary[key][BASE]:
                raise ValueError(f"Baseline/gallery mismatch: {key}")
            rankings[key] = {
                BASE: base,
                TIE: boundary[key][TIE],
                SUPPORT: (
                    base
                    if query["text"]
                    else _support_order(
                        base, views[qid]["global"], views[qid]["region_title"]
                    )
                ),
            }
        labels.extend(
            dict(row, query_id=f"{cohort}/{row['query_id']}")
            for row in evidence.labels(
                root / "judgments.ai.jsonl", selection[f"{prefix}/judgments.ai.jsonl"]
            )
        )
    return rankings, labels, queries, catalog


def _historical(evidence: Evidence) -> tuple[dict, list[dict], list[dict], list[dict]]:
    receipt = FOCUS / "completion_receipt.json"
    pins = evidence.json(receipt)["files"]
    source = evidence.json(FOCUS / "inputs.public.json", pins["inputs.public.json"])
    archived = evidence.json(FOCUS / "rankings.json", pins["rankings.json"])
    replay_receipt = evidence.json(FOLLOWUP / "results.json")
    evidence.digest(receipt, replay_receipt["input_receipt_sha256"])
    replay = evidence.json(
        FOLLOWUP / "rankings.json", replay_receipt["rankings_sha256"]
    )
    rankings, galleries = {}, {}
    for query in source["queries"]:
        qid = query["id"]
        old = archived[qid]
        keys = (HISTORICAL, BLIND_POLICY)
        baselines = {key: replay[qid][key] for key in keys}
        orders = {
            f"{key}_{SUPPORT}": (
                base
                if query["text"]
                else _support_order(base, old["global"], old["region_title"])
            )
            for key, base in baselines.items()
        }
        rankings[qid] = {**baselines, **orders}
        galleries[qid] = set(baselines[HISTORICAL])
        expected = {item["id"] for item in source["catalog"]} - {query["exclude_id"]}
        if galleries[qid] != expected:
            raise ValueError(f"Historical gallery differs from the catalog: {qid}")
    labels = [
        row
        for row in evidence.labels(
            FOCUS / "judgments.ai.jsonl", pins["judgments.ai.jsonl"]
        )
        if row["candidate_id"] in galleries[row["query_id"]]
    ]
    queries = [dict(row, cohort="focus") for row in source["queries"]]
    return rankings, labels, queries, source["catalog"]


def _stress(evidence: Evidence) -> tuple[dict, list[dict], list[dict]]:
    pins = evidence.json(STRESS / "inference_protocol.json")["files"]
    receipt = evidence.json(STRESS / "evaluation_receipt.json")["files"]
    qpath, lpath = STRESS / "queries.public.json", STRESS / "judgments.ai.jsonl"
    queries = evidence.json(qpath, pins[qpath.relative_to(PROJECT_ROOT).as_posix()])
    labels = evidence.labels(lpath, pins[lpath.relative_to(PROJECT_ROOT).as_posix()])
    archived = evidence.json(STRESS / "rankings.json", receipt["rankings.json"])
    rankings = {
        qid: {
            CONTROL: row[CONTROL],
            SUPPORT: _support_order(row[CONTROL], row["global"], row["region_title"]),
        }
        for qid, row in archived.items()
    }
    return rankings, labels, [dict(row, cohort="stress") for row in queries]


def _report(
    rankings: dict, labels: list[dict], queries: list[dict], comparisons: list
) -> dict:
    modes = {row["id"]: "text" if row["text"] else "image" for row in queries}
    metrics = evaluate_intervals(rankings, labels, modes)
    paired = paired_querywise(rankings, labels, modes, comparisons)
    return dict(metrics=metrics, paired=paired)


def _human_snapshot(evidence: Evidence) -> dict[tuple[str, str], int | None]:
    files = evidence.json(TARGET / "poster_text_replay/summary.json")["files"]
    path = "subject_challengers/results.json"
    snapshot = evidence.json(TARGET / path, files[path])["human_review_diagnostic"]
    grades = {}
    for group in snapshot.values():
        for row in group["records"]:
            key = row["query"], row["candidate_id"]
            if key in grades:
                raise ValueError(f"Duplicate human decision: {key}")
            grades[key] = REVIEW_GRADES[row["human_grade"]]
    return grades


def _mixed(
    rankings: dict, labels: list[dict], queries: list[dict], human: dict, contract: dict
) -> dict:
    grades: dict[str, dict] = {key: {} for key in rankings}
    for row in labels:
        key = row["query_id"], row["candidate_id"]
        grades[key[0]][key[1]] = human.get(key, row["grade"])
    if any(cid not in grades[qid] for qid, cid in human):
        raise ValueError("Human judgment is outside the evaluation gallery")
    rows = {
        qid: {name: query_bounds(order, grades[qid]) for name, order in methods.items()}
        for qid, methods in rankings.items()
    }
    output = {}
    for scope in contract["report_scopes"]:
        keys = [
            q["id"]
            for q in queries
            if scope in ("all", "text" if q["text"] else "image")
        ]
        methods = {}
        for name in next(iter(rankings.values())):
            values = [rows[qid][name] for qid in keys]
            ranges = {
                metric: [
                    sum(row["ranges"][metric][i] for row in values) / len(values)
                    for i in range(2)
                ]
                for metric in METRICS
                if metric != "recall@10"
            }
            ranges["recall@10"] = list(
                optional_mean_bounds(
                    [
                        tuple(row["ranges"]["recall@10"])
                        for row in values
                        if row["known_positive"]
                    ],
                    [
                        tuple(row["ranges"]["recall@10"])
                        for row in values
                        if not row["known_positive"] and row["unknown"]
                    ],
                )
            )
            methods[name] = ranges
        output[scope] = dict(
            queries=len(keys),
            methods=methods,
            capacity=oracle_capacity([rows[qid][BASE] for qid in keys], contract),
        )
    return dict(
        scopes=output, human_pairs=len(human), other_pairs=len(labels) - len(human)
    )


def _media(evidence: Evidence, value: str) -> dict:
    path = project_path(value)
    return dict(
        image_path=path.relative_to(PROJECT_ROOT).as_posix(),
        image_sha256=evidence.digest(path),
    )


def _union(evidence: Evidence, panels: list, human: dict) -> dict:
    catalog: dict = {}
    queries: dict = {}
    grades: dict[tuple[str, str], list[tuple[bool, int | None]]] = {}
    source_to_key = {}
    for documents, source, labels in panels:
        for row in documents:
            cid = row["id"]
            item = dict(
                id=cid, title=row["title"], **_media(evidence, row["image_path"])
            )
            old = catalog.setdefault(cid, item)
            if any(old[field] != item[field] for field in ("title", "image_sha256")):
                raise ValueError(f"Conflicting source product identity: {cid}")
        aliases = {}
        for row in source:
            if row["text"]:
                text_key = " ".join(normalize("NFKC", row["text"]).casefold().split())
                key = f"text/{text_key}"
                item = dict(text=row["text"], mode="text")
            else:
                media = _media(evidence, row["image_path"])
                key = f"image/{media['image_sha256']}"
                item = dict(mode="image", **media)
            alias = (
                f"{row['cohort']}/{row['id']}"
                if row["cohort"] in ("focus", "stress", "dual")
                else row["id"]
            )
            aliases[row["id"]] = key
            source_to_key[alias] = key
            query = queries.setdefault(
                key,
                dict(
                    item,
                    sources=[],
                    exclusions=[],
                    judgments={},
                    unresolved_pairs=0,
                ),
            )
            query["sources"].append(alias)
            if row.get("exclude_id"):
                query["exclusions"].append(row["exclude_id"])
        for row in labels:
            qid, cid = row["query_id"], row["candidate_id"]
            grades.setdefault((aliases[qid], cid), []).append((False, row["grade"]))
    for (alias, cid), grade in human.items():
        pair = source_to_key[alias], cid
        grades.setdefault(pair, []).append((True, grade))
    counts: Counter[str] = Counter()
    for (key, cid), values in grades.items():
        if cid not in catalog:
            raise ValueError(f"Judgment outside the combined catalog: {cid}")
        query = queries[key]
        if cid in query["exclusions"]:
            continue
        human_values = [grade for reviewed, grade in values if reviewed]
        binary = {grade for _, grade in values if grade is not None}
        counts["known_pairs"] += 1
        counts["conflicting_binary_pairs"] += len(binary) > 1
        accepted = set(human_values) if human_values else binary
        grade = next(iter(accepted)) if len(accepted) == 1 else None
        unresolved = grade is None
        counts["unresolved_pairs"] += unresolved
        query["judgments"][cid] = dict(grade=grade, human=bool(human_values))
        query["unresolved_pairs"] += unresolved
    for query in queries.values():
        query["exclusions"] = sorted(set(query["exclusions"]))
        query["expected_pairs"] = len(catalog.keys() - set(query["exclusions"]))
        query["observed_pairs"] = len(query["judgments"])
        query["missing_pairs"] = query["expected_pairs"] - query["observed_pairs"]
    expected = sum(row["expected_pairs"] for row in queries.values())
    image_counts = Counter(item["image_sha256"] for item in catalog.values())
    return dict(
        products=len(catalog),
        unique_product_image_bytes=len(image_counts),
        queries=len(queries),
        modes=dict(Counter(item["mode"] for item in queries.values())),
        original_query_instances=sum(len(item["sources"]) for item in queries.values()),
        expected_pairs=expected,
        observed_pairs=counts["known_pairs"],
        missing_pairs=expected - counts["known_pairs"],
        unresolved_pairs=counts["unresolved_pairs"],
        conflicting_binary_pairs=counts["conflicting_binary_pairs"],
        catalog=sorted(catalog.values(), key=lambda row: row["id"]),
        query_manifest=queries,
    )


def main() -> None:
    """Run the label-free development challengers and shared-gallery coverage audit."""
    config = PROJECT_ROOT / "configs"
    evidence = Evidence(load(config / "evaluation_registry.json"))
    contract = load(config / "module2_target_p80_r90.json")
    if contract["k"] != K:
        raise ValueError("Benchmark requires the fixed Top10 contract")
    br, bl, bq, bc = _balanced(evidence)
    hr, hl, hq, hc = _historical(evidence)
    sr, sl, sq = _stress(evidence)
    human = _human_snapshot(evidence)
    recall = evidence.json(DUAL / "recall_results.json")
    dual = evidence.json(DUAL / "recall_protocol.json", recall["protocol_sha256"])
    score = evidence.json(DUAL / "rerank_results.json")
    audit = evidence.json(
        DUAL / "human_review/adjudication.json", score["human_review_sha256"]
    )
    query = dual["query"]
    if _media(evidence, query["image"])["image_sha256"] != query["image_sha256"]:
        raise ValueError("Dual-use query image changed")
    dq = [dict(id=query["qid"], cohort="dual", image_path=query["image"], text=None)]
    dc = [dict(row, image_path=row["image"]) for row in dual["candidates"]]
    union_human = {
        (f"dual/{query['qid']}", cid): REVIEW_GRADES[row["grade"]]
        for cid, row in audit["rows"].items()
    }
    shared = _union(
        evidence,
        [(bc, bq, bl), (hc, hq, hl), ([], sq, sl), (dc, dq, [])],
        {**human, **union_human},
    )
    bcomp = [(TIE, BASE), (SUPPORT, BASE), (SUPPORT, TIE)]
    hcomp = [(f"{route}_{SUPPORT}", route) for route in (HISTORICAL, BLIND_POLICY)]
    same_gallery = [row for row in bq if row["cohort"] in ("tune", "fresh")]
    common_ids = {row["id"] for row in same_gallery}
    if len({frozenset(br[qid][BASE]) for qid in common_ids}) != 1:
        raise ValueError("Shared 315-item panel must use exactly the same products")
    reports = dict(
        original_315_panels=_report(br, bl, bq, bcomp),
        shared_315_panel=_report(
            {qid: br[qid] for qid in sorted(common_ids)},
            [row for row in bl if row["query_id"] in common_ids],
            same_gallery,
            bcomp,
        ),
        historical_418=_report(hr, hl, hq, hcomp),
        stress_418=_report(sr, sl, sq, [(SUPPORT, CONTROL)]),
        partial_human_315=_mixed(br, bl, bq, human, contract),
    )
    write_json(OUTPUT / "shared_gallery.json", shared)
    write_json(OUTPUT / "rankings.json", dict(balanced=br, historical=hr, stress=sr))
    write_json(
        OUTPUT / "results.json",
        dict(
            role="exposed_development",
            k=K,
            targets=contract,
            policy=(
                "Image only; keep ranks 1–9. Replace rank 10 only if absent from both "
                + "retrieval heads and the first eligible rank 11–20 is Top10 in both."
            ),
            aggregation="Macro over queries; Recall uses all relevant gallery items.",
            shared_gallery_sha256=sha256_file(OUTPUT / "shared_gallery.json"),
            rankings_sha256=sha256_file(OUTPUT / "rankings.json"),
            files=evidence.files,
            source_sha256=sha256_file(Path(__file__)),
            panels=reports,
            new_model_calls=0,
            formal_metrics=None,
        ),
    )
    print(
        f"Verified {shared['queries']} unique queries and {shared['products']} products."
    )


if __name__ == "__main__":
    main()
