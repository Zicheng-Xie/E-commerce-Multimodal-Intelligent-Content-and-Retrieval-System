"""Recompute six weekly metrics from complete frozen AI evidence, offline."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import run  # noqa: F401 - initialize this checkout before importing its package
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.safety import PROJECT_ROOT


def load_case(
    name: str, root: Path | None = None
) -> tuple[dict, list[dict], dict, dict]:
    """Decode a complete ranking and AI label bank without filling unknowns.

    Args:
        name: Frozen case identifier in the weekly evidence manifest.
        root: Checkout containing evidence; defaults to this project.

    Returns:
        Full rankings, grades with normalized AI provenance, modes and expected scopes.

    Raises:
        ValueError: The digest, schema, dimensions or ranking is invalid.
    """
    directory = (root or PROJECT_ROOT) / "evidence"
    raw = (directory / "weekly_replay.json.gz").read_bytes()
    provenance = json.loads((directory / "provenance.json").read_text())
    if hashlib.sha256(raw).hexdigest() != provenance["artifact_sha256"]:
        raise ValueError("Weekly evidence checksum mismatch")
    data = json.loads(gzip.decompress(raw))
    if data["schema_version"] != 1 or data["fixed_k"] != 10:
        raise ValueError("Unsupported evidence schema or changed K")
    if name not in data["cases"]:
        raise ValueError(f"Unknown frozen case: {name}")
    case = data["cases"][name]
    bank = data["label_banks"][case["label_bank"]]
    gallery, queries = bank["gallery_ids"], bank["query_ids"]
    if len(gallery) != len(set(gallery)) or len(queries) != len(bank["grades"]):
        raise ValueError("Duplicate gallery or incomplete label matrix")
    labels: list[dict] = []
    for query, grades in zip(queries, bank["grades"]):
        if len(grades) != len(gallery):
            raise ValueError("Judgments must cover the entire gallery")
        labels.extend(
            dict(
                query_id=query,
                candidate_id=cid,
                grade=grade,
                review_origin="model",
                human_label=False,
            )
            for cid, grade in zip(gallery, grades)
        )
    rankings: dict = {}
    for query, methods in case["rankings"].items():
        rankings[query] = {}
        for method, reference in methods.items():
            vector = data["orders"][reference]
            if (
                len(vector) != len(gallery)
                or len(vector) != len(set(vector))
                or any(type(i) is not int or not 0 <= i < len(gallery) for i in vector)
            ):
                raise ValueError("Every method needs a complete unique gallery order")
            rankings[query][method] = [gallery[i] for i in vector]
    return rankings, labels, bank["modes"], case["expected_scopes"]


def verify_case(name: str, root: Path | None = None) -> dict:
    """Recompute a frozen case and compare all bounds, denominators and flags.

    Args:
        name: Frozen blind or development case to replay.
        root: Optional checkout for isolated replay or integrity tests.

    Returns:
        Dimensions and independently recomputed six-metric scopes.

    Raises:
        ValueError: Any method, scope, bound, denominator or target flag differs.
    """
    rankings, labels, modes, expected = load_case(name, root)
    actual = evaluate_intervals(rankings, labels, modes)["scopes"]
    if set(actual) != set(expected):
        raise ValueError("Metric scope coverage differs")
    for scope, methods in expected.items():
        if set(methods) != set(actual[scope]):
            raise ValueError("Method coverage differs")
        for method, reference in methods.items():
            computed = actual[scope][method]
            if set(computed) != set(reference):
                raise ValueError("Metric or denominator fields differ")
            for field, value in reference.items():
                if field != "ranges":
                    if computed[field] != value:
                        raise ValueError(f"Changed {name}/{scope}/{method}/{field}")
                    continue
                if set(computed[field]) != set(value):
                    raise ValueError("Six-metric coverage differs")
                for metric, bounds in value.items():
                    for measured, recorded in zip(computed[field][metric], bounds):
                        equal = (
                            (measured is recorded)
                            if measured is None or recorded is None
                            else math.isclose(
                                measured, recorded, rel_tol=0, abs_tol=1e-12
                            )
                        )
                        if not equal:
                            raise ValueError(
                                f"Changed {name}/{scope}/{method}/{metric}"
                            )
    return dict(
        case=name,
        queries=len(rankings),
        relations=len(labels),
        complete_rankings=sum(len(m) for m in rankings.values()),
        unknown=sum(r["grade"] is None for r in labels),
        scopes=actual,
        verified=True,
        human_labels=0,
        formal_metrics=None,
    )


def main() -> None:
    """Replay all weekly cases, or one selected case, without model loading."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case")
    parser.add_argument("--evidence-root", type=Path, default=PROJECT_ROOT)
    options = parser.parse_args()
    artifact = options.evidence_root / "evidence/weekly_replay.json.gz"
    if not artifact.is_file():
        parser.error(
            "Private weekly evidence is required; supply --evidence-root after obtaining authorization"
        )
    data = json.loads(gzip.decompress(artifact.read_bytes()))
    names = [options.case] if options.case else list(data["cases"])
    results = [verify_case(name, options.evidence_root) for name in names]
    output = PROJECT_ROOT / "artifacts/weekly_replay/results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                verified_cases=len(results),
                complete_rankings=sum(r["complete_rankings"] for r in results),
                report=str(output),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
