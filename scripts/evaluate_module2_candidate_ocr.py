"""Compare candidate-side OCR evidence on the same completely scored Top30."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import run  # noqa: F401
from module2_candidate_ocr import ROOT
from module2_poster_text_replay import (
    CUE,
    EXPANDED,
)
from module2_poster_text_replay import ROOT as PREVIOUS
from module2_poster_text_replay import (
    SIMPLE,
    TARGET,
    TITLE,
    candidate_signature,
    kind,
    query_signature,
    subject_hints,
)
from module2_subject_pool_evaluation import blend, corroborated_profiles
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from paper_fusion_experiment import COHORTS, DATA, load
from primary_subject_ranker import NOUNS, penalty, signature
from querywise_paired_bounds import paired_querywise

FILL = "candidate_ocr_fill"
OFFER = "candidate_ocr_offer_override"
PHRASE = "candidate_ocr_short_phrase"
CONFLICT = "candidate_ocr_confirmed_conflict"
METHODS = (FILL, OFFER, PHRASE, CONFLICT)
CONTROLS = ("global", "region_title")
ALIASES = {alias for _, _, terms in NOUNS for alias in terms}
HOST_CONTEXT = re.compile(r"适用|适配|兼容|不含|不带|仅展示|示意|仅配|支持")


def phrase_alias(text: str) -> str | None:
    """Find a short clear product noun suffix without a host-context cue.

    Args:
        text: Normalized OCR line or an explicit offer's text.

    Returns:
        Longest matching suffix when its type agrees with the full phrase.
    """
    if (
        not 2 <= len(text) <= 20
        or not re.fullmatch(r"[A-Za-z0-9\u3400-\u9fff]+", text)
        or HOST_CONTEXT.search(text)
    ):
        return None
    suffixes = [alias for alias in ALIASES if len(alias) >= 2 and text.endswith(alias)]
    longest = max(suffixes, key=lambda alias: (len(alias), alias), default=None)
    if longest and kind(signature(longest)) == kind(signature(text)):
        return longest
    return None


def ocr_signature(lines: list[dict], qualified: bool = False) -> dict | None:
    """Extract one unambiguous known product noun from reliable OCR lines.

    Args:
        lines: Text and confidence from the original candidate image.
        qualified: Accept short type-consistent descriptive phrases.

    Returns:
        Candidate type with the explicit-offer indicator, or abstention.
    """
    offers: set[tuple[str | None, str | None]] = set()
    literals: set[tuple[str | None, str | None]] = set()
    for line in lines:
        if line["confidence"] < 0.70:
            continue
        text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", line["text"]))
        offer = CUE.fullmatch(text)
        noun = offer.group(1) if offer else text
        if HOST_CONTEXT.search(noun):
            continue
        alias = noun if noun in ALIASES else phrase_alias(noun) if qualified else None
        if alias is not None:
            selected = offers if offer else literals
            selected.add(kind(signature(alias)))
    hints = offers or literals
    if len({family for family, _ in hints}) != 1:
        return None
    specific = {(family, subtype) for family, subtype in hints if subtype is not None}
    if len(specific) > 1:
        return None
    family, subtype = next(iter(specific or hints))
    return dict(family=family, subtype=subtype, structures=[], offer=bool(offers))


def merge_candidate(base: dict, title: dict, ocr: dict | None, override: bool) -> dict:
    """Apply only type evidence corroborated by the merchant title.

    Args:
        base: Previous image-and-title candidate signature.
        title: Type extracted from the original merchant title.
        ocr: Candidate original's unambiguous OCR evidence, if available.
        override: Allow an explicit offer to correct a conflicting prior type.

    Returns:
        Original or corroborated candidate type.
    """
    if ocr is None or title["family"] != ocr["family"]:
        return base
    detail = title["subtype"] if title["subtype"] == ocr["subtype"] else None
    if title["subtype"] and ocr["subtype"] and detail is None:
        return base
    corroborated = dict(family=ocr["family"], subtype=detail, structures=[])
    if base["family"] is None:
        return corroborated
    if (
        base["family"] == ocr["family"]
        and base["subtype"] is None
        and detail is not None
    ):
        return corroborated
    if override and ocr["offer"] and kind(title) == kind(ocr):
        conflict = base["family"] != ocr["family"] or (
            detail is not None and base["subtype"] != detail
        )
        if conflict:
            return corroborated
    return base


def confirmed_penalty(query: dict, subject: dict, evidence: dict) -> dict:
    """Strengthen conflicts supported by both OCR and the merchant title.

    Args:
        query: Earlier corroborated query subject.
        subject: Candidate subject after the short-phrase fill rule.
        evidence: Product type inferred from matching title and OCR alone.

    Returns:
        Existing penalty, optionally increased by a fixed 0.20 rank units.
    """
    original = penalty(query, subject)
    broad_conflict = (
        evidence["family"] is not None
        and evidence["family"] == subject["family"]
        and query["family"] is not None
        and query["family"] != subject["family"]
    )
    incompatible_bag_role = (
        query["family"] == subject["family"] == "bag"
        and evidence["subtype"] is not None
        and kind(evidence) == kind(subject)
        and query["subtype"] is not None
        and query["subtype"] != subject["subtype"]
    )
    if original["value"] > 0 and (broad_conflict or incompatible_bag_role):
        return {**original, "value": original["value"] + 0.20}
    return original


def replay(ocr: dict, jobs: list[dict]) -> tuple[dict, dict]:
    """Replay the earlier Top30 and substitute only candidate subject evidence.

    Args:
        ocr: Verified complete map of attempted candidate OCR results.
        jobs: Frozen existing exact score plan.

    Returns:
        Full rankings and a compact count of changed candidate evidence.
    """
    base_profiles = corroborated_profiles()
    role, ground = subject_hints()
    unknown = signature("")
    doc_inputs = load(SIMPLE / "inputs.json")["documents"]
    prior_ranks = load(PREVIOUS / "rankings.json")
    cross = load(TARGET / "rankings.json")
    query_lines = load(PREVIOUS / "ocr.json")
    signatures = {
        cid: ocr_signature(row["observations"])
        for cid, row in ocr.items()
        if not row.get("error")
    }
    phrases = {
        cid: ocr_signature(row["observations"], qualified=True)
        for cid, row in ocr.items()
        if not row.get("error")
    }
    rankings = {
        key: {
            TITLE: row[TITLE],
            **{name: row[TITLE] for name in METHODS},
            **{
                name: cross[key.split("/")[0]][key.split("/")[1]][name]
                for name in CONTROLS
            },
        }
        for key, row in prior_ranks.items()
    }
    changed: Counter[str] = Counter(
        strict_type_images=sum(value is not None for value in signatures.values()),
        short_phrase_type_images=sum(value is not None for value in phrases.values()),
    )
    for job in jobs:
        c, q = job["cohort"], job["query"]["id"]
        key = f"{c}/{q}"
        query = base_profiles.get(key, unknown)
        same_image = ground.get(key)
        if same_image and same_image[0] == job["query"]["source_sha256"]:
            query = query_signature(query_lines[key], query, same_image[1])
        cache = load(EXPANDED / "sid_pairs" / c / f"{q}.json")
        if cache["identity"] != dict(
            protocol_sha256=sha256_file(EXPANDED / "sid_protocol.json"),
            query_sha256=job["query"]["source_sha256"],
            extra=job["extra"],
        ) or set(cache["scores"]) != set(job["extra"]):
            raise ValueError(
                "Prior exact Qwen scores do not cover the frozen candidate pool"
            )
        scores = {**job["original_scores"], **cache["scores"]}
        baseline = prior_ranks[key][TITLE]
        if len(scores) != 30 or set(scores) != set(baseline[:30]):
            raise ValueError(
                "The candidate pool differs from the previous complete Top30"
            )
        offsets: dict[str, dict] = {name: {} for name in (TITLE, *METHODS)}
        for cid in scores:
            doc = doc_inputs[cid]
            observations = [
                value[1]
                for value in (role.get(cid), ground.get(cid))
                if value and value[0] == doc["source_sha256"]
            ]
            standard = candidate_signature(
                doc, base_profiles.get(cid, unknown), observations
            )
            title = signature(doc["title"])
            phrase = merge_candidate(standard, title, phrases.get(cid), True)
            independent = merge_candidate(unknown, title, phrases.get(cid), False)
            for method, subject in (
                (TITLE, standard),
                (FILL, merge_candidate(standard, title, signatures.get(cid), False)),
                (OFFER, merge_candidate(standard, title, signatures.get(cid), True)),
                (PHRASE, phrase),
            ):
                offsets[method][cid] = penalty(query, subject)
                if method != TITLE and kind(subject) != kind(standard):
                    changed[method] += 1
            offsets[CONFLICT][cid] = confirmed_penalty(query, phrase, independent)
            if offsets[CONFLICT][cid]["value"] > offsets[PHRASE][cid]["value"]:
                changed["strengthened_conflict_pairs"] += 1
        for method in offsets:
            order = blend(baseline, cross[c][q]["cross20"], scores, offsets[method])
            if method == TITLE and order != baseline:
                raise ValueError(
                    "Baseline title-assisted ranking could not be replayed"
                )
            rankings[key][method] = order
    return rankings, dict(changed)


def audit_positions(rankings: dict, source: dict) -> dict:
    """Measure already supplied human judgments independently of AI metrics.

    Args:
        rankings: Fixed complete rankings produced before reading review labels.
        source: Verified previous review diagnostic snapshot.

    Returns:
        Per-design counts and rank movement for the original reviewed relations.
    """
    result = {}
    for design, group in source["human_review_diagnostic"].items():
        records = [
            dict(
                query=row["query"],
                human_grade=row["human_grade"],
                ranks={
                    name: rankings[row["query"]][name].index(row["candidate_id"]) + 1
                    for name in (TITLE, *METHODS)
                },
            )
            for row in group["records"]
        ]
        result[design] = dict(
            reviewed=len(records),
            records=records,
            image_top10={
                name: {
                    grade: sum(
                        "/image_" in row["query"]
                        and row["human_grade"] == grade
                        and row["ranks"][name] <= 10
                        for row in records
                    )
                    for grade in ("0", "1")
                }
                for name in (TITLE, *METHODS)
            },
        )
    return result


def main() -> None:
    """Verify existing data and compare the complete candidate-OCR ablations."""
    registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
    labels_paths = [DATA / c / "judgments.ai.jsonl" for c in COHORTS]
    human_path = TARGET / "subject_challengers/results.json"
    summary = PREVIOUS / "summary.json"
    files = [
        ROOT / "receipt.json",
        ROOT / "protocol.json",
        ROOT / "ocr.json",
        summary,
        PREVIOUS / "results.json",
        PREVIOUS / "rankings.json",
        PREVIOUS / "ocr.json",
        EXPANDED / "sid_protocol.json",
        SIMPLE / "inputs.json",
        TARGET / "rankings.json",
        *labels_paths,
        *sorted((EXPANDED / "sid_pairs").glob("*/*.json")),
        human_path,
    ]
    require_data_access(files, "development", registry)
    for name, digest in load(ROOT / "receipt.json")["files"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Frozen candidate OCR artifact changed")
    for name, digest in load(summary)["files"].items():
        if sha256_file(TARGET / name) != digest:
            raise ValueError("Previous OCR or human-review result changed")
    for name, digest in load(PREVIOUS / "results.json")["inputs"].items():
        if sha256_file(PROJECT_ROOT / name) != digest:
            raise ValueError(f"Prior score or development input changed: {name}")
    protocol = load(ROOT / "protocol.json")
    ocr = load(ROOT / "ocr.json")
    if set(ocr) != {row["id"] for row in protocol["items"]}:
        raise ValueError("OCR did not attempt every original candidate image")
    rankings, changed = replay(ocr, load(EXPANDED / "sid_protocol.json")["jobs"])
    labels = []
    for path in labels_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            labels.append(dict(row, query_id=f"{path.parent.name}/{row['query_id']}"))
    modes = {key: "image" if "/image_" in key else "text" for key in rankings}
    metrics = evaluate_intervals(rankings, labels, modes)
    pairs = [(name, old) for name in METHODS for old in (TITLE, *CONTROLS)]
    paired = paired_querywise(
        rankings, labels, modes, [*pairs, (OFFER, FILL), (CONFLICT, PHRASE)]
    )
    human = audit_positions(rankings, load(human_path))
    write_json(ROOT / "rankings.json", rankings)
    write_json(
        ROOT / "results.json",
        dict(
            OCR=load(ROOT / "receipt.json"),
            evidence_counts=changed,
            AI_development=metrics,
            paired_AI_development=paired,
            human_review_diagnostic=human,
            real_reused_Qwen_scores=720,
            new_Qwen_pairs=0,
            frozen_acceptance_model_calls=0,
            formal_metrics=None,
            inputs={
                str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in files
            },
            evaluation_source_sha256=sha256_file(Path(__file__)),
        ),
    )
    print(f"Replayed {len(rankings)} queries, candidate changes {changed}.")


if __name__ == "__main__":
    main()
