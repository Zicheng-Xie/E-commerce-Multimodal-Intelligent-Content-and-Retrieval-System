"""Test original-poster text as a corroborating subject cue on the Mac."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

import run  # noqa: F401
from module2_subject_challengers import BASE, EXPANDED, TARGET
from module2_subject_pool_evaluation import blend, corroborated_profiles
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from paper_fusion_experiment import COHORTS, DATA, GROUND, ROLE, SIMPLE, load
from primary_subject_ranker import penalty, signature
from querywise_paired_bounds import paired_querywise

ROOT = TARGET / "poster_text_replay"
SWIFT = PROJECT_ROOT / "scripts/module2_poster_ocr.swift"
POSTER = "poster_corroborated_query"
TITLE = "poster_and_candidate_title"
CUE = re.compile(r"^(?:专属|专用|替换)(?:的)?\s*(.{2,20})$")


def read_posters(jobs: list[dict], registry: dict) -> dict:
    """Read the existing original query images with local Apple Vision OCR.

    Args:
        jobs: Frozen development queries and original image hashes.
        registry: Data-role rules.

    Returns:
        Observed text lines with recognition confidence per query.
    """
    paths = [Path(job["query"]["image_path"]) for job in jobs]
    require_data_access(paths, "development", registry)
    for path, job in zip(paths, jobs, strict=True):
        if sha256_file(path) != job["query"]["source_sha256"]:
            raise ValueError("Original OCR input changed")
    with tempfile.TemporaryDirectory(dir=ROOT) as folder:
        directory = Path(folder)
        binary = directory / "poster-ocr"
        subprocess.run(
            [
                "swiftc",
                "-module-cache-path",
                str(directory / "cache"),
                "-o",
                str(binary),
                str(SWIFT),
            ],
            check=True,
        )
        run_result = subprocess.run(
            [str(binary)],
            input=json.dumps([str(path) for path in paths]),
            text=True,
            capture_output=True,
            check=True,
        )
    observations = json.loads(run_result.stdout)
    if any(row.get("error") for row in observations):
        raise ValueError("At least one original query could not be read by Vision")
    return {
        f"{job['cohort']}/{job['query']['id']}": record["observations"]
        for job, record in zip(jobs, observations, strict=True)
    }


def kind(profile: dict) -> tuple[str | None, str | None]:
    """Return family and subtype from the existing subject taxonomy.

    Args:
        profile: Subject signature from one evidence source.

    Returns:
        A two-component type identity.
    """
    return profile["family"], profile["subtype"]


def subject_hints() -> tuple[dict, dict]:
    """Load original-image observations as separate sources for agreement checks.

    Returns:
        Role and semantic maps keyed by record ID with original input hashes.
    """
    role_hashes = {
        row["key"]: row["source_sha256"]
        for row in load(ROLE / "profile_inputs.public.json")
    }
    ground_hashes = {
        row["key"]: row["source_sha256"]
        for row in load(GROUND / "views_inputs.public.json")
    }
    role = {
        row["key"]: (
            role_hashes[row["key"]],
            signature(
                row["value"]["sale_item"], uncertain=row["value"]["primary_unclear"]
            ),
        )
        for row in load(ROLE / "profiles.json")
    }
    ground = {
        row["key"]: (
            ground_hashes[row["key"]],
            signature(row["native_value"]["sale_item"]),
        )
        for row in load(GROUND / "semantic_results.json")
        if row.get("native_value")
        and row["native_value"].get("box_correct_sale_item") is True
    }
    return role, ground


def query_signature(lines: list[dict], broad: dict, visual: dict) -> dict:
    """Resolve a subtype only when poster text and a same-image observation agree.

    Args:
        lines: OCR lines of the original query.
        broad: Existing corroborated query signature.
        visual: Same-image semantic observation.

    Returns:
        Original signature or its independently supported finer subtype.
    """
    if broad["subtype"] is not None:
        return broad
    hints = set()
    for row in lines:
        match = CUE.fullmatch(row["text"].strip())
        if match and row["confidence"] >= 0.70:
            detected = kind(signature(match.group(1)))
            if (
                detected[0] == broad["family"]
                and detected[1]
                and detected == kind(visual)
            ):
                hints.add(detected)
    if len(hints) == 1:
        _, subtype = hints.pop()
        return {**broad, "subtype": subtype}
    return broad


def candidate_signature(document: dict, base: dict, observed: list[dict]) -> dict:
    """Refine ambiguous candidate types with a corroborated original title.

    Args:
        document: Original candidate image identity and merchant title.
        base: Existing two-observation signature.
        observed: Same-image original observations eligible for support.

    Returns:
        Conservative original or title-corroborated subject signature.
    """
    title = signature(document["title"])
    if title["family"] is None or kind(title) not in {kind(row) for row in observed}:
        return base
    if any(
        row["family"] is not None
        and row["family"] != title["family"]
        or row["subtype"] is not None
        and row["subtype"] != title["subtype"]
        for row in observed
    ):
        return base
    return title if base["family"] is None or base["subtype"] is None else base


def main() -> None:
    """Measure original-image OCR and candidate-title evidence as separate ablations."""
    registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
    labels_paths = [DATA / c / "judgments.ai.jsonl" for c in COHORTS]
    files = [
        EXPANDED / "summary.json",
        EXPANDED / "sid_protocol.json",
        EXPANDED / "rankings.json",
        EXPANDED / "subject_evidence.json",
        TARGET / "rankings.json",
        TARGET / "target_status.json",
        TARGET / "selection.json",
        SIMPLE / "inputs.json",
        ROLE / "profile_inputs.public.json",
        ROLE / "profiles.json",
        GROUND / "views_inputs.public.json",
        GROUND / "semantic_results.json",
        TARGET / "human_review/adjudications.json",
        *labels_paths,
        *sorted((EXPANDED / "sid_pairs").glob("*/*.json")),
    ]
    require_data_access(files, "development", registry)
    receipt = load(EXPANDED / "summary.json")
    for name, digest in receipt["files"].items():
        if sha256_file(EXPANDED / name) != digest:
            raise ValueError(f"Previous experiment changed: {name}")
    if (
        sha256_file(TARGET / "rankings.json")
        != load(TARGET / "target_status.json")["files"]["rankings.json"]
    ):
        raise ValueError("Cross20 support ranking changed")
    label_receipt = load(TARGET / "selection.json")["inputs"]
    for path in labels_paths:
        if sha256_file(path) != label_receipt[str(path.relative_to(PROJECT_ROOT))]:
            raise ValueError("Frozen AI judgments changed")
    jobs = load(EXPANDED / "sid_protocol.json")["jobs"]
    previous = load(EXPANDED / "rankings.json")
    support = load(TARGET / "rankings.json")
    inputs = load(SIMPLE / "inputs.json")
    base_profiles = corroborated_profiles()
    role, ground = subject_hints()
    unknown = signature("")
    ROOT.mkdir(parents=True, exist_ok=True)
    ocr = read_posters(jobs, registry)
    rankings = {
        f"{c}/{q}": {name: row[BASE] for name in (BASE, POSTER, TITLE)}
        for c, group in previous.items()
        for q, row in group.items()
    }
    refined = []
    title_fallbacks = 0
    for job in jobs:
        c, q = job["cohort"], job["query"]["id"]
        key = f"{c}/{q}"
        query = base_profiles.get(key, unknown)
        observation = ground.get(key)
        subject = query
        if observation and observation[0] == job["query"]["source_sha256"]:
            subject = query_signature(ocr[key], query, observation[1])
        if kind(subject) != kind(query):
            refined.append(key)
        cache = load(EXPANDED / "sid_pairs" / c / f"{q}.json")
        if cache["identity"] != dict(
            protocol_sha256=sha256_file(EXPANDED / "sid_protocol.json"),
            query_sha256=job["query"]["source_sha256"],
            extra=job["extra"],
        ) or set(cache["scores"]) != set(job["extra"]):
            raise ValueError("Real Qwen score cache identity changed")
        scores = {**job["original_scores"], **cache["scores"]}
        base = rankings[key][BASE]
        cross = support[c][q]["cross20"]
        if (
            set(scores) != set(base[:30])
            or blend(
                base,
                cross,
                scores,
                {
                    cid: penalty(query, base_profiles.get(cid, unknown))
                    for cid in scores
                },
            )
            != base
        ):
            raise ValueError("Original corroborated subject ranking replay differed")
        poster_offsets = {}
        title_offsets = {}
        for cid in scores:
            doc = inputs["documents"][cid]
            standard = base_profiles.get(cid, unknown)
            sources = [
                value[1]
                for value in [role.get(cid), ground.get(cid)]
                if value and value[0] == doc["source_sha256"]
            ]
            assisted = candidate_signature(doc, standard, sources)
            title_fallbacks += kind(assisted) != kind(standard)
            poster_offsets[cid] = penalty(subject, standard)
            title_offsets[cid] = penalty(subject, assisted)
        rankings[key][POSTER] = blend(base, cross, scores, poster_offsets)
        rankings[key][TITLE] = blend(base, cross, scores, title_offsets)
    labels = []
    for path in labels_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            labels.append(dict(row, query_id=f"{path.parent.name}/{row['query_id']}"))
    modes = {key: "image" if "/image_" in key else "text" for key in rankings}
    absolute = evaluate_intervals(rankings, labels, modes)
    paired = paired_querywise(
        rankings, labels, modes, [(POSTER, BASE), (TITLE, BASE), (TITLE, POSTER)]
    )
    human: dict[str, dict] = {}
    for row in load(TARGET / "human_review/adjudications.json"):
        key = f"{row['cohort']}/{row['query_id']}"
        positions = human.setdefault(key, {})
        positions[row["candidate_id"]] = dict(
            grade=row["human_grade"],
            ranks={
                method: order.index(row["candidate_id"]) + 1
                for method, order in rankings[key].items()
            },
        )
    write_json(ROOT / "ocr.json", ocr)
    write_json(ROOT / "rankings.json", rankings)
    write_json(
        ROOT / "results.json",
        dict(
            actual_local_OCR_queries=len(ocr),
            query_subtypes_refined=refined,
            title_refined_candidate_pairs=title_fallbacks,
            new_Qwen_pairs=0,
            AI_development=absolute,
            paired_AI_development=paired,
            targeted_review_diagnostic=human,
            formal_metrics=None,
            inputs={
                str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
                for path in files + [SWIFT, Path(__file__).resolve()]
            },
        ),
    )
    print(
        f"OCR {len(ocr)} queries; refined query subjects {len(refined)}; title pairs {title_fallbacks}"
    )


if __name__ == "__main__":
    main()
