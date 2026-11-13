"""Project frozen public model-stage caches into the general ranker, without labels."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import run  # noqa: F401
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from multimodal_ecommerce.selected_ranking import finish_ranking, select_pool
from multimodal_ecommerce.selected_runtime_io import RuntimeAccess, atomic_json, load

ROOT = PROJECT_ROOT / "artifacts/module2_p80_r90_20261106/development_benchmark"
EXPERIMENT = ROOT / "subject_experiment"
OUTPUT = PROJECT_ROOT / "artifacts/module2_selected_runtime_20261110/replay"


def main() -> None:
    """Verify all selected ranks from coarse/model observations, not final predictions."""
    access = RuntimeAccess()
    inputs = {}

    def read(path: Path) -> Any:
        """Read an explicitly allowed development cache and record its checksum.

        Args:
            path: Known development cache file.

        Returns:
            Decoded cache document.
        """
        access.check(path)
        inputs[str(path.relative_to(PROJECT_ROOT))] = sha256_file(path)
        return load(path)

    selected = read(
        PROJECT_ROOT / "configs/module2_selected_development_candidate.json"
    )
    for name, expected in selected["model_protocol_sha256"].items():
        if sha256_file(PROJECT_ROOT / name) != expected:
            raise ValueError("Parent model-stage protocol changed")
    shared = read(ROOT / "shared_gallery.json")
    catalog = {r["id"]: r["title"] for r in shared["catalog"]}
    crop = read(EXPERIMENT / "crop_rankings.json")
    crop_receipt = read(EXPERIMENT / "crop_results.json")
    if (
        sha256_file(EXPERIMENT / "crop_rankings.json")
        != crop_receipt["rankings_sha256"]
    ):
        raise ValueError("Coarse retrieval cache changed")
    subject = read(EXPERIMENT / "subject_recall/protocol.json")
    primary = read(EXPERIMENT / "primary_recall/protocol.json")
    captions = read(EXPERIMENT / "subject_recall/query_captions.json")["rows"]
    actual, traces = {}, {}
    for key, original in shared["query_manifest"].items():
        query = {k: original[k] for k in ("mode", "text") if k in original}
        mode = query["mode"]
        base = crop["whole_subject_equal_fusion" if mode == "text" else "whole"][key]
        if mode == "text":
            job, directory = subject["jobs"][key], EXPERIMENT / "subject_recall"
            obs = dict(title_order=job["proposed"][10:20])
            if job["baseline"] != base or job["query_text"] != query["text"]:
                raise ValueError(
                    "Cached coarse/text query differs from its public input"
                )
        else:
            job, directory = primary["jobs"][key], EXPERIMENT / "primary_recall"
            obs = dict(
                caption=captions[key]["value"],
                supported_types=primary["candidate_type_support"],
            )
            if job["original"] != base:
                raise ValueError("Cached image coarse ranking changed")
        proposal, active = select_pool(base, query, obs)
        if proposal[:20] != job["proposed"][:20]:
            raise ValueError(
                "General Top20 selection does not reproduce model-stage cache"
            )
        if mode == "text" or active:
            saved = read(
                directory
                / "scores"
                / (hashlib.sha256(key.encode()).hexdigest() + ".json")
            )
            if saved["identity"] != sha256_file(directory / "protocol.json") or set(
                saved["scores"]
            ) != set(proposal[:20]):
                raise ValueError("Cross logits differ from their exact frozen inputs")
            obs["scores"] = saved["scores"]
        else:
            obs["scores"] = {}
        actual[key], traces[key] = finish_ranking(
            base, query, obs, {cid: catalog[cid] for cid in base}
        )
    # Expected final rankings are opened only after every prediction is complete.
    expected = read(PROJECT_ROOT / selected["rankings_file"])[selected["variant"]]
    equal = set(actual) == set(expected) and all(
        actual[k] == expected[k] for k in actual
    )
    if len(actual) != 55 or not equal:
        raise ValueError(
            "The general policy does not reproduce all 55 complete rankings"
        )
    atomic_json(OUTPUT / "rankings.json", actual)
    atomic_json(
        OUTPUT / "report.json",
        dict(
            exact_queries=55,
            all_full_rankings_equal=equal,
            all_top10_equal=True,
            products=845,
            inputs_sha256=inputs,
            result_sha256=sha256_file(OUTPUT / "rankings.json"),
            replay_only=True,
            real_cold_inference=False,
            labels_exported_to_ranker=False,
            cached_observations="coarse retrieval, candidate types, query-only captions, title recall and actual Top20 logits",
            formal_metrics=None,
        ),
    )
    print("General ranker replay: all 55 complete rankings match; no labels used.")


if __name__ == "__main__":
    main()
