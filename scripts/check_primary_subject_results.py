"""Validate completed model artifacts and the static result viewer before sealing."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

import run  # noqa: F401
from acquire_primary_subjects import ROOT
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from primary_subject_experiment import guard
from subject_roi_experiment import load, once


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through once, output.parent.mkdir, output.write_text.
    """
    guard()
    verify_files(PROJECT_ROOT, load(ROOT / "execution_source_manifest.json")["files"])
    assert load(ROOT / "pre_inference_checks.json")["passed"]
    assert load(ROOT / "verification.json")["passed"]
    engineering = load(ROOT / "engineering_checks.pre.json")
    assert engineering["passed"]
    for name in [
        "label_receipt.json",
        "encoding_receipt.json",
        "detection_receipt.json",
        "fg_encoding_receipt.json",
        "reranker_receipt.json",
        "prediction_receipt.json",
    ]:
        verify_files(ROOT, load(ROOT / name)["files"])
    html = (ROOT / "internal_comparison.html").read_text(encoding="utf-8")
    match = re.search(
        r'<script id="data" type="application/json">(.*?)</script>', html, re.S
    )
    assert match
    rows = json.loads(match.group(1))
    assert len(rows) == 40
    cards = 0
    for row in rows:
        assert set(row["results"]) == {
            "cross20",
            "subject_caption20",
            "subject_guard20",
        }
        images = [row["image"]] if row.get("image") else []
        for results in row["results"].values():
            assert len(results) == len({r["id"] for r in results}) == 20
            assert all(
                r["title"] and r["category"] and isinstance(r["score"], float)
                for r in results
            )
            images.extend(r["image"] for r in results)
            cards += len(results)
        for image in images:
            path = (ROOT / unquote(image)).resolve()
            assert path.is_relative_to(PROJECT_ROOT) and path.is_file()
    scripts = [
        s
        for s in re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
        if s.lstrip().startswith("const rows=")
    ]
    assert len(scripts) == 1
    output = ROOT / "qa/viewer.js"
    output.parent.mkdir(exist_ok=True)
    output.write_text(scripts[0], encoding="utf-8")
    syntax = subprocess.run(
        ["E:/Node.js/node.exe", "--check", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr
    once(
        ROOT / "checks.json",
        dict(
            passed=True,
            source_sha256=sha256_file(Path(__file__)),
            engineering=engineering,
            all_full_model_receipts_verified=True,
            original_and_changed_context_cache_checks=True,
            complete_AI_label_grid=11840,
            original_labels_unchanged=7720,
            full_rankings_replayed=320,
            result_viewer_queries=40,
            result_cards=cards,
            static_JS_syntax=True,
            all_image_references_exist=True,
            browser_interaction_tested=False,
            online_latency_acceptance=False,
            formal_metrics=None,
        ),
    )
    print(dict(result_checks_passed=True, cards=cards, formal_metrics=None), flush=True)


if __name__ == "__main__":
    main()
