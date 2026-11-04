"""Measure actual loopback JSON retrieval calls on the old development workload."""

from __future__ import annotations

import json
import time

import httpx
import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

OUT = PROJECT_ROOT / "artifacts/module2_continuation/service"
BASE_URL = "http://127.0.0.1:8765"


def main() -> None:
    """Measure actual loopback JSON retrieval calls on the old development workload."""
    if (OUT / "benchmark.json").exists():
        raise FileExistsError("Preserve the completed resident benchmark")
    OUT.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(timeout=120, trust_env=False)
    health = client.get(BASE_URL + "/health")
    health.raise_for_status()
    health = health.json()
    samples = client.get(BASE_URL + "/api/samples").json()
    cases = [dict(text=q["text"]) for q in samples["text_queries"]]
    cases.extend(dict(image_id=q["id"]) for q in samples["image_queries"])
    cases.extend(
        dict(image_id=q["id"], roi=[0.1, 0.1, 0.9, 0.9])
        for q in samples["image_queries"]
    )
    for i, case in enumerate(cases):
        case.update(top_k=10, method=samples["methods"][i % len(samples["methods"])])
    code = [
        "scripts/benchmark_resident.py",
        "scripts/serve_retrieval.py",
        "src/multimodal_ecommerce/resident_retrieval.py",
        "src/multimodal_ecommerce/retrieval_http.py",
        "assets/resident_retrieval.html",
    ]
    protocol = dict(
        cases=cases,
        service_instance=health.get("instance_id"),
        repeats=2,
        warmup_calls=2,
        implementation={name: sha256_file(PROJECT_ROOT / name) for name in code},
        scope="Sequential loopback HTTP JSON response including guards, preprocessing, embedding, ranking and serialization/client parse; excludes service cold startup and browser image rendering",
        formal_performance_acceptance=False,
        formal_quality_acceptance=False,
        acceptance_model_predictions=0,
    )
    write_json(OUT / "benchmark_protocol.json", protocol)
    for query in [cases[0], cases[len(samples["text_queries"])]]:
        result = client.post(BASE_URL + "/api/search", json=query)
        result.raise_for_status()
    records = []
    for repeat in range(2):
        for i, case in enumerate(cases):
            started = time.perf_counter()
            response = client.post(BASE_URL + "/api/search", json=case)
            response.raise_for_status()
            result = response.json()
            elapsed = time.perf_counter() - started
            assert (
                len(result["results"]) == 10
                and len({r["id"] for r in result["results"]}) == 10
            )
            assert (
                result["embedding_calls"] == 1 and result["result_cache_used"] is False
            )
            assert (
                result["formal_metrics"] is None
                and result["formal_acceptance"] is False
            )
            records.append(
                dict(
                    repeat=repeat,
                    case=i,
                    modality=(
                        "text"
                        if "text" in case
                        else "image_roi" if "roi" in case else "image"
                    ),
                    method=case["method"],
                    http_seconds=elapsed,
                    server_seconds=result["processing_seconds"],
                    top_ids=[r["id"] for r in result["results"]],
                )
            )
            if (i + 1) % 6 == 0:
                print(
                    json.dumps(dict(repeat=repeat, completed=i + 1, total=len(cases))),
                    flush=True,
                )
    write_json(OUT / "benchmark_records.json", records)
    before = client.get(BASE_URL + "/health").json()["requests"]
    for query in [
        dict(text="商品", image_id=samples["image_queries"][0]["id"]),
        dict(image_id=samples["image_queries"][0]["id"], roi=[0, 0, 2, 1]),
        dict(text="商品", category="answer"),
    ]:
        response = client.post(BASE_URL + "/api/search", json=query)
        assert response.status_code == 400
    response = client.post(
        BASE_URL + "/api/search",
        json=cases[0],
        headers={"Origin": "https://example.invalid"},
    )
    assert response.status_code == 403
    registry = PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl"
    reserved = read_rows(registry)[0]
    response = client.post(
        BASE_URL + "/api/search",
        json=dict(image_path=str((registry.parent / reserved["image_path"]).resolve())),
    )
    assert (
        response.status_code == 400
        and "Reserved acceptance" in response.json()["error"]
    )
    assert client.get(BASE_URL + "/health").json()["requests"] == before
    image = client.get(
        BASE_URL
        + client.post(BASE_URL + "/api/search", json=cases[0]).json()["results"][0][
            "image_url"
        ]
    )
    assert image.status_code == 200 and image.headers["Content-Type"].startswith(
        "image/"
    )

    def stats(values: list[float]) -> dict:
        """Return stats for this workflow.

        Args:
            values: Values used by this operation.

        Returns:
            Mapping with fields: count, p50_seconds, p95_seconds, maximum_seconds,
            fraction_below_five_seconds.
        """
        return dict(
            count=len(values),
            p50_seconds=float(np.percentile(values, 50)),
            p95_seconds=float(np.percentile(values, 95)),
            maximum_seconds=max(values),
            fraction_below_five_seconds=sum(v < 5 for v in values) / len(values),
        )

    summary = dict(
        requests=len(records),
        case_count=len(cases),
        startup_seconds=health["startup_seconds"],
        http=stats([r["http_seconds"] for r in records]),
        by_modality={
            mode: stats([r["http_seconds"] for r in records if r["modality"] == mode])
            for mode in ["text", "image", "image_roi"]
        },
        guard_checks_passed=5,
        guards_triggered_no_retrieval=True,
        image_asset_served=True,
        embedding_recomputed_every_request=True,
        formal_performance_acceptance=False,
        formal_quality_metrics=None,
        goal_met=False,
        acceptance_model_predictions=0,
        scope=protocol["scope"],
    )
    write_json(OUT / "benchmark_records.json", records)
    write_json(OUT / "benchmark.json", summary)
    write_json(
        OUT / "benchmark_receipt.json",
        dict(
            files={
                name: sha256_file(OUT / name)
                for name in [
                    "benchmark_protocol.json",
                    "benchmark_records.json",
                    "benchmark.json",
                ]
            }
        ),
    )
    print(json.dumps(summary))
    client.close()


if __name__ == "__main__":
    main()
