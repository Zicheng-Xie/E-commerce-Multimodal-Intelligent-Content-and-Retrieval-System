"""Full pinned Qwen development-gallery extension, retaining original vectors."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.precision_reranking import color_descriptor
from multimodal_ecommerce.regional_retrieval import (
    RegionalRetrievalEncodeViewsInputs,
    encode_views,
)
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from primary_form_coverage import receipt
from subject_coverage_acquisition import ROOT, check
from subject_roi_experiment import COVERAGE, load, once


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve completed Qwen features.

    Notes:
        Writes derived artifacts through cache.mkdir, np.save, np.savez_compressed,
        once, receipt, target.mkdir.
    """
    check()
    verify_files(ROOT, load(ROOT / "label_receipt.json")["files"])
    if (ROOT / "encoding_receipt.json").exists():
        raise FileExistsError("Preserve completed Qwen features")
    folder = ROOT / "additions/index"
    docs = load(ROOT / "additions/catalog.final.public.json")
    profiles = read_rows(ROOT / "additions/profiles.public.jsonl")
    texts = [
        "商品主体：" + p["product_type"] + "。可见结构：" + p["appearance"] + "。"
        for p in profiles
    ]
    config = load(COVERAGE / "protocol.json")["regions"]
    files = [
        Path(__file__),
        ROOT / "label_receipt.json",
        PROJECT_ROOT / "configs/embedding_model.json",
        *sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")),
    ]
    once(
        ROOT / "encoding_protocol.json",
        dict(
            files={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in files
            },
            regions=config,
            full_model=True,
            original_vectors_reused_exactly=True,
            new_public_profiles_used=True,
        ),
    )
    verify_files(PROJECT_ROOT, load(ROOT / "encoding_protocol.json")["files"])
    encoder = QwenEmbedder()
    records, regions, titles, semantic, all_calls = [], [], [], [], []
    try:
        for i, (doc, text) in enumerate(zip(docs, texts), 1):
            cache = folder / "products" / f"{i:03d}"
            cache.mkdir(parents=True, exist_ok=True)
            if not (cache / "vectors.npz").exists():
                before = len(encoder.diagnostics()["calls"])
                views, metadata = encode_views(
                    RegionalRetrievalEncodeViewsInputs(
                        encoder, Path(doc["image_path"]), cache / "views", config
                    )
                )
                title, profile = encoder.encode(text=doc["title"]), encoder.encode(
                    text=text
                )
                once(cache / "calls.json", encoder.diagnostics()["calls"][before:])
                once(cache / "views.json", metadata)
                np.savez_compressed(
                    cache / "vectors.npz", views=views, title=title, profile=profile
                )
            with np.load(cache / "vectors.npz", allow_pickle=False) as bank:
                padded = np.zeros((7, 2048), dtype=np.float32)
                padded[: len(bank["views"])] = bank["views"]
                regions.append(padded)
                titles.append(bank["title"].copy())
                semantic.append(bank["profile"].copy())
            all_calls += load(cache / "calls.json")
            records.append(dict(**doc, views=load(cache / "views.json")))
            print(
                dict(encoded=i, total=len(docs), full_calls=len(all_calls)), flush=True
            )
        runtime = encoder.diagnostics()
    finally:
        encoder.close()
    np.savez_compressed(
        folder / "vectors.npz",
        regions=np.array(regions),
        texts=np.array(titles),
        profiles=np.array(semantic),
    )
    np.save(
        folder / "colors.npy",
        np.array([color_descriptor(Path(d["image_path"])) for d in docs]),
    )
    once(folder / "records.json", records)
    once(folder / "profile_texts.json", texts)
    once(folder / "runtime.json", dict(**runtime, all_completed_calls=all_calls))
    for cohort in ["tune", "regression"]:
        parent = COVERAGE / cohort / "index"
        target = ROOT / cohort / "index"
        target.mkdir(parents=True, exist_ok=True)
        with (
            np.load(parent / "vectors.npz", allow_pickle=False) as a,
            np.load(folder / "vectors.npz", allow_pickle=False) as b,
        ):
            np.savez_compressed(
                target / "vectors.npz",
                allow_pickle=False,
                **{
                    k: np.concatenate([a[k], b[k]])
                    for k in ["regions", "texts", "profiles"]
                },
            )
        once(target / "records.json", load(parent / "records.json") + records)
        once(target / "profile_texts.json", load(parent / "profile_texts.json") + texts)
        np.save(
            target / "colors.npy",
            np.concatenate(
                [
                    np.load(parent / "colors.npy", allow_pickle=False),
                    np.load(folder / "colors.npy", allow_pickle=False),
                ]
            ),
        )
        receipt(
            target,
            "manifest.json",
            sorted(p for p in target.iterdir() if p.is_file()),
            model=load(PROJECT_ROOT / "configs/embedding_model.json"),
            regions=config,
            upstream_files={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                for p in [
                    parent / "manifest.json",
                    parent / "vectors.npz",
                    folder / "vectors.npz",
                ]
            },
        )
    receipt(
        ROOT,
        "encoding_receipt.json",
        [
            p
            for base in [folder, ROOT / "tune/index", ROOT / "regression/index"]
            for p in sorted(base.rglob("*"))
            if p.is_file()
        ],
        actual_full_embedding_calls=len(all_calls),
        additions=len(docs),
        gallery_per_cohort=168 + len(docs),
        original_gallery_vectors_reused_exactly=True,
    )
    print(dict(done=True, calls=len(all_calls)), flush=True)


if __name__ == "__main__":
    main()
