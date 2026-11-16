"""CPU binary clothing segmentation smoke run, not formal model acceptance."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch import nn
from multimodal_ecommerce.segmentation import polygon_mask, mask_metrics, sha256_file


def rows(path, count):
    selected = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["eligible"]:
                selected.append(row)
            if len(selected) == count:
                break
    return selected


def tensors(records):
    images, masks = [], []
    for row in records:
        with Image.open(row["image_path"]) as im:
            images.append(
                np.asarray(
                    im.convert("RGB").resize((96, 96)), dtype=np.float32
                ).transpose(2, 0, 1)
                / 255
            )
        polygons = [p for o in row["objects"] for p in o["polygons"]]
        mask = polygon_mask((row["width"], row["height"]), polygons)
        masks.append(
            np.asarray(
                mask.resize((96, 96), Image.Resampling.NEAREST), dtype=np.float32
            )[None]
        )
    return torch.tensor(np.stack(images)), torch.tensor(np.stack(masks))


def main(data, output):
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20261116)
    torch.set_num_threads(2)
    train = rows(data / "train.jsonl", 32)
    val = rows(data / "validation.jsonl", 16)
    x, y = tensors(train)
    vx, vy = tensors(val)
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 1, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    losses = []
    started = time.perf_counter()
    for epoch in range(5):
        model.train()
        total = 0
        for i in range(0, len(x), 4):
            opt.zero_grad()
            logits = model(x[i : i + 4])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y[i : i + 4])
            loss.backward()
            opt.step()
            total += float(loss.detach())
        losses.append(total / 8)
    checkpoint = output / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 1, 1),
    )
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    restored.eval()
    with torch.no_grad():
        predictions = (restored(vx).sigmoid() >= 0.5).numpy()[:, 0]
        assert torch.allclose(restored(vx[:1]), restored(vx)[:1], atol=1e-6)
    measurements = []
    for i, (row, pred) in enumerate(zip(val, predictions)):
        predicted = Image.fromarray(pred.astype("uint8") * 255)
        target = Image.fromarray(vy[i, 0].numpy().astype("uint8") * 255)
        measurements.append(
            dict(image_id=row["image_id"], **mask_metrics(predicted, target))
        )
        predicted.save(output / f"prediction_{i:02}.png")
        with Image.open(row["image_path"]) as im:
            panel = Image.new("RGB", (288, 96))
            panel.paste(im.convert("RGB").resize((96, 96)), (0, 0))
            panel.paste(target.convert("RGB"), (96, 0))
            panel.paste(predicted.convert("RGB"), (192, 0))
            panel.save(output / f"comparison_{i:02}.png")
    baselines = {}
    for name, fill in (("empty", 0), ("full", 255)):
        values = [
            mask_metrics(
                Image.new("L", (96, 96), fill),
                Image.fromarray(vy[i, 0].numpy().astype("uint8") * 255),
            )
            for i in range(len(val))
        ]
        baselines[name] = {
            key: float(np.mean([v[key] for v in values])) for key in ("iou", "dice")
        }
    groups = {}
    for field in ("category_name", "occlusion", "viewpoint", "scale", "zoom_in"):
        buckets = {}
        for row, result in zip(val, measurements):
            for label in {str(obj[field]) for obj in row["objects"]}:
                buckets.setdefault(label, []).append(result["iou"])
        groups[field] = {
            label: dict(
                images=len(values), mean_semantic_union_iou=float(np.mean(values))
            )
            for label, values in buckets.items()
        }
    report = dict(
        baselines=baselines,
        group_metrics=groups,
        status="cpu_smoke_only",
        model="three_layer_binary_cnn_random_initialization",
        seed=20261116,
        train_ids=[r["image_id"] for r in train],
        validation_ids=[r["image_id"] for r in val],
        epochs=5,
        losses=losses,
        metrics=measurements,
        mean_iou=float(np.mean([r["iou"] for r in measurements])),
        elapsed_seconds=time.perf_counter() - started,
        checkpoint_sha256=sha256_file(checkpoint),
        dataset_manifest_sha256=sha256_file(data / "manifest.json"),
        torch_version=torch.__version__,
        device="cpu",
        checkpoint_reload=True,
        single_batch_consistent=True,
        test_used=False,
        limitations="Fixed prefix subset; binary semantic union, no instance detection or class metrics; not official validation acceptance",
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    main(a.data, a.output)
