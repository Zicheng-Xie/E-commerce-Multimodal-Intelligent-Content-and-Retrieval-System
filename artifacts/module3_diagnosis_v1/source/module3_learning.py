"""Gated CPU learning experiment. Never reads the official test split."""

import argparse
import hashlib
import heapq
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image
from module3_instances import build, sample, image_input
from multimodal_ecommerce.instance_metrics import instance_ap
from multimodal_ecommerce.segmentation import sha256_file

CONFIG = dict(
    seed=42, lr=0.001, momentum=0.9, min_size=128, max_size=192, backbone_frozen=True
)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2))


def subset(data, split, minimum):
    """Keep bounded lowest-hash candidates per class, then greedy cover."""
    heaps = {c: [] for c in range(1, 14)}
    with (data / f"{split}.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if not r["eligible"]:
                continue
            score = int(hashlib.sha256(r["image_id"].encode()).hexdigest(), 16)
            for c in {o["category_id"] for o in r["objects"]}:
                h = heaps[c]
                item = (-score, r["image_id"], r)
                if len(h) < minimum:
                    heapq.heappush(h, item)
                elif item[:2] > h[0][:2]:
                    heapq.heapreplace(h, item)
    candidates = {item[1]: item[2] for h in heaps.values() for item in h}
    counts = Counter()
    selected = []
    for r in sorted(
        candidates.values(),
        key=lambda r: hashlib.sha256(r["image_id"].encode()).hexdigest(),
    ):
        classes = Counter(o["category_id"] for o in r["objects"])
        if any(counts[c] < minimum for c in classes):
            selected.append(r)
            counts.update(classes)
    if not all(counts[c] >= minimum for c in heaps):
        raise ValueError("Insufficient class coverage")
    return selected, dict(counts)


def evaluate(model, rows, folder=None):
    model.eval()
    items = []
    timing = time.perf_counter()
    with torch.no_grad():
        for r in rows:
            pred = model([image_input(r["image_path"])])[0]
            _, gt = sample(r)
            masks = pred["masks"][:, 0] >= 0.5
            items.append(
                dict(
                    gt_labels=gt["labels"],
                    gt_masks=gt["masks"],
                    labels=pred["labels"],
                    masks=masks,
                    scores=pred["scores"],
                )
            )
            if folder:
                name = r["image_id"].replace("/", "_")
                np.savez_compressed(
                    folder / (name + ".npz"),
                    masks=masks.numpy(),
                    labels=pred["labels"].numpy(),
                    scores=pred["scores"].numpy(),
                    boxes=pred["boxes"].numpy(),
                )
                with Image.open(r["image_path"]) as im:
                    base = im.convert("RGB")
                    panels = []
                    for ms, labels, scores in [
                        (gt["masks"], gt["labels"], [1] * len(gt["labels"])),
                        (masks, pred["labels"], pred["scores"]),
                    ]:
                        layer = base.copy()
                        draw = ImageDraw.Draw(layer)
                        for i, (mask, label, score) in enumerate(
                            zip(ms, labels, scores)
                        ):
                            if float(score) < 0.05:
                                continue
                            arr = np.asarray(mask, dtype=bool)
                            color = (
                                (i * 73 + 80) % 255,
                                (i * 91 + 50) % 255,
                                (i * 41 + 120) % 255,
                            )
                            layer.paste(
                                Image.new("RGB", layer.size, color),
                                mask=Image.fromarray(arr.astype("uint8") * 100),
                            )
                            draw.text(
                                (5, 5 + 12 * i),
                                f"{int(label)} {float(score):.2f}",
                                fill="red",
                            )
                        panels.append(layer.resize((300, 300)))
                    canvas = Image.new("RGB", (900, 300))
                    canvas.paste(base.resize((300, 300)), (0, 0))
                    canvas.paste(panels[0], (300, 0))
                    canvas.paste(panels[1], (600, 0))
                    canvas.save(folder / (name + ".png"))
    result = instance_ap(items)
    result["seconds"] = time.perf_counter() - timing
    result["gt_counts"] = dict(Counter(int(c) for v in items for c in v["gt_labels"]))
    result["prediction_counts"] = dict(
        Counter(int(c) for v in items for c in v["labels"])
    )
    # Diagnostic association only; AP above uses strict independent matching.
    diagnoses = []
    for r, v in zip(rows, items):
        for k, (label, mask) in enumerate(zip(v["gt_labels"], v["gt_masks"])):
            overlaps = []
            for predicted in v["masks"]:
                a = np.asarray(mask, dtype=bool)
                b = np.asarray(predicted, dtype=bool)
                union = np.logical_or(a, b).sum()
                overlaps.append(
                    float(np.logical_and(a, b).sum() / union) if union else 0
                )
            nearby = [j for j, u in enumerate(overlaps) if u >= 0.5]
            correct = [j for j in nearby if int(v["labels"][j]) == int(label)]
            reason = (
                "duplicate"
                if len(correct) > 1
                else (
                    "matched"
                    if correct
                    else (
                        "wrong_class"
                        if nearby
                        else (
                            "boundary_shift"
                            if max(overlaps, default=0) > 0.1
                            else "missed"
                        )
                    )
                )
            )
            diagnoses.append(
                dict(
                    image_id=r["image_id"],
                    object_index=k,
                    category=int(label),
                    reason=reason,
                    best_iou=max(overlaps, default=0),
                )
            )
    result["diagnoses"] = diagnoses
    return result


def save(path, model, opt, step, identity, epoch=None):
    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=opt.state_dict(),
            step=step,
            epoch=step // 4 if epoch is None else epoch,
            rng=torch.get_rng_state(),
            config=CONFIG,
            identity=identity,
        ),
        path,
    )


def restore(path, model, opt, identity):
    state = torch.load(path, weights_only=True)
    if state["identity"] != identity or state["config"] != CONFIG:
        raise ValueError("Checkpoint configuration/subset mismatch")
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng"])
    return state["step"]


def update(model, opt, row):
    model.train()
    im, gt = sample(row)
    opt.zero_grad()
    parts = model([im], [gt])
    total = sum(parts.values())
    if not torch.isfinite(total):
        raise ValueError("Nonfinite loss")
    total.backward()
    grad = float(model.roi_heads.box_predictor.cls_score.weight.grad.norm())
    opt.step()
    return dict(
        losses={k: float(v.detach()) for k, v in parts.items()},
        classifier_gradient_norm=grad,
    )


def main(a):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    a.output.mkdir(parents=True, exist_ok=False)
    train, tc = subset(a.data, "train", 20)
    val, vc = subset(a.data, "validation", 10)
    dump(
        a.output / "subsets.json",
        dict(train=train, validation=val, train_counts=tc, validation_counts=vc),
    )
    four = []
    # Include both single- and multiple-object images, selected before training.
    for multi in (False, True):
        four.extend([r for r in train if (len(r["objects"]) > 1) == multi][:2])
    if len(four) != 4:
        raise ValueError("Missing single/multi cases")
    dump(a.output / "overfit_subset.json", four)
    identity = dict(
        manifest=sha256_file(a.data / "manifest.json"),
        subsets=sha256_file(a.output / "subsets.json"),
        overfit=sha256_file(a.output / "overfit_subset.json"),
    )
    dump(
        a.output / "protocol.json",
        dict(
            config=CONFIG,
            identity=identity,
            gate_ap50=0.8,
            max_updates=100,
            test_used=False,
            scope="development_only",
            checkpoint_selection="mask_map_then_ap50_then_earliest_epoch",
            weights_sha256=sha256_file(a.weights),
        ),
    )
    replay = []
    for r in val:
        _, gt = sample(r)
        replay.append(
            dict(
                gt_labels=gt["labels"],
                gt_masks=gt["masks"],
                labels=gt["labels"],
                masks=gt["masks"],
                scores=[1.0] * len(gt["labels"]),
            )
        )
    oracle = instance_ap(replay)
    assert len(oracle["per_class"]) == 13 and all(
        v["ap"] == v["ap50"] == 1 for v in oracle["per_class"].values()
    )
    dump(a.output / "gt_replay.json", oracle)
    model = build(a.weights)
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=CONFIG["lr"],
        momentum=CONFIG["momentum"],
    )
    start_step = restore(a.resume, model, opt, identity) if a.resume else 0
    if start_step not in (0, 20, 40, 60, 80):
        raise ValueError("Resume must be a 20/40/60/80 step checkpoint")
    history = []
    logs = []
    started = time.perf_counter()
    for step in range(start_step + 1, 101):
        logs.append(update(model, opt, four[(step - 1) % 4]))
        if step % 20 == 0:
            folder = a.output / f"step_{step:03}"
            folder.mkdir()
            save(folder / "checkpoint.pt", model, opt, step, identity)
            metrics = evaluate(model, four, folder)
            history.append(dict(step=step, **metrics))
            dump(folder / "metrics.json", metrics)
            print(f'step={step} AP50={metrics["mask_ap50"]:.4f}', flush=True)
    # Compare next update after disk restoration with uninterrupted update.
    continuous_log = update(model, opt, four[0])
    continuous = {k: v.clone() for k, v in model.state_dict().items()}
    restore(a.output / "step_100/checkpoint.pt", model, opt, identity)
    restored_log = update(model, opt, four[0])
    same = all(torch.equal(v, model.state_dict()[k]) for k, v in continuous.items())
    assert same and continuous_log == restored_log
    dump(
        a.output / "resume_verification.json",
        dict(next_update_all_parameters_equal=same, optimizer_rng_restored=True),
    )
    restore(a.output / "step_100/checkpoint.pt", model, opt, identity)
    passed = history[-1]["mask_ap50"] >= 0.8
    summary = dict(
        gate_passed=passed,
        overfit_history=history,
        training_log=logs,
        elapsed_seconds=time.perf_counter() - started,
        test_used=False,
        development_training_started=False,
    )
    dump(a.output / "summary.json", summary)
    if not passed:
        print("Gate failed; no expanded training.", flush=True)
        return
    torch.manual_seed(42)
    model = build(a.weights)
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=CONFIG["lr"],
        momentum=CONFIG["momentum"],
    )
    epochs = []
    for epoch in range(1, 6):
        losses = [update(model, opt, r) for r in train]
        folder = a.output / f"dev_epoch_{epoch}"
        folder.mkdir()
        save(
            folder / "checkpoint.pt",
            model,
            opt,
            epoch * len(train),
            identity,
            epoch=epoch,
        )
        metrics = evaluate(model, val)
        dump(folder / "metrics.json", dict(metrics=metrics, losses=losses))
        epochs.append(dict(epoch=epoch, **metrics))
        print(f'dev epoch={epoch} AP={metrics["mask_map"]:.4f}', flush=True)
    best = max(epochs, key=lambda e: (e["mask_map"], e["mask_ap50"], -e["epoch"]))
    dump(
        a.output / "best_checkpoint.json",
        dict(
            epoch=best["epoch"],
            path=f"dev_epoch_{best['epoch']}/checkpoint.pt",
            sha256=sha256_file(a.output / f"dev_epoch_{best['epoch']}/checkpoint.pt"),
        ),
    )
    summary.update(
        development_training_started=True,
        development_epochs=epochs,
        best_epoch=best["epoch"],
    )
    dump(a.output / "summary.json", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", type=Path)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
