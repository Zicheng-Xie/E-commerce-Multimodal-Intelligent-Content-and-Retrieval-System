"""Mask R-CNN training/resume/evaluation/prediction on official module3 v2."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms.functional import to_tensor
from multimodal_ecommerce.segmentation import polygon_mask, sha256_file
from multimodal_ecommerce.instance_metrics import instance_ap


def load_rows(data, split, limit):
    rows = []
    with (data / f"{split}.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if split == "test" or row["eligible"]:
                rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def image_input(path):
    """Inference loads pixels only, never target annotations."""
    with Image.open(path) as im:
        return to_tensor(im.convert("RGB"))


def sample(row):
    image = image_input(row["image_path"])
    boxes = []
    masks = []
    labels = []
    for obj in row["objects"]:
        x, y, w, h = obj["bbox"]
        boxes.append([x, y, x + w, y + h])
        labels.append(obj["category_id"])
        masks.append(
            np.asarray(
                polygon_mask((row["width"], row["height"]), obj["polygons"]),
                dtype=np.uint8,
            )
        )
    return image, dict(
        boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        labels=torch.tensor(labels, dtype=torch.int64),
        masks=(
            torch.tensor(np.stack(masks))
            if masks
            else torch.zeros((0, row["height"], row["width"]), dtype=torch.uint8)
        ),
    )


def build(weights=None):
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        min_size=128,
        max_size=192,
        rpn_pre_nms_top_n_train=200,
        rpn_post_nms_top_n_train=100,
        rpn_pre_nms_top_n_test=100,
        rpn_post_nms_top_n_test=50,
        box_detections_per_img=20,
        box_score_thresh=0.05,
    )
    if weights:
        if not sha256_file(weights).startswith("bf2d0c1e"):
            raise ValueError("Official weight hash mismatch")
        model.load_state_dict(torch.load(weights, weights_only=True))
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, 14
    )
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 14
    )
    for param in model.backbone.parameters():
        param.requires_grad = False
    return model


def main(a):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    a.output.mkdir(parents=True, exist_ok=False)
    digest = sha256_file(a.data / "manifest.json")
    model = build(a.weights if not a.checkpoint else None)
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.001, momentum=0.9
    )
    epoch = 0
    if a.checkpoint:
        state = torch.load(a.checkpoint, weights_only=True)
        if state["manifest_sha256"] != digest:
            raise ValueError("Dataset mismatch")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        epoch = state["epoch"]
        torch.set_rng_state(state["rng"])
    elif not a.weights:
        raise ValueError("Supply official --weights or --checkpoint")
    losses = []
    started = time.perf_counter()
    if a.command == "train":
        for _ in range(a.epochs):
            model.train()
            for row in load_rows(a.data, "train", a.limit):
                im, target = sample(row)
                opt.zero_grad()
                parts = model([im], [target])
                loss = sum(parts.values())
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss")
                loss.backward()
                opt.step()
                losses.append({k: float(v.detach()) for k, v in parts.items()})
            epoch += 1
        torch.save(
            dict(
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                epoch=epoch,
                rng=torch.get_rng_state(),
                manifest_sha256=digest,
            ),
            a.output / "checkpoint.pt",
        )
    split = "validation" if a.command == "train" else a.split
    rows = load_rows(a.data, split, a.eval_limit)
    model.eval()
    evaluation = []
    receipts = []
    with torch.no_grad():
        for offset in range(0, len(rows), a.batch_size):
            batch = rows[offset : offset + a.batch_size]
            predictions = model([image_input(row["image_path"]) for row in batch])
            for row, pred in zip(batch, predictions):
                masks = pred["masks"][:, 0] >= 0.5
                path = a.output / (row["image_id"].replace("/", "_") + ".npz")
                np.savez_compressed(
                    path,
                    masks=masks.numpy(),
                    boxes=pred["boxes"].numpy(),
                    labels=pred["labels"].numpy(),
                    scores=pred["scores"].numpy(),
                )
                receipts.append(
                    dict(
                        image_id=row["image_id"],
                        file=path.name,
                        sha256=sha256_file(path),
                        instances=len(masks),
                    )
                )
                if split != "test":
                    _, target = sample(row)
                    evaluation.append(
                        dict(
                            gt_labels=target["labels"],
                            gt_masks=target["masks"],
                            labels=pred["labels"],
                            masks=masks,
                            scores=pred["scores"],
                        )
                    )
    report = dict(
        model="maskrcnn_resnet50_fpn_13_clothing_classes",
        epoch=epoch,
        split=split,
        losses=losses,
        training_ids=(
            [r["image_id"] for r in load_rows(a.data, "train", a.limit)]
            if a.command == "train"
            else []
        ),
        source_sha256=sha256_file(Path(__file__)),
        learning_rate=0.001,
        momentum=0.9,
        predictions=receipts,
        metrics=instance_ap(evaluation) if evaluation else None,
        elapsed_seconds=time.perf_counter() - started,
        manifest_sha256=digest,
        initial_weights_sha256=sha256_file(a.weights) if a.weights else None,
        resumed_from_sha256=sha256_file(a.checkpoint) if a.checkpoint else None,
        device="cpu",
        torch_version=str(torch.__version__),
        configuration=dict(min_size=128, max_size=192, backbone_frozen=True, seed=42),
        scope="bounded_development_not_full_validation",
        test_used=split == "test",
    )
    (a.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "predict"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weights", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--eval-limit", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    main(p.parse_args())
