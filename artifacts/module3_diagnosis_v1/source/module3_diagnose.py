"""Four-image controlled diagnostics: RPN recall, classification and learning budget."""

import argparse
import json
import time
from pathlib import Path
import torch
from torchvision.ops import box_iou
from module3_instances import build, sample, image_input
from module3_learning import update, evaluate, dump
from multimodal_ecommerce.segmentation import sha256_file


def proposal_diagnostics(model, rows):
    """Inspect predictions before consulting GT; no GT boxes enter the model."""
    model.eval()
    result = []
    with torch.no_grad():
        for row in rows:
            image = image_input(row["image_path"])
            images, _ = model.transform([image], None)
            features = model.backbone(images.tensors)
            proposals, _ = model.rpn(images, features, None)
            pooled = model.roi_heads.box_roi_pool(
                features, proposals, images.image_sizes
            )
            logits, regression = model.roi_heads.box_predictor(
                model.roi_heads.box_head(pooled)
            )
            scores = logits.softmax(-1)
            decoded = model.roi_heads.box_coder.decode(regression, proposals)
            pred = model([image])[0]
            _, gt = sample(row)
            height, width = images.image_sizes[0]
            original_h, original_w = image.shape[-2:]
            factor = torch.tensor(
                [
                    width / original_w,
                    height / original_h,
                    width / original_w,
                    height / original_h,
                ]
            )
            boxes = gt["boxes"] * factor
            rpn_iou = box_iou(boxes, proposals[0])
            final_iou = box_iou(gt["boxes"], pred["boxes"])
            for k, label in enumerate(gt["labels"]):
                candidates = rpn_iou[k] >= 0.5
                best = int(rpn_iou[k].argmax()) if len(proposals[0]) else None
                max_score = (
                    float(scores[candidates, int(label)].max())
                    if candidates.any()
                    else None
                )
                correct = pred["labels"] == label
                result.append(
                    dict(
                        image_id=row["image_id"],
                        object_index=k,
                        category=int(label),
                        rpn_best_iou=float(rpn_iou[k].max()) if best is not None else 0,
                        rpn_recalled_50=bool(candidates.any()),
                        correct_class_score_on_recalled_proposals=max_score,
                        best_proposal_background_score=(
                            float(scores[best, 0]) if best is not None else None
                        ),
                        best_proposal_top_class=(
                            int(scores[best].argmax()) if best is not None else None
                        ),
                        final_any_class_best_box_iou=(
                            float(final_iou[k].max()) if final_iou.shape[1] else 0
                        ),
                        final_correct_class_best_box_iou=(
                            float(final_iou[k, correct].max()) if correct.any() else 0
                        ),
                    )
                )
    return dict(
        objects=result,
        rpn_recall50=sum(v["rpn_recalled_50"] for v in result) / len(result),
    )


def run(a):
    torch.set_num_threads(2)
    rows = json.loads((a.parent / "overfit_subset.json").read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(
        parent_checkpoint_sha256=sha256_file(a.parent / "step_100/checkpoint.pt"),
        subset_sha256=sha256_file(a.parent / "overfit_subset.json"),
        weights_sha256=sha256_file(a.weights),
        cases=[
            dict(name="baseline100", min_size=128, max_size=192, total_steps=100),
            dict(name="resolution100", min_size=256, max_size=384, total_steps=100),
            dict(name="long500", min_size=128, max_size=192, total_steps=500),
        ],
        seed=42,
        lr=0.001,
        momentum=0.9,
        backbone_frozen=True,
        gate_ap50=0.8,
        test_used=False,
        scope="same_four_train_images_only",
    )
    dump(a.output / "protocol.json", protocol)
    summaries = []
    for case in protocol["cases"]:
        name = case["name"]
        folder = a.output / name
        folder.mkdir()
        torch.manual_seed(42)
        model = build(a.weights)
        model.transform.min_size = (case["min_size"],)
        model.transform.max_size = case["max_size"]
        opt = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.001, momentum=0.9
        )
        start = 0
        if name != "resolution100":
            state = torch.load(a.parent / "step_100/checkpoint.pt", weights_only=True)
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["rng"])
            start = 100
        logs = []
        begin = time.perf_counter()
        for step in range(start + 1, case["total_steps"] + 1):
            logs.append(dict(step=step, **update(model, opt, rows[(step - 1) % 4])))
            if step % 100 == 0:
                print(f"{name}: {step}", flush=True)
        train_seconds = time.perf_counter() - begin
        torch.save(
            dict(
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                rng=torch.get_rng_state(),
                case=case,
                protocol=protocol,
            ),
            folder / "checkpoint.pt",
        )
        metrics = evaluate(model, rows, folder)
        probes = proposal_diagnostics(model, rows)
        report = dict(
            case=case,
            metrics=metrics,
            probes=probes,
            training_log=logs,
            additional_train_seconds=train_seconds,
            checkpoint_sha256=sha256_file(folder / "checkpoint.pt"),
        )
        dump(folder / "report.json", report)
        summaries.append(
            dict(
                case=case,
                ap50=metrics["mask_ap50"],
                ap=metrics["mask_map"],
                rpn_recall50=probes["rpn_recall50"],
                additional_train_seconds=train_seconds,
                inference_and_evaluation_seconds=metrics["seconds"],
            )
        )
        print(summaries[-1], flush=True)
        dump(a.output / "summary.json", summaries)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
