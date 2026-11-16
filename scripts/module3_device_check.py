"""Same four images/config on explicit CPU or MPS; no test/validation access."""

import argparse
import json
import os
import platform
import time
from pathlib import Path
import torch
from module3_instances import build, image_input
from module3_learning import update, evaluate, dump
from multimodal_ecommerce.segmentation import sha256_file
from multimodal_ecommerce.segmentation_device import (
    select_device,
    synchronize,
    memory_sample,
    cpu_prediction,
)


def main(a):
    device = select_device(a.device)
    if a.device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable MPS CPU fallback for this device verification")
    rows = json.loads(a.subset.read_text())
    if len(rows) != 4 or any(r["split"] != "train" for r in rows):
        raise ValueError("Expected four official train images")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if a.device == "mps":
        torch.mps.manual_seed(42)
    a.output.mkdir(parents=True, exist_ok=False)
    config = dict(
        device=a.device,
        steps=a.steps,
        min_size=256,
        max_size=384,
        seed=42,
        lr=0.001,
        momentum=0.9,
        backbone_frozen=True,
    )
    identity = dict(
        subset_sha256=sha256_file(a.subset), weights_sha256=sha256_file(a.weights)
    )
    dump(
        a.output / "protocol.json",
        dict(
            config=config,
            identity=identity,
            torch=str(torch.__version__),
            platform=platform.platform(),
            mps_available=torch.backends.mps.is_available(),
            fallback_env=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            test_used=False,
        ),
    )
    model = build(a.weights)
    model.transform.min_size = (256,)
    model.transform.max_size = 384
    model.to(device)
    assert next(model.parameters()).device.type == a.device
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.001, momentum=0.9
    )
    logs = []
    samples = []
    synchronize(device)
    started = time.perf_counter()
    for step in range(1, a.steps + 1):
        log = update(model, opt, rows[(step - 1) % 4])
        logs.append(dict(step=step, **log))
        samples.append(memory_sample(device))
        if step % 20 == 0:
            print(f"{a.device} step={step}", flush=True)
    synchronize(device)
    train_seconds = time.perf_counter() - started
    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=opt.state_dict(),
            torch_rng=torch.get_rng_state(),
            mps_rng=torch.mps.get_rng_state() if a.device == "mps" else None,
            config=config,
            identity=identity,
            step=a.steps,
        ),
        a.output / "checkpoint.pt",
    )
    synchronize(device)
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        im = image_input(rows[0]["image_path"]).to(device)
        before = cpu_prediction(model([im])[0])
    synchronize(device)
    first_prediction_seconds = time.perf_counter() - start
    state = torch.load(
        a.output / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    with torch.no_grad():
        after = cpu_prediction(model([im])[0])
    assert all(torch.allclose(v, after[k], atol=1e-5) for k, v in before.items())
    metrics = evaluate(model, rows, a.output)
    report = dict(
        config=config,
        identity=identity,
        training_seconds=train_seconds,
        first_prediction_seconds=first_prediction_seconds,
        metrics=metrics,
        losses=logs,
        model_parameter_device=str(next(model.parameters()).device),
        checkpoint_reload_prediction_equal=True,
        checkpoint_sha256=sha256_file(a.output / "checkpoint.pt"),
        memory_samples=samples,
        max_sampled_current_bytes=max(
            (v["current_allocated_bytes"] or 0) for v in samples
        ),
        max_sampled_driver_bytes=max(
            (v["driver_allocated_bytes"] or 0) for v in samples
        ),
        memory_note="MPS live/driver memory sampled after each update, not true process peak",
        test_used=False,
        gate_passed=metrics["mask_ap50"] >= 0.8,
    )
    dump(a.output / "report.json", report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("losses", "memory_samples", "metrics")
            }
        ),
        flush=True,
    )
    print("AP50", metrics["mask_ap50"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "mps"], required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--subset", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
