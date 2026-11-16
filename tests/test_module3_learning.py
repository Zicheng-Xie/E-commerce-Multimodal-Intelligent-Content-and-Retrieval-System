"""Coordinate contracts and prediction input boundary."""

import numpy as np
import torch
from PIL import Image
from module3_instances import image_input, sample
from torchvision.models.detection.transform import GeneralizedRCNNTransform


def test_xywh_xyxy_scaling_and_mask_dimensions(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (40, 20), "white").save(path)
    r = dict(
        image_path=str(path),
        width=40,
        height=20,
        objects=[
            dict(
                bbox=[4, 2, 20, 10],
                category_id=13,
                polygons=[[4, 2, 24, 2, 24, 12, 4, 12]],
            )
        ],
    )
    image, gt = sample(r)
    assert gt["labels"].tolist() == [13]
    assert gt["boxes"].tolist() == [[4, 2, 24, 12]]
    assert gt["masks"].shape == (1, 20, 40)
    transform = GeneralizedRCNNTransform(40, 80, [0, 0, 0], [1, 1, 1])
    transform.eval()
    images, scaled = transform([image], [gt])
    assert scaled[0]["boxes"].tolist() == [[8, 4, 48, 24]]
    assert scaled[0]["masks"].shape == (1, 40, 80)
    output = transform.postprocess(
        [dict(boxes=scaled[0]["boxes"])], images.image_sizes, [(20, 40)]
    )
    assert torch.allclose(output[0]["boxes"], gt["boxes"])


def test_pixels_only_input_does_not_require_annotation(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (40, 20), "white").save(path)
    assert image_input(path).shape == (3, 20, 40)
