# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for COCO SSL proposal loading."""

import json

import torch
from PIL import Image

from rfdetr.datasets.coco import CocoDetection


def _write_minimal_coco(path, image_name: str, category_id: int = 1) -> None:
    coco = {
        "images": [{"id": 1, "file_name": image_name, "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": category_id,
                "bbox": [10, 20, 30, 40],
                "area": 1200,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": category_id, "name": "obj"}],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f)


def test_coco_detection_loads_ssl_boxes_by_image_id(tmp_path):
    image_path = tmp_path / "img1.jpg"
    Image.new("RGB", (100, 100)).save(image_path)

    ann_file = tmp_path / "ann.json"
    ssl_file = tmp_path / "ssl.json"
    _write_minimal_coco(ann_file, image_name=image_path.name)
    _write_minimal_coco(ssl_file, image_name=image_path.name, category_id=7)

    dataset = CocoDetection(tmp_path, ann_file, transforms=None, ssl_annotations_file=ssl_file)
    _, target = dataset[0]

    assert "ssl_boxes" in target
    assert target["ssl_boxes"].shape == (1, 4)
    assert torch.allclose(target["ssl_boxes"], torch.tensor([[10.0, 20.0, 40.0, 60.0]]))


def test_coco_detection_emits_empty_ssl_boxes_when_missing_image_id(tmp_path):
    image_path = tmp_path / "img1.jpg"
    Image.new("RGB", (100, 100)).save(image_path)

    ann_file = tmp_path / "ann.json"
    _write_minimal_coco(ann_file, image_name=image_path.name)

    ssl_content = {
        "images": [{"id": 999, "file_name": "other.jpg", "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 999,
                "category_id": 1,
                "bbox": [1, 1, 2, 2],
                "area": 4,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "obj"}],
    }
    ssl_file = tmp_path / "ssl.json"
    with open(ssl_file, "w", encoding="utf-8") as f:
        json.dump(ssl_content, f)

    dataset = CocoDetection(tmp_path, ann_file, transforms=None, ssl_annotations_file=ssl_file)
    _, target = dataset[0]

    assert target["ssl_boxes"].shape == (0, 4)
