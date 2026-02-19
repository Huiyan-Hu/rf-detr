# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for SSL-aware SetCriterion classification behavior."""

import torch

from rfdetr.models.lwdetr import SetCriterion


class _StubMatcher:
    def __call__(self, outputs, targets, group_detr=1):
        del outputs, group_detr
        return [
            (
                torch.arange(len(target["boxes"]), dtype=torch.int64),
                torch.arange(len(target["boxes"]), dtype=torch.int64),
            )
            for target in targets
        ]


def _build_criterion(ssl_iou_threshold: float = 0.5, ssl_objectness_loss_coef: float = 0.0) -> SetCriterion:
    return SetCriterion(
        num_classes=2,
        matcher=_StubMatcher(),
        weight_dict={"loss_ce": 1.0, "loss_bbox": 1.0, "loss_giou": 1.0, "loss_ssl_objectness": ssl_objectness_loss_coef},
        focal_alpha=0.25,
        losses=["labels", "boxes", "cardinality"],
        group_detr=1,
        ssl_iou_threshold=ssl_iou_threshold,
        ssl_objectness_loss_coef=ssl_objectness_loss_coef,
    )


def test_ssl_overlap_reduces_supervised_classification_penalty():
    criterion = _build_criterion(ssl_iou_threshold=0.5, ssl_objectness_loss_coef=0.0)

    outputs = {
        "pred_logits": torch.full((1, 3, 2), -8.0),
        "pred_boxes": torch.tensor([
            [
                [0.50, 0.50, 0.20, 0.20],
                [0.15, 0.15, 0.20, 0.20],
                [0.80, 0.80, 0.20, 0.20],
            ]
        ]),
    }
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.50, 0.50, 0.20, 0.20]]),
            "ssl_boxes": torch.tensor([[0.15, 0.15, 0.20, 0.20]]),
        }
    ]

    with_ssl = criterion(outputs, targets)["loss_ce"]

    targets_without_ssl = [{k: v for k, v in targets[0].items() if k != "ssl_boxes"}]
    without_ssl = criterion(outputs, targets_without_ssl)["loss_ce"]

    assert with_ssl < without_ssl


def test_ssl_objectness_loss_is_returned_when_enabled():
    criterion = _build_criterion(ssl_iou_threshold=0.5, ssl_objectness_loss_coef=0.25)

    outputs = {
        "pred_logits": torch.full((1, 2, 2), -2.0),
        "pred_boxes": torch.tensor([
            [
                [0.50, 0.50, 0.20, 0.20],
                [0.15, 0.15, 0.20, 0.20],
            ]
        ]),
    }
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.50, 0.50, 0.20, 0.20]]),
            "ssl_boxes": torch.tensor([[0.15, 0.15, 0.20, 0.20]]),
        }
    ]

    losses = criterion(outputs, targets)

    assert "loss_ssl_objectness" in losses
    assert losses["loss_ssl_objectness"] > 0
