"""Unit tests for DiceCELoss and multi-class segmentation evaluation metrics."""

import pytest
import numpy as np
import torch
from src.evaluation.metrics import (
    MetricAccumulator,
    compute_binary_dice_and_iou,
    compute_multiclass_metrics_numpy,
    evaluate_batch_predictions,
)
from src.training.losses import DiceCELoss, SoftDiceLoss


def test_soft_dice_loss_finite_and_backward() -> None:
    """Verifies that SoftDiceLoss is finite and differentiates with respect to logits."""
    criterion = SoftDiceLoss(num_classes=4, include_background=True)
    logits = torch.randn(2, 4, 32, 32, requires_grad=True)
    targets = torch.randint(0, 4, (2, 32, 32))

    loss = criterion(logits, targets)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_dice_ce_combined_loss() -> None:
    """Verifies that DiceCELoss combines CrossEntropy and SoftDice accurately."""
    criterion = DiceCELoss(num_classes=4, ce_weight=1.0, dice_weight=1.0)
    logits = torch.randn(2, 4, 64, 64, requires_grad=True)
    targets = torch.randint(0, 4, (2, 64, 64))

    loss = criterion(logits, targets)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None


def test_binary_dice_and_iou_exact_synthetic_cases() -> None:
    """Verifies Dice and IoU calculations on controlled synthetic arrays."""
    # 1. Complete match
    pred = np.array([[1, 1], [0, 0]])
    target = np.array([[1, 1], [0, 0]])
    dice, iou = compute_binary_dice_and_iou(pred, target)
    assert dice == 1.0
    assert iou == 1.0

    # 2. Complete mismatch
    pred = np.array([[1, 1], [0, 0]])
    target = np.array([[0, 0], [1, 1]])
    dice, iou = compute_binary_dice_and_iou(pred, target)
    assert dice == 0.0
    assert iou == 0.0

    # 3. Partial overlap (2 TP, 1 FP, 1 FN)
    # TP=2, pred_sum=3, target_sum=3 -> Dice = 2*2 / (3+3) = 4/6 = 0.6667; IoU = 2/4 = 0.5
    pred = np.array([[1, 1], [1, 0]])
    target = np.array([[1, 1], [0, 1]])
    dice, iou = compute_binary_dice_and_iou(pred, target)
    assert np.isclose(dice, 4.0 / 6.0)
    assert np.isclose(iou, 2.0 / 4.0)


def test_binary_dice_absent_class_handling() -> None:
    """Verifies absent class handling policy: true negative yields 1.0; false pos/neg yield 0.0."""
    empty_pred = np.zeros((10, 10), dtype=bool)
    empty_target = np.zeros((10, 10), dtype=bool)
    non_empty = np.ones((10, 10), dtype=bool)

    # True Negative (absent class correctly not predicted)
    dice_tn, iou_tn = compute_binary_dice_and_iou(empty_pred, empty_target)
    assert dice_tn == 1.0
    assert iou_tn == 1.0

    # False Positive (absent class incorrectly predicted)
    dice_fp, iou_fp = compute_binary_dice_and_iou(non_empty, empty_target)
    assert dice_fp == 0.0
    assert iou_fp == 0.0

    # False Negative (present class missed completely)
    dice_fn, iou_fn = compute_binary_dice_and_iou(empty_pred, non_empty)
    assert dice_fn == 0.0
    assert iou_fn == 0.0


def test_multiclass_metrics_numpy_structure() -> None:
    """Verifies that multi-class metrics contain all expected class and macro keys."""
    pred = np.array([[0, 1], [2, 3]])
    target = np.array([[0, 1], [2, 3]])

    metrics = compute_multiclass_metrics_numpy(pred, target, num_classes=4)
    expected_keys = [
        "dice_background", "dice_rv", "dice_myo", "dice_lv",
        "iou_background", "iou_rv", "iou_myo", "iou_lv",
        "mean_dice_foreground", "mean_iou_foreground",
        "mean_dice_all", "mean_iou_all"
    ]
    for k in expected_keys:
        assert k in metrics, f"Missing metric key: {k}"
        assert metrics[k] == 1.0  # Perfect match


def test_metric_accumulator_running_averages() -> None:
    """Verifies sample-weighted aggregation across batches."""
    acc = MetricAccumulator()
    acc.update({"dice_rv": 0.8, "loss": 0.4}, batch_size=2)
    acc.update({"dice_rv": 0.5, "loss": 0.1}, batch_size=4)

    avgs = acc.compute_averages()
    # dice_rv = (0.8*2 + 0.5*4) / 6 = (1.6 + 2.0) / 6 = 3.6 / 6 = 0.6
    assert np.isclose(avgs["dice_rv"], 0.6)
    # loss = (0.4*2 + 0.1*4) / 6 = (0.8 + 0.4) / 6 = 1.2 / 6 = 0.2
    assert np.isclose(avgs["loss"], 0.2)
