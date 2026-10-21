"""Evaluation metrics, Dice, IoU, and validation tracking."""

from src.evaluation.metrics import (
    CLASS_NAME_MAP,
    FOREGROUND_CLASS_INDICES,
    MetricAccumulator,
    compute_binary_dice_and_iou,
    compute_multiclass_metrics_numpy,
    evaluate_batch_predictions,
)

__all__ = [
    "CLASS_NAME_MAP",
    "FOREGROUND_CLASS_INDICES",
    "MetricAccumulator",
    "compute_binary_dice_and_iou",
    "compute_multiclass_metrics_numpy",
    "evaluate_batch_predictions",
]
