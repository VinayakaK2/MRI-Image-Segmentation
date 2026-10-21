"""Evaluation metrics for multi-class cardiac MRI segmentation."""

from typing import Dict, List, Tuple
import numpy as np
import torch


CLASS_NAME_MAP: Dict[int, str] = {
    0: "background",
    1: "rv",
    2: "myo",
    3: "lv",
}
FOREGROUND_CLASS_INDICES: Tuple[int, ...] = (1, 2, 3)


def compute_binary_dice_and_iou(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
) -> Tuple[float, float]:
    """Computes binary Dice and IoU with explicit, mathematically rigorous absent-class handling.

    Absent class policy:
    - If neither ground-truth nor prediction contains the class: Dice = 1.0, IoU = 1.0 (True Negative).
    - If ground-truth has the class but prediction does not: Dice = 0.0, IoU = 0.0 (False Negative).
    - If ground-truth does not have the class but prediction does: Dice = 0.0, IoU = 0.0 (False Positive).
    - If both contain positive pixels: 2*TP / (2*TP + FP + FN) and TP / (TP + FP + FN).
    """
    pred_bool = pred_mask.astype(bool)
    target_bool = target_mask.astype(bool)

    intersection = int(np.sum(pred_bool & target_bool))
    pred_sum = int(np.sum(pred_bool))
    target_sum = int(np.sum(target_bool))
    union = int(np.sum(pred_bool | target_bool))

    # Case 1: Both ground truth and prediction are empty
    if target_sum == 0 and pred_sum == 0:
        return 1.0, 1.0

    # Case 2: One is empty while the other contains predictions
    if target_sum == 0 or pred_sum == 0 or union == 0:
        return 0.0, 0.0

    dice = (2.0 * intersection) / (pred_sum + target_sum)
    iou = intersection / union
    return float(dice), float(iou)


def compute_multiclass_metrics_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 4,
) -> Dict[str, float]:
    """Computes per-class and macro-averaged Dice and IoU metrics on discrete label arrays."""
    metrics: Dict[str, float] = {}
    fg_dices: List[float] = []
    fg_ious: List[float] = []
    all_dices: List[float] = []
    all_ious: List[float] = []

    for c in range(num_classes):
        c_name = CLASS_NAME_MAP.get(c, f"class_{c}")
        c_pred = (pred == c)
        c_target = (target == c)

        c_dice, c_iou = compute_binary_dice_and_iou(c_pred, c_target)
        metrics[f"dice_{c_name}"] = c_dice
        metrics[f"iou_{c_name}"] = c_iou

        all_dices.append(c_dice)
        all_ious.append(c_iou)
        if c in FOREGROUND_CLASS_INDICES:
            fg_dices.append(c_dice)
            fg_ious.append(c_iou)

    metrics["mean_dice_foreground"] = float(np.mean(fg_dices)) if fg_dices else 0.0
    metrics["mean_iou_foreground"] = float(np.mean(fg_ious)) if fg_ious else 0.0
    metrics["mean_dice_all"] = float(np.mean(all_dices))
    metrics["mean_iou_all"] = float(np.mean(all_ious))

    return metrics


def evaluate_batch_predictions(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
) -> Dict[str, float]:
    """Converts model logits to discrete predictions and computes batch-averaged segmentation metrics."""
    if logits.dim() == 4:
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
    elif logits.dim() == 3:
        preds = logits.detach().cpu().numpy()
    else:
        raise ValueError(f"Unexpected prediction tensor dimensions: {logits.shape}")

    targets_np = targets.detach().cpu().numpy()
    batch_size = preds.shape[0]

    batch_accum: Dict[str, float] = {}
    for i in range(batch_size):
        sample_metrics = compute_multiclass_metrics_numpy(preds[i], targets_np[i], num_classes=num_classes)
        for k, v in sample_metrics.items():
            batch_accum[k] = batch_accum.get(k, 0.0) + v

    # Average across the batch
    return {k: v / batch_size for k, v in batch_accum.items()}


class MetricAccumulator:
    """Thread-safe metric running accumulator for epoch-level evaluation aggregation."""

    def __init__(self) -> None:
        """Initializes accumulator stores."""
        self.reset()

    def reset(self) -> None:
        """Resets all metric sums and sample counts to zero."""
        self.totals: Dict[str, float] = {}
        self.count: int = 0

    def update(self, batch_metrics: Dict[str, float], batch_size: int = 1) -> None:
        """Accumulates batch metric values weighted by batch size."""
        for k, v in batch_metrics.items():
            self.totals[k] = self.totals.get(k, 0.0) + (v * batch_size)
        self.count += batch_size

    def compute_averages(self) -> Dict[str, float]:
        """Calculates final sample-weighted averages for all tracked metrics."""
        if self.count == 0:
            return {}
        return {k: float(v / self.count) for k, v in self.totals.items()}
