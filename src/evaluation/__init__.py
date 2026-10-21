"""Evaluation module: metrics, accumulators, and patient-level pooling."""

from src.evaluation.metrics import (
    CLASS_NAME_MAP,
    FOREGROUND_CLASS_INDICES,
    MetricAccumulator,
    compute_binary_dice_and_iou,
    compute_multiclass_metrics_numpy,
    evaluate_batch_predictions,
)
from src.evaluation.patient_metrics import (
    PatientMetricAccumulator,
    adjust_pvalues_holm,
    compute_paired_patient_comparison,
)

__all__ = [
    "CLASS_NAME_MAP",
    "FOREGROUND_CLASS_INDICES",
    "MetricAccumulator",
    "compute_binary_dice_and_iou",
    "compute_multiclass_metrics_numpy",
    "evaluate_batch_predictions",
    "PatientMetricAccumulator",
    "adjust_pvalues_holm",
    "compute_paired_patient_comparison",
]
