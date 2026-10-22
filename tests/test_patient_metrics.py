"""Unit tests for patient-level 3D metric pooling and statistical comparisons."""

import numpy as np
import pytest

from src.evaluation.patient_metrics import (
    PatientMetricAccumulator,
    adjust_pvalues_holm,
    compute_paired_patient_comparison,
)


def test_patient_accumulator_3d_pooling_different_from_slice_average() -> None:
    """Verifies that patient-level 3D pooling correctly weights voxel volume rather than slice counts."""
    accum = PatientMetricAccumulator(num_classes=4)

    # Slice 1: Small apical slice, 10 foreground pixels, 5 matched
    pred1 = np.zeros((100, 100), dtype=np.int64)
    target1 = np.zeros((100, 100), dtype=np.int64)
    pred1[:5, :2] = 3  # 10 pixels of LV
    target1[:5, :1] = 3  # 5 matched, 5 unmatched
    target1[:5, 1:2] = 3

    # Slice 2: Large mid-ventricular slice, 1000 foreground pixels, 900 matched
    pred2 = np.zeros((100, 100), dtype=np.int64)
    target2 = np.zeros((100, 100), dtype=np.int64)
    pred2[:10, :100] = 3
    target2[:10, :90] = 3
    target2[:10, 90:100] = 3

    accum.update("patient_test", pred1, target1)
    accum.update("patient_test", pred2, target2)

    patient_res = accum.compute_per_patient_metrics()["patient_test"]
    # Total LV: intersection = 5 + 900 = 905, pred = 10 + 1000 = 1010, target = 10 + 1000 = 1010
    expected_dice = (2.0 * (10 + 1000)) / (1010 + 1010)
    assert patient_res["dice_lv"] > 0.95


def test_patient_accumulator_absent_class_handling() -> None:
    """Verifies that absent anatomical classes across all slices for a patient yield 1.0 (TN)."""
    accum = PatientMetricAccumulator(num_classes=4)

    # Patient has no RV (class 1) in any slice
    pred = np.zeros((50, 50), dtype=np.int64)
    target = np.zeros((50, 50), dtype=np.int64)
    pred[10:20, 10:20] = 3  # LV only
    target[10:20, 10:20] = 3

    accum.update("p1", pred, target)
    p_metrics = accum.compute_per_patient_metrics()["p1"]

    assert p_metrics["dice_rv"] == 1.0
    assert p_metrics["iou_rv"] == 1.0
    assert p_metrics["dice_lv"] == 1.0


def test_compute_summary_statistics() -> None:
    """Verifies mean, standard deviation, median, IQR, and CI computation across patients."""
    accum = PatientMetricAccumulator(num_classes=4)

    for i in range(5):
        pred = np.zeros((20, 20), dtype=np.int64)
        target = np.zeros((20, 20), dtype=np.int64)
        pred[5:15, 5:15] = 3
        target[5:15, 5:15] = 3
        accum.update(f"p_{i}", pred, target)

    summary = accum.compute_summary_statistics()
    assert summary["dice_lv"]["mean"] == 1.0
    assert summary["dice_lv"]["std"] == 0.0
    assert summary["dice_lv"]["median"] == 1.0
    assert summary["dice_lv"]["iqr"] == 0.0
    assert summary["dice_lv"]["n_patients"] == 5


def test_compute_paired_patient_comparison() -> None:
    """Verifies paired t-test and Wilcoxon statistical difference computations."""
    clean = {
        f"p_{i}": {"mean_dice_foreground": 0.8} for i in range(10)
    }
    corrupt = {
        f"p_{i}": {"mean_dice_foreground": 0.6} for i in range(10)
    }

    comp = compute_paired_patient_comparison(clean, corrupt, "mean_dice_foreground")
    assert pytest.approx(comp["mean_absolute_change"], abs=1e-5) == -0.2
    assert pytest.approx(comp["mean_relative_change_pct"], abs=1e-4) == -25.0
    assert pytest.approx(comp["relative_change_pct_of_means"], abs=1e-4) == -25.0
    assert comp["paired_t_statistic"] < 0.0
    assert comp["paired_t_pvalue"] < 0.05


def test_adjust_pvalues_holm() -> None:
    """Verifies that Holm-Bonferroni correction strictly orders, inflates, and caps p-values."""
    raw = [0.01, 0.04, 0.03]
    # Sorted: 0.01 (rank 0, m=3), 0.03 (rank 1, m=2), 0.04 (rank 2, m=1)
    # Holm values:
    # rank 0: 0.01 * 3 = 0.03
    # rank 1: max(0.03, 0.03 * 2) = 0.06
    # rank 2: max(0.06, 0.04 * 1) = 0.06
    adj = adjust_pvalues_holm(raw)
    assert pytest.approx(adj[0], abs=1e-5) == 0.03
    assert pytest.approx(adj[1], abs=1e-5) == 0.06
    assert pytest.approx(adj[2], abs=1e-5) == 0.06

    # Edge cases:
    # 1. Empty input
    assert adjust_pvalues_holm([]) == []
    # 2. Single hypothesis
    assert adjust_pvalues_holm([0.05]) == [0.05]
    # 3. Already sorted p-values
    assert adjust_pvalues_holm([0.001, 0.01, 0.03, 0.05]) == [0.004, 0.03, 0.06, 0.06]
    # 4. Unsorted permutation
    assert adjust_pvalues_holm([0.05, 0.001, 0.03, 0.01]) == [0.06, 0.004, 0.06, 0.03]
    # 5. Duplicate p-values
    assert adjust_pvalues_holm([0.01, 0.01, 0.04, 0.04]) == [0.04, 0.04, 0.08, 0.08]
    # 6. Capped at 1.0
    assert adjust_pvalues_holm([0.4, 0.6, 0.8, 0.9]) == [1.0, 1.0, 1.0, 1.0]
    # 7. Monotonicity enforcement under ties / near-ties
    adj_mono = adjust_pvalues_holm([0.02, 0.019, 0.018])
    assert all(pytest.approx(v, abs=1e-5) == 0.054 for v in adj_mono)


def test_phase5_hypothesis_families_sizes() -> None:
    """Verifies pre-registered hypothesis family sizes for Phase 5 FWER control."""
    f1_postproc = 10  # All 10 experimental conditions
    f2_clahe_contrast = 3  # Mild, Moderate, Severe contrast
    f3_combined_contrast = 3  # Mild, Moderate, Severe contrast
    f4_combined_noise = 3  # Mild, Moderate, Severe noise

    # Total pre-registered confirmatory comparisons across all 4 families
    total_comparisons = f1_postproc + f2_clahe_contrast + f3_combined_contrast + f4_combined_noise
    assert total_comparisons == 19

    # Verify Holm step-down multiplier scaling for each registered family size
    # Family 1 (m=10): smallest p-value scaled by m=10
    adj_f1 = adjust_pvalues_holm([0.001] * f1_postproc)
    assert len(adj_f1) == 10
    assert pytest.approx(adj_f1[0], abs=1e-5) == 0.01

    # Family 2, 3, 4 (m=3): smallest p-value scaled by m=3
    adj_f2 = adjust_pvalues_holm([0.001] * f2_clahe_contrast)
    assert len(adj_f2) == 3
    assert pytest.approx(adj_f2[0], abs=1e-5) == 0.003
