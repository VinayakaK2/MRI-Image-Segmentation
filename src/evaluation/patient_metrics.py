"""Patient-level 3D volume metric pooling and statistical comparison framework."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats

from src.evaluation.metrics import CLASS_NAME_MAP, FOREGROUND_CLASS_INDICES


class PatientMetricAccumulator:
    """Accumulates voxel-level intersections and cardinalities per patient across all 2D slices.

    This implements patient-level 3D pooling where volume overlap is computed over
    the union of all slices belonging to a patient, rather than averaging per-slice scores.
    """

    def __init__(self, num_classes: int = 4) -> None:
        """Initializes accumulator stores."""
        self.num_classes = num_classes
        # Structure: {patient_id: {class_idx: {'intersection': int, 'pred_sum': int, 'target_sum': int}}}
        self.stores: Dict[str, Dict[int, Dict[str, int]]] = {}

    def update(
        self,
        patient_id: str,
        pred_slice: np.ndarray,
        target_slice: np.ndarray,
    ) -> None:
        """Accumulates voxel overlap counts for a single 2D slice into the patient's store."""
        if patient_id not in self.stores:
            self.stores[patient_id] = {
                c: {"intersection": 0, "pred_sum": 0, "target_sum": 0}
                for c in range(self.num_classes)
            }

        for c in range(self.num_classes):
            c_pred = (pred_slice == c)
            c_target = (target_slice == c)

            inter = int(np.sum(c_pred & c_target))
            p_sum = int(np.sum(c_pred))
            t_sum = int(np.sum(c_target))

            self.stores[patient_id][c]["intersection"] += inter
            self.stores[patient_id][c]["pred_sum"] += p_sum
            self.stores[patient_id][c]["target_sum"] += t_sum

    def compute_per_patient_metrics(self) -> Dict[str, Dict[str, float]]:
        """Computes 3D pooled Dice and IoU for each registered patient."""
        results: Dict[str, Dict[str, float]] = {}

        for patient_id, class_data in self.stores.items():
            patient_metrics: Dict[str, float] = {}
            fg_dices: List[float] = []
            fg_ious: List[float] = []

            for c in range(self.num_classes):
                c_name = CLASS_NAME_MAP.get(c, f"class_{c}")
                inter = class_data[c]["intersection"]
                p_sum = class_data[c]["pred_sum"]
                t_sum = class_data[c]["target_sum"]
                union = p_sum + t_sum - inter

                if t_sum == 0 and p_sum == 0:
                    dice = 1.0
                    iou = 1.0
                elif t_sum == 0 or p_sum == 0 or union == 0:
                    dice = 0.0
                    iou = 0.0
                else:
                    dice = (2.0 * inter) / (p_sum + t_sum)
                    iou = inter / union

                patient_metrics[f"dice_{c_name}"] = float(dice)
                patient_metrics[f"iou_{c_name}"] = float(iou)

                if c in FOREGROUND_CLASS_INDICES:
                    fg_dices.append(float(dice))
                    fg_ious.append(float(iou))

            patient_metrics["mean_dice_foreground"] = float(np.mean(fg_dices)) if fg_dices else 0.0
            patient_metrics["mean_iou_foreground"] = float(np.mean(fg_ious)) if fg_ious else 0.0
            results[patient_id] = patient_metrics

        return results

    def compute_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculates cohort-level summary statistics across all evaluated patients."""
        per_patient = self.compute_per_patient_metrics()
        if not per_patient:
            return {}

        metric_names = list(next(iter(per_patient.values())).keys())
        summary: Dict[str, Dict[str, float]] = {}

        n_patients = len(per_patient)
        # Critical value for 95% two-tailed confidence interval (Student-t with df = n-1)
        t_crit = float(stats.t.ppf(0.975, df=max(n_patients - 1, 1)))

        for m in metric_names:
            vals = np.array([p[m] for p in per_patient.values()], dtype=np.float64)
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if n_patients > 1 else 0.0
            sem = std / np.sqrt(n_patients) if n_patients > 0 else 0.0
            q25, median, q75 = np.percentile(vals, [25, 50, 75])

            summary[m] = {
                "mean": mean,
                "std": std,
                "median": float(median),
                "iqr": float(q75 - q25),
                "q25": float(q25),
                "q75": float(q75),
                "ci_95_lower": float(mean - t_crit * sem),
                "ci_95_upper": float(mean + t_crit * sem),
                "n_patients": n_patients,
            }

        return summary


def adjust_pvalues_holm(p_values: List[float]) -> List[float]:
    """Applies the Holm-Bonferroni step-down correction to control Family-Wise Error Rate (FWER).

    Given m p-values, sorts them in ascending order: p_(1) <= ... <= p_(m).
    Adjusted p-value is: p_adj,(k) = min(1.0, max_{j <= k} (m - j + 1) * p_(j)).

    Args:
        p_values: List of uncorrected float p-values.

    Returns:
        List of Holm-adjusted p-values in the original corresponding input order.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Track original indices to restore order after step-down adjustment
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted_sorted = [0.0] * m
    running_max = 0.0

    for rank, (orig_idx, p_val) in enumerate(indexed):
        mult = m - rank
        raw_adj = p_val * mult
        running_max = max(running_max, raw_adj)
        adjusted_sorted[rank] = min(1.0, float(running_max))

    result = [0.0] * m
    for (orig_idx, _), adj_p in zip(indexed, adjusted_sorted):
        result[orig_idx] = adj_p
    return result


def compute_paired_patient_comparison(
    clean_metrics: Dict[str, Dict[str, float]],
    corrupt_metrics: Dict[str, Dict[str, float]],
    target_metric: str = "mean_dice_foreground",
) -> Dict[str, Any]:
    """Performs paired statistical testing across identical patients under clean vs corrupted conditions."""
    common_patients = sorted(list(set(clean_metrics.keys()) & set(corrupt_metrics.keys())))
    if len(common_patients) < 2:
        raise ValueError(f"Need at least 2 common patients for paired comparison, found {len(common_patients)}")

    clean_vals = np.array([clean_metrics[p][target_metric] for p in common_patients], dtype=np.float64)
    corrupt_vals = np.array([corrupt_metrics[p][target_metric] for p in common_patients], dtype=np.float64)

    # Absolute difference: corrupt - clean
    diffs = corrupt_vals - clean_vals

    # Definition A: Mean of patient-wise relative percentage changes
    # Formula: (1/N) * sum_p ((Dice_corrupt,p - Dice_clean,p) / Dice_clean,p) * 100%
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diffs = np.where(clean_vals > 0, (diffs / clean_vals) * 100.0, 0.0)

    # Definition B: Relative percentage change of cohort means
    # Formula: ((Mean_corrupt - Mean_clean) / Mean_clean) * 100%
    mean_clean = float(np.mean(clean_vals))
    mean_corrupt = float(np.mean(corrupt_vals))
    rel_change_of_means = ((mean_corrupt - mean_clean) / mean_clean * 100.0) if mean_clean > 0 else 0.0

    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(corrupt_vals, clean_vals)

    # Wilcoxon signed-rank test (non-parametric fallback)
    try:
        w_stat, w_pval = stats.wilcoxon(diffs, zero_method="wilcox")
    except Exception:
        w_stat, w_pval = 0.0, 1.0

    return {
        "metric": target_metric,
        "n_patients": len(common_patients),
        "mean_clean": mean_clean,
        "mean_corrupt": mean_corrupt,
        "mean_absolute_change": float(np.mean(diffs)),
        "std_absolute_change": float(np.std(diffs, ddof=1)),
        "median_absolute_change": float(np.median(diffs)),
        "mean_relative_change_pct": float(np.mean(rel_diffs)),
        "mean_relative_change_pct_patient_wise": float(np.mean(rel_diffs)),
        "relative_change_pct_of_means": float(rel_change_of_means),
        "median_relative_change_pct": float(np.median(rel_diffs)),
        "paired_t_statistic": float(t_stat) if not np.isnan(t_stat) else 0.0,
        "paired_t_pvalue": float(t_pval) if not np.isnan(t_pval) else 1.0,
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_pval),
        "per_patient_differences": {
            p: {
                "clean": float(c),
                "corrupted": float(k),
                "absolute_change": float(d),
                "relative_change_pct": float(r),
            }
            for p, c, k, d, r in zip(common_patients, clean_vals, corrupt_vals, diffs, rel_diffs)
        },
    }
