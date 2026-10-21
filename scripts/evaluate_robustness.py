"""Full robustness evaluation pipeline across 10 experimental conditions on held-out test cohort."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.corruptions import apply_corruption
from src.data.dataset import ACDCSliceDataset, acdc_collate_fn
from src.evaluation.metrics import (
    CLASS_NAME_MAP,
    FOREGROUND_CLASS_INDICES,
    MetricAccumulator,
    compute_multiclass_metrics_numpy,
)
from src.evaluation.patient_metrics import (
    PatientMetricAccumulator,
    adjust_pvalues_holm,
    compute_paired_patient_comparison,
)
from src.models.unet2d import LightweightUNet2D
from src.utils.config_parser import load_yaml_config
from src.utils.reproducibility import get_device, set_seed


CONDITIONS: List[Dict[str, Any]] = [
    {
        "condition_id": "clean_baseline",
        "family": "clean",
        "severity": "none",
        "param_value": 0.0,
        "units": "none",
        "display_name": "Clean Baseline",
    },
    {
        "condition_id": "gaussian_noise_mild",
        "family": "gaussian_noise",
        "severity": "mild",
        "param_value": 0.02,
        "units": "normalized std",
        "display_name": "Gaussian Noise (Mild, std=0.02)",
    },
    {
        "condition_id": "gaussian_noise_moderate",
        "family": "gaussian_noise",
        "severity": "moderate",
        "param_value": 0.05,
        "units": "normalized std",
        "display_name": "Gaussian Noise (Mod, std=0.05)",
    },
    {
        "condition_id": "gaussian_noise_severe",
        "family": "gaussian_noise",
        "severity": "severe",
        "param_value": 0.10,
        "units": "normalized std",
        "display_name": "Gaussian Noise (Sev, std=0.10)",
    },
    {
        "condition_id": "gaussian_blur_mild",
        "family": "gaussian_blur",
        "severity": "mild",
        "param_value": 0.75,
        "units": "pixels (sigma)",
        "display_name": "Gaussian Blur (Mild, sigma=0.75)",
    },
    {
        "condition_id": "gaussian_blur_moderate",
        "family": "gaussian_blur",
        "severity": "moderate",
        "param_value": 1.50,
        "units": "pixels (sigma)",
        "display_name": "Gaussian Blur (Mod, sigma=1.50)",
    },
    {
        "condition_id": "gaussian_blur_severe",
        "family": "gaussian_blur",
        "severity": "severe",
        "param_value": 2.50,
        "units": "pixels (sigma)",
        "display_name": "Gaussian Blur (Sev, sigma=2.50)",
    },
    {
        "condition_id": "contrast_reduction_mild",
        "family": "contrast_reduction",
        "severity": "mild",
        "param_value": 0.75,
        "units": "scaling factor alpha",
        "display_name": "Contrast Reduction (Mild, alpha=0.75)",
    },
    {
        "condition_id": "contrast_reduction_moderate",
        "family": "contrast_reduction",
        "severity": "moderate",
        "param_value": 0.50,
        "units": "scaling factor alpha",
        "display_name": "Contrast Reduction (Mod, alpha=0.50)",
    },
    {
        "condition_id": "contrast_reduction_severe",
        "family": "contrast_reduction",
        "severity": "severe",
        "param_value": 0.25,
        "units": "scaling factor alpha",
        "display_name": "Contrast Reduction (Sev, alpha=0.25)",
    },
]


def compute_sha256(path: Path) -> str:
    """Computes SHA-256 hash of a file on disk."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_condition(
    model: torch.nn.Module,
    loader: DataLoader,
    condition: Dict[str, Any],
    device: torch.device,
    base_seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    """Evaluates the model under a single corruption condition on the test DataLoader."""
    model.eval()
    slice_accumulator = MetricAccumulator()
    patient_accumulator = PatientMetricAccumulator(num_classes=4)
    representative_samples: List[Dict[str, Any]] = []

    fam = condition["family"]
    sev = condition["severity"]

    with torch.no_grad():
        for batch_idx, (images, targets, metadatas) in enumerate(loader):
            # Batch size is 1 for test evaluation
            image = images.to(device)
            target = targets.to(device)
            meta = metadatas[0]
            patient_id = meta["patient_id"]
            slice_idx = meta["slice_idx"]

            # Deterministic per-slice seed
            seed = base_seed + (batch_idx * 17) + int(slice_idx)

            # Apply corruption directly in model-input normalized space
            corrupted_img, _ = apply_corruption(
                image=image,
                corruption_name=fam,
                severity=sev,
                seed=seed,
            )

            logits = model(corrupted_img)
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy()[0]
            target_np = target.detach().cpu().numpy()[0]

            # Slice-level metrics
            slice_metrics = compute_multiclass_metrics_numpy(pred, target_np, num_classes=4)
            slice_accumulator.update(slice_metrics, batch_size=1)

            # Patient-level 3D volume pooling
            patient_accumulator.update(patient_id, pred, target_np)

            # Capture mid-ventricular representative samples for plotting
            if meta.get("has_foreground", False) and len(representative_samples) < 3 and batch_idx % 20 == 0:
                representative_samples.append({
                    "patient_id": patient_id,
                    "slice_idx": slice_idx,
                    "clean_img": image.cpu().numpy()[0, 0],
                    "corrupt_img": corrupted_img.cpu().numpy()[0, 0],
                    "target_mask": target_np,
                    "pred_mask": pred,
                })

    slice_averages = slice_accumulator.compute_averages()
    per_patient = patient_accumulator.compute_per_patient_metrics()
    patient_summary = patient_accumulator.compute_summary_statistics()

    return slice_averages, per_patient, patient_summary, representative_samples


def generate_robustness_plots(
    all_results: Dict[str, Any],
    representative_overlays: Dict[str, List[Dict[str, Any]]],
    report_dir: Path,
) -> None:
    """Generates severity-vs-Dice curves and visual MRI segmentation overlays."""
    # 1. Severity vs Macro Foreground Dice Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    families = ["gaussian_noise", "gaussian_blur", "contrast_reduction"]
    family_titles = ["Gaussian Noise", "Gaussian Blur", "Contrast Reduction"]
    x_labels = ["Noise std (normalized)", "Blur sigma (pixels)", "Contrast factor alpha"]

    for ax, fam, title, xlabel in zip(axes, families, family_titles, x_labels):
        # Gather conditions for this family
        clean_res = all_results["conditions"]["clean_baseline"]
        fam_conds = [
            c for c in CONDITIONS if c["family"] == fam
        ]

        x_vals = [0.0] + [c["param_value"] for c in fam_conds]
        dice_fg_slice = [clean_res["slice_level"]["mean_dice_foreground"]] + [
            all_results["conditions"][c["condition_id"]]["slice_level"]["mean_dice_foreground"] for c in fam_conds
        ]
        dice_fg_patient = [clean_res["patient_summary"]["mean_dice_foreground"]["mean"]] + [
            all_results["conditions"][c["condition_id"]]["patient_summary"]["mean_dice_foreground"]["mean"] for c in fam_conds
        ]
        dice_rv = [clean_res["slice_level"]["dice_rv"]] + [
            all_results["conditions"][c["condition_id"]]["slice_level"]["dice_rv"] for c in fam_conds
        ]
        dice_myo = [clean_res["slice_level"]["dice_myo"]] + [
            all_results["conditions"][c["condition_id"]]["slice_level"]["dice_myo"] for c in fam_conds
        ]
        dice_lv = [clean_res["slice_level"]["dice_lv"]] + [
            all_results["conditions"][c["condition_id"]]["slice_level"]["dice_lv"] for c in fam_conds
        ]

        ax.plot(x_vals, dice_fg_slice, "o-", label="Macro FG (Slice Avg)", linewidth=2.5, color="#1f77b4")
        ax.plot(x_vals, dice_fg_patient, "s--", label="Macro FG (Patient 3D)", linewidth=2.0, color="#2ca02c")
        ax.plot(x_vals, dice_rv, "^:", label="RV Dice", linewidth=1.5, color="#ff7f0e")
        ax.plot(x_vals, dice_myo, "d:", label="MYO Dice", linewidth=1.5, color="#9467bd")
        ax.plot(x_vals, dice_lv, "x:", label="LV Dice", linewidth=1.5, color="#d62728")

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Dice Score", fontsize=12)
        ax.set_ylim([0.0, 1.0])
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    curves_path = report_dir / "severity_vs_dice.png"
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"Saved severity curve plot to: {curves_path}")

    # 2. Representative MRI Overlays
    # Pick a sample from clean and severe conditions
    sample_key = "clean_baseline"
    if sample_key in representative_overlays and len(representative_overlays[sample_key]) > 0:
        sample = representative_overlays[sample_key][0]
        patient_id = sample["patient_id"]
        slice_idx = sample["slice_idx"]

        severe_conditions = [
            "clean_baseline",
            "gaussian_noise_severe",
            "gaussian_blur_severe",
            "contrast_reduction_severe",
        ]
        col_titles = ["Clean", "Noise (Sev, 0.10)", "Blur (Sev, 2.5px)", "Contrast (Sev, 0.25)"]

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        for col, (cond_id, c_title) in enumerate(zip(severe_conditions, col_titles)):
            if cond_id in representative_overlays and len(representative_overlays[cond_id]) > 0:
                s = representative_overlays[cond_id][0]
                img = s["corrupt_img"]
                pred = s["pred_mask"]
                tgt = s["target_mask"]

                # Row 0: MRI image
                axes[0, col].imshow(img, cmap="gray")
                axes[0, col].set_title(f"{c_title}\nMRI Slice", fontsize=11, fontweight="bold")
                axes[0, col].axis("off")

                # Row 1: Ground truth overlay
                axes[1, col].imshow(img, cmap="gray")
                axes[1, col].imshow(tgt, cmap="autumn", alpha=0.5, interpolation="none")
                axes[1, col].set_title("Ground Truth", fontsize=10)
                axes[1, col].axis("off")

                # Row 2: Prediction overlay
                axes[2, col].imshow(img, cmap="gray")
                axes[2, col].imshow(pred, cmap="winter", alpha=0.5, interpolation="none")
                axes[2, col].set_title("Model Prediction", fontsize=10)
                axes[2, col].axis("off")

        plt.suptitle(f"Representative MRI Predictions Under Severe Perturbations ({patient_id}, slice {slice_idx})", fontsize=14, y=0.98)
        plt.tight_layout()
        overlay_path = report_dir / "representative_overlays.png"
        plt.savefig(overlay_path, dpi=200)
        plt.close()
        print(f"Saved representative overlays to: {overlay_path}")


def main() -> None:
    """Executes the full controlled robustness evaluation suite."""
    parser = argparse.ArgumentParser(description="Evaluate baseline model under controlled corruptions.")
    parser.add_argument("--config", type=str, default="configs/baseline_unet.yaml", help="Path to config file.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_frozen.pt", help="Path to frozen checkpoint.")
    parser.add_argument("--device", type=str, default="auto", help="Execution device.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)
    device = get_device(args.device)

    # 1. Load frozen checkpoint and verify integrity
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/best_model.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    ckpt_hash = compute_sha256(ckpt_path)
    print(f"Loading frozen checkpoint: {ckpt_path}")
    print(f"Checkpoint SHA-256: {ckpt_hash}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = LightweightUNet2D().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model state loaded and set to eval() mode.")

    # 2. Load held-out test dataset
    manifest_path = PROJECT_ROOT / "data" / "processed" / "acdc_slice_manifest.json"
    target_size = tuple(cfg.get("dataset", {}).get("target_spatial_size", [256, 256]))
    norm_method = cfg.get("preprocessing", {}).get("baseline", {}).get("normalization", "z_score")
    clip_percentiles = tuple(cfg.get("preprocessing", {}).get("baseline", {}).get("clip_percentiles", [0.5, 99.5]))

    test_dataset = ACDCSliceDataset(
        manifest_path=manifest_path,
        split="test",
        target_spatial_size=target_size,
        normalization_method=norm_method,
        clip_percentiles=clip_percentiles,
        cache_data=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=acdc_collate_fn,
        num_workers=0,
    )
    print(f"Loaded held-out test split: {len(test_dataset)} slices across 15 patients.")

    # 3. Iterate over all 10 experimental conditions
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Any] = {
        "metadata": {
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": ckpt_hash,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": seed,
            "device": str(device),
            "total_test_slices": len(test_dataset),
            "n_patients": 15,
        },
        "conditions": {},
        "paired_comparisons": {},
    }

    clean_patient_metrics: Dict[str, Dict[str, float]] = {}
    representative_overlays: Dict[str, List[Dict[str, Any]]] = {}

    print("\n" + "=" * 70)
    print("         EXECUTING PHASE 4 CONTROLLED ROBUSTNESS SUITE")
    print("=" * 70)

    for cond in CONDITIONS:
        cond_id = cond["condition_id"]
        d_name = cond["display_name"]
        print(f"\nEvaluating Condition: {d_name}...")

        slice_avg, per_patient, patient_summary, rep_samples = evaluate_condition(
            model=model,
            loader=test_loader,
            condition=cond,
            device=device,
            base_seed=seed,
        )

        all_results["conditions"][cond_id] = {
            "condition_info": cond,
            "slice_level": slice_avg,
            "patient_level": per_patient,
            "patient_summary": patient_summary,
        }
        representative_overlays[cond_id] = rep_samples

        if cond["family"] == "clean":
            clean_patient_metrics = per_patient

        # Print live condition summary
        sl_fg_dice = slice_avg.get("mean_dice_foreground", 0.0)
        pt_fg_dice = patient_summary.get("mean_dice_foreground", {}).get("mean", 0.0)
        pt_fg_std = patient_summary.get("mean_dice_foreground", {}).get("std", 0.0)
        print(f"  -> Slice-level Macro FG Dice: {sl_fg_dice:.4f} | IoU: {slice_avg.get('mean_iou_foreground', 0.0):.4f}")
        print(f"  -> Patient-level 3D FG Dice:  {pt_fg_dice:.4f} +/- {pt_fg_std:.4f}")
        print(f"     (RV: {slice_avg.get('dice_rv', 0.0):.4f}, MYO: {slice_avg.get('dice_myo', 0.0):.4f}, LV: {slice_avg.get('dice_lv', 0.0):.4f})")

    # 4. Compute paired patient-level comparisons against clean baseline
    paired_cond_ids = [c["condition_id"] for c in CONDITIONS if c["family"] != "clean"]
    raw_t_pvals: List[float] = []
    raw_w_pvals: List[float] = []

    for cond_id in paired_cond_ids:
        corrupt_patient_metrics = all_results["conditions"][cond_id]["patient_level"]
        paired_stats = compute_paired_patient_comparison(
            clean_metrics=clean_patient_metrics,
            corrupt_metrics=corrupt_patient_metrics,
            target_metric="mean_dice_foreground",
        )
        all_results["paired_comparisons"][cond_id] = paired_stats
        raw_t_pvals.append(paired_stats["paired_t_pvalue"])
        raw_w_pvals.append(paired_stats["wilcoxon_pvalue"])

    # Family-Wise Error Rate (FWER) control via Holm-Bonferroni correction (m=9 hypotheses)
    holm_t_pvals = adjust_pvalues_holm(raw_t_pvals)
    holm_w_pvals = adjust_pvalues_holm(raw_w_pvals)

    for cond_id, h_t, h_w in zip(paired_cond_ids, holm_t_pvals, holm_w_pvals):
        all_results["paired_comparisons"][cond_id]["paired_t_pvalue_holm"] = h_t
        all_results["paired_comparisons"][cond_id]["wilcoxon_pvalue_holm"] = h_w

    # 5. Export JSON report
    json_path = report_dir / "robustness_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWritten full JSON results to: {json_path}")

    # 6. Export CSV summary table
    csv_path = report_dir / "robustness_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition_id",
            "family",
            "severity",
            "param_value",
            "units",
            "slice_dice_fg",
            "slice_iou_fg",
            "slice_dice_rv",
            "slice_dice_myo",
            "slice_dice_lv",
            "patient_dice_fg_mean",
            "patient_dice_fg_std",
            "patient_dice_fg_median",
            "patient_dice_fg_iqr",
            "delta_dice_abs_mean",
            "delta_dice_rel_pct_patient_mean",
            "delta_dice_rel_pct_cohort_mean",
            "paired_t_pvalue_raw",
            "paired_t_pvalue_holm",
            "wilcoxon_pvalue_raw",
            "wilcoxon_pvalue_holm",
        ])

        clean_cohort_mean = all_results["conditions"]["clean_baseline"]["patient_summary"]["mean_dice_foreground"]["mean"]

        for cond in CONDITIONS:
            cond_id = cond["condition_id"]
            c_data = all_results["conditions"][cond_id]
            sl = c_data["slice_level"]
            ps = c_data["patient_summary"]["mean_dice_foreground"]
            paired = all_results["paired_comparisons"].get(cond_id, {})

            # Cohort mean percentage change
            curr_cohort_mean = ps["mean"]
            rel_pct_cohort = ((curr_cohort_mean - clean_cohort_mean) / clean_cohort_mean * 100.0) if clean_cohort_mean > 0 else 0.0

            writer.writerow([
                cond_id,
                cond["family"],
                cond["severity"],
                cond["param_value"],
                cond["units"],
                round(sl.get("mean_dice_foreground", 0.0), 4),
                round(sl.get("mean_iou_foreground", 0.0), 4),
                round(sl.get("dice_rv", 0.0), 4),
                round(sl.get("dice_myo", 0.0), 4),
                round(sl.get("dice_lv", 0.0), 4),
                round(ps["mean"], 4),
                round(ps["std"], 4),
                round(ps["median"], 4),
                round(ps["iqr"], 4),
                round(paired.get("mean_absolute_change", 0.0), 4),
                round(paired.get("mean_relative_change_pct_patient_wise", 0.0), 2),
                round(rel_pct_cohort, 2),
                f"{paired.get('paired_t_pvalue', 1.0):.4e}" if "paired_t_pvalue" in paired else "N/A",
                f"{paired.get('paired_t_pvalue_holm', 1.0):.4e}" if "paired_t_pvalue_holm" in paired else "N/A",
                f"{paired.get('wilcoxon_pvalue', 1.0):.4e}" if "wilcoxon_pvalue" in paired else "N/A",
                f"{paired.get('wilcoxon_pvalue_holm', 1.0):.4e}" if "wilcoxon_pvalue_holm" in paired else "N/A",
            ])
    print(f"Written summary CSV to: {csv_path}")

    # 7. Generate plots
    generate_robustness_plots(all_results, representative_overlays, report_dir)

    print("\nRobustness evaluation completed successfully across all 10 conditions!")


if __name__ == "__main__":
    main()
