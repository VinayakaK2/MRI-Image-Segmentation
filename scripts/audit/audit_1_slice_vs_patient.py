"""Audit 1: Slice-Level vs Patient-Level 3D Dice Verification and Mathematical Audit."""

import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ACDCSliceDataset, acdc_collate_fn
from src.evaluation.metrics import (
    CLASS_NAME_MAP,
    FOREGROUND_CLASS_INDICES,
    MetricAccumulator,
    compute_binary_dice_and_iou,
    compute_multiclass_metrics_numpy,
)
from src.evaluation.patient_metrics import PatientMetricAccumulator
from src.models.unet2d import LightweightUNet2D
from src.utils.config_parser import load_yaml_config


def run_audit_1():
    print("======================================================================")
    print("   AUDIT 1: SLICE-LEVEL VS PATIENT-LEVEL 3D DICE VERIFICATION")
    print("======================================================================")

    # 1. Inspect existing 50-epoch Phase 4 JSON
    json_path = PROJECT_ROOT / "experiments" / "phase4_robustness_50epoch_best_epoch44" / "robustness_results.json"
    if not json_path.exists():
        json_path = PROJECT_ROOT / "reports" / "phase4_50epochs" / "robustness_results.json"
    
    with open(json_path, "r", encoding="utf-8") as f:
        reported_data = json.load(f)

    clean_slice = reported_data["conditions"]["clean_baseline"]["slice_level"]
    clean_patient = reported_data["conditions"]["clean_baseline"]["patient_level"]
    clean_summary = reported_data["conditions"]["clean_baseline"]["patient_summary"]

    reported_slice_fg = clean_slice["mean_dice_foreground"]
    reported_patient_mean = clean_summary["mean_dice_foreground"]["mean"]
    reported_patient_std = clean_summary["mean_dice_foreground"]["std"]

    print(f"Reported Clean Slice FG Dice:       {reported_slice_fg:.4f} ({reported_slice_fg})")
    print(f"Reported Clean Patient 3D FG Dice:  {reported_patient_mean:.4f} +/- {reported_patient_std:.4f}")
    print(f"Reported Number of Patients:        {clean_summary['mean_dice_foreground']['n_patients']}")
    print(f"Reported Patient IDs:               {sorted(list(clean_patient.keys()))}")

    # 2. Independent evaluation from raw checkpoint and dataset
    config_path = PROJECT_ROOT / "configs" / "baseline_unet.yaml"
    cfg = load_yaml_config(config_path)

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
        num_workers=0,
        collate_fn=acdc_collate_fn,
    )

    ckpt_path = PROJECT_ROOT / "checkpoints" / "50epochs" / "best_model_50epochs.pt"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "checkpoints" / "best_model_50epochs.pt"

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = LightweightUNet2D()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    slice_acc = MetricAccumulator()
    patient_acc = PatientMetricAccumulator(num_classes=4)

    slice_records = []
    patient_slice_counts = {}

    with torch.no_grad():
        for batch_idx, (images, targets, metadatas) in enumerate(test_loader):
            meta = metadatas[0]
            pid = meta["patient_id"]
            sidx = meta["slice_idx"]

            patient_slice_counts[pid] = patient_slice_counts.get(pid, 0) + 1

            logits = model(images)
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
            target_np = targets.cpu().numpy()[0]

            # Slice-level metrics
            s_metrics = compute_multiclass_metrics_numpy(pred, target_np, num_classes=4)
            slice_acc.update(s_metrics, batch_size=1)

            # Patient-level accumulation
            patient_acc.update(pid, pred, target_np)

            # Record slice anatomical info for mathematical investigation
            rv_vox = int(np.sum(target_np == 1))
            myo_vox = int(np.sum(target_np == 2))
            lv_vox = int(np.sum(target_np == 3))
            fg_vox = rv_vox + myo_vox + lv_vox

            slice_records.append({
                "patient_id": pid,
                "slice_idx": sidx,
                "fg_voxels": fg_vox,
                "rv_voxels": rv_vox,
                "myo_voxels": myo_vox,
                "lv_voxels": lv_vox,
                "slice_dice_fg": s_metrics["mean_dice_foreground"],
                "slice_dice_rv": s_metrics["dice_rv"],
                "slice_dice_myo": s_metrics["dice_myo"],
                "slice_dice_lv": s_metrics["dice_lv"],
            })

    # Computed metrics
    comp_slice_averages = slice_acc.compute_averages()
    comp_per_patient = patient_acc.compute_per_patient_metrics()
    comp_patient_summary = patient_acc.compute_summary_statistics()

    comp_slice_fg = comp_slice_averages["mean_dice_foreground"]
    comp_patient_mean = comp_patient_summary["mean_dice_foreground"]["mean"]
    comp_patient_std = comp_patient_summary["mean_dice_foreground"]["std"]

    print("\n--- INDEPENDENT RECALCULATION OUTCOME ---")
    print(f"Total Test Slices Evaluated:        {len(slice_records)}")
    print(f"Total Patients Evaluated:          {len(comp_per_patient)}")
    print(f"Recalculated Slice FG Dice:        {comp_slice_fg:.4f} ({comp_slice_fg})")
    print(f"Recalculated Patient 3D FG Dice:   {comp_patient_mean:.4f} +/- {comp_patient_std:.4f}")

    slice_match = np.isclose(comp_slice_fg, reported_slice_fg, atol=1e-6)
    patient_mean_match = np.isclose(comp_patient_mean, reported_patient_mean, atol=1e-6)
    patient_std_match = np.isclose(comp_patient_std, reported_patient_std, atol=1e-6)

    print(f"Slice FG Dice Match:               {slice_match}")
    print(f"Patient 3D Mean Match:             {patient_mean_match}")
    print(f"Patient 3D Std Match:              {patient_std_match}")

    # 3. Mathematical Analysis: Why does slice-level differ from patient-level?
    print("\n--- MATHEMATICAL EXPLANATION: WHY SLICE (0.8297) != PATIENT 3D (0.8958) ---")
    # Sort slices by foreground size
    sorted_by_size = sorted(slice_records, key=lambda x: x["fg_voxels"])
    bottom_10 = sorted_by_size[:15]
    top_10 = sorted_by_size[-15:]

    avg_dice_smallest = np.mean([s["slice_dice_fg"] for s in bottom_10])
    avg_dice_largest = np.mean([s["slice_dice_fg"] for s in top_10])
    avg_vox_smallest = np.mean([s["fg_voxels"] for s in bottom_10])
    avg_vox_largest = np.mean([s["fg_voxels"] for s in top_10])

    print(f"Smallest 15 slices (apical/basal):  Mean voxels = {avg_vox_smallest:.1f}, Mean Slice Dice = {avg_dice_smallest:.4f}")
    print(f"Largest 15 slices (mid-ventricle):  Mean voxels = {avg_vox_largest:.1f}, Mean Slice Dice = {avg_dice_largest:.4f}")

    # Also inspect absent classes
    absent_rv_slices = [s for s in slice_records if s["rv_voxels"] == 0]
    print(f"Slices where RV is absent in GT:    {len(absent_rv_slices)} out of 131 slices")
    for s in absent_rv_slices[:5]:
        print(f"  Patient {s['patient_id']} slice {s['slice_idx']}: GT RV=0, Slice RV Dice={s['slice_dice_rv']:.4f}")

    audit_1_result = {
        "verdict": "VERIFIED" if (slice_match and patient_mean_match and patient_std_match) else "FAILED",
        "slice_fg_dice_reported": reported_slice_fg,
        "slice_fg_dice_recalculated": comp_slice_fg,
        "patient_3d_mean_reported": reported_patient_mean,
        "patient_3d_mean_recalculated": comp_patient_mean,
        "patient_3d_std_reported": reported_patient_std,
        "patient_3d_std_recalculated": comp_patient_std,
        "n_patients": len(comp_per_patient),
        "n_slices": len(slice_records),
        "patient_slice_counts": patient_slice_counts,
        "mathematical_reasons": [
            "Slice-level macro Dice assigns equal weight (1/131) to small apical/basal slices where minor boundary misalignments drastically reduce Dice.",
            "Patient-level 3D Dice pools intersection and cardinality over the full 3D cardiac volume, naturally weighting each slice proportional to its voxel volume.",
            "Empty/absent class handling on individual 2D slices penalizes slice averages if false positive pixels exist, whereas in 3D pooling the class is present across the full volume."
        ]
    }

    out_file = PROJECT_ROOT / "scripts" / "audit" / "audit_1_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_1_result, f, indent=2)

    print(f"\nAudit 1 Verdict: {audit_1_result['verdict']}")
    return audit_1_result


if __name__ == "__main__":
    run_audit_1()
