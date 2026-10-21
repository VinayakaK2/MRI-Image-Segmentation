"""Command-line runner for training and evaluating the baseline Lightweight 2D U-Net."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import ACDCSliceDataset, acdc_collate_fn
from src.models.unet2d import LightweightUNet2D
from src.training.trainer import Trainer
from src.utils.config_parser import get_project_root, load_yaml_config, resolve_project_path
from src.utils.reproducibility import get_device, set_seed


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train baseline Lightweight 2D U-Net on ACDC Cardiac MRI")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_unet.yaml",
        help="Path to YAML experiment configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for maximum training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for training batch size",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Runs an abbreviated smoke run (1 epoch, minimal batches) to verify execution",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run training on ('cuda', 'cpu', or 'auto')",
    )
    return parser.parse_args()


def run_baseline_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Executes deterministic seed setup, dataset loading, model training, and final test evaluation."""
    project_root = get_project_root()
    cfg = load_yaml_config(args.config)

    # 1. Reproducibility & Device setup
    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed=seed, deterministic_cudnn=cfg.get("project", {}).get("deterministic", True))

    device_str = args.device if args.device is not None else cfg.get("project", {}).get("device", "auto")
    device = get_device(device_str)
    print(f"Executing pipeline on device: {device} (Seed: {seed})")

    # 2. Dataset manifests and loaders
    manifest_rel = "data/processed/acdc_slice_manifest.json"
    manifest_path = project_root / manifest_rel
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Slice manifest not found at {manifest_path}. Build Phase 2 manifests first."
        )

    batch_size = args.batch_size if args.batch_size is not None else cfg.get("training", {}).get("batch_size", 16)
    eval_batch_size = cfg.get("evaluation", {}).get("eval_batch_size", 1)
    target_size = tuple(cfg.get("dataset", {}).get("target_spatial_size", [256, 256]))

    norm_cfg = cfg.get("preprocessing", {}).get("baseline", {})
    norm_method = norm_cfg.get("normalization", "z_score")
    clip_percentiles = tuple(norm_cfg.get("clip_percentiles", [0.5, 99.5]))

    print(f"Loading datasets from manifest: {manifest_path}")
    train_dataset = ACDCSliceDataset(
        manifest_path=manifest_path,
        split="train",
        target_spatial_size=target_size,
        normalization_method=norm_method,
        clip_percentiles=clip_percentiles,
        cache_data=True,
    )
    val_dataset = ACDCSliceDataset(
        manifest_path=manifest_path,
        split="val",
        target_spatial_size=target_size,
        normalization_method=norm_method,
        clip_percentiles=clip_percentiles,
        cache_data=True,
    )
    test_dataset = ACDCSliceDataset(
        manifest_path=manifest_path,
        split="test",
        target_spatial_size=target_size,
        normalization_method=norm_method,
        clip_percentiles=clip_percentiles,
        cache_data=True,
    )

    print(f"Dataset partitions loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Worker count adjusted defensively
    num_workers = 0 if sys.platform.startswith("win") else 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=acdc_collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=acdc_collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=acdc_collate_fn,
        num_workers=num_workers,
    )

    # 3. Model instantiation
    model = LightweightUNet2D.from_config(cfg)
    param_counts = model.count_parameters()
    print(f"Instantiated LightweightUNet2D with {param_counts['total']:,} parameters.")

    # 4. Trainer setup
    checkpoint_dir = resolve_project_path(cfg.get("training", {}).get("checkpoint_dir", "checkpoints"))
    report_dir = resolve_project_path(cfg.get("evaluation", {}).get("report_dir", "reports"))

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device,
        checkpoint_dir=checkpoint_dir,
        report_dir=report_dir,
    )

    # 5. Training execution
    max_epochs = 1 if args.smoke else (args.epochs if args.epochs is not None else cfg.get("training", {}).get("epochs", 50))
    train_summary = trainer.train(max_epochs=max_epochs)

    # 6. Held-out test evaluation using best saved model
    best_model_path = checkpoint_dir / "best_model.pt"
    if not best_model_path.exists():
        best_model_path = checkpoint_dir / "last_checkpoint.pt"

    print(f"\nLoading best checkpoint for final evaluation: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Evaluating held-out 15-patient test cohort...")
    test_metrics = trainer.evaluate_dataset(test_loader)
    print("\n" + "=" * 60)
    print("      HELD-OUT 15-PATIENT TEST EVALUATION RESULTS")
    print("=" * 60)
    print(f"Macro Foreground Dice: {test_metrics.get('mean_dice_foreground', 0.0):.4f}")
    print(f"Macro Foreground IoU:  {test_metrics.get('mean_iou_foreground', 0.0):.4f}")
    print(f"Right Ventricle (RV) Dice: {test_metrics.get('dice_rv', 0.0):.4f} | IoU: {test_metrics.get('iou_rv', 0.0):.4f}")
    print(f"Myocardium (MYO) Dice:     {test_metrics.get('dice_myo', 0.0):.4f} | IoU: {test_metrics.get('iou_myo', 0.0):.4f}")
    print(f"Left Ventricle (LV) Dice:  {test_metrics.get('dice_lv', 0.0):.4f} | IoU: {test_metrics.get('iou_lv', 0.0):.4f}")
    print(f"Background Dice:           {test_metrics.get('dice_background', 0.0):.4f} | IoU: {test_metrics.get('iou_background', 0.0):.4f}")
    print("=" * 60)

    # Save test evaluation report
    test_report_path = report_dir / "baseline_test_evaluation.json"
    report_payload = {
        "model_architecture": "LightweightUNet2D",
        "parameters": param_counts,
        "best_epoch": train_summary["best_epoch"],
        "best_val_foreground_dice": train_summary["best_val_foreground_dice"],
        "test_metrics": test_metrics,
        "config": cfg,
    }
    with open(test_report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"Test evaluation results written to: {test_report_path}")

    return report_payload


if __name__ == "__main__":
    cli_args = parse_arguments()
    run_baseline_pipeline(cli_args)
