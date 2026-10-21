"""Training engine, epoch optimization loop, validation, and checkpoint management."""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.evaluation.metrics import MetricAccumulator, evaluate_batch_predictions
from src.models.unet2d import LightweightUNet2D
from src.training.losses import DiceCELoss
from src.utils.reproducibility import get_device, set_seed


class Trainer:
    """Manages model optimization, validation evaluation, metric logging, and checkpoint persistence."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[Path] = None,
        report_dir: Optional[Path] = None,
    ) -> None:
        """Initializes the training engine with model, loaders, loss function, and optimizer."""
        self.config = config
        self.device = device if device is not None else get_device(config.get("project", {}).get("device", "auto"))
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        training_cfg = config.get("training", {})
        self.max_epochs = training_cfg.get("epochs", 50)
        self.learning_rate = training_cfg.get("learning_rate", 0.001)
        self.weight_decay = training_cfg.get("weight_decay", 0.0001)
        self.min_lr = training_cfg.get("min_lr", 0.00001)
        self.patience = training_cfg.get("early_stopping_patience", 10)
        self.mixed_precision = training_cfg.get("mixed_precision", False) and self.device.type == "cuda"

        # Objective criterion
        self.criterion = DiceCELoss(
            num_classes=config.get("model", {}).get("num_classes", 4),
            ce_weight=1.0,
            dice_weight=1.0,
            include_background_dice=True,
        )

        # Optimizer & Scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_epochs,
            eta_min=self.min_lr,
        )

        # Checkpoints & Reports directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.report_dir = Path(report_dir) if report_dir else Path("reports")
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Optional Google Drive durable backup path if mounted
        self.drive_backup_dir: Optional[Path] = None
        if os.path.exists("/content/drive/MyDrive"):
            self.drive_backup_dir = Path("/content/drive/MyDrive/cardiac_mri_checkpoints")
            self.drive_backup_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.best_val_dice: float = -1.0
        self.best_epoch: int = -1
        self.patience_counter: int = 0
        self.history: List[Dict[str, Any]] = []

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Executes one training epoch across the training DataLoader."""
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        metric_acc = MetricAccumulator()

        for batch_idx, (images, targets, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"Loss diverged to NaN/Inf at epoch {epoch}, batch {batch_idx}")

            loss.backward()
            # Gradient clipping to maintain stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            batch_count += batch_size

            batch_metrics = evaluate_batch_predictions(logits, targets)
            metric_acc.update(batch_metrics, batch_size=batch_size)

        avg_loss = total_loss / max(batch_count, 1)
        epoch_metrics = metric_acc.compute_averages()
        epoch_metrics["loss"] = avg_loss
        return epoch_metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """Evaluates model performance across the validation DataLoader without gradient computation."""
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        metric_acc = MetricAccumulator()

        with torch.no_grad():
            for images, targets, _ in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, targets)

                batch_size = images.size(0)
                total_loss += float(loss.item()) * batch_size
                batch_count += batch_size

                batch_metrics = evaluate_batch_predictions(logits, targets)
                metric_acc.update(batch_metrics, batch_size=batch_size)

        avg_loss = total_loss / max(batch_count, 1)
        val_metrics = metric_acc.compute_averages()
        val_metrics["loss"] = avg_loss
        return val_metrics

    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, float], is_best: bool = False) -> Path:
        """Saves a model checkpoint bundle containing state dicts, configuration, and metrics."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_metrics": val_metrics,
            "best_val_dice": self.best_val_dice,
            "config": self.config,
            "seed": self.config.get("project", {}).get("seed", 42),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save last checkpoint
        last_path = self.checkpoint_dir / "last_checkpoint.pt"
        torch.save(state, last_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)

            # Replicate to Google Drive if active
            if self.drive_backup_dir is not None:
                try:
                    drive_best = self.drive_backup_dir / "best_model.pt"
                    shutil.copyfile(best_path, drive_best)
                except Exception as err:
                    print(f"Warning: Failed to back up best checkpoint to Drive: {err}")

            return best_path

        return last_path

    def train(self, max_epochs: Optional[int] = None) -> Dict[str, Any]:
        """Runs the complete training and validation cycle with early stopping and best checkpoint selection."""
        epochs_to_run = max_epochs if max_epochs is not None else self.max_epochs
        print(f"Starting training on device: {self.device} (Total Epochs: {epochs_to_run})")

        start_time = time.time()

        for epoch in range(1, epochs_to_run + 1):
            ep_start = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            self.scheduler.step()

            current_lr = self.scheduler.get_last_lr()[0]
            val_fg_dice = val_metrics.get("mean_dice_foreground", 0.0)

            # Check for improvement on validation macro foreground Dice
            is_best = val_fg_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = val_fg_dice
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, val_metrics, is_best=False)

            ep_time = time.time() - ep_start
            log_entry = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_dice_foreground": train_metrics.get("mean_dice_foreground", 0.0),
                "val_loss": val_metrics["loss"],
                "val_dice_foreground": val_fg_dice,
                "val_dice_rv": val_metrics.get("dice_rv", 0.0),
                "val_dice_myo": val_metrics.get("dice_myo", 0.0),
                "val_dice_lv": val_metrics.get("dice_lv", 0.0),
                "val_iou_foreground": val_metrics.get("mean_iou_foreground", 0.0),
                "time_sec": ep_time,
                "is_best": is_best,
            }
            self.history.append(log_entry)

            print(
                f"Epoch [{epoch:02d}/{epochs_to_run:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val FG Dice: {val_fg_dice:.4f} "
                f"(RV: {val_metrics.get('dice_rv', 0.0):.3f}, "
                f"MYO: {val_metrics.get('dice_myo', 0.0):.3f}, "
                f"LV: {val_metrics.get('dice_lv', 0.0):.3f}) | "
                f"{'*BEST*' if is_best else ''} "
                f"Time: {ep_time:.1f}s"
            )

            # Early stopping check
            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered: No validation improvement for {self.patience} epochs.")
                break

        total_elapsed = time.time() - start_time
        print(f"Training completed in {total_elapsed:.1f}s. Best Val FG Dice: {self.best_val_dice:.4f} at Epoch {self.best_epoch}.")

        # Save training history JSON
        history_path = self.report_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({
                "best_epoch": self.best_epoch,
                "best_val_foreground_dice": self.best_val_dice,
                "total_time_seconds": total_elapsed,
                "history": self.history,
            }, f, indent=2)

        return {
            "best_epoch": self.best_epoch,
            "best_val_foreground_dice": self.best_val_dice,
            "total_time_seconds": total_elapsed,
            "history": self.history,
        }

    def evaluate_dataset(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluates model performance over an arbitrary DataLoader (e.g. held-out test set)."""
        self.model.eval()
        metric_acc = MetricAccumulator()

        with torch.no_grad():
            for images, targets, _ in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(images)
                batch_metrics = evaluate_batch_predictions(logits, targets)
                metric_acc.update(batch_metrics, batch_size=images.size(0))

        return metric_acc.compute_averages()
