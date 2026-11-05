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
        checkpoint_suffix: str = "",
        early_stopping_enabled: bool = True,
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
        self.checkpoint_suffix = checkpoint_suffix
        self.early_stopping_enabled = early_stopping_enabled

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
        self.resumed_from: Optional[str] = None
        self.resumed_epoch: int = 0
        self.resumed_with_optimizer: bool = False
        self.resumed_with_scheduler: bool = False

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

    def resume_from_checkpoint(
        self,
        checkpoint_path: Any,
        history_path: Optional[Any] = None,
    ) -> int:
        """Restores model weights, optimizer, scheduler, epoch, and metric history from a saved checkpoint bundle."""
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Resume checkpoint not found at: {checkpoint_path}")

        print(f"Resuming training state from checkpoint: {ckpt_p}")
        checkpoint = torch.load(ckpt_p, map_location=self.device, weights_only=False)

        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint at {ckpt_p} is not a valid dictionary bundle.")

        if "model_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint at {ckpt_p} does not contain 'model_state_dict'.")

        # 1. Restore model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print("-> Model weights loaded successfully.")

        # 2. Restore optimizer state if present
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Ensure optimizer tensor state is placed on target device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            self.resumed_with_optimizer = True
            print("-> Optimizer state dict restored successfully.")
        else:
            self.resumed_with_optimizer = False
            print("-> Warning: Checkpoint does not contain optimizer_state_dict. Optimizer initialized fresh.")

        # 3. Restore scheduler state if present
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.resumed_with_scheduler = True
            print("-> Scheduler state dict restored successfully.")
        else:
            self.resumed_with_scheduler = False
            print("-> Warning: Checkpoint does not contain scheduler_state_dict. Scheduler initialized fresh.")

        # 4. Restore completed epoch
        completed_epoch = int(checkpoint.get("epoch", 0))
        self.resumed_epoch = completed_epoch
        self.resumed_from = str(ckpt_p)

        # 5. Restore best validation metrics and best epoch
        self.best_val_dice = float(checkpoint.get("best_val_dice", -1.0))
        self.best_epoch = int(checkpoint.get("epoch", completed_epoch))

        # 6. Preload training history if available
        if "history" in checkpoint and isinstance(checkpoint["history"], list):
            self.history = [
                h for h in checkpoint["history"] if int(h.get("epoch", 0)) <= completed_epoch
            ]
            print(f"-> Preloaded {len(self.history)} previous epoch history records directly from checkpoint bundle.")
        else:
            hist_p = Path(history_path) if history_path else None
            if hist_p is None:
                suffixed_hist = self.report_dir / f"training_history_{self.checkpoint_suffix}.json" if self.checkpoint_suffix else None
                default_hist = self.report_dir / "training_history.json"
                if suffixed_hist and suffixed_hist.exists():
                    hist_p = suffixed_hist
                elif default_hist.exists():
                    hist_p = default_hist

            if hist_p is not None and hist_p.exists():
                try:
                    with open(hist_p, "r", encoding="utf-8") as f:
                        hist_data = json.load(f)
                    if "history" in hist_data and isinstance(hist_data["history"], list):
                        # Filter history to keep records up to the completed resume epoch
                        self.history = [
                            h for h in hist_data["history"] if int(h.get("epoch", 0)) <= completed_epoch
                        ]
                        print(f"-> Preloaded {len(self.history)} previous epoch history records from {hist_p}.")
                    if "best_epoch" in hist_data:
                        self.best_epoch = int(hist_data["best_epoch"])
                    if "best_val_foreground_dice" in hist_data:
                        self.best_val_dice = max(self.best_val_dice, float(hist_data["best_val_foreground_dice"]))
                except Exception as err:
                    print(f"-> Warning: Failed to load previous history from {hist_p}: {err}")

        # 7. Compute patience counter relative to best epoch
        self.patience_counter = max(0, completed_epoch - self.best_epoch)
        print(
            f"-> Resumed at Completed Epoch: {completed_epoch} "
            f"(Best Epoch: {self.best_epoch}, Best Val Dice: {self.best_val_dice:.4f}, "
            f"Patience Counter: {self.patience_counter})"
        )

        return completed_epoch

    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, float], is_best: bool = False) -> Path:
        """Saves a model checkpoint bundle containing state dicts, configuration, and metrics."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_metrics": val_metrics,
            "best_val_dice": self.best_val_dice,
            "best_epoch": self.best_epoch,
            "config": self.config,
            "seed": self.config.get("project", {}).get("seed", 42),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "architecture": self.config.get("model", {}).get("architecture", "LightweightUNet2D"),
            "num_classes": self.config.get("model", {}).get("num_classes", 4),
            "total_completed_epochs": epoch,
            "dataset_name": self.config.get("dataset", {}).get("name", "ACDC"),
            "patient_split_ratios": self.config.get("dataset", {}).get("patient_split", {}),
            "optimizer_name": self.config.get("training", {}).get("optimizer", "AdamW"),
            "scheduler_name": self.config.get("training", {}).get("lr_scheduler", "CosineAnnealingLR"),
            "torch_version": torch.__version__,
            "resumed_from": self.resumed_from,
            "resumed_epoch": self.resumed_epoch,
            "resumed_with_optimizer": self.resumed_with_optimizer,
            "resumed_with_scheduler": self.resumed_with_scheduler,
            "history": self.history,
        }

        # Resolve filenames respecting optional checkpoint suffix
        last_filename = f"last_checkpoint_{self.checkpoint_suffix}.pt" if self.checkpoint_suffix else "last_checkpoint.pt"
        best_filename = f"best_model_{self.checkpoint_suffix}.pt" if self.checkpoint_suffix else "best_model.pt"

        # Save last checkpoint
        last_path = self.checkpoint_dir / last_filename
        torch.save(state, last_path)

        if is_best:
            best_path = self.checkpoint_dir / best_filename
            torch.save(state, best_path)

            # Replicate to Google Drive if active
            if self.drive_backup_dir is not None:
                try:
                    drive_best = self.drive_backup_dir / best_filename
                    shutil.copyfile(best_path, drive_best)
                except Exception as err:
                    print(f"Warning: Failed to back up best checkpoint to Drive: {err}")

            return best_path

        return last_path

    def train(
        self,
        max_epochs: Optional[int] = None,
        start_epoch: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Runs the complete training and validation cycle with early stopping and best checkpoint selection."""
        epochs_to_run = max_epochs if max_epochs is not None else self.max_epochs
        start_ep = start_epoch if start_epoch is not None else (self.resumed_epoch + 1)
        print(f"Starting training on device: {self.device} (Epochs: {start_ep}..{epochs_to_run}, Total Target: {epochs_to_run})")

        start_time = time.time()

        for epoch in range(start_ep, epochs_to_run + 1):
            ep_start = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            self.scheduler.step()

            current_lr = self.scheduler.get_last_lr()[0]
            val_fg_dice = val_metrics.get("mean_dice_foreground", 0.0)

            # Check for improvement on validation macro foreground Dice
            is_best = val_fg_dice > self.best_val_dice

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

            # Update best metrics and save checkpoint
            if is_best:
                self.best_val_dice = val_fg_dice
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, val_metrics, is_best=False)

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

            # Early stopping check only if enabled
            if self.early_stopping_enabled and self.patience_counter >= self.patience:
                print(f"Early stopping triggered: No validation improvement for {self.patience} epochs.")
                break

        total_elapsed = time.time() - start_time
        print(f"Training completed in {total_elapsed:.1f}s. Best Val FG Dice: {self.best_val_dice:.4f} at Epoch {self.best_epoch}.")

        # Save training history JSON respecting suffix
        history_filename = f"training_history_{self.checkpoint_suffix}.json" if self.checkpoint_suffix else "training_history.json"
        history_path = self.report_dir / history_filename
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
