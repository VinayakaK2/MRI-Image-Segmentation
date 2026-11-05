"""Unit and integration tests for Trainer checkpoint saving, resume functionality, and metadata integrity."""

import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.unet2d import LightweightUNet2D
from src.training.trainer import Trainer


@pytest.fixture
def dummy_loaders():
    """Provides small synthetic training and validation DataLoaders for testing Trainer mechanics."""
    # 8 samples of (1, 32, 32) and target (32, 32)
    images = torch.randn(8, 1, 32, 32)
    targets = torch.randint(0, 4, (8, 32, 32))
    dataset = TensorDataset(images, targets)
    
    def collate_fn(batch):
        imgs = torch.stack([b[0] for b in batch])
        tgts = torch.stack([b[1] for b in batch])
        return imgs, tgts, [{}] * len(batch)
    
    train_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


@pytest.fixture
def minimal_config():
    """Provides a minimal experiment configuration for Trainer initialization."""
    return {
        "project": {"seed": 42, "device": "cpu"},
        "model": {"architecture": "LightweightUNet2D", "num_classes": 4, "base_channels": 4, "depth": 2},
        "training": {
            "epochs": 10,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "min_lr": 0.00001,
            "early_stopping_patience": 5,
            "mixed_precision": False,
        },
    }


def test_save_and_resume_checkpoint(tmp_path: Path, dummy_loaders, minimal_config):
    """Verifies that Trainer saves complete state dicts and successfully restores all states on resume."""
    train_loader, val_loader = dummy_loaders
    ckpt_dir = tmp_path / "checkpoints"
    report_dir = tmp_path / "reports"

    model = LightweightUNet2D(in_channels=1, num_classes=4, base_channels=4, depth=2)
    trainer1 = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=minimal_config,
        device=torch.device("cpu"),
        checkpoint_dir=ckpt_dir,
        report_dir=report_dir,
        checkpoint_suffix="test",
    )

    # Run 2 epochs
    trainer1.train(max_epochs=2)

    last_ckpt = ckpt_dir / "last_checkpoint_test.pt"
    best_ckpt = ckpt_dir / "best_model_test.pt"
    assert last_ckpt.exists(), "last_checkpoint_test.pt must exist"
    assert best_ckpt.exists(), "best_model_test.pt must exist"

    # Inspect saved metadata
    bundle = torch.load(last_ckpt, map_location="cpu", weights_only=False)
    assert bundle["epoch"] == 2
    assert "model_state_dict" in bundle
    assert "optimizer_state_dict" in bundle
    assert "scheduler_state_dict" in bundle
    assert bundle["architecture"] == "LightweightUNet2D"
    assert bundle["total_completed_epochs"] == 2

    # Resume in a second trainer instance
    model2 = LightweightUNet2D(in_channels=1, num_classes=4, base_channels=4, depth=2)
    trainer2 = Trainer(
        model=model2,
        train_loader=train_loader,
        val_loader=val_loader,
        config=minimal_config,
        device=torch.device("cpu"),
        checkpoint_dir=ckpt_dir,
        report_dir=report_dir,
        checkpoint_suffix="test",
    )

    resumed_ep = trainer2.resume_from_checkpoint(last_ckpt)
    assert resumed_ep == 2
    assert trainer2.resumed_epoch == 2
    assert trainer2.resumed_with_optimizer is True
    assert trainer2.resumed_with_scheduler is True
    assert len(trainer2.history) == 2

    # Verify model weights were loaded identically
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)

    # Continue training for 1 more epoch (to total 3)
    train_summary = trainer2.train(max_epochs=3)
    assert train_summary["best_epoch"] >= 1
    assert len(trainer2.history) == 3


def test_resume_nonexistent_checkpoint_raises(tmp_path: Path, dummy_loaders, minimal_config):
    """Verifies that attempting to resume from a missing path raises FileNotFoundError."""
    train_loader, val_loader = dummy_loaders
    model = LightweightUNet2D(in_channels=1, num_classes=4, base_channels=4, depth=2)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=minimal_config,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
        report_dir=tmp_path / "reports",
    )

    with pytest.raises(FileNotFoundError):
        trainer.resume_from_checkpoint(tmp_path / "nonexistent.pt")


def test_checkpoint_suffix_isolation(tmp_path: Path, dummy_loaders, minimal_config):
    """Verifies that saving with a suffix does not overwrite the default unsuffixed checkpoint."""
    train_loader, val_loader = dummy_loaders
    ckpt_dir = tmp_path / "checkpoints"
    report_dir = tmp_path / "reports"

    # Create dummy original files
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    orig_best = ckpt_dir / "best_model.pt"
    orig_best.write_bytes(b"original_frozen_content")

    model = LightweightUNet2D(in_channels=1, num_classes=4, base_channels=4, depth=2)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=minimal_config,
        device=torch.device("cpu"),
        checkpoint_dir=ckpt_dir,
        report_dir=report_dir,
        checkpoint_suffix="50epochs",
    )

    trainer.train(max_epochs=1)

    # Original must remain untouched
    assert orig_best.read_bytes() == b"original_frozen_content"
    # New suffix checkpoint exists
    new_best = ckpt_dir / "best_model_50epochs.pt"
    assert new_best.exists()
