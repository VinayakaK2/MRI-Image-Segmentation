"""Loss functions for multi-class cardiac MRI segmentation."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_DICE_SMOOTH: float = 1e-5


class SoftDiceLoss(nn.Module):
    """Multi-class soft Dice loss supporting optional background exclusion and numerical smoothing."""

    def __init__(
        self,
        num_classes: int = 4,
        include_background: bool = True,
        smooth: float = DEFAULT_DICE_SMOOTH,
    ) -> None:
        """Initializes the soft Dice loss criterion with class count and smoothing parameters."""
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes soft Dice loss across spatial dimensions and averages across classes."""
        # 1. Input assertions
        if logits.dim() != 4:
            raise ValueError(f"Expected 4D logits (B, C, H, W), got shape: {logits.shape}")
        if targets.dim() != 3:
            raise ValueError(f"Expected 3D integer targets (B, H, W), got shape: {targets.shape}")

        batch_size, num_c, h, w = logits.shape
        if num_c != self.num_classes:
            raise ValueError(f"Logits channels ({num_c}) does not match expected num_classes ({self.num_classes})")

        # 2. Probability distribution via softmax
        probs = F.softmax(logits, dim=1)

        # 3. One-hot encoding of targets (B, H, W) -> (B, C, H, W)
        targets_clamped = torch.clamp(targets, min=0, max=self.num_classes - 1)
        targets_one_hot = F.one_hot(targets_clamped, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # 4. Flatten spatial dimensions for stable batch-wise reduction
        probs_flat = probs.view(batch_size, num_c, -1)
        targets_flat = targets_one_hot.view(batch_size, num_c, -1)

        # 5. Compute intersection and cardinality per class
        intersection = torch.sum(probs_flat * targets_flat, dim=-1)
        cardinality = torch.sum(probs_flat, dim=-1) + torch.sum(targets_flat, dim=-1)

        # Soft Dice per batch and class: 2 * (I + eps) / (C + eps)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score

        # 6. Select evaluated classes
        start_idx = 0 if self.include_background else 1
        return torch.mean(dice_loss[:, start_idx:])


class DiceCELoss(nn.Module):
    """Composite objective combining Cross-Entropy and Soft Dice loss with configurable weights."""

    def __init__(
        self,
        num_classes: int = 4,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        include_background_dice: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        smooth: float = DEFAULT_DICE_SMOOTH,
    ) -> None:
        """Initializes combined loss criteria with balancing hyperparameters."""
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = SoftDiceLoss(
            num_classes=num_classes,
            include_background=include_background_dice,
            smooth=smooth,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates weighted linear combination: w_ce * CE(logits, targets) + w_dice * Dice(logits, targets)."""
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return (self.ce_weight * ce) + (self.dice_weight * dice)
