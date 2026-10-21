"""Unit tests for LightweightUNet2D architecture, gradient flow, and parameter count."""

import pytest
import torch
from src.models.unet2d import DoubleConvBlock, DownsampleBlock, LightweightUNet2D, UpsampleBlock


def test_double_conv_block_output_shape() -> None:
    """Verifies that DoubleConvBlock preserves spatial resolution and changes channel depth."""
    block = DoubleConvBlock(in_channels=1, out_channels=16, use_batchnorm=True)
    x = torch.randn(2, 1, 64, 64)
    out = block(x)
    assert out.shape == (2, 16, 64, 64)
    assert not torch.isnan(out).any()


def test_unet_output_shape_and_finite_values() -> None:
    """Verifies that LightweightUNet2D produces (B, 4, 256, 256) logits without NaNs or Infs."""
    model = LightweightUNet2D(
        in_channels=1,
        num_classes=4,
        base_channels=16,
        depth=4,
        dropout=0.0,
        use_batchnorm=True,
        bilinear_upsample=True,
    )
    x = torch.randn(2, 1, 256, 256)
    logits = model(x)

    assert logits.shape == (2, 4, 256, 256)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_unet_parameter_count_lightweight() -> None:
    """Verifies that the parameter count is under 3 million parameters."""
    model = LightweightUNet2D(base_channels=16, depth=4)
    counts = model.count_parameters()

    assert counts["total"] > 1_000_000
    assert counts["total"] < 3_000_000
    assert counts["total"] == counts["trainable"]


def test_unet_gradient_flow_backward() -> None:
    """Verifies that gradients propagate back to all trainable parameters."""
    model = LightweightUNet2D(base_channels=16, depth=4)
    x = torch.randn(2, 1, 128, 128)
    logits = model(x)
    loss = logits.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"


def test_unet_from_config_instantiation() -> None:
    """Verifies that LightweightUNet2D instantiates accurately from a configuration mapping."""
    sample_cfg = {
        "model": {
            "in_channels": 1,
            "num_classes": 4,
            "base_channels": 16,
            "depth": 4,
            "dropout": 0.1,
            "use_batchnorm": True,
            "bilinear_upsample": True,
        }
    }
    model = LightweightUNet2D.from_config(sample_cfg)
    assert model.in_channels == 1
    assert model.num_classes == 4
    assert model.base_channels == 16
    assert model.depth == 4
