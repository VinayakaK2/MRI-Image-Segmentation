"""Lightweight 2D U-Net architecture for multi-class cardiac MRI segmentation."""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):
    """Standard double convolution block: (Conv2d -> BatchNorm -> ReLU) x 2 with optional Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Initializes the double convolution sequence with defensive argument validation."""
        super().__init__()
        layers: List[nn.Module] = []

        # First convolution layer
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))

        # Second convolution layer
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes input tensor through the sequential double convolution block."""
        return self.block(x)


class DownsampleBlock(nn.Module):
    """Encoder downsampling stage composed of 2x2 MaxPool followed by a DoubleConvBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Initializes max pooling and subsequent convolutional feature extraction."""
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsamples feature map spatially and computes extracted feature representations."""
        return self.conv(self.pool(x))


class UpsampleBlock(nn.Module):
    """Decoder upsampling stage combining feature expansion, skip connection concatenation, and conv block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Initializes upsampling layer (bilinear or transposed conv) and fusion convolutions."""
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                use_batchnorm=use_batchnorm,
                dropout=dropout,
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                use_batchnorm=use_batchnorm,
                dropout=dropout,
            )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsamples decoder features, aligns spatial dimensions with skip features, and concatenates."""
        x_up = self.up(x)

        # Defensive geometry alignment: Pad x_up if spatial dimensions differ slightly due to odd resolutions
        diff_y = skip.size()[2] - x_up.size()[2]
        diff_x = skip.size()[3] - x_up.size()[3]

        if diff_y != 0 or diff_x != 0:
            x_up = F.pad(x_up, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        # Concatenate along channel dimension (B, C_up + C_skip, H, W)
        fused = torch.cat([skip, x_up], dim=1)
        return self.conv(fused)


class LightweightUNet2D(nn.Module):
    """Lightweight 2D U-Net for rapid, robust semantic segmentation of cardiac MRI short-axis slices."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 16,
        depth: int = 4,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        bilinear_upsample: bool = True,
    ) -> None:
        """Initializes the multi-scale encoder-decoder architecture with configurable depth and width."""
        super().__init__()
        if depth < 2:
            raise ValueError(f"U-Net depth must be >= 2, received depth={depth}")
        if base_channels < 4:
            raise ValueError(f"base_channels must be >= 4, received base_channels={base_channels}")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.depth = depth
        self.bilinear = bilinear_upsample

        # Channel width at each depth level: [16, 32, 64, 128, ...]
        channel_widths = [base_channels * (2 ** i) for i in range(depth)]

        # Initial stem block (Level 0)
        self.inc = DoubleConvBlock(
            in_channels=in_channels,
            out_channels=channel_widths[0],
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )

        # Encoder downsampling stages
        self.down_blocks = nn.ModuleList()
        for i in range(depth - 1):
            self.down_blocks.append(
                DownsampleBlock(
                    in_channels=channel_widths[i],
                    out_channels=channel_widths[i + 1],
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                )
            )

        # Bottleneck stage
        bottleneck_channels = channel_widths[-1] * 2
        self.bottleneck = DownsampleBlock(
            in_channels=channel_widths[-1],
            out_channels=bottleneck_channels,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )

        # Decoder upsampling stages
        self.up_blocks = nn.ModuleList()
        # Decode from bottleneck back to level 0
        dec_in = bottleneck_channels
        for i in reversed(range(depth)):
            skip_channels = channel_widths[i]
            # When concatenating skip (skip_channels) with upsampled features (dec_in // 2 or dec_in)
            # Total channels entering conv = skip_channels + dec_in // 2 (or dec_in in bilinear mode)
            fused_in = skip_channels + (dec_in // 2 if not bilinear_upsample else dec_in)
            self.up_blocks.append(
                UpsampleBlock(
                    in_channels=fused_in,
                    out_channels=skip_channels,
                    bilinear=bilinear_upsample,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                )
            )
            dec_in = skip_channels

        # Final 1x1 classification head
        self.outc = nn.Conv2d(channel_widths[0], num_classes, kernel_size=1)

        # Initialize network weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Applies Kaiming/He normal initialization to convolutions and ones to batchnorm."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes forward pass through encoder, bottleneck, skip connections, and decoder."""
        # 1. Encoder forward pass with skip feature retention
        skips: List[torch.Tensor] = []
        feat = self.inc(x)
        skips.append(feat)

        for down in self.down_blocks:
            feat = down(feat)
            skips.append(feat)

        # 2. Bottleneck
        feat = self.bottleneck(feat)

        # 3. Decoder forward pass with skip fusion
        for up in self.up_blocks:
            skip = skips.pop()
            feat = up(feat, skip)

        # 4. Final classification logits
        logits = self.outc(feat)
        return logits

    def count_parameters(self) -> Dict[str, int]:
        """Calculates total and trainable parameter counts for reporting."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LightweightUNet2D":
        """Instantiates LightweightUNet2D directly from a parsed configuration dictionary."""
        model_cfg = cfg.get("model", {})
        return cls(
            in_channels=model_cfg.get("in_channels", 1),
            num_classes=model_cfg.get("num_classes", 4),
            base_channels=model_cfg.get("base_channels", 16),
            depth=model_cfg.get("depth", 4),
            dropout=model_cfg.get("dropout", 0.1),
            use_batchnorm=model_cfg.get("use_batchnorm", True),
            bilinear_upsample=model_cfg.get("bilinear_upsample", True),
        )
