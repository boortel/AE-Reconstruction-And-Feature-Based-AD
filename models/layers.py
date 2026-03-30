# -*- coding: utf-8 -*-
"""
PyTorch encoder and decoder layer configurations.

This module provides the six encoder-decoder configurations (ConvM1-ConvM6)
used by the autoencoder models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass
class LayerConfig:
    """Configuration for encoder-decoder pair."""

    name: str
    reduction_factor: int  # How much the spatial dimensions are reduced
    bottleneck_channels: int


LAYER_CONFIGS: dict[str, LayerConfig] = {
    "ConvM1": LayerConfig("ConvM1", 32, 64),
    "ConvM2": LayerConfig("ConvM2", 16, 1),
    "ConvM3": LayerConfig("ConvM3", 4, 64),
    "ConvM4": LayerConfig("ConvM4", 4, 4),
    "ConvM5": LayerConfig("ConvM5", 8, 4),
    "ConvM6": LayerConfig("ConvM6", 8, 4),
}


class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU/LeakyReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

        if activation == "leaky_relu":
            self.act = nn.LeakyReLU(inplace=True)
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvTransposeBNReLU(nn.Module):
    """Transposed Convolution + BatchNorm + ReLU/LeakyReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

        if activation == "leaky_relu":
            self.act = nn.LeakyReLU(inplace=True)
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# =============================================================================
# Encoder implementations
# =============================================================================


class EncoderConvM1(nn.Module):
    """ConvM1 encoder: 5 blocks with stride 2, reduction factor 32."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNReLU(in_channels, 32, 3, stride=2, padding=1),
            ConvBNReLU(32, 64, 3, stride=2, padding=1),
            ConvBNReLU(64, 64, 3, stride=2, padding=1),
            ConvBNReLU(64, 64, 3, stride=2, padding=1),
            ConvBNReLU(64, 64, 3, stride=2, padding=1),
        )
        self.out_channels = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class EncoderConvM2(nn.Module):
    """ConvM2 encoder: MVT structure with sigmoid activation."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNReLU(in_channels, 32, 4, stride=2, padding=1, activation="sigmoid"),
            ConvBNReLU(32, 32, 4, stride=2, padding=1, activation="sigmoid"),
            ConvBNReLU(32, 32, 3, stride=1, padding=1, activation="sigmoid"),
            ConvBNReLU(32, 64, 4, stride=2, padding=1, activation="sigmoid"),
            ConvBNReLU(64, 64, 3, stride=1, padding=1, activation="sigmoid"),
            ConvBNReLU(64, 128, 4, stride=2, padding=1, activation="sigmoid"),
            ConvBNReLU(128, 64, 3, stride=1, padding=1, activation="sigmoid"),
            ConvBNReLU(64, 32, 3, stride=1, padding=1, activation="sigmoid"),
            nn.Conv2d(32, 1, 8, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(1),
        )
        self.out_channels = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class EncoderConvM3(nn.Module):
    """ConvM3 encoder: Basic 2-block structure."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNReLU(in_channels, 32, 3, stride=2, padding=1, activation="sigmoid"),
            ConvBNReLU(32, 64, 3, stride=2, padding=1, activation="sigmoid"),
        )
        self.out_channels = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class EncoderConvM4(nn.Module):
    """ConvM4 encoder: Conv + MaxPool structure."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, 5, padding=2, bias=False),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 4, 3, padding=1, bias=False),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
        )
        self.out_channels = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class EncoderConvM5(nn.Module):
    """ConvM5 encoder: 3-block Conv + MaxPool structure."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, 3, padding=1, bias=False),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 4, 3, padding=1, bias=False),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
        )
        self.out_channels = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class EncoderConvM6(nn.Module):
    """ConvM6 encoder: Same as ConvM5 (asymmetric with ConvM4 decoder)."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, 3, padding=1, bias=False),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 4, 3, padding=1, bias=False),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
        )
        self.out_channels = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# =============================================================================
# Decoder implementations
# =============================================================================


class DecoderConvM1(nn.Module):
    """ConvM1 decoder: 5 transposed conv blocks."""

    def __init__(self, in_channels: int = 64, out_channels: int = 3):
        super().__init__()
        self.decoder = nn.Sequential(
            ConvTransposeBNReLU(in_channels, 64, 3, stride=2, padding=1, output_padding=1),
            ConvTransposeBNReLU(64, 64, 3, stride=2, padding=1, output_padding=1),
            ConvTransposeBNReLU(64, 64, 3, stride=2, padding=1, output_padding=1),
            ConvTransposeBNReLU(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(32, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class DecoderConvM2(nn.Module):
    """ConvM2 decoder: MVT structure with upsampling."""

    def __init__(self, in_channels: int = 1, out_channels: int = 3):
        super().__init__()
        self.decoder = nn.Sequential(
            ConvBNReLU(in_channels, 16, 3, stride=1, padding=1, activation="sigmoid"),
            ConvBNReLU(16, 64, 3, stride=1, padding=1, activation="sigmoid"),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.BatchNorm2d(128),
            ConvBNReLU(128, 64, 3, stride=1, padding=1, activation="sigmoid"),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.BatchNorm2d(64),
            ConvBNReLU(64, 32, 3, stride=1, padding=1, activation="sigmoid"),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=4, mode="nearest"),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, out_channels, 8, stride=1, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class DecoderConvM3(nn.Module):
    """ConvM3 decoder: Basic 3-block transposed conv."""

    def __init__(self, in_channels: int = 64, out_channels: int = 3):
        super().__init__()
        self.decoder = nn.Sequential(
            ConvTransposeBNReLU(
                in_channels, 64, 3, stride=2, padding=1, output_padding=1, activation="sigmoid"
            ),
            ConvTransposeBNReLU(64, 32, 3, stride=2, padding=1, output_padding=1, activation="sigmoid"),
            nn.ConvTranspose2d(32, out_channels, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class DecoderConvM4(nn.Module):
    """ConvM4 decoder: Conv + Upsample structure."""

    def __init__(self, in_channels: int = 4, out_channels: int = 3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 4, 3, padding=1, bias=False),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 5, padding=2, bias=False),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class DecoderConvM5(nn.Module):
    """ConvM5 decoder: 3-block Conv + Upsample structure."""

    def __init__(self, in_channels: int = 4, out_channels: int = 3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 4, 3, padding=1, bias=False),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 3, padding=1, bias=False),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class DecoderConvM6(nn.Module):
    """ConvM6 decoder: Asymmetric (ConvM4-like) for ConvM5 encoder."""

    def __init__(self, in_channels: int = 4, out_channels: int = 3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 4, 3, padding=1, bias=False),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=4, mode="nearest"),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 5, padding=2, bias=False),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


# =============================================================================
# Factory functions
# =============================================================================

ENCODERS: dict[str, type[nn.Module]] = {
    "ConvM1": EncoderConvM1,
    "ConvM2": EncoderConvM2,
    "ConvM3": EncoderConvM3,
    "ConvM4": EncoderConvM4,
    "ConvM5": EncoderConvM5,
    "ConvM6": EncoderConvM6,
}

DECODERS: dict[str, type[nn.Module]] = {
    "ConvM1": DecoderConvM1,
    "ConvM2": DecoderConvM2,
    "ConvM3": DecoderConvM3,
    "ConvM4": DecoderConvM4,
    "ConvM5": DecoderConvM5,
    "ConvM6": DecoderConvM6,
}


def get_encoder(layer_name: str, in_channels: int = 3) -> nn.Module:
    """
    Get encoder by layer configuration name.

    Args:
        layer_name: One of ConvM1-ConvM6
        in_channels: Number of input channels (1 or 3)

    Returns:
        Encoder module
    """
    if layer_name not in ENCODERS:
        raise ValueError(f"Unknown layer name: {layer_name}. Available: {list(ENCODERS.keys())}")
    return ENCODERS[layer_name](in_channels)


def get_decoder(layer_name: str, in_channels: int, out_channels: int = 3) -> nn.Module:
    """
    Get decoder by layer configuration name.

    Args:
        layer_name: One of ConvM1-ConvM6
        in_channels: Number of input channels from encoder
        out_channels: Number of output channels (1 or 3)

    Returns:
        Decoder module
    """
    if layer_name not in DECODERS:
        raise ValueError(f"Unknown layer name: {layer_name}. Available: {list(DECODERS.keys())}")
    return DECODERS[layer_name](in_channels, out_channels)
