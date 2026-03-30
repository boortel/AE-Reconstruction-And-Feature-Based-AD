# -*- coding: utf-8 -*-
"""
HardNet descriptor network implementation in PyTorch.

HardNet is a compact learned local feature descriptor that achieves
state-of-the-art results on standard matching benchmarks.

Reference:
    Mishchuk et al., "Working hard to know your neighbor's margins:
    Local descriptor learning loss", NeurIPS 2017
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HardNet(nn.Module):
    """
    HardNet descriptor network.

    A CNN that takes a 32x32 grayscale patch and outputs a 128-dimensional
    L2-normalized descriptor.

    Architecture:
        - 7 conv layers with batch normalization
        - Dropout (0.3) before final conv
        - Output: 128-dim L2-normalized descriptor
    """

    def __init__(self, pretrained: bool = True, checkpoint_path: str | Path | None = None):
        """
        Initialize HardNet.

        Args:
            pretrained: Whether to load pretrained weights
            checkpoint_path: Path to checkpoint file (default: ./hardnet_checkpoint.pt)
        """
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(inplace=True),
            # Block 2
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(inplace=True),
            # Block 3 (stride 2)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
            # Block 4
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
            # Block 5 (stride 2)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(inplace=True),
            # Block 6
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(inplace=True),
            # Dropout
            nn.Dropout(0.3),
            # Block 7 (8x8 kernel for 32x32 -> 1x1)
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

        if pretrained:
            self._load_pretrained(checkpoint_path)

    def _load_pretrained(self, checkpoint_path: str | Path | None = None) -> None:
        """Load pretrained weights."""
        if checkpoint_path is None:
            checkpoint_path = Path("./hardnet_checkpoint.pt")
        else:
            checkpoint_path = Path(checkpoint_path)

        # Try loading PyTorch checkpoint first
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.load_state_dict(state_dict)
            return

        # Try loading from pickle (legacy format)
        pickle_path = checkpoint_path.with_suffix(".pickle")
        if pickle_path.exists():
            import pickle

            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
                self._load_from_pickle(data["weights"])
            return

        print(f"Warning: No pretrained weights found at {checkpoint_path}")

    def _load_from_pickle(self, weights: dict) -> None:
        """Load weights from pickle format (legacy TensorFlow format)."""
        state_dict = {}

        # Map pickle keys to PyTorch state dict keys
        layer_mapping = {
            "features.0": "features.0",   # Conv
            "features.1": "features.1",   # BN
            "features.3": "features.3",   # Conv
            "features.4": "features.4",   # BN
            "features.6": "features.6",   # Conv
            "features.7": "features.7",   # BN
            "features.9": "features.9",   # Conv
            "features.10": "features.10", # BN
            "features.12": "features.12", # Conv
            "features.13": "features.13", # BN
            "features.15": "features.15", # Conv
            "features.16": "features.16", # BN
            "features.19": "features.19", # Conv
            "features.20": "features.20", # BN
        }

        for pickle_key, value in weights.items():
            parts = pickle_key.split(".")
            layer_idx = f"features.{parts[1]}"

            if "weight" in pickle_key:
                # Conv weights: pickle is [out, in, h, w], same as PyTorch
                state_dict[f"{layer_idx}.weight"] = torch.from_numpy(value)
            elif "running_mean" in pickle_key:
                state_dict[f"{layer_idx}.running_mean"] = torch.from_numpy(value)
            elif "running_var" in pickle_key:
                state_dict[f"{layer_idx}.running_var"] = torch.from_numpy(value)

        self.load_state_dict(state_dict, strict=False)

    def input_norm(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Normalize input patches.

        Args:
            x: Input tensor (B, 1, H, W)
            eps: Small constant for numerical stability

        Returns:
            Normalized tensor
        """
        
         # Convert numpy arrays automatically to torch.Tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        # Ensure float type for computation
        x = x.float()
        
        # Flatten spatial dimensions
        b, c, h, w = x.shape
        x_flat = x.view(b, -1)

        # Compute mean and std per sample
        mean = x_flat.mean(dim=1, keepdim=True)
        std = x_flat.std(dim=1, keepdim=True)

        # Normalize
        x_norm = (x_flat - mean) / (std + eps)

        return x_norm.view(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input patches (B, 1, 32, 32) - grayscale 32x32 patches

        Returns:
            L2-normalized descriptors (B, 128)
        """
        
        # Convert numpy arrays automatically to torch.Tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        # Ensure float type for computation
        x = x.float()

        # If channels are last, convert to NCHW
        if x.ndim == 4 and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2) 
            
        # Normalize input
        x = self.input_norm(x)

        # Extract features
        features = self.features(x)

        # Flatten and L2 normalize
        features = features.view(features.size(0), -1)
        features = F.normalize(features, p=2, dim=1)

        return features

    @torch.no_grad()
    def extract_descriptors(
        self,
        image: torch.Tensor,
        keypoints: list[tuple[int, int]],
        patch_size: int = 32,
    ) -> torch.Tensor:
        """
        Extract descriptors for keypoints in an image.

        Args:
            image: Grayscale image tensor (1, H, W) or (H, W)
            keypoints: List of (x, y) keypoint coordinates
            patch_size: Size of patches to extract (default 32)

        Returns:
            Descriptors tensor (N, 128)
        """
        
         # Convert numpy arrays automatically to torch.Tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        
        if image.dim() == 2:
            image = image.unsqueeze(0)  # Add channel dim

        half_size = patch_size // 2
        patches = []

        for x, y in keypoints:
            # Extract patch with boundary handling
            y1 = max(0, y - half_size)
            y2 = min(image.shape[1], y + half_size)
            x1 = max(0, x - half_size)
            x2 = min(image.shape[2], x + half_size)

            patch = image[:, y1:y2, x1:x2]

            # Pad if necessary
            if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                pad_y = patch_size - patch.shape[1]
                pad_x = patch_size - patch.shape[2]
                patch = F.pad(patch, (0, pad_x, 0, pad_y))

            patches.append(patch)

        if not patches:
            return torch.empty(0, 128)

        patches = torch.stack(patches)
        return self(patches)


def convert_pickle_to_pytorch(
    pickle_path: str | Path,
    output_path: str | Path | None = None,
) -> None:
    """
    Convert legacy pickle checkpoint to PyTorch format.

    Args:
        pickle_path: Path to pickle checkpoint
        output_path: Output path for PyTorch checkpoint
    """
    import pickle

    pickle_path = Path(pickle_path)
    if output_path is None:
        output_path = pickle_path.with_suffix(".pt")

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    model = HardNet(pretrained=False)
    model._load_from_pickle(data["weights"])

    torch.save(model.state_dict(), output_path)
    print(f"Saved PyTorch checkpoint to {output_path}")
