# -*- coding: utf-8 -*-
"""
PyTorch data loaders for anomaly detection datasets.

This module provides Dataset and DataLoader classes for loading
images organized in the one-class classification directory structure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class AnomalyDetectionDataset(Dataset):
    """
    Dataset for anomaly detection with OK/NOK labels.

    Expected directory structure:
        dataset_root/
        ├── train/ok/       (label = 1)
        ├── valid/ok/       (label = 1)
        └── test/
            ├── ok/         (label = 1)
            └── nok/        (label = -1)
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        image_size: tuple[int, int] = (256, 256),
        channels: int = 3,
        transform: Callable | None = None,
        include_nok: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            root_dir: Root directory of the dataset
            split: One of 'train', 'valid', 'test'
            image_size: Target image size (height, width)
            channels: Number of color channels (1 or 3)
            transform: Optional additional transforms
            include_nok: Whether to include NOK samples (for test split)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.channels = channels
        self.include_nok = include_nok

        # Build default transform
        base_transforms = [
            transforms.Resize(image_size),
            transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
        ]

        if channels == 1:
            base_transforms.insert(0, transforms.Grayscale(num_output_channels=1))

        self.base_transform = transforms.Compose(base_transforms)
        self.additional_transform = transform

        # Collect image paths and labels
        self.samples: list[tuple[Path, int]] = []
        self._collect_samples()

        logging.info(f"Loaded {len(self.samples)} samples for {split} split")

    def _collect_samples(self) -> None:
        """Collect all image paths and their labels."""
        split_dir = self.root_dir / self.split

        # OK samples (label = 1)
        ok_dir = split_dir / "ok"
        if ok_dir.exists():
            for img_path in ok_dir.iterdir():
                if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.samples.append((img_path, 1))

        # NOK samples (label = -1) - only for test split
        if self.include_nok and self.split == "test":
            nok_dir = split_dir / "nok"
            if nok_dir.exists():
                for img_path in nok_dir.iterdir():
                    if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                        self.samples.append((img_path, -1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        """
        Get a sample.

        Returns:
            Tuple of (image_tensor, label, image_path)
        """
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path)
        if self.channels == 3 and image.mode != "RGB":
            image = image.convert("RGB")
        elif self.channels == 1 and image.mode != "L":
            image = image.convert("L")

        # Apply transforms
        image = self.base_transform(image)
        if self.additional_transform is not None:
            image = self.additional_transform(image)

        return image, label, str(img_path)


class InferenceDataset(Dataset):
    """
    Dataset for inference on a directory of images (no labels).
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

    def __init__(
        self,
        image_dir: str | Path,
        image_size: tuple[int, int] = (256, 256),
        channels: int = 3,
        transform: Callable | None = None,
    ):
        """
        Initialize inference dataset.

        Args:
            image_dir: Directory containing images
            image_size: Target image size (height, width)
            channels: Number of color channels (1 or 3)
            transform: Optional additional transforms
        """
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.channels = channels

        # Build transform
        base_transforms = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]

        if channels == 1:
            base_transforms.insert(0, transforms.Grayscale(num_output_channels=1))

        self.base_transform = transforms.Compose(base_transforms)
        self.additional_transform = transform

        # Collect image paths
        self.image_paths: list[Path] = []
        self._collect_images()

    def _collect_images(self) -> None:
        """Collect all image paths."""
        for img_path in sorted(self.image_dir.iterdir()):
            if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                self.image_paths.append(img_path)

        # Also search subdirectories
        for subdir in self.image_dir.iterdir():
            if subdir.is_dir():
                for img_path in sorted(subdir.iterdir()):
                    if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                        self.image_paths.append(img_path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """
        Get a sample.

        Returns:
            Tuple of (image_tensor, image_path)
        """
        img_path = self.image_paths[idx]

        # Load image
        image = Image.open(img_path)
        if self.channels == 3 and image.mode != "RGB":
            image = image.convert("RGB")
        elif self.channels == 1 and image.mode != "L":
            image = image.convert("L")

        # Apply transforms
        image = self.base_transform(image)
        if self.additional_transform is not None:
            image = self.additional_transform(image)

        return image, str(img_path)


def create_data_loaders(
    dataset_path: str | Path,
    image_dim: tuple[int, int, int],
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform: Callable | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.

    Args:
        dataset_path: Path to dataset root directory
        image_dim: Image dimensions as (height, width, channels)
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        train_transform: Optional augmentation transform for training

    Returns:
        Tuple of (train_loader, valid_loader, test_loader)
    """
    height, width, channels = image_dim
    image_size = (height, width)

    # Create datasets
    train_dataset = AnomalyDetectionDataset(
        dataset_path,
        split="train",
        image_size=image_size,
        channels=channels,
        transform=train_transform,
        include_nok=False,
    )

    valid_dataset = AnomalyDetectionDataset(
        dataset_path,
        split="valid",
        image_size=image_size,
        channels=channels,
        include_nok=False,
    )

    test_dataset = AnomalyDetectionDataset(
        dataset_path,
        split="test",
        image_size=image_size,
        channels=channels,
        include_nok=True,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, valid_loader, test_loader


def create_inference_loader(
    image_dir: str | Path,
    image_dim: tuple[int, int, int],
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create data loader for inference.

    Args:
        image_dir: Directory containing images
        image_dim: Image dimensions as (height, width, channels)
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        DataLoader for inference
    """
    height, width, channels = image_dim

    dataset = InferenceDataset(
        image_dir,
        image_size=(height, width),
        channels=channels,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
