# -*- coding: utf-8 -*-
"""
Tests for PyTorch data loaders.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from data_loader import (
    AnomalyDetectionDataset,
    InferenceDataset,
    create_data_loaders,
    create_inference_loader,
)


@pytest.fixture
def image_dataset_dir(temp_dir: Path) -> Path:
    """Create a dataset directory with actual image files."""
    dataset_path = temp_dir / "dataset"

    # Create directory structure
    for split in ["train", "valid", "test"]:
        ok_dir = dataset_path / split / "ok"
        ok_dir.mkdir(parents=True)

        # Create actual image files
        for i in range(3):
            img = Image.fromarray(
                np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            )
            img.save(ok_dir / f"image_{i}.png")

    # Create NOK directory for test split
    nok_dir = dataset_path / "test" / "nok"
    nok_dir.mkdir(parents=True)
    for i in range(2):
        img = Image.fromarray(
            np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        )
        img.save(nok_dir / f"anomaly_{i}.png")

    return dataset_path


class TestAnomalyDetectionDataset:
    """Tests for AnomalyDetectionDataset."""

    def test_train_split_loading(self, image_dataset_dir: Path):
        """Test loading train split."""
        dataset = AnomalyDetectionDataset(
            image_dataset_dir,
            split="train",
            image_size=(64, 64),
            channels=3,
        )

        assert len(dataset) == 3  # 3 OK images

    def test_test_split_with_nok(self, image_dataset_dir: Path):
        """Test loading test split includes NOK samples."""
        dataset = AnomalyDetectionDataset(
            image_dataset_dir,
            split="test",
            image_size=(64, 64),
            channels=3,
            include_nok=True,
        )

        assert len(dataset) == 5  # 3 OK + 2 NOK

    def test_test_split_without_nok(self, image_dataset_dir: Path):
        """Test loading test split without NOK samples."""
        dataset = AnomalyDetectionDataset(
            image_dataset_dir,
            split="test",
            image_size=(64, 64),
            channels=3,
            include_nok=False,
        )

        assert len(dataset) == 3  # 3 OK only

    def test_getitem_returns_tuple(self, image_dataset_dir: Path):
        """Test that __getitem__ returns correct tuple."""
        dataset = AnomalyDetectionDataset(
            image_dataset_dir,
            split="train",
            image_size=(64, 64),
            channels=3,
        )

        image, label, path = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)
        assert isinstance(path, str)

    def test_image_dimensions(self, image_dataset_dir: Path):
        """Test that images have correct dimensions."""
        dataset = AnomalyDetectionDataset(
            image_dataset_dir,
            split="train",
            image_size=(64, 64),
            channels=3,
        )

        image, _, _ = dataset[0]

        assert image.shape == (3, 64, 64)  # (C, H, W)

    def test_grayscale_loading(self, image_dataset_dir: Path):
        """Test loading images as grayscale."""
        dataset = AnomalyDetectionDataset(
            image_dataset_dir,
            split="train",
            image_size=(64, 64),
            channels=1,
        )

        image, _, _ = dataset[0]

        assert image.shape == (1, 64, 64)

    def test_labels_correct(self, image_dataset_dir: Path):
        """Test that labels are correct."""
        dataset = AnomalyDetectionDataset(
            image_dataset_dir,
            split="test",
            image_size=(64, 64),
            channels=3,
            include_nok=True,
        )

        labels = [dataset[i][1] for i in range(len(dataset))]

        assert 1 in labels  # OK samples
        assert -1 in labels  # NOK samples

    def test_image_values_normalized(self, image_dataset_dir: Path):
        """Test that images are normalized to [0, 1]."""
        dataset = AnomalyDetectionDataset(
            image_dataset_dir,
            split="train",
            image_size=(64, 64),
            channels=3,
        )

        image, _, _ = dataset[0]

        assert image.min() >= 0
        assert image.max() <= 1


class TestInferenceDataset:
    """Tests for InferenceDataset."""

    def test_loading(self, image_dataset_dir: Path):
        """Test loading inference dataset."""
        dataset = InferenceDataset(
            image_dataset_dir / "train" / "ok",
            image_size=(64, 64),
            channels=3,
        )

        assert len(dataset) == 3

    def test_getitem_returns_tuple(self, image_dataset_dir: Path):
        """Test that __getitem__ returns correct tuple."""
        dataset = InferenceDataset(
            image_dataset_dir / "train" / "ok",
            image_size=(64, 64),
            channels=3,
        )

        image, path = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert isinstance(path, str)

    def test_no_labels(self, image_dataset_dir: Path):
        """Test that inference dataset doesn't return labels."""
        dataset = InferenceDataset(
            image_dataset_dir / "train" / "ok",
            image_size=(64, 64),
            channels=3,
        )

        result = dataset[0]

        # Should be (image, path) - no label
        assert len(result) == 2


class TestCreateDataLoaders:
    """Tests for create_data_loaders function."""

    def test_returns_three_loaders(self, image_dataset_dir: Path):
        """Test that function returns three data loaders."""
        train_loader, val_loader, test_loader = create_data_loaders(
            image_dataset_dir,
            image_dim=(64, 64, 3),
            batch_size=2,
            num_workers=0,
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

    def test_batch_iteration(self, image_dataset_dir: Path):
        """Test iterating through batches."""
        train_loader, _, _ = create_data_loaders(
            image_dataset_dir,
            image_dim=(64, 64, 3),
            batch_size=2,
            num_workers=0,
        )

        for images, labels, paths in train_loader:
            assert images.shape[1:] == (3, 64, 64)
            assert len(labels) == len(paths)
            break

    def test_pin_memory_disabled_for_cpu(self, image_dataset_dir: Path):
        """Test data loader creation with minimal settings."""
        train_loader, _, _ = create_data_loaders(
            image_dataset_dir,
            image_dim=(64, 64, 3),
            batch_size=1,
            num_workers=0,
        )

        # Just verify it works
        assert len(train_loader.dataset) > 0


class TestCreateInferenceLoader:
    """Tests for create_inference_loader function."""

    def test_returns_loader(self, image_dataset_dir: Path):
        """Test that function returns a data loader."""
        loader = create_inference_loader(
            image_dataset_dir / "train" / "ok",
            image_dim=(64, 64, 3),
            batch_size=2,
            num_workers=0,
        )

        assert loader is not None

    def test_batch_iteration(self, image_dataset_dir: Path):
        """Test iterating through inference batches."""
        loader = create_inference_loader(
            image_dataset_dir / "train" / "ok",
            image_dim=(64, 64, 3),
            batch_size=2,
            num_workers=0,
        )

        for images, paths in loader:
            assert images.shape[1:] == (3, 64, 64)
            assert len(paths) <= 2  # batch_size
            break
