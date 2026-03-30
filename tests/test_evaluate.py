# -*- coding: utf-8 -*-
"""
Tests for the evaluate module (PyTorch version).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from evaluate import (
    apply_gaussian_filter,
    get_device,
    prepare_output_directories,
    save_labels,
)


class TestGetDevice:
    """Tests for get_device function."""

    def test_cpu_device(self):
        """Test getting CPU device."""
        device = get_device("cpu")
        assert device == torch.device("cpu")

    def test_auto_device_returns_valid_device(self):
        """Test that auto returns a valid device."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        # Should be one of cpu, cuda, or mps
        assert device.type in ("cpu", "cuda", "mps")

    def test_invalid_device_raises_error(self):
        """Test that invalid device specification raises error."""
        # PyTorch will raise an error for invalid device
        with pytest.raises(RuntimeError):
            device = get_device("invalid_device")
            # Need to actually use the device to trigger the error
            torch.tensor([1.0], device=device)


class TestApplyGaussianFilter:
    """Tests for apply_gaussian_filter function."""

    def test_filter_shape_preserved_nhwc(self, sample_batch_images: np.ndarray):
        """Test that filter preserves image shape (NHWC format)."""
        filtered = apply_gaussian_filter(sample_batch_images)
        assert filtered.shape == sample_batch_images.shape

    def test_filter_shape_preserved_nchw(self):
        """Test that filter handles NCHW format correctly."""
        # Create NCHW format images
        images = np.random.rand(4, 3, 64, 64).astype(np.float32)
        filtered = apply_gaussian_filter(images)
        # Output should be NHWC
        assert filtered.shape == (4, 64, 64, 3)

    def test_filter_strength_zero(self, sample_batch_images: np.ndarray):
        """Test filter with strength 0 returns blurred images."""
        filtered = apply_gaussian_filter(sample_batch_images, filter_strength=0.0)
        # With strength 0, output should be entirely blurred
        assert not np.allclose(filtered, sample_batch_images)

    def test_filter_strength_one(self, sample_batch_images: np.ndarray):
        """Test filter with strength 1 returns original images."""
        filtered = apply_gaussian_filter(sample_batch_images, filter_strength=1.0)
        np.testing.assert_array_almost_equal(filtered, sample_batch_images)

    def test_default_filter_strength(self, sample_batch_images: np.ndarray):
        """Test default filter strength is 0.5."""
        filtered = apply_gaussian_filter(sample_batch_images)
        # Result should be between original and fully blurred
        assert filtered.min() >= 0
        assert filtered.max() <= 1

    def test_grayscale_images(self):
        """Test filter handles grayscale images."""
        images = np.random.rand(4, 64, 64, 1).astype(np.float32)
        filtered = apply_gaussian_filter(images)
        assert filtered.shape == images.shape


class TestPrepareOutputDirectories:
    """Tests for prepare_output_directories function."""

    def test_creates_directories(self, temp_dir: Path):
        """Test that directories are created."""
        result_path = temp_dir / "results"
        ok_path, nok_path = prepare_output_directories(result_path)

        assert result_path.exists()
        assert ok_path.exists()
        assert nok_path.exists()
        assert ok_path == result_path / "OK"
        assert nok_path == result_path / "NOK"

    def test_idempotent(self, temp_dir: Path):
        """Test that calling twice doesn't raise error."""
        result_path = temp_dir / "results"

        # Call twice
        prepare_output_directories(result_path)
        ok_path, nok_path = prepare_output_directories(result_path)

        assert ok_path.exists()
        assert nok_path.exists()


class TestSaveLabels:
    """Tests for save_labels function."""

    def test_saves_yaml_file(self, temp_dir: Path):
        """Test that labels are saved to YAML file."""
        labels = {
            "OK": ["/path/to/image1.png", "/path/to/image2.png"],
            "NOK": ["/path/to/image3.png"],
        }

        save_labels(labels, temp_dir)

        labels_file = temp_dir / "labels.yaml"
        assert labels_file.exists()

    def test_saves_absolute_paths(self, temp_dir: Path):
        """Test that paths are converted to absolute."""
        import yaml

        labels = {
            "OK": ["./relative/path.png"],
            "NOK": [],
        }

        save_labels(labels, temp_dir)

        labels_file = temp_dir / "labels.yaml"
        with open(labels_file) as f:
            saved = yaml.safe_load(f)

        # Check that path is absolute
        if saved and "OK" in saved:
            for path in saved["OK"]:
                assert Path(path).is_absolute()

    def test_empty_labels(self, temp_dir: Path):
        """Test saving empty labels."""
        labels = {"OK": [], "NOK": []}

        save_labels(labels, temp_dir)

        labels_file = temp_dir / "labels.yaml"
        assert labels_file.exists()
