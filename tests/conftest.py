# -*- coding: utf-8 -*-
"""
Pytest configuration and shared fixtures.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import torch


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_ini_content() -> str:
    """Sample INI configuration content for testing."""
    return """
[General]
labelInfo = TestDataset
modelBasePath = ./data/test
npzSave = False
imHeight = 256
imWidth = 256
imChannel = 3
imIndxList = 0, 50, 100

[Training]
layerSel = ConvM1, ConvM2
modelSel = BAE1, BAE2
datasetPath = ./test_data
batchSize = 10
numEpoch = 5

[Prediction]
predictionDataPath = ./test_data/predict
predictionResultPath = ./output
modelName = BAE1
layerName = ConvM1
featureExtractorName = ErrM
anomalyAlgorithmName = Isolation Forest
batchSize = 50
"""


@pytest.fixture
def sample_ini_file(temp_dir: Path, sample_ini_content: str) -> Path:
    """Create a sample INI file for testing."""
    ini_path = temp_dir / "test_config.ini"
    ini_path.write_text(sample_ini_content)
    return ini_path


@pytest.fixture
def sample_dataset_dir(temp_dir: Path) -> Path:
    """Create a sample dataset directory structure."""
    dataset_path = temp_dir / "test_data"

    # Create directory structure
    for split in ["train", "valid", "test"]:
        ok_dir = dataset_path / split / "ok"
        ok_dir.mkdir(parents=True)

        # Create dummy image files
        for i in range(5):
            (ok_dir / f"image_{i}.png").write_bytes(b"dummy")

    # Create NOK directory for test split
    nok_dir = dataset_path / "test" / "nok"
    nok_dir.mkdir(parents=True)
    for i in range(3):
        (nok_dir / f"anomaly_{i}.png").write_bytes(b"dummy")

    return dataset_path


@pytest.fixture
def sample_image_array() -> np.ndarray:
    """Create a sample image array for testing (NHWC format)."""
    return np.random.rand(256, 256, 3).astype(np.float32)


@pytest.fixture
def sample_batch_images() -> np.ndarray:
    """Create a batch of sample images for testing (NHWC format)."""
    return np.random.rand(10, 256, 256, 3).astype(np.float32)


@pytest.fixture
def sample_model_data(sample_batch_images: np.ndarray) -> dict:
    """Create sample model data dictionary for testing."""
    batch_size = sample_batch_images.shape[0]
    return {
        "Train": {
            "Org": sample_batch_images,
            "Dec": sample_batch_images + np.random.randn(*sample_batch_images.shape) * 0.1,
            "Lab": np.ones(batch_size),
        },
        "Test": {
            "Org": sample_batch_images,
            "Dec": sample_batch_images + np.random.randn(*sample_batch_images.shape) * 0.1,
            "Lab": np.concatenate([np.ones(batch_size // 2), -np.ones(batch_size // 2)]),
        },
    }


# PyTorch-specific fixtures


@pytest.fixture
def sample_tensor_batch() -> torch.Tensor:
    """Create a batch of sample image tensors (NCHW format)."""
    return torch.randn(4, 3, 256, 256)


@pytest.fixture
def sample_grayscale_tensor_batch() -> torch.Tensor:
    """Create a batch of grayscale image tensors (NCHW format)."""
    return torch.randn(4, 1, 256, 256)


@pytest.fixture
def device() -> torch.device:
    """Get the default test device (CPU for CI compatibility)."""
    return torch.device("cpu")
