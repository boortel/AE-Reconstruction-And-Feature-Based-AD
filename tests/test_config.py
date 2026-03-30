# -*- coding: utf-8 -*-
"""
Tests for the config module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config import (
    ConfigurationError,
    ExperimentConfig,
    GeneralConfig,
    ImageDimensions,
    PredictionConfig,
    TrainingConfig,
    load_all_configs,
)


class TestImageDimensions:
    """Tests for ImageDimensions dataclass."""

    def test_valid_dimensions(self):
        """Test creating valid image dimensions."""
        dims = ImageDimensions(height=256, width=256, channels=3)
        assert dims.height == 256
        assert dims.width == 256
        assert dims.channels == 3

    def test_as_tuple(self):
        """Test converting dimensions to tuple."""
        dims = ImageDimensions(height=128, width=64, channels=1)
        assert dims.as_tuple() == (128, 64, 1)

    def test_invalid_height(self):
        """Test that invalid height raises error."""
        with pytest.raises(ConfigurationError, match="height must be positive"):
            ImageDimensions(height=0, width=256, channels=3)

    def test_invalid_width(self):
        """Test that invalid width raises error."""
        with pytest.raises(ConfigurationError, match="width must be positive"):
            ImageDimensions(height=256, width=-1, channels=3)

    def test_invalid_channels(self):
        """Test that invalid channels raises error."""
        with pytest.raises(ConfigurationError, match="channels must be 1 or 3"):
            ImageDimensions(height=256, width=256, channels=4)


class TestGeneralConfig:
    """Tests for GeneralConfig dataclass."""

    def test_valid_config(self):
        """Test creating valid general config."""
        dims = ImageDimensions(256, 256, 3)
        config = GeneralConfig(
            label_info="TestDataset",
            model_base_path=Path("./data"),
            image_dim=dims,
        )
        assert config.label_info == "TestDataset"
        assert config.model_base_path == Path("./data")

    def test_empty_label_raises_error(self):
        """Test that empty label raises error."""
        dims = ImageDimensions(256, 256, 3)
        with pytest.raises(ConfigurationError, match="label_info cannot be empty"):
            GeneralConfig(
                label_info="",
                model_base_path=Path("./data"),
                image_dim=dims,
            )

    def test_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        dims = ImageDimensions(256, 256, 3)
        config = GeneralConfig(
            label_info="Test",
            model_base_path="./data/output",
            image_dim=dims,
        )
        assert isinstance(config.model_base_path, Path)


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_invalid_batch_size(self, sample_dataset_dir: Path):
        """Test that invalid batch size raises error."""
        with pytest.raises(ConfigurationError, match="Batch size must be positive"):
            TrainingConfig(
                dataset_path=sample_dataset_dir,
                layers=["ConvM1"],
                models=["BAE1"],
                batch_size=0,
            )

    def test_invalid_layer_name(self, sample_dataset_dir: Path):
        """Test that invalid layer name raises error."""
        with pytest.raises(ConfigurationError, match="Unknown layer"):
            TrainingConfig(
                dataset_path=sample_dataset_dir,
                layers=["InvalidLayer"],
                models=["BAE1"],
            )

    def test_invalid_model_name(self, sample_dataset_dir: Path):
        """Test that invalid model name raises error."""
        with pytest.raises(ConfigurationError, match="Unknown model"):
            TrainingConfig(
                dataset_path=sample_dataset_dir,
                layers=["ConvM1"],
                models=["InvalidModel"],
            )


class TestPredictionConfig:
    """Tests for PredictionConfig dataclass."""

    def test_invalid_feature_extractor(self, temp_dir: Path):
        """Test that invalid feature extractor raises error."""
        with pytest.raises(ConfigurationError, match="Unknown feature extractor"):
            PredictionConfig(
                data_path=temp_dir,
                result_path=temp_dir,
                model_name="BAE1",
                layer_name="ConvM1",
                feature_extractor="InvalidExtractor",
                anomaly_algorithm="Isolation Forest",
            )

    def test_invalid_anomaly_algorithm(self, temp_dir: Path):
        """Test that invalid anomaly algorithm raises error."""
        with pytest.raises(ConfigurationError, match="Unknown anomaly algorithm"):
            PredictionConfig(
                data_path=temp_dir,
                result_path=temp_dir,
                model_name="BAE1",
                layer_name="ConvM1",
                feature_extractor="ErrM",
                anomaly_algorithm="InvalidAlgorithm",
            )


class TestExperimentConfig:
    """Tests for ExperimentConfig class."""

    def test_from_ini_file(self, sample_ini_file: Path, sample_dataset_dir: Path):
        """Test loading config from INI file."""
        # Update INI to use actual dataset path
        content = sample_ini_file.read_text()
        content = content.replace("./test_data", str(sample_dataset_dir))
        sample_ini_file.write_text(content)

        config = ExperimentConfig.from_ini_file(sample_ini_file)

        assert config.general.label_info == "TestDataset"
        assert config.general.image_dim.height == 256
        assert config.training is not None
        assert "ConvM1" in config.training.layers
        assert config.prediction is not None
        assert config.prediction.model_name == "BAE1"

    def test_missing_file_raises_error(self, temp_dir: Path):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            ExperimentConfig.from_ini_file(temp_dir / "nonexistent.ini")

    def test_missing_general_section_raises_error(self, temp_dir: Path):
        """Test that missing General section raises error."""
        ini_path = temp_dir / "bad_config.ini"
        ini_path.write_text("[Training]\nlayerSel = ConvM1\n")

        with pytest.raises(ConfigurationError, match="Missing \\[General\\] section"):
            ExperimentConfig.from_ini_file(ini_path)


class TestLoadAllConfigs:
    """Tests for load_all_configs function."""

    def test_load_from_directory(self, temp_dir: Path, sample_ini_content: str, sample_dataset_dir: Path):
        """Test loading all configs from a directory."""
        init_dir = temp_dir / "init"
        init_dir.mkdir()

        # Create two config files
        content = sample_ini_content.replace("./test_data", str(sample_dataset_dir))
        (init_dir / "config1.ini").write_text(content)
        (init_dir / "config2.ini").write_text(content)

        configs = load_all_configs(init_dir)
        assert len(configs) == 2

    def test_empty_directory(self, temp_dir: Path):
        """Test loading from empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        configs = load_all_configs(empty_dir)
        assert len(configs) == 0

    def test_nonexistent_directory(self, temp_dir: Path):
        """Test loading from nonexistent directory returns empty list."""
        configs = load_all_configs(temp_dir / "nonexistent")
        assert len(configs) == 0
