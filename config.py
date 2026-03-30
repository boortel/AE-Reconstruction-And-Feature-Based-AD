# -*- coding: utf-8 -*-
"""
Configuration management with validation and type safety.

This module provides dataclasses for all configuration types, with proper
validation and sensible defaults.
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from registry import (
    DEFAULT_MODEL_PARAMS,
    LAYER_CONFIGURATIONS,
    MODEL_ARCHITECTURES,
    get_anomaly_algorithms,
    get_feature_extractors,
)


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

    pass


@dataclass
class ImageDimensions:
    """Image dimension configuration."""

    height: int
    width: int
    channels: int

    def __post_init__(self) -> None:
        if self.height <= 0:
            raise ConfigurationError(f"Image height must be positive, got {self.height}")
        if self.width <= 0:
            raise ConfigurationError(f"Image width must be positive, got {self.width}")
        if self.channels not in (1, 3):
            raise ConfigurationError(f"Image channels must be 1 or 3, got {self.channels}")

    def as_tuple(self) -> tuple[int, int, int]:
        """Return dimensions as (height, width, channels) tuple."""
        return (self.height, self.width, self.channels)


@dataclass
class GeneralConfig:
    """General experiment configuration."""

    label_info: str
    model_base_path: Path
    image_dim: ImageDimensions
    npz_save: bool = False
    image_indices: list[int] = field(default_factory=lambda: [0, 100, 200])

    def __post_init__(self) -> None:
        if not self.label_info:
            raise ConfigurationError("label_info cannot be empty")
        self.model_base_path = Path(self.model_base_path)


@dataclass
class TrainingConfig:
    """Training configuration."""

    dataset_path: Path
    layers: list[str]
    models: list[str]
    batch_size: int = 20
    num_epochs: int = 200

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path)
        if not self.dataset_path.exists():
            raise ConfigurationError(f"Dataset path does not exist: {self.dataset_path}")
        if self.batch_size <= 0:
            raise ConfigurationError(f"Batch size must be positive, got {self.batch_size}")
        if self.num_epochs <= 0:
            raise ConfigurationError(f"Number of epochs must be positive, got {self.num_epochs}")

        # Validate layer and model names
        for layer in self.layers:
            if layer not in LAYER_CONFIGURATIONS:
                raise ConfigurationError(
                    f"Unknown layer '{layer}'. Available: {LAYER_CONFIGURATIONS}"
                )
        for model in self.models:
            if model not in MODEL_ARCHITECTURES:
                raise ConfigurationError(
                    f"Unknown model '{model}'. Available: {MODEL_ARCHITECTURES}"
                )


@dataclass
class PredictionConfig:
    """Prediction/evaluation configuration."""

    data_path: Path
    result_path: Path
    model_name: str
    layer_name: str
    feature_extractor: str
    anomaly_algorithm: str
    batch_size: int = 100

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.result_path = Path(self.result_path)

        if self.batch_size <= 0:
            raise ConfigurationError(f"Batch size must be positive, got {self.batch_size}")

        # Validate model and layer
        if self.model_name not in MODEL_ARCHITECTURES:
            raise ConfigurationError(
                f"Unknown model '{self.model_name}'. Available: {MODEL_ARCHITECTURES}"
            )
        if self.layer_name not in LAYER_CONFIGURATIONS:
            raise ConfigurationError(
                f"Unknown layer '{self.layer_name}'. Available: {LAYER_CONFIGURATIONS}"
            )

        # Validate feature extractor and anomaly algorithm
        extractors = get_feature_extractors()
        if self.feature_extractor not in extractors:
            raise ConfigurationError(
                f"Unknown feature extractor '{self.feature_extractor}'. "
                f"Available: {list(extractors.keys())}"
            )

        algorithms = get_anomaly_algorithms()
        if self.anomaly_algorithm not in algorithms:
            raise ConfigurationError(
                f"Unknown anomaly algorithm '{self.anomaly_algorithm}'. "
                f"Available: {list(algorithms.keys())}"
            )


@dataclass
class ModelHyperparameters:
    """Autoencoder model hyperparameters."""

    data_variance: float = DEFAULT_MODEL_PARAMS["dataVariance"]
    intermediate_dim: int = DEFAULT_MODEL_PARAMS["intermediateDim"]
    latent_dim: int = DEFAULT_MODEL_PARAMS["latentDim"]
    num_embeddings: int = DEFAULT_MODEL_PARAMS["num_embeddings"]


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    general: GeneralConfig
    training: TrainingConfig | None = None
    prediction: PredictionConfig | None = None
    hyperparameters: ModelHyperparameters = field(default_factory=ModelHyperparameters)

    @classmethod
    def from_ini_file(cls, ini_path: str | Path, require_training: bool = False, require_prediction: bool = False) -> Self:
        """
        Load configuration from an INI file.

        Args:
            ini_path: Path to the INI file
            require_training: If True, raise error if [Training] section is missing
            require_prediction: If True, raise error if [Prediction] section is missing

        Returns:
            ExperimentConfig instance

        Raises:
            ConfigurationError: If required sections are missing or validation fails
            FileNotFoundError: If the INI file doesn't exist
        """
        ini_path = Path(ini_path)
        if not ini_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {ini_path}")

        # Create a fresh ConfigParser to avoid accumulation issues
        cfg = configparser.ConfigParser()
        cfg.read(ini_path)

        # Parse General section (required)
        if "General" not in cfg:
            raise ConfigurationError(f"Missing [General] section in {ini_path}")

        general_section = cfg["General"]
        image_dim = ImageDimensions(
            height=general_section.getint("imHeight"),
            width=general_section.getint("imWidth"),
            channels=general_section.getint("imChannel"),
        )

        # Parse image indices
        indices_str = general_section.get("imIndxList", "0, 100, 200")
        image_indices = [int(x.strip()) for x in indices_str.split(",")]

        general = GeneralConfig(
            label_info=general_section.get("labelInfo"),
            model_base_path=Path(general_section.get("modelBasePath")),
            image_dim=image_dim,
            npz_save=general_section.getboolean("npzSave", fallback=False),
            image_indices=image_indices,
        )

        # Parse Training section (optional)
        training = None
        if "Training" in cfg:
            training_section = cfg["Training"]
            layers = [x.strip() for x in training_section.get("layerSel", "").split(",") if x.strip()]
            models = [x.strip() for x in training_section.get("modelSel", "").split(",") if x.strip()]

            dataset_path = Path(training_section.get("datasetPath"))
            # Resolve relative paths from INI file location
            if not dataset_path.is_absolute():
                dataset_path = ini_path.parent / dataset_path

            training = TrainingConfig(
                dataset_path=dataset_path,
                layers=layers,
                models=models,
                batch_size=training_section.getint("batchSize", fallback=20),
                num_epochs=training_section.getint("numEpoch", fallback=200),
            )
        elif require_training:
            raise ConfigurationError(f"Missing required [Training] section in {ini_path}")

        # Parse Prediction section (optional)
        prediction = None
        if "Prediction" in cfg:
            pred_section = cfg["Prediction"]

            data_path = Path(pred_section.get("predictionDataPath"))
            result_path = Path(pred_section.get("predictionResultPath"))

            # Resolve relative paths from INI file location
            if not data_path.is_absolute():
                data_path = ini_path.parent / data_path
            if not result_path.is_absolute():
                result_path = ini_path.parent / result_path

            prediction = PredictionConfig(
                data_path=data_path,
                result_path=result_path,
                model_name=pred_section.get("modelName"),
                layer_name=pred_section.get("layerName"),
                feature_extractor=pred_section.get("featureExtractorName"),
                anomaly_algorithm=pred_section.get("anomalyAlgorithmName"),
                batch_size=pred_section.getint("batchSize", fallback=100),
            )
        elif require_prediction:
            raise ConfigurationError(f"Missing required [Prediction] section in {ini_path}")

        return cls(
            general=general,
            training=training,
            prediction=prediction,
        )


def load_all_configs(init_dir: str | Path = "./init") -> list[tuple[Path, ExperimentConfig]]:
    """
    Load all INI configuration files from a directory.

    Args:
        init_dir: Directory containing INI files

    Returns:
        List of (path, config) tuples for successfully loaded configs

    Note:
        Files that fail to load are logged but don't stop processing.
    """
    import logging

    init_dir = Path(init_dir)
    configs = []

    if not init_dir.exists():
        logging.warning(f"Init directory does not exist: {init_dir}")
        return configs

    for ini_file in sorted(init_dir.glob("**/*.ini")):
        try:
            config = ExperimentConfig.from_ini_file(ini_file)
            configs.append((ini_file, config))
        except (ConfigurationError, FileNotFoundError) as e:
            logging.warning(f"Failed to load {ini_file}: {e}")

    return configs
