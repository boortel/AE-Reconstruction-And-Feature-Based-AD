# -*- coding: utf-8 -*-
"""
Central registry for feature extractors and anomaly detection algorithms.

This module provides a single source of truth for all available components,
eliminating duplication across evaluation scripts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

if TYPE_CHECKING:
    from ModelClassificationBase import ModelClassificationBase


def get_feature_extractors() -> dict[str, type["ModelClassificationBase"]]:
    """
    Get all available feature extractors.

    Lazy import to avoid circular dependencies and reduce startup time.
    """
    from ModelClassificationEnc import ModelClassificationEnc
    from ModelClassificationErrM import ModelClassificationErrM
    from ModelClassificationHardNet1 import ModelClassificationHardNet1
    from ModelClassificationHardNet2 import ModelClassificationHardNet2
    from ModelClassificationHardNet3 import ModelClassificationHardNet3
    from ModelClassificationHardNet4 import ModelClassificationHardNet4
    from ModelClassificationSIFT import ModelClassificationSIFT

    return {
        "Enc": ModelClassificationEnc,
        "ErrM": ModelClassificationErrM,
        "SIFT": ModelClassificationSIFT,
        "HardNet1": ModelClassificationHardNet1,
        "HardNet2": ModelClassificationHardNet2,
        "HardNet3": ModelClassificationHardNet3,
        "HardNet4": ModelClassificationHardNet4,
    }


def get_feature_extractor(name: str) -> type["ModelClassificationBase"]:
    """
    Get a specific feature extractor by name.

    Args:
        name: The feature extractor name (e.g., 'ErrM', 'HardNet1')

    Returns:
        The feature extractor class

    Raises:
        KeyError: If the feature extractor is not found
    """
    extractors = get_feature_extractors()
    if name not in extractors:
        available = ", ".join(sorted(extractors.keys()))
        raise KeyError(f"Unknown feature extractor '{name}'. Available: {available}")
    return extractors[name]


def get_anomaly_algorithms(outliers_fraction: float = 0.01) -> dict[str, object]:
    """
    Get all available anomaly detection algorithms with default configuration.

    Args:
        outliers_fraction: Expected proportion of outliers in the data

    Returns:
        Dictionary mapping algorithm names to configured instances
    """
    return {
        "Robust covariance": EllipticEnvelope(
            contamination=outliers_fraction, support_fraction=0.9
        ),
        "One-Class SVM": svm.OneClassSVM(
            nu=outliers_fraction, kernel="rbf", gamma="scale"
        ),
        "Isolation Forest": IsolationForest(
            contamination=outliers_fraction, random_state=42
        ),
        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors=15, contamination=outliers_fraction, novelty=True
        ),
    }


def get_anomaly_algorithm(name: str, outliers_fraction: float = 0.01) -> object:
    """
    Get a specific anomaly detection algorithm by name.

    Args:
        name: The algorithm name (e.g., 'Robust covariance', 'Isolation Forest')
        outliers_fraction: Expected proportion of outliers

    Returns:
        Configured algorithm instance

    Raises:
        KeyError: If the algorithm is not found
    """
    algorithms = get_anomaly_algorithms(outliers_fraction)
    if name not in algorithms:
        available = ", ".join(sorted(algorithms.keys()))
        raise KeyError(f"Unknown anomaly algorithm '{name}'. Available: {available}")
    return algorithms[name]


# Model architecture names
MODEL_ARCHITECTURES = ["BAE1", "BAE2", "VAE1", "VAE2", "VQVAE1"]

# Layer configuration names
LAYER_CONFIGURATIONS = ["ConvM1", "ConvM2", "ConvM3", "ConvM4", "ConvM5", "ConvM6"]

# Default model hyperparameters
DEFAULT_MODEL_PARAMS = {
    "dataVariance": 0.5,
    "intermediateDim": 64,
    "latentDim": 32,
    "num_embeddings": 32,
}
