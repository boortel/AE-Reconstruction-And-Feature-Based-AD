# -*- coding: utf-8 -*-
"""
Tests for the registry module.
"""

from __future__ import annotations

import pytest
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from registry import (
    DEFAULT_MODEL_PARAMS,
    LAYER_CONFIGURATIONS,
    MODEL_ARCHITECTURES,
    get_anomaly_algorithm,
    get_anomaly_algorithms,
    get_feature_extractor,
    get_feature_extractors,
)


class TestFeatureExtractors:
    """Tests for feature extractor registry."""

    def test_get_all_extractors(self):
        """Test getting all feature extractors."""
        extractors = get_feature_extractors()

        assert isinstance(extractors, dict)
        assert len(extractors) >= 7  # At least 7 extractors

        expected_keys = ["Enc", "ErrM", "SIFT", "HardNet1", "HardNet2", "HardNet3", "HardNet4"]
        for key in expected_keys:
            assert key in extractors, f"Missing extractor: {key}"

    def test_get_specific_extractor(self):
        """Test getting a specific feature extractor."""
        extractor = get_feature_extractor("ErrM")
        assert extractor is not None
        assert extractor.__name__ == "ModelClassificationErrM"

    def test_get_invalid_extractor_raises_error(self):
        """Test that getting invalid extractor raises KeyError."""
        with pytest.raises(KeyError, match="Unknown feature extractor"):
            get_feature_extractor("InvalidExtractor")

    def test_extractors_are_classes(self):
        """Test that all extractors are classes."""
        extractors = get_feature_extractors()
        for name, extractor in extractors.items():
            assert isinstance(extractor, type), f"{name} is not a class"


class TestAnomalyAlgorithms:
    """Tests for anomaly algorithm registry."""

    def test_get_all_algorithms(self):
        """Test getting all anomaly algorithms."""
        algorithms = get_anomaly_algorithms()

        assert isinstance(algorithms, dict)
        assert len(algorithms) == 4

        expected_keys = [
            "Robust covariance",
            "One-Class SVM",
            "Isolation Forest",
            "Local Outlier Factor",
        ]
        for key in expected_keys:
            assert key in algorithms, f"Missing algorithm: {key}"

    def test_algorithm_types(self):
        """Test that algorithms are of correct types."""
        algorithms = get_anomaly_algorithms()

        assert isinstance(algorithms["Robust covariance"], EllipticEnvelope)
        assert isinstance(algorithms["One-Class SVM"], OneClassSVM)
        assert isinstance(algorithms["Isolation Forest"], IsolationForest)
        assert isinstance(algorithms["Local Outlier Factor"], LocalOutlierFactor)

    def test_get_specific_algorithm(self):
        """Test getting a specific algorithm."""
        algorithm = get_anomaly_algorithm("Isolation Forest")
        assert isinstance(algorithm, IsolationForest)

    def test_get_invalid_algorithm_raises_error(self):
        """Test that getting invalid algorithm raises KeyError."""
        with pytest.raises(KeyError, match="Unknown anomaly algorithm"):
            get_anomaly_algorithm("InvalidAlgorithm")

    def test_custom_outliers_fraction(self):
        """Test that outliers fraction is configurable."""
        algorithms = get_anomaly_algorithms(outliers_fraction=0.05)

        # Check that contamination is set correctly
        assert algorithms["Robust covariance"].contamination == 0.05
        assert algorithms["Isolation Forest"].contamination == 0.05


class TestConstants:
    """Tests for registry constants."""

    def test_model_architectures(self):
        """Test model architecture constants."""
        expected = ["BAE1", "BAE2", "VAE1", "VAE2", "VQVAE1"]
        assert MODEL_ARCHITECTURES == expected

    def test_layer_configurations(self):
        """Test layer configuration constants."""
        expected = ["ConvM1", "ConvM2", "ConvM3", "ConvM4", "ConvM5", "ConvM6"]
        assert LAYER_CONFIGURATIONS == expected

    def test_default_model_params(self):
        """Test default model parameters."""
        assert "dataVariance" in DEFAULT_MODEL_PARAMS
        assert "intermediateDim" in DEFAULT_MODEL_PARAMS
        assert "latentDim" in DEFAULT_MODEL_PARAMS
        assert "num_embeddings" in DEFAULT_MODEL_PARAMS

        assert DEFAULT_MODEL_PARAMS["dataVariance"] == 0.5
        assert DEFAULT_MODEL_PARAMS["intermediateDim"] == 64
        assert DEFAULT_MODEL_PARAMS["latentDim"] == 32
        assert DEFAULT_MODEL_PARAMS["num_embeddings"] == 32
