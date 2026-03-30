# -*- coding: utf-8 -*-
"""
Tests for the metrics module.
"""

from __future__ import annotations

import numpy as np
import pytest

from metrics import (
    ClassificationMetrics,
    apply_threshold,
    compute_confusion_metrics,
    compute_roc_metrics,
)


class TestApplyThreshold:
    """Tests for apply_threshold function."""

    def test_basic_threshold(self):
        """Test basic threshold application."""
        scores = np.array([0.2, 0.5, 0.8, 0.3, 0.9])
        threshold = 0.5

        predictions, boolean_labels = apply_threshold(scores, threshold)

        # Scores >= 0.5 should be OK (1, True)
        expected_predictions = np.array([-1, 1, 1, -1, 1])
        expected_booleans = np.array([False, True, True, False, True])

        np.testing.assert_array_equal(predictions, expected_predictions)
        np.testing.assert_array_equal(boolean_labels, expected_booleans)

    def test_all_above_threshold(self):
        """Test when all scores are above threshold."""
        scores = np.array([0.6, 0.7, 0.8, 0.9])
        threshold = 0.5

        predictions, boolean_labels = apply_threshold(scores, threshold)

        assert np.all(predictions == 1)
        assert np.all(boolean_labels)

    def test_all_below_threshold(self):
        """Test when all scores are below threshold."""
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        threshold = 0.5

        predictions, boolean_labels = apply_threshold(scores, threshold)

        assert np.all(predictions == -1)
        assert not np.any(boolean_labels)

    def test_edge_case_at_threshold(self):
        """Test scores exactly at threshold."""
        scores = np.array([0.5, 0.5, 0.5])
        threshold = 0.5

        predictions, boolean_labels = apply_threshold(scores, threshold)

        # Scores >= threshold should be OK
        assert np.all(predictions == 1)
        assert np.all(boolean_labels)


class TestComputeConfusionMetrics:
    """Tests for compute_confusion_metrics function."""

    def test_perfect_classification(self):
        """Test with perfect classification."""
        y_true = np.array([1, 1, 1, -1, -1, -1])
        y_pred = np.array([1, 1, 1, -1, -1, -1])

        metrics = compute_confusion_metrics(y_true, y_pred, "Test")

        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0

    def test_all_wrong_classification(self):
        """Test with all wrong predictions."""
        y_true = np.array([1, 1, 1])
        y_pred = np.array([-1, -1, -1])

        metrics = compute_confusion_metrics(y_true, y_pred, "Test")

        assert metrics.recall == 0.0

    def test_returns_classification_metrics(self):
        """Test that function returns ClassificationMetrics object."""
        y_true = np.array([1, -1, 1, -1])
        y_pred = np.array([1, -1, -1, -1])

        metrics = compute_confusion_metrics(y_true, y_pred, "Test")

        assert isinstance(metrics, ClassificationMetrics)


class TestComputeRocMetrics:
    """Tests for compute_roc_metrics function."""

    def test_returns_threshold(self):
        """Test that function returns a threshold value."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

        threshold = compute_roc_metrics(y_true, scores, "Test")

        assert isinstance(threshold, (int, float, np.floating))

    def test_perfect_separation(self):
        """Test with perfectly separable classes."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([1.0, 0.9, 0.8, 0.2, 0.1, 0.0])

        threshold = compute_roc_metrics(y_true, scores, "Test")

        # Threshold should be between the two classes
        assert 0.2 < threshold < 0.8


class TestClassificationMetrics:
    """Tests for ClassificationMetrics dataclass."""

    def test_dataclass_creation(self):
        """Test creating ClassificationMetrics instance."""
        metrics = ClassificationMetrics(
            roc_auc=0.95,
            prc_auc=0.90,
            precision=0.85,
            recall=0.80,
            f1_score=0.82,
            tpr=0.80,
            tnr=0.90,
            optimal_threshold=0.5,
        )

        assert metrics.roc_auc == 0.95
        assert metrics.precision == 0.85
        assert metrics.f1_score == 0.82
