# -*- coding: utf-8 -*-
"""
Visualization utilities for anomaly detection results.

This module provides functions for plotting feature spaces, classification
results, and autoencoder outputs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def visualize_feature_space(
    metrics: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: str | Path,
    perplexity: int = 30,
) -> Figure:
    """
    Visualize feature space using t-SNE and PCA.

    Args:
        metrics: Feature matrix (n_samples, n_features)
        labels: Class labels (1 for OK, -1 for NOK)
        title: Plot title
        output_path: Path to save the figure
        perplexity: t-SNE perplexity parameter

    Returns:
        matplotlib Figure object
    """
    # Reduce dimensionality for t-SNE if needed
    if metrics.shape[0] > 50 and metrics.shape[1] > 50:
        pca_reducer = PCA(n_components=50)
        metrics_reduced = pca_reducer.fit_transform(metrics)
    else:
        metrics_reduced = metrics

    # Adjust perplexity for small datasets
    if metrics.shape[0] < 30:
        perplexity = int(metrics.shape[0] / 2)

    # Perform t-SNE
    tsne_metrics = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=1000,
        learning_rate=100,
        init="pca",
    ).fit_transform(metrics_reduced)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_metrics = pca.fit_transform(metrics_reduced)

    # Create visualization
    fig, axarr = plt.subplots(2)
    fig.set_size_inches(8, 8)
    fig.suptitle(title, fontsize=12)

    ok_idx = np.where(labels == 1)
    nok_idx = np.where(labels == -1)

    # t-SNE plot
    axarr[0].scatter(tsne_metrics[ok_idx, 0], tsne_metrics[ok_idx, 1], s=4)
    axarr[0].scatter(tsne_metrics[nok_idx, 0], tsne_metrics[nok_idx, 1], s=4)
    axarr[0].set(xlabel="t-SNE 1", ylabel="t-SNE 2")
    axarr[0].set_title("t-SNE Feature space visualisation")
    axarr[0].legend(["OK", "NOK"], loc="upper right")

    # PCA plot
    axarr[1].scatter(pca_metrics[ok_idx, 0], pca_metrics[ok_idx, 1], s=4)
    axarr[1].scatter(pca_metrics[nok_idx, 0], pca_metrics[nok_idx, 1], s=4)
    axarr[1].set(xlabel="PCA 1", ylabel="PCA 2")
    axarr[1].set_title("PCA Feature space visualisation")
    axarr[1].legend(["OK", "NOK"], loc="upper right")

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    fig.savefig(output_path)

    return fig


def visualize_classification_results(
    n_algorithms: int,
    title: str,
) -> tuple[Figure, np.ndarray]:
    """
    Create a figure for classification results visualization.

    Args:
        n_algorithms: Number of anomaly detection algorithms
        title: Figure title

    Returns:
        Tuple of (figure, axes_array)
    """
    fig, axarr = plt.subplots(2, n_algorithms)
    fig.set_size_inches(16, 8)
    fig.suptitle(title, fontsize=16)

    return fig, axarr


def save_classification_figure(
    fig: Figure,
    output_path: str | Path,
) -> None:
    """
    Save classification results figure.

    Args:
        fig: matplotlib Figure
        output_path: Path to save the figure
    """
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(output_path)


def visualize_training_loss(
    train_loss: list[float],
    val_loss: list[float],
    model_name: str,
    layer_name: str,
    label_info: str,
    output_path: str | Path,
    val_label: str = "Validation loss [-]",
) -> Figure:
    """
    Visualize training and validation loss curves.

    Args:
        train_loss: Training loss history
        val_loss: Validation/other loss history
        model_name: Name of the model
        layer_name: Name of the layer configuration
        label_info: Dataset label
        output_path: Path to save the figure
        val_label: Label for the validation loss axis

    Returns:
        matplotlib Figure object
    """
    title = f"Training and Validation Loss of {layer_name}-{model_name}_{label_info} model"

    fig, axarr = plt.subplots(2)
    fig.suptitle(title, fontsize=14)

    axarr[0].plot(train_loss)
    axarr[0].set(xlabel="Number of Epochs", ylabel="Training Loss [-]")

    axarr[1].plot(val_loss)
    axarr[1].set(xlabel="Number of Epochs", ylabel=val_label)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    fig.savefig(output_path)

    return fig


def visualize_autoencoder_results(
    original: np.ndarray,
    encoded: np.ndarray,
    decoded: np.ndarray,
    indices: list[int],
    model_name: str,
    layer_name: str,
    label_info: str,
    ae_type: str,
    output_path: str | Path,
) -> Figure:
    """
    Visualize autoencoder input/output for selected samples.

    Args:
        original: Original images
        encoded: Encoded representations
        decoded: Reconstructed images
        indices: Indices of samples to visualize
        model_name: Name of the model
        layer_name: Name of the layer configuration
        label_info: Dataset label
        ae_type: Type of autoencoder (BAE1, VAE1, etc.)
        output_path: Path to save the figure

    Returns:
        matplotlib Figure object
    """
    import cv2 as cv

    # Compute difference images
    diff_data = np.subtract(original, decoded)

    # Create figure
    fig, axarr = plt.subplots(len(indices), 4)
    title = f"Visualisations of the {layer_name}-{model_name}_{label_info} model"

    fig.suptitle(title, fontsize=18)
    fig.set_size_inches(4 * len(indices), 16)

    img_sources = [original, encoded, decoded, diff_data]
    img_titles = ["Original", "Encoded", "Decoded", "Difference"]

    for v_idx, img_idx in enumerate(indices):
        for h_idx, (img_title, img_source) in enumerate(zip(img_titles, img_sources)):
            axarr[v_idx, h_idx].set_title(img_title)

            if img_title == "Encoded":
                if ae_type in ("VAE1", "VAE2"):
                    axarr[v_idx, h_idx].scatter(
                        img_source[img_idx, :, 0], img_source[img_idx, :, 1], s=4
                    )
                    axarr[v_idx, h_idx].set(
                        xlabel="Mean", ylabel="Variance", xlim=(-10, 10), ylim=(-10, 10)
                    )
                elif ae_type in ("BAE1", "BAE2"):
                    normalized = cv.normalize(
                        img_source[img_idx].mean(axis=2), None, 0, 255, cv.NORM_MINMAX, cv.CV_8U
                    )
                    axarr[v_idx, h_idx].imshow(normalized)
                    axarr[v_idx, h_idx].axis("off")
                elif ae_type == "VQVAE1":
                    normalized = cv.normalize(
                        img_source[img_idx], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U
                    )
                    axarr[v_idx, h_idx].imshow(normalized)
                    axarr[v_idx, h_idx].axis("off")
            else:
                normalized = cv.normalize(
                    img_source[img_idx], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U
                )
                axarr[v_idx, h_idx].imshow(normalized)
                axarr[v_idx, h_idx].axis("off")

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(output_path)

    return fig
