# -*- coding: utf-8 -*-
"""
Unified evaluation script for the AE Reconstruction and Feature Based AD framework.

This module provides evaluation capabilities using PyTorch models.

Usage:
    python evaluate.py --config init/experiment.ini
    python evaluate.py --init-dir ./init --save-images
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import yaml
from tqdm import tqdm

from config import ExperimentConfig, load_all_configs
from data_loader import create_inference_loader
from models import create_autoencoder
from registry import get_feature_extractor


# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for inference."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def apply_gaussian_filter(
    images: np.ndarray,
    filter_strength: float = 0.5,
    kernel_size: tuple[int, int] = (25, 25),
) -> np.ndarray:
    """
    Apply Gaussian blur filter to images.

    Args:
        images: Input images array (N, H, W, C) or (N, C, H, W)
        filter_strength: Blend ratio (0.5 = equal blend of original and blurred)
        kernel_size: Gaussian kernel size for smoothing

    Returns:
        Filtered images array
    """
    # Ensure images are in (N, H, W, C) format
    if images.ndim == 4 and images.shape[1] in (1, 3):
        # Convert from (N, C, H, W) to (N, H, W, C)
        images = np.transpose(images, (0, 2, 3, 1))

    filtered = []
    for img in images:
        blurred = cv.GaussianBlur(img, kernel_size, 0)
        if blurred.ndim == 2:
            blurred = blurred[..., np.newaxis]
        blended = filter_strength * img + (1 - filter_strength) * blurred
        filtered.append(blended)

    return np.array(filtered)


def build_model(
    config: ExperimentConfig,
    device: torch.device,
) -> torch.nn.Module:
    """
    Build and load the autoencoder model.

    Args:
        config: Experiment configuration
        device: Device to load model on

    Returns:
        Loaded PyTorch model with weights
    """
    if config.prediction is None:
        raise ValueError("Prediction configuration is required")

    # Create model
    model = create_autoencoder(
        model_name=config.prediction.model_name,
        layer_name=config.prediction.layer_name,
        image_dim=config.general.image_dim.as_tuple(),
        latent_dim=config.hyperparameters.latent_dim,
        intermediate_dim=config.hyperparameters.intermediate_dim,
        num_embeddings=config.hyperparameters.num_embeddings,
        data_variance=config.hyperparameters.data_variance,
    )

    # Load weights
    base_path = (
        config.general.model_base_path
        / f"{config.prediction.layer_name}_{config.general.label_info}"
        / config.prediction.model_name
    )
    weights_path = base_path / "model.weights.pt"

    if not weights_path.exists():
        # Try legacy path
        weights_path = base_path / "final_model.pt"

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    # Load weights
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    logging.info(f"Loaded model from {weights_path}")
    return model


def prepare_output_directories(result_path: Path) -> tuple[Path, Path]:
    """
    Prepare output directories for OK and NOK classifications.

    Args:
        result_path: Base result directory

    Returns:
        Tuple of (ok_path, nok_path)
    """
    result_path.mkdir(parents=True, exist_ok=True)

    ok_path = result_path / "OK"
    nok_path = result_path / "NOK"

    ok_path.mkdir(exist_ok=True)
    nok_path.mkdir(exist_ok=True)

    return ok_path, nok_path


@torch.no_grad()
def evaluate(
    config: ExperimentConfig,
    model: torch.nn.Module,
    device: torch.device,
    save_images: bool = False,
) -> dict[str, list[str]]:
    """
    Evaluate images using the trained model.

    Args:
        config: Experiment configuration
        model: Loaded autoencoder model
        device: Device for inference
        save_images: Whether to save sorted images to OK/NOK folders

    Returns:
        Dictionary with 'OK' and 'NOK' lists of file paths
    """
    if config.prediction is None:
        raise ValueError("Prediction configuration is required")

    pred_config = config.prediction
    image_dim = config.general.image_dim

    # Create data loader
    data_loader = create_inference_loader(
        pred_config.data_path,
        image_dim.as_tuple(),
        batch_size=pred_config.batch_size,
        num_workers=4,
    )

    if len(data_loader.dataset) == 0:
        logging.warning(f"No images found in {pred_config.data_path}")
        return {"OK": [], "NOK": []}

    start_time = time.time()

    # Process all batches
    all_originals = []
    all_reconstructions = []
    all_paths = []

    model.eval()
    pbar = tqdm(data_loader, desc="Evaluating", leave=False)

    for images, paths in pbar:
        images = images.to(device)

        # Get reconstructions
        output = model(images)

        # Handle different model outputs (VAE returns tuple, basic AE returns tensor)
        if isinstance(output, tuple):
            reconstructions = output[0]
        else:
            reconstructions = output

        # Store results (convert to numpy)
        all_originals.append(images.cpu().numpy())
        all_reconstructions.append(reconstructions.cpu().numpy())
        all_paths.extend(paths)

    # Concatenate all batches
    orig_data = np.concatenate(all_originals, axis=0)
    recon_data = np.concatenate(all_reconstructions, axis=0)

    # Convert from (N, C, H, W) to (N, H, W, C) for feature extraction
    orig_data = np.transpose(orig_data, (0, 2, 3, 1))
    recon_data = np.transpose(recon_data, (0, 2, 3, 1))

    # Apply Gaussian filter
    orig_data = apply_gaussian_filter(orig_data)

    # Build prediction data for feature extractor
    prediction_data = {
        "Predict": {
            "Org": orig_data,
            "Dec": recon_data,
            "Lab": [None for _ in recon_data],
        }
    }

    # Run feature extraction and classification
    base_path = (
        config.general.model_base_path
        / f"{pred_config.layer_name}_{config.general.label_info}"
        / pred_config.model_name
    )

    extractor_class = get_feature_extractor(pred_config.feature_extractor)
    extractor = extractor_class(
        str(base_path / "modelData"),
        str(pred_config.result_path),
        pred_config.model_name,
        pred_config.layer_name,
        "Classification test",
        image_dim.as_tuple(),
        prediction_data,
        [pred_config.anomaly_algorithm],
        False,  # visualize=False
    )

    labels = extractor.predictedLabels
    sorted_labels = {path: label for path, label in zip(all_paths, labels)}

    ok_files = [path for path, label in sorted_labels.items() if label]
    nok_files = [path for path, label in sorted_labels.items() if not label]

    logging.info(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
    logging.info(f"Results: {len(ok_files)} OK, {len(nok_files)} NOK")

    # Save sorted images if requested
    if save_images:
        ok_path, nok_path = prepare_output_directories(pred_config.result_path)
        for src in ok_files:
            shutil.copy(src, ok_path)
        for src in nok_files:
            shutil.copy(src, nok_path)

    return {"OK": ok_files, "NOK": nok_files}


def evaluate_config(
    config: ExperimentConfig,
    save_images: bool = False,
    device: str = "auto",
) -> dict[str, list[str]]:
    """
    Evaluate a single configuration.

    Args:
        config: Experiment configuration
        save_images: Whether to save sorted images
        device: Device to use for inference

    Returns:
        Dictionary with classification results
    """
    if config.prediction is None:
        raise ValueError("Configuration must have prediction section")

    torch_device = get_device(device)
    logging.info(f"Using device: {torch_device}")

    model = build_model(config, torch_device)
    return evaluate(config, model, torch_device, save_images)


def save_labels(labels: dict[str, list[str]], output_path: Path) -> None:
    """Save classification labels to YAML file."""
    labels_path = output_path / "labels.yaml"

    # Convert to absolute paths
    labels_abs = {
        key: [str(Path(p).absolute()) for p in paths]
        for key, paths in labels.items()
        if paths
    }

    with open(labels_path, "w") as f:
        yaml.safe_dump(labels_abs, f)

    logging.info(f"Labels saved to {labels_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained autoencoder models for anomaly detection"
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to specific INI config file (default: process all in ./init)",
    )

    parser.add_argument(
        "--init-dir",
        type=str,
        default="./init",
        help="Directory containing INI config files",
    )

    parser.add_argument(
        "--save-images",
        "-s",
        action="store_true",
        help="Save sorted images to OK/NOK folders",
    )

    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use for inference",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    args = parse_args()

    if args.config:
        # Process single config file
        configs = [
            (
                Path(args.config),
                ExperimentConfig.from_ini_file(args.config, require_prediction=True),
            )
        ]
    else:
        # Process all configs in directory
        configs = load_all_configs(args.init_dir)

    for config_path, config in configs:
        if config.prediction is None:
            logging.warning(f"Skipping {config_path}: no prediction configuration")
            continue

        logging.info(f"Processing: {config_path}")
        logging.info(
            f"  Model: {config.prediction.model_name}, Layer: {config.prediction.layer_name}"
        )
        logging.info(f"  Feature extractor: {config.prediction.feature_extractor}")
        logging.info(f"  Anomaly algorithm: {config.prediction.anomaly_algorithm}")

        try:
            labels = evaluate_config(config, args.save_images, args.device)
            save_labels(labels, config.prediction.result_path)
        except FileNotFoundError as e:
            logging.error(f"Failed to process {config_path}: {e}")
        except Exception as e:
            logging.exception(f"Error processing {config_path}: {e}")


if __name__ == "__main__":
    main()
