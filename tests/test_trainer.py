# -*- coding: utf-8 -*-
"""
Tests for PyTorch trainer module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models import BAE1, VAE1, VQVAE1
from trainer import EarlyStopping, Trainer, TrainingConfig, TrainingHistory


@pytest.fixture
def simple_train_loader() -> DataLoader:
    """Create a simple training data loader."""
    # Create random data
    images = torch.randn(20, 3, 64, 64)
    labels = torch.ones(20, dtype=torch.long)
    paths = [f"image_{i}.png" for i in range(20)]

    # Create dataset that returns (image, label, path)
    class SimpleDataset:
        def __init__(self, images, labels, paths):
            self.images = images
            self.labels = labels
            self.paths = paths

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx], self.paths[idx]

    dataset = SimpleDataset(images, labels, paths)
    return DataLoader(dataset, batch_size=4, shuffle=True)


@pytest.fixture
def simple_val_loader() -> DataLoader:
    """Create a simple validation data loader."""
    images = torch.randn(10, 3, 64, 64)
    labels = torch.ones(10, dtype=torch.long)
    paths = [f"val_image_{i}.png" for i in range(10)]

    class SimpleDataset:
        def __init__(self, images, labels, paths):
            self.images = images
            self.labels = labels
            self.paths = paths

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx], self.paths[idx]

    dataset = SimpleDataset(images, labels, paths)
    return DataLoader(dataset, batch_size=4, shuffle=False)


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.num_epochs == 200
        assert config.learning_rate == 1e-3
        assert config.patience == 10
        assert config.device == "auto"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            num_epochs=100,
            learning_rate=1e-4,
            patience=5,
            device="cpu",
        )

        assert config.num_epochs == 100
        assert config.learning_rate == 1e-4
        assert config.patience == 5
        assert config.device == "cpu"


class TestTrainingHistory:
    """Tests for TrainingHistory dataclass."""

    def test_default_values(self):
        """Test default history values."""
        history = TrainingHistory()

        assert history.train_loss == []
        assert history.val_loss == []
        assert history.best_loss == float("inf")
        assert history.best_epoch == 0

    def test_appending_losses(self):
        """Test appending losses to history."""
        history = TrainingHistory()
        history.train_loss.append(0.5)
        history.train_loss.append(0.4)

        assert len(history.train_loss) == 2
        assert history.train_loss[0] == 0.5


class TestEarlyStopping:
    """Tests for EarlyStopping class."""

    def test_no_stop_when_improving(self):
        """Test that training continues when loss improves."""
        early_stopping = EarlyStopping(patience=3)

        # Decreasing loss - should not stop
        assert not early_stopping(0.5)
        assert not early_stopping(0.4)
        assert not early_stopping(0.3)

    def test_stop_after_patience(self):
        """Test that training stops after patience is exhausted."""
        early_stopping = EarlyStopping(patience=3)

        # Initial improvement
        early_stopping(0.5)

        # No improvement for patience iterations
        assert not early_stopping(0.6)
        assert not early_stopping(0.6)
        assert early_stopping(0.6)  # Should stop after 3 non-improvements

    def test_counter_resets_on_improvement(self):
        """Test that counter resets when loss improves."""
        early_stopping = EarlyStopping(patience=3)

        early_stopping(0.5)
        early_stopping(0.6)  # Counter = 1
        early_stopping(0.6)  # Counter = 2

        # Improvement resets counter
        early_stopping(0.4)
        assert early_stopping.counter == 0

    def test_min_delta(self):
        """Test minimum delta for improvement."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)

        early_stopping(0.5)

        # Improvement less than min_delta doesn't count
        assert not early_stopping(0.495)  # Counter = 1 (improvement < 0.01)


class TestTrainer:
    """Tests for Trainer class."""

    def test_trainer_creation_bae(self):
        """Test creating trainer with BAE model."""
        model = BAE1(layer_name="ConvM1", image_dim=(64, 64, 3))
        config = TrainingConfig(device="cpu", num_epochs=1)

        trainer = Trainer(model, config)

        assert trainer.model is not None
        assert trainer.device == torch.device("cpu")

    def test_trainer_creation_vae(self):
        """Test creating trainer with VAE model."""
        model = VAE1(layer_name="ConvM1", image_dim=(64, 64, 3), latent_dim=16)
        config = TrainingConfig(device="cpu", num_epochs=1)

        trainer = Trainer(model, config)

        assert trainer.is_vae

    def test_trainer_creation_vqvae(self):
        """Test creating trainer with VQ-VAE model."""
        model = VQVAE1(
            layer_name="ConvM1", image_dim=(64, 64, 3), num_embeddings=16
        )
        config = TrainingConfig(device="cpu", num_epochs=1)

        trainer = Trainer(model, config)

        assert trainer.is_vqvae

    def test_train_single_epoch(self, simple_train_loader: DataLoader):
        """Test training for a single epoch."""
        model = BAE1(layer_name="ConvM1", image_dim=(64, 64, 3))
        config = TrainingConfig(device="cpu", num_epochs=1)

        trainer = Trainer(model, config)
        losses = trainer.train_epoch(simple_train_loader)

        assert "total_loss" in losses
        assert losses["total_loss"] > 0

    def test_validate(
        self,
        simple_train_loader: DataLoader,
        simple_val_loader: DataLoader,
    ):
        """Test validation."""
        model = BAE1(layer_name="ConvM1", image_dim=(64, 64, 3))
        config = TrainingConfig(device="cpu", num_epochs=1)

        trainer = Trainer(model, config)
        losses = trainer.validate(simple_val_loader)

        assert "total_loss" in losses
        assert losses["total_loss"] > 0

    def test_full_training(
        self,
        simple_train_loader: DataLoader,
        simple_val_loader: DataLoader,
        temp_dir: Path,
    ):
        """Test full training loop."""
        model = BAE1(layer_name="ConvM1", image_dim=(64, 64, 3))
        config = TrainingConfig(device="cpu", num_epochs=2, patience=10)

        trainer = Trainer(model, config)
        history = trainer.train(
            train_loader=simple_train_loader,
            val_loader=simple_val_loader,
            save_dir=temp_dir,
        )

        assert len(history.train_loss) == 2
        assert len(history.val_loss) == 2

    def test_checkpoint_saving(
        self,
        simple_train_loader: DataLoader,
        simple_val_loader: DataLoader,
        temp_dir: Path,
    ):
        """Test that checkpoints are saved."""
        model = BAE1(layer_name="ConvM1", image_dim=(64, 64, 3))
        config = TrainingConfig(device="cpu", num_epochs=1, save_best=True)

        trainer = Trainer(model, config)
        trainer.train(
            train_loader=simple_train_loader,
            val_loader=simple_val_loader,
            save_dir=temp_dir,
        )

        # Check that files were saved
        assert (temp_dir / "final_model.pt").exists()
        assert (temp_dir / "model.weights.pt").exists()

    def test_vae_loss_computation(self, simple_train_loader: DataLoader):
        """Test VAE loss computation includes KL loss."""
        model = VAE1(layer_name="ConvM1", image_dim=(64, 64, 3), latent_dim=16)
        config = TrainingConfig(device="cpu", num_epochs=1)

        trainer = Trainer(model, config)
        losses = trainer.train_epoch(simple_train_loader)

        assert "kl_loss" in losses

    def test_vqvae_loss_computation(self, simple_train_loader: DataLoader):
        """Test VQ-VAE loss computation includes VQ loss."""
        model = VQVAE1(
            layer_name="ConvM1", image_dim=(64, 64, 3), num_embeddings=16
        )
        config = TrainingConfig(device="cpu", num_epochs=1)

        trainer = Trainer(model, config)
        losses = trainer.train_epoch(simple_train_loader)

        assert "vq_loss" in losses

    def test_save_and_load_checkpoint(
        self,
        simple_train_loader: DataLoader,
        temp_dir: Path,
    ):
        """Test saving and loading checkpoints."""
        model = BAE1(layer_name="ConvM1", image_dim=(64, 64, 3))
        config = TrainingConfig(device="cpu", num_epochs=1)

        trainer = Trainer(model, config)

        # Train a bit
        trainer.train_epoch(simple_train_loader)
        trainer.history.train_loss.append(0.5)

        # Save checkpoint
        checkpoint_path = temp_dir / "checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path)

        # Create new trainer and load
        model2 = BAE1(layer_name="ConvM1", image_dim=(64, 64, 3))
        trainer2 = Trainer(model2, config)
        trainer2.load_checkpoint(checkpoint_path)

        assert trainer2.history.train_loss == trainer.history.train_loss

    def test_save_and_load_weights(self, temp_dir: Path):
        """Test saving and loading weights only."""
        model = BAE1(layer_name="ConvM1", image_dim=(64, 64, 3))
        config = TrainingConfig(device="cpu")

        trainer = Trainer(model, config)

        # Save weights
        weights_path = temp_dir / "weights.pt"
        trainer.save_weights(weights_path)

        # Load into new model
        model2 = BAE1(layer_name="ConvM1", image_dim=(64, 64, 3))
        trainer2 = Trainer(model2, config)
        trainer2.load_weights(weights_path)

        # Check weights match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            torch.testing.assert_close(p1, p2)
