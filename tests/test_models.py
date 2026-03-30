# -*- coding: utf-8 -*-
"""
Tests for PyTorch models.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models import BAE1, BAE2, VAE1, VAE2, VQVAE1, HardNet, create_autoencoder
from models.layers import (
    LAYER_CONFIGS,
    get_decoder,
    get_encoder,
)


class TestLayerConfigs:
    """Tests for layer configurations."""

    def test_all_configs_present(self):
        """Test that all expected layer configs are present."""
        expected = ["ConvM1", "ConvM2", "ConvM3", "ConvM4", "ConvM5", "ConvM6"]
        for name in expected:
            assert name in LAYER_CONFIGS

    def test_config_structure(self):
        """Test that configs have required keys."""
        for name, config in LAYER_CONFIGS.items():
            assert "encoder" in config
            assert "decoder" in config


class TestEncoders:
    """Tests for encoder classes."""

    @pytest.mark.parametrize("layer_name", ["ConvM1", "ConvM2", "ConvM3", "ConvM4", "ConvM5", "ConvM6"])
    def test_encoder_creation(self, layer_name: str):
        """Test creating encoders for all layer configs."""
        encoder = get_encoder(layer_name, in_channels=3)
        assert isinstance(encoder, nn.Module)

    @pytest.mark.parametrize("layer_name", ["ConvM1", "ConvM2", "ConvM3"])
    def test_encoder_forward_pass(self, layer_name: str):
        """Test encoder forward pass."""
        encoder = get_encoder(layer_name, in_channels=3)
        x = torch.randn(2, 3, 256, 256)

        output = encoder(x)

        assert isinstance(output, torch.Tensor)
        assert output.dim() == 4  # (N, C, H, W)
        assert output.shape[0] == 2  # Batch size preserved

    def test_encoder_grayscale(self):
        """Test encoder with grayscale input."""
        encoder = get_encoder("ConvM1", in_channels=1)
        x = torch.randn(2, 1, 256, 256)

        output = encoder(x)

        assert isinstance(output, torch.Tensor)


class TestDecoders:
    """Tests for decoder classes."""

    @pytest.mark.parametrize("layer_name", ["ConvM1", "ConvM2", "ConvM3", "ConvM4", "ConvM5", "ConvM6"])
    def test_decoder_creation(self, layer_name: str):
        """Test creating decoders for all layer configs."""
        decoder = get_decoder(layer_name, out_channels=3)
        assert isinstance(decoder, nn.Module)

    @pytest.mark.parametrize("layer_name", ["ConvM1", "ConvM2", "ConvM3"])
    def test_encoder_decoder_roundtrip(self, layer_name: str):
        """Test that encoder-decoder produces correct output shape."""
        encoder = get_encoder(layer_name, in_channels=3)
        decoder = get_decoder(layer_name, out_channels=3)

        x = torch.randn(2, 3, 256, 256)
        encoded = encoder(x)
        decoded = decoder(encoded)

        assert decoded.shape == x.shape


class TestBAE1:
    """Tests for BAE1 autoencoder."""

    def test_creation(self):
        """Test creating BAE1."""
        model = BAE1(layer_name="ConvM1", image_dim=(256, 256, 3))
        assert isinstance(model, nn.Module)

    def test_forward_pass(self):
        """Test BAE1 forward pass."""
        model = BAE1(layer_name="ConvM1", image_dim=(256, 256, 3))
        x = torch.randn(2, 3, 256, 256)

        output = model(x)

        assert output.shape == x.shape

    def test_grayscale(self):
        """Test BAE1 with grayscale images."""
        model = BAE1(layer_name="ConvM1", image_dim=(256, 256, 1))
        x = torch.randn(2, 1, 256, 256)

        output = model(x)

        assert output.shape == x.shape


class TestBAE2:
    """Tests for BAE2 autoencoder."""

    def test_creation(self):
        """Test creating BAE2."""
        model = BAE2(
            layer_name="ConvM1",
            image_dim=(256, 256, 3),
            latent_dim=32,
        )
        assert isinstance(model, nn.Module)

    def test_forward_pass(self):
        """Test BAE2 forward pass."""
        model = BAE2(
            layer_name="ConvM1",
            image_dim=(256, 256, 3),
            latent_dim=32,
        )
        x = torch.randn(2, 3, 256, 256)

        output = model(x)

        assert output.shape == x.shape


class TestVAE1:
    """Tests for VAE1 variational autoencoder."""

    def test_creation(self):
        """Test creating VAE1."""
        model = VAE1(
            layer_name="ConvM1",
            image_dim=(256, 256, 3),
            latent_dim=32,
        )
        assert isinstance(model, nn.Module)

    def test_forward_pass(self):
        """Test VAE1 forward pass."""
        model = VAE1(
            layer_name="ConvM1",
            image_dim=(256, 256, 3),
            latent_dim=32,
        )
        x = torch.randn(2, 3, 256, 256)

        output = model(x)

        # VAE returns tuple (reconstruction, mu, logvar)
        assert isinstance(output, tuple)
        assert len(output) == 3
        recon, mu, logvar = output
        assert recon.shape == x.shape

    def test_kl_loss_computed(self):
        """Test that KL loss is computed during forward pass."""
        model = VAE1(
            layer_name="ConvM1",
            image_dim=(256, 256, 3),
            latent_dim=32,
        )
        x = torch.randn(2, 3, 256, 256)

        _ = model(x)

        assert hasattr(model, "kl_loss")
        assert model.kl_loss >= 0


class TestVAE2:
    """Tests for VAE2 variational autoencoder."""

    def test_creation(self):
        """Test creating VAE2."""
        model = VAE2(
            layer_name="ConvM1",
            image_dim=(256, 256, 3),
            latent_dim=32,
            intermediate_dim=64,
        )
        assert isinstance(model, nn.Module)

    def test_forward_pass(self):
        """Test VAE2 forward pass."""
        model = VAE2(
            layer_name="ConvM1",
            image_dim=(256, 256, 3),
            latent_dim=32,
            intermediate_dim=64,
        )
        x = torch.randn(2, 3, 256, 256)

        output = model(x)

        # VAE returns tuple (reconstruction, mu, logvar)
        assert isinstance(output, tuple)
        assert len(output) == 3


class TestVQVAE1:
    """Tests for VQVAE1 vector-quantized autoencoder."""

    def test_creation(self):
        """Test creating VQVAE1."""
        model = VQVAE1(
            layer_name="ConvM1",
            image_dim=(256, 256, 3),
            num_embeddings=32,
        )
        assert isinstance(model, nn.Module)

    def test_forward_pass(self):
        """Test VQVAE1 forward pass."""
        model = VQVAE1(
            layer_name="ConvM1",
            image_dim=(256, 256, 3),
            num_embeddings=32,
        )
        x = torch.randn(2, 3, 256, 256)

        output = model(x)

        # VQ-VAE returns tuple (reconstruction, indices)
        assert isinstance(output, tuple)
        assert len(output) == 2
        recon, indices = output
        assert recon.shape == x.shape


class TestCreateAutoencoder:
    """Tests for create_autoencoder factory function."""

    def test_create_bae1(self):
        """Test creating BAE1 via factory."""
        model = create_autoencoder(
            model_name="BAE1",
            layer_name="ConvM1",
            image_dim=(256, 256, 3),
        )
        assert isinstance(model, BAE1)

    def test_create_bae2(self):
        """Test creating BAE2 via factory."""
        model = create_autoencoder(
            model_name="BAE2",
            layer_name="ConvM1",
            image_dim=(256, 256, 3),
            latent_dim=32,
        )
        assert isinstance(model, BAE2)

    def test_create_vae1(self):
        """Test creating VAE1 via factory."""
        model = create_autoencoder(
            model_name="VAE1",
            layer_name="ConvM1",
            image_dim=(256, 256, 3),
            latent_dim=32,
        )
        assert isinstance(model, VAE1)

    def test_create_vae2(self):
        """Test creating VAE2 via factory."""
        model = create_autoencoder(
            model_name="VAE2",
            layer_name="ConvM1",
            image_dim=(256, 256, 3),
            latent_dim=32,
            intermediate_dim=64,
        )
        assert isinstance(model, VAE2)

    def test_create_vqvae1(self):
        """Test creating VQVAE1 via factory."""
        model = create_autoencoder(
            model_name="VQVAE1",
            layer_name="ConvM1",
            image_dim=(256, 256, 3),
            num_embeddings=32,
        )
        assert isinstance(model, VQVAE1)

    def test_invalid_model_raises_error(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError, match="Unknown model"):
            create_autoencoder(
                model_name="InvalidModel",
                layer_name="ConvM1",
                image_dim=(256, 256, 3),
            )

    def test_invalid_layer_raises_error(self):
        """Test that invalid layer name raises error."""
        with pytest.raises(ValueError, match="Unknown layer"):
            create_autoencoder(
                model_name="BAE1",
                layer_name="InvalidLayer",
                image_dim=(256, 256, 3),
            )


class TestHardNet:
    """Tests for HardNet descriptor network."""

    def test_creation(self):
        """Test creating HardNet."""
        model = HardNet(pretrained=False)
        assert isinstance(model, nn.Module)

    def test_forward_pass(self):
        """Test HardNet forward pass."""
        model = HardNet(pretrained=False)
        # HardNet expects 32x32 grayscale patches
        x = torch.randn(2, 1, 32, 32)

        output = model(x)

        # Output should be 128-dimensional descriptor
        assert output.shape == (2, 128)

    def test_output_normalized(self):
        """Test that HardNet output is L2 normalized."""
        model = HardNet(pretrained=False)
        x = torch.randn(4, 1, 32, 32)

        output = model(x)

        # Check L2 norm is approximately 1
        norms = torch.norm(output, dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


class TestModelGradients:
    """Tests for model gradient computation."""

    @pytest.mark.parametrize("model_name", ["BAE1", "BAE2", "VAE1", "VAE2", "VQVAE1"])
    def test_gradients_flow(self, model_name: str):
        """Test that gradients flow through the model."""
        kwargs = {"layer_name": "ConvM1", "image_dim": (256, 256, 3)}
        if model_name in ["BAE2", "VAE1", "VAE2"]:
            kwargs["latent_dim"] = 32
        if model_name == "VAE2":
            kwargs["intermediate_dim"] = 64
        if model_name == "VQVAE1":
            kwargs["num_embeddings"] = 32

        model = create_autoencoder(model_name=model_name, **kwargs)
        model.train()

        x = torch.randn(2, 3, 256, 256, requires_grad=True)
        output = model(x)

        # Get reconstruction
        if isinstance(output, tuple):
            recon = output[0]
        else:
            recon = output

        # Compute loss and backward
        loss = torch.nn.functional.mse_loss(recon, x)
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape
