# -*- coding: utf-8 -*-
"""
PyTorch autoencoder architectures for anomaly detection.

This module provides five autoencoder architectures:
- BAE1: Basic autoencoder (direct encoder-decoder connection)
- BAE2: Basic autoencoder with fully-connected bottleneck
- VAE1: Variational autoencoder
- VAE2: Variational autoencoder with fully-connected bottleneck
- VQVAE1: Vector-quantized variational autoencoder
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LAYER_CONFIGS, get_decoder, get_encoder


class BaseAutoencoder(nn.Module):
    """Base class for all autoencoders."""

    def __init__(
        self,
        layer_name: str,
        image_dim: tuple[int, int, int],
        latent_dim: int = 32,
    ):
        super().__init__()
        self.layer_name = layer_name
        self.image_dim = image_dim  # (H, W, C) format
        self.latent_dim = latent_dim

        self.in_channels = image_dim[2]
        self.height = image_dim[0]
        self.width = image_dim[1]

        # Get layer config
        self.layer_config = LAYER_CONFIGS[layer_name]
        self.reduced_height = self.height // self.layer_config.reduction_factor
        self.reduced_width = self.width // self.layer_config.reduction_factor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder."""
        raise NotImplementedError


class BAE1(BaseAutoencoder):
    """
    Basic Autoencoder 1: Direct encoder-decoder connection.

    The encoder output is passed directly to the decoder without
    any intermediate fully-connected layers.
    """

    def __init__(
        self,
        layer_name: str,
        image_dim: tuple[int, int, int],
        latent_dim: int = 32,
        **kwargs: Any,
    ):
        super().__init__(layer_name, image_dim, latent_dim)

        self.encoder = get_encoder(layer_name, self.in_channels)
        self.decoder = get_decoder(
            layer_name, self.layer_config.bottleneck_channels, self.in_channels
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


class BAE2(BaseAutoencoder):
    """
    Basic Autoencoder 2: With fully-connected bottleneck.

    Adds dense layers between encoder and decoder for a more
    compressed latent representation.
    """

    def __init__(
        self,
        layer_name: str,
        image_dim: tuple[int, int, int],
        latent_dim: int = 32,
        intermediate_dim: int = 64,
        **kwargs: Any,
    ):
        super().__init__(layer_name, image_dim, latent_dim)

        self.encoder = get_encoder(layer_name, self.in_channels)

        # Calculate flattened size after encoder
        self.flat_size = (
            self.layer_config.bottleneck_channels
            * self.reduced_height
            * self.reduced_width
        )

        # Bottleneck layers
        self.fc_encode1 = nn.Linear(self.flat_size, intermediate_dim)
        self.fc_encode2 = nn.Linear(intermediate_dim, latent_dim)
        self.fc_decode1 = nn.Linear(latent_dim, intermediate_dim)
        self.fc_decode2 = nn.Linear(intermediate_dim, self.flat_size)

        self.decoder = get_decoder(
            layer_name, self.layer_config.bottleneck_channels, self.in_channels
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        h = F.relu(self.fc_encode1(h))
        return F.relu(self.fc_encode2(h))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc_decode1(z))
        h = F.relu(self.fc_decode2(h))
        h = h.view(
            h.size(0),
            self.layer_config.bottleneck_channels,
            self.reduced_height,
            self.reduced_width,
        )
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


class VAE1(BaseAutoencoder):
    """
    Variational Autoencoder 1: Basic VAE with reparameterization.

    Uses the reparameterization trick to sample from the latent distribution.
    """

    def __init__(
        self,
        layer_name: str,
        image_dim: tuple[int, int, int],
        latent_dim: int = 32,
        **kwargs: Any,
    ):
        super().__init__(layer_name, image_dim, latent_dim)

        self.encoder = get_encoder(layer_name, self.in_channels)

        # Calculate flattened size after encoder
        self.flat_size = (
            self.layer_config.bottleneck_channels
            * self.reduced_height
            * self.reduced_width
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)

        # Decoder input
        self.fc_decode = nn.Linear(latent_dim, self.flat_size)

        self.decoder = get_decoder(
            layer_name, self.layer_config.bottleneck_channels, self.in_channels
        )

        # Store KL loss for training
        self.kl_loss = 0.0

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + sigma * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc_decode(z))
        h = h.view(
            h.size(0),
            self.layer_config.bottleneck_channels,
            self.reduced_height,
            self.reduced_width,
        )
        return self.decoder(h)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        # Compute KL divergence
        self.kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return recon, mu, logvar


class VAE2(BaseAutoencoder):
    """
    Variational Autoencoder 2: VAE with additional FC layers in bottleneck.

    Similar to VAE1 but with extra dense layers before computing
    mu and logvar.
    """

    def __init__(
        self,
        layer_name: str,
        image_dim: tuple[int, int, int],
        latent_dim: int = 32,
        intermediate_dim: int = 64,
        **kwargs: Any,
    ):
        super().__init__(layer_name, image_dim, latent_dim)

        self.encoder = get_encoder(layer_name, self.in_channels)

        # Calculate flattened size after encoder
        self.flat_size = (
            self.layer_config.bottleneck_channels
            * self.reduced_height
            * self.reduced_width
        )

        # Pre-latent FC layer
        self.fc_pre = nn.Linear(self.flat_size, intermediate_dim)

        # Latent space parameters
        self.fc_mu = nn.Linear(intermediate_dim, latent_dim)
        self.fc_logvar = nn.Linear(intermediate_dim, latent_dim)

        # Decoder input
        self.fc_decode = nn.Linear(latent_dim, self.flat_size)

        self.decoder = get_decoder(
            layer_name, self.layer_config.bottleneck_channels, self.in_channels
        )

        # Store KL loss for training
        self.kl_loss = 0.0

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        h = F.relu(self.fc_pre(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + sigma * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc_decode(z))
        h = h.view(
            h.size(0),
            self.layer_config.bottleneck_channels,
            self.reduced_height,
            self.reduced_width,
        )
        return self.decoder(h)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        # Compute KL divergence
        self.kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return recon, mu, logvar


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for VQ-VAE.

    Implements the vector quantization operation with straight-through
    gradient estimation.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )

        # Store VQ loss for training
        self.vq_loss = 0.0

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # z: (B, C, H, W) -> (B, H, W, C)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)

        # Calculate distances to embeddings
        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # Get nearest embedding indices
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        # Quantize
        z_q = torch.matmul(encodings, self.embedding.weight)
        z_q = z_q.view(z.shape)

        # Compute loss
        e_latent_loss = F.mse_loss(z_q.detach(), z)
        q_latent_loss = F.mse_loss(z_q, z.detach())
        self.vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        # Back to (B, C, H, W)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, encoding_indices.view(z.shape[:-1])


class VQVAE1(BaseAutoencoder):
    """
    Vector-Quantized Variational Autoencoder.

    Uses discrete latent codes from a learned codebook instead of
    continuous latent variables.
    """

    def __init__(
        self,
        layer_name: str,
        image_dim: tuple[int, int, int],
        latent_dim: int = 32,
        num_embeddings: int = 32,
        data_variance: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(layer_name, image_dim, latent_dim)

        self.data_variance = data_variance

        self.encoder = get_encoder(layer_name, self.in_channels)

        # Pre-quantization conv to match embedding dim
        self.pre_quantize = nn.Conv2d(
            self.layer_config.bottleneck_channels, latent_dim, kernel_size=1
        )

        # Vector quantizer
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim)

        # Post-quantization conv
        self.post_quantize = nn.Conv2d(
            latent_dim, self.layer_config.bottleneck_channels, kernel_size=1
        )

        self.decoder = get_decoder(
            layer_name, self.layer_config.bottleneck_channels, self.in_channels
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.pre_quantize(h)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        h = self.post_quantize(z_q)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        z_q, indices = self.quantizer(z)
        recon = self.decode(z_q)
        return recon, indices
    
class DAE(BAE2):
    """
    Denoising Autoencoder.
    
    Inherits from BAE2 (with FC bottleneck) but adds Gaussian noise
    to the input during the training phase. This forces the model to 
    learn robust features rather than the identity function.
    """
    def __init__(
        self,
        layer_name: str,
        image_dim: tuple[int, int, int],
        latent_dim: int = 32,
        intermediate_dim: int = 64,
        noise_factor: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(layer_name, image_dim, latent_dim, intermediate_dim, **kwargs)
        self.noise_factor = noise_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply noise only if the model is in training mode
        if self.training:
            noise = torch.randn_like(x) * self.noise_factor
            x_noisy = x + noise
            # Assuming normalized inputs (e.g., [0, 1]). Adjust clamp values if needed.
            x_noisy = torch.clamp(x_noisy, 0.0, 1.0) 
        else:
            x_noisy = x
            
        z = self.encode(x_noisy)
        return self.decode(z)


class SAE(BAE2):
    """
    Sparse Autoencoder.
    
    Adds an L1 penalty (sparsity constraint) to the latent space activations.
    This encourages the model to use fewer active neurons to represent the input.
    """
    def __init__(
        self,
        layer_name: str,
        image_dim: tuple[int, int, int],
        latent_dim: int = 32,
        intermediate_dim: int = 64,
        **kwargs: Any,
    ):
        super().__init__(layer_name, image_dim, latent_dim, intermediate_dim, **kwargs)
        self.sparsity_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        
        # Calculate L1 penalty on the latent vector z
        self.sparsity_loss = torch.mean(torch.abs(z))
        
        return self.decode(z)


class AttnAE(BaseAutoencoder):
    """
    Attention Autoencoder.
    
    Applies Multi-Head Self-Attention in the bottleneck to capture global 
    context and relationships before final compression.
    """
    def __init__(
        self,
        layer_name: str,
        image_dim: tuple[int, int, int],
        latent_dim: int = 32,
        intermediate_dim: int = 64,
        num_heads: int = 4,
        **kwargs: Any,
    ):
        super().__init__(layer_name, image_dim, latent_dim)

        self.encoder = get_encoder(layer_name, self.in_channels)

        self.flat_size = (
            self.layer_config.bottleneck_channels
            * self.reduced_height
            * self.reduced_width
        )

        # Attention projection
        self.attn = nn.MultiheadAttention(embed_dim=self.flat_size, num_heads=num_heads, batch_first=True)
        
        # Latent compression
        self.fc_encode = nn.Linear(self.flat_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_size)

        self.decoder = get_decoder(
            layer_name, self.layer_config.bottleneck_channels, self.in_channels
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        
        # Attention requires (Batch, Seq_len, Embed_dim). 
        # Here we treat the entire flattened image features as a sequence of length 1.
        h_seq = h.unsqueeze(1) 
        attn_out, _ = self.attn(h_seq, h_seq, h_seq)
        attn_out = attn_out.squeeze(1)
        
        return F.relu(self.fc_encode(attn_out))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc_decode(z))
        h = h.view(
            h.size(0),
            self.layer_config.bottleneck_channels,
            self.reduced_height,
            self.reduced_width,
        )
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


# =============================================================================
# Factory function
# =============================================================================

AUTOENCODERS: dict[str, type[BaseAutoencoder]] = {
    "BAE1": BAE1,
    "BAE2": BAE2,
    "VAE1": VAE1,
    "VAE2": VAE2,
    "VQVAE1": VQVAE1,
    "DAE": DAE,       
    "SAE": SAE,       
    "AttnAE": AttnAE,
}


def create_autoencoder(
    model_name: str,
    layer_name: str,
    image_dim: tuple[int, int, int],
    latent_dim: int = 32,
    intermediate_dim: int = 64,
    num_embeddings: int = 32,
    data_variance: float = 0.5,
) -> BaseAutoencoder:
    """
    Create an autoencoder model.

    Args:
        model_name: One of BAE1, BAE2, VAE1, VAE2, VQVAE1
        layer_name: One of ConvM1-ConvM6
        image_dim: Image dimensions as (height, width, channels)
        latent_dim: Dimension of latent space
        intermediate_dim: Dimension of intermediate FC layers (BAE2, VAE2)
        num_embeddings: Number of codebook entries (VQVAE1)
        data_variance: Data variance for VQVAE loss (VQVAE1)

    Returns:
        Autoencoder model

    Raises:
        ValueError: If model_name or layer_name is invalid
    """
    if model_name not in AUTOENCODERS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(AUTOENCODERS.keys())}"
        )

    if layer_name not in LAYER_CONFIGS:
        raise ValueError(
            f"Unknown layer: {layer_name}. Available: {list(LAYER_CONFIGS.keys())}"
        )

    return AUTOENCODERS[model_name](
        layer_name=layer_name,
        image_dim=image_dim,
        latent_dim=latent_dim,
        intermediate_dim=intermediate_dim,
        num_embeddings=num_embeddings,
        data_variance=data_variance,
        noise_factor=noise_factor,
        num_heads=num_heads,
    )
