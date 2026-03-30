# -*- coding: utf-8 -*-
"""
PyTorch equivalent of VAE / VQ-VAE helper classes

@author: Simon Bilik
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class MeanMetric:
    def __init__(self, name):
        self.name = name
        self.reset()

    def update(self, value):
        self.total += float(value)
        self.count += 1

    def result(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def reset(self):
        self.total = 0.0
        self.count = 0


class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        batch, dim = z_mean.shape
        epsilon = torch.randn(batch, dim, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon



class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.total_loss_tracker = MeanMetric("total_loss")
        self.reconstruction_loss_tracker = MeanMetric("reconstruction_loss")
        self.kl_loss_tracker = MeanMetric("kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def forward(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var

    def train_step(self, x, y, optimizer):
        optimizer.zero_grad()

        reconstruction, z_mean, z_log_var = self.forward(x)

        reconstruction_loss = torch.mean(
            torch.sum(
                F.binary_cross_entropy(reconstruction, y, reduction='none'),
                dim=(1, 2)
            )
        )

        # KL loss
        kl_loss = -0.5 * (1 + z_log_var - z_mean**2 - torch.exp(z_log_var))
        kl_loss = torch.mean(torch.sum(kl_loss, dim=1))

        total_loss = reconstruction_loss + kl_loss

        total_loss.backward()
        optimizer.step()

        # Update metrics
        self.total_loss_tracker.update(total_loss.item())
        self.reconstruction_loss_tracker.update(reconstruction_loss.item())
        self.kl_loss_tracker.update(kl_loss.item())

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }



class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.embeddings = nn.Parameter(
            torch.rand(embedding_dim, num_embeddings)
        )

    def forward(self, x):
        input_shape = x.shape
        flattened = x.view(-1, self.embedding_dim)

        encoding_indices = self.get_code_indices(flattened)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        quantized = torch.matmul(encodings, self.embeddings.t())
        quantized = quantized.view(input_shape)


        commitment_loss = torch.mean((quantized.detach() - x) ** 2)
        codebook_loss = torch.mean((quantized - x.detach()) ** 2)
        vq_loss = self.beta * commitment_loss + codebook_loss

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        return quantized, vq_loss

    def get_code_indices(self, flattened_inputs):
        similarity = torch.matmul(flattened_inputs, self.embeddings)

        distances = (
            torch.sum(flattened_inputs ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings ** 2, dim=0)
            - 2 * similarity
        )

        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices

class VQVAETrainer(nn.Module):
    def __init__(self, vqvae, train_variance):
        super().__init__()
        self.vqvae = vqvae
        self.train_variance = train_variance

        self.total_loss_tracker = MeanMetric("total_loss")
        self.reconstruction_loss_tracker = MeanMetric("reconstruction_loss")
        self.vq_loss_tracker = MeanMetric("vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def forward(self, x):
        return self.vqvae(x)

    def train_step(self, x, y, optimizer):
        optimizer.zero_grad()

        reconstructions, vq_loss = self.vqvae(x)

        reconstruction_loss = torch.mean((y - reconstructions) ** 2) / self.train_variance
        total_loss = reconstruction_loss + vq_loss

        total_loss.backward()
        optimizer.step()

        # Update metrics
        self.total_loss_tracker.update(total_loss.item())
        self.reconstruction_loss_tracker.update(reconstruction_loss.item())
        self.vq_loss_tracker.update(vq_loss.item())

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }



def reset_metrics(model):
    for metric in model.metrics:
        metric.reset()