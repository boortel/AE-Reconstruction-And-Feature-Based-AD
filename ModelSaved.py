# -*- coding: utf-8 -*-
"""
Created on Wed Jan 6 2021
@author: Simon Bilik (Adapted to PyTorch)

This class returns compiled autoencoder model later used in the ModelTrainAndEval.py script. Feel free to define any new models if necessary.

"""

import logging
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelLayers import get_encoder, get_decoder, LAYER_CONFIGS



class Sampling(nn.Module):
    """Uses (z_mean, z_log_var) to sample z."""
    def forward(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

class VectorQuantizer(nn.Module):
    """Vector Quantization Layer for VQ-VAE"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embeddings.weight).view(inputs.permute(0, 2, 3, 1).shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs.permute(0, 2, 3, 1))
        q_latent_loss = F.mse_loss(quantized, inputs.permute(0, 2, 3, 1).detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs.permute(0, 2, 3, 1) + (quantized - inputs.permute(0, 2, 3, 1)).detach()
        return quantized.permute(0, 3, 1, 2).contiguous(), loss


# --- Model Architectures ---

class BAE1_Net(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))

class BAE2_Net(nn.Module):
    def __init__(self, encoder, decoder, flat_dim, latent_dim, filterCount, redEncHeight, redEncWidth):
        super().__init__()
        self.encoder = encoder
        self.fc_enc = nn.Linear(flat_dim, 10)
        self.fc_latent = nn.Linear(10, latent_dim)
        self.fc_dec1 = nn.Linear(latent_dim, 10)
        self.fc_dec2 = nn.Linear(10, flat_dim)
        self.decoder = decoder
        
        self.filterCount = filterCount
        self.redEncHeight = redEncHeight
        self.redEncWidth = redEncWidth

    def forward(self, x):
        x = torch.flatten(self.encoder(x), start_dim=1)
        x = F.relu(self.fc_enc(x))
        encoded = F.relu(self.fc_latent(x))
        x = F.relu(self.fc_dec1(encoded))
        x = F.relu(self.fc_dec2(x))
        x = x.view(-1, self.filterCount, self.redEncHeight, self.redEncWidth)
        return self.decoder(x)

class VAE1_Net(nn.Module):
    def __init__(self, encoder, decoder, flat_dim, latent_dim, filterCount, redEncHeight, redEncWidth):
        super().__init__()
        self.encoder = encoder
        self.fc_mean = nn.Linear(flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(flat_dim, latent_dim)
        self.sampling = Sampling()
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.decoder = decoder
        
        self.filterCount = filterCount
        self.redEncHeight = redEncHeight
        self.redEncWidth = redEncWidth

    def forward(self, x):
        enc_out = self.encoder(x)
        flatten = torch.flatten(enc_out, start_dim=1)
        
        z_mean = self.fc_mean(flatten)
        z_log_var = self.fc_log_var(flatten)
        z = self.sampling(z_mean, z_log_var)
        
        x_dec = F.relu(self.fc_dec(z))
        x_dec = x_dec.view(-1, self.filterCount, self.redEncHeight, self.redEncWidth)
        reconstructions = self.decoder(x_dec)
        
        return reconstructions, z_mean, z_log_var

class VAE2_Net(nn.Module):
    def __init__(self, encoder, decoder, flat_dim, latent_dim, filterCount, redEncHeight, redEncWidth):
        super().__init__()
        self.encoder = encoder
        self.fc_dense = nn.Linear(flat_dim, 16)
        self.fc_mean = nn.Linear(16, latent_dim)
        self.fc_log_var = nn.Linear(16, latent_dim)
        self.sampling = Sampling()
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.decoder = decoder
        
        self.filterCount = filterCount
        self.redEncHeight = redEncHeight
        self.redEncWidth = redEncWidth

    def forward(self, x):
        enc_out = self.encoder(x)
        x_flat = torch.flatten(enc_out, start_dim=1)
        x_dense = F.relu(self.fc_dense(x_flat))
        
        z_mean = self.fc_mean(x_dense)
        z_log_var = self.fc_log_var(x_dense)
        z = self.sampling(z_mean, z_log_var)
        
        x_dec = F.relu(self.fc_dec(z))
        x_dec = x_dec.view(-1, self.filterCount, self.redEncHeight, self.redEncWidth)
        reconstructions = self.decoder(x_dec)
        
        return reconstructions, z_mean, z_log_var

class VQVAE_Net(nn.Module):
    def __init__(self, encoder, decoder, latent_dim, num_embeddings, filterCount):
        super().__init__()
        self.encoder = encoder
        self.pre_vq_conv = nn.Conv2d(filterCount, latent_dim, kernel_size=1)
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim)
        self.post_vq_conv = nn.Conv2d(latent_dim, filterCount, kernel_size=1)
        self.decoder = decoder

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        quantized_inputs = self.pre_vq_conv(encoder_outputs)
        quantized_latents, vq_loss = self.vq_layer(quantized_inputs)
        decoder_inputs = self.post_vq_conv(quantized_latents)
        reconstructions = self.decoder(decoder_inputs)
        return reconstructions, vq_loss


class DAE_Net(nn.Module):
    """Denoising Autoencoder Network"""
    def __init__(self, encoder, decoder, flat_dim, latent_dim, filterCount, redEncHeight, redEncWidth, noise_factor=0.1):
        super().__init__()
        self.encoder = encoder
        self.noise_factor = noise_factor
        self.fc_enc = nn.Linear(flat_dim, 10)
        self.fc_latent = nn.Linear(10, latent_dim)
        self.fc_dec1 = nn.Linear(latent_dim, 10)
        self.fc_dec2 = nn.Linear(10, flat_dim)
        self.decoder = decoder
        
        self.filterCount = filterCount
        self.redEncHeight = redEncHeight
        self.redEncWidth = redEncWidth

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.noise_factor
            x = torch.clamp(x + noise, 0.0, 1.0)
            
        x = torch.flatten(self.encoder(x), start_dim=1)
        x = F.relu(self.fc_enc(x))
        encoded = F.relu(self.fc_latent(x))
        x = F.relu(self.fc_dec1(encoded))
        x = F.relu(self.fc_dec2(x))
        x = x.view(-1, self.filterCount, self.redEncHeight, self.redEncWidth)
        return self.decoder(x)

class SAE_Net(nn.Module):
    """Sparse Autoencoder Network"""
    def __init__(self, encoder, decoder, flat_dim, latent_dim, filterCount, redEncHeight, redEncWidth):
        super().__init__()
        self.encoder = encoder
        self.fc_enc = nn.Linear(flat_dim, 10)
        self.fc_latent = nn.Linear(10, latent_dim)
        self.fc_dec1 = nn.Linear(latent_dim, 10)
        self.fc_dec2 = nn.Linear(10, flat_dim)
        self.decoder = decoder
        
        self.filterCount = filterCount
        self.redEncHeight = redEncHeight
        self.redEncWidth = redEncWidth
        
        self.sparsity_loss = 0.0

    def forward(self, x):
        x = torch.flatten(self.encoder(x), start_dim=1)
        x = F.relu(self.fc_enc(x))
        encoded = F.relu(self.fc_latent(x))
        
        self.sparsity_loss = torch.mean(torch.abs(encoded))
        
        x = F.relu(self.fc_dec1(encoded))
        x = F.relu(self.fc_dec2(x))
        x = x.view(-1, self.filterCount, self.redEncHeight, self.redEncWidth)
        return self.decoder(x)

class AttnAE_Net(nn.Module):
    """Attention Autoencoder Network"""
    def __init__(self, encoder, decoder, flat_dim, latent_dim, filterCount, redEncHeight, redEncWidth, num_heads=4):
        super().__init__()
        self.encoder = encoder
        
        self.attn_embed_dim = 256 
        self.proj_in = nn.Linear(flat_dim, self.attn_embed_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim=self.attn_embed_dim, num_heads=num_heads, batch_first=True)
        
        self.proj_out = nn.Linear(self.attn_embed_dim, flat_dim)
        
        self.fc_enc = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.decoder = decoder
        
        self.filterCount = filterCount
        self.redEncHeight = redEncHeight
        self.redEncWidth = redEncWidth

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        
        x_proj = self.proj_in(x)
        
        x_seq = x_proj.unsqueeze(1)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        attn_out = attn_out.squeeze(1)
        
        attn_restored = self.proj_out(attn_out)
        
        encoded = F.relu(self.fc_enc(attn_restored))
        x = F.relu(self.fc_dec(encoded))
        x = x.view(-1, self.filterCount, self.redEncHeight, self.redEncWidth)
        return self.decoder(x)


class ModelSaved():

    def __init__(self, modelSel, layerSel, imageDim, dataVariance = 0.5, intermediateDim = 64, latentDim = 32, num_embeddings = 32, noiseFactor=0.1, numHeads=4):

        self.modelName = modelSel
        self.layerSel = layerSel
        self.im_height, self.im_width, self.im_channel = imageDim
        self.intermediateDim = intermediateDim
        self.num_embeddings = num_embeddings
        self.dataVariance = dataVariance
        self.latentDim = latentDim
        self.noiseFactor = noiseFactor
        self.numHeads = numHeads

        self.base_encoder = get_encoder(self.layerSel, in_channels=self.im_channel)

        dummy_input = torch.zeros(1, self.im_channel, self.im_height, self.im_width)
        
        with torch.no_grad():
            dummy_output = self.base_encoder(dummy_input)

        _, self.filterCount, self.redEncHeight, self.redEncWidth = dummy_output.shape
        self.flatDim = self.filterCount * self.redEncHeight * self.redEncWidth

        self.base_decoder = get_decoder(self.layerSel, in_channels=self.filterCount, out_channels=self.im_channel)

        try:
            if self.modelName == 'VAE1':
                self.model = self.build_vae1_model()

            elif self.modelName == 'VAE2':
                self.model = self.build_vae2_model()
                
            elif self.modelName == 'VQVAE1':
                self.model = self.build_vqvaeC1_model()

            elif self.modelName == 'BAE1':
                self.model = self.build_bae1_model()

            elif self.modelName == 'BAE2':
                self.model = self.build_bae2_model()
                
            elif self.modelName == 'DAE':
                self.model = self.build_dae_model()
                
            elif self.modelName == 'SAE':
                self.model = self.build_sae_model()
                
            elif self.modelName == 'AttnAE':
                self.model = self.build_attnae_model()

            else:
                logging.error('Unknown model name: ' + self.modelName)
                raise ValueError('Unknown model name: ' + self.modelName)
                return

            logging.info(f"Model {self.modelName} initialized successfully. Flat dimension calculated as: {self.flatDim}")

        except:
            logging.error('Initialization of the selected model: ' + self.modelName + ' failed....')
            traceback.print_exc()

    ## Variational autoencoder 1
    def build_vae1_model(self):
        self.typeAE = 'VAE1'
        
        return VAE1_Net(self.base_encoder, self.base_decoder, self.flatDim, 
                        self.latentDim, self.filterCount, self.redEncHeight, self.redEncWidth)

    ## Variational autoencoder 2 (with fully connected layers before z-parameters computation)
    def build_vae2_model(self):
        self.typeAE = 'VAE2'
        
        return VAE2_Net(self.base_encoder, self.base_decoder, self.flatDim, 
                        self.latentDim, self.filterCount, self.redEncHeight, self.redEncWidth)

    ## Convolutional VQ-VAE
    def build_vqvaeC1_model(self):
        self.typeAE = 'VQVAE1'
        
        return VQVAE_Net(self.base_encoder, self.base_decoder, 
                         self.latentDim, self.num_embeddings, self.filterCount)

    ## Basic autoencoder model
    def build_bae1_model(self):
        self.typeAE = 'BAE1'
        
        return BAE1_Net(self.base_encoder, self.base_decoder)

    ## Basic autoencoder model with fully connected layers before encoding
    def build_bae2_model(self):
        self.typeAE = 'BAE2'
        
        return BAE2_Net(self.base_encoder, self.base_decoder, self.flatDim, 
                        self.latentDim, self.filterCount, self.redEncHeight, self.redEncWidth)
                        

    ## Denoising Autoencoder
    def build_dae_model(self):
        self.typeAE = 'DAE'
        return DAE_Net(self.base_encoder, self.base_decoder, self.flatDim, 
                       self.latentDim, self.filterCount, self.redEncHeight, self.redEncWidth, self.noiseFactor)

    ## Sparse Autoencoder
    def build_sae_model(self):
        self.typeAE = 'SAE'
        return SAE_Net(self.base_encoder, self.base_decoder, self.flatDim, 
                       self.latentDim, self.filterCount, self.redEncHeight, self.redEncWidth)

    ## Attention Autoencoder
    def build_attnae_model(self):
        self.typeAE = 'AttnAE'
        return AttnAE_Net(self.base_encoder, self.base_decoder, self.flatDim, 
                          self.latentDim, self.filterCount, self.redEncHeight, self.redEncWidth, self.numHeads)


    def get_model(self) -> nn.Module:
        return self.model