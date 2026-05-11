import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import optuna


DATASET_PATH = "IndustryBiscuit_Folders/"
IMG_SIZE = 256


def get_encoder(layer_sel, in_channels, f_base, k_size):
    pad = k_size // 2
    
    if layer_sel == 'ConvM1':
        return nn.Sequential(
            nn.Conv2d(in_channels, f_base, kernel_size=k_size, stride=2, padding=pad), nn.BatchNorm2d(f_base), nn.LeakyReLU(),
            nn.Conv2d(f_base, f_base*2, kernel_size=k_size, stride=2, padding=pad), nn.BatchNorm2d(f_base*2), nn.LeakyReLU(),
            nn.Conv2d(f_base*2, f_base*2, kernel_size=k_size, stride=2, padding=pad), nn.BatchNorm2d(f_base*2), nn.LeakyReLU(),
            nn.Conv2d(f_base*2, f_base*2, kernel_size=k_size, stride=2, padding=pad), nn.BatchNorm2d(f_base*2), nn.LeakyReLU(),
            nn.Conv2d(f_base*2, f_base*2, kernel_size=k_size, stride=2, padding=pad), nn.BatchNorm2d(f_base*2), nn.LeakyReLU()
        )
    elif layer_sel == 'ConvM2':
        return nn.Sequential(
            nn.Conv2d(in_channels, f_base, kernel_size=k_size, stride=2, padding=pad), nn.BatchNorm2d(f_base), nn.Sigmoid(),
            nn.Conv2d(f_base, f_base, kernel_size=k_size, stride=2, padding=pad), nn.BatchNorm2d(f_base), nn.Sigmoid(),
            nn.Conv2d(f_base, f_base, kernel_size=k_size, stride=1, padding='same'), nn.BatchNorm2d(f_base), nn.Sigmoid(),
            nn.Conv2d(f_base, f_base*2, kernel_size=k_size, stride=2, padding=pad), nn.BatchNorm2d(f_base*2), nn.Sigmoid(),
            nn.Conv2d(f_base*2, f_base*2, kernel_size=k_size, stride=1, padding='same'), nn.BatchNorm2d(f_base*2), nn.Sigmoid(),
            nn.Conv2d(f_base*2, f_base*4, kernel_size=k_size, stride=2, padding=pad), nn.BatchNorm2d(f_base*4), nn.Sigmoid(),
            nn.Conv2d(f_base*4, f_base*2, kernel_size=k_size, stride=1, padding='same'), nn.BatchNorm2d(f_base*2), nn.Sigmoid(),
            nn.Conv2d(f_base*2, f_base, kernel_size=k_size, stride=1, padding='same'), nn.BatchNorm2d(f_base), nn.Sigmoid(),
            nn.Conv2d(f_base, 1, kernel_size=k_size, stride=1, padding='same'), nn.BatchNorm2d(1)
        )
    elif layer_sel == 'ConvM3':
        return nn.Sequential(
            nn.Conv2d(in_channels, f_base, kernel_size=k_size, stride=2, padding=pad), nn.BatchNorm2d(f_base), nn.Sigmoid(),
            nn.Conv2d(f_base, f_base*2, kernel_size=k_size, stride=2, padding=pad), nn.BatchNorm2d(f_base*2), nn.Sigmoid()
        )
    elif layer_sel == 'ConvM4':
        return nn.Sequential(
            nn.Conv2d(in_channels, f_base, kernel_size=k_size, stride=1, padding='same'), nn.MaxPool2d(2, 2), nn.BatchNorm2d(f_base), nn.Sigmoid(),
            nn.Conv2d(f_base, max(1, f_base//2), kernel_size=k_size, stride=1, padding='same'), nn.MaxPool2d(2, 2), nn.BatchNorm2d(max(1, f_base//2)), nn.Sigmoid()
        )
    elif layer_sel == 'ConvM5' or layer_sel == 'ConvM6':
        return nn.Sequential(
            nn.Conv2d(in_channels, f_base, kernel_size=k_size, stride=1, padding='same'), nn.MaxPool2d(2, 2), nn.BatchNorm2d(f_base), nn.Sigmoid(),
            nn.Conv2d(f_base, max(1, f_base//2), kernel_size=k_size, stride=1, padding='same'), nn.MaxPool2d(2, 2), nn.BatchNorm2d(max(1, f_base//2)), nn.Sigmoid(),
            nn.Conv2d(max(1, f_base//2), max(1, f_base//4), kernel_size=k_size, stride=1, padding='same'), nn.MaxPool2d(2, 2), nn.BatchNorm2d(max(1, f_base//4)), nn.Sigmoid()
        )
    else:
        raise ValueError(f'Unknown convolutional net name: {layer_sel}')

def get_decoder(layer_sel, in_channels, out_channels, f_base, k_size):
    pad = k_size // 2
    
    if layer_sel == 'ConvM1':
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=k_size, stride=2, padding=pad, output_padding=1), nn.BatchNorm2d(in_channels), nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels, f_base*2, kernel_size=k_size, stride=2, padding=pad, output_padding=1), nn.BatchNorm2d(f_base*2), nn.LeakyReLU(),
            nn.ConvTranspose2d(f_base*2, f_base*2, kernel_size=k_size, stride=2, padding=pad, output_padding=1), nn.BatchNorm2d(f_base*2), nn.LeakyReLU(),
            nn.ConvTranspose2d(f_base*2, f_base, kernel_size=k_size, stride=2, padding=pad, output_padding=1), nn.BatchNorm2d(f_base), nn.LeakyReLU(),
            nn.ConvTranspose2d(f_base, out_channels, kernel_size=k_size, stride=2, padding=pad, output_padding=1), nn.Sigmoid()
        )
    elif layer_sel == 'ConvM2':
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=k_size, stride=1, padding='same'), nn.Sigmoid(),
            nn.Conv2d(in_channels, f_base*2, kernel_size=k_size, stride=1, padding='same'), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(f_base*2),
            nn.Conv2d(f_base*2, f_base*4, kernel_size=k_size, stride=2, padding=pad), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(f_base*4),
            nn.Conv2d(f_base*4, f_base*2, kernel_size=k_size, stride=1, padding='same'), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(f_base*2),
            nn.Conv2d(f_base*2, f_base*2, kernel_size=k_size, stride=2, padding=pad), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(f_base*2),
            nn.Conv2d(f_base*2, f_base, kernel_size=k_size, stride=1, padding='same'), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(f_base),
            nn.Conv2d(f_base, f_base, kernel_size=k_size, stride=2, padding=pad), nn.Sigmoid(),
            nn.Upsample(scale_factor=4, mode='nearest'), nn.BatchNorm2d(f_base),
            nn.Conv2d(f_base, f_base, kernel_size=k_size, stride=2, padding=pad), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(f_base),
            nn.Conv2d(f_base, out_channels, kernel_size=k_size, stride=1, padding='same'), nn.Sigmoid()
        )
    elif layer_sel == 'ConvM3':
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=k_size, stride=2, padding=pad, output_padding=1), nn.Sigmoid(),
            nn.ConvTranspose2d(in_channels, f_base, kernel_size=k_size, stride=2, padding=pad, output_padding=1), nn.Sigmoid(),
            nn.BatchNorm2d(f_base),
            nn.ConvTranspose2d(f_base, out_channels, kernel_size=k_size, stride=1, padding=pad), nn.Sigmoid() 
        )
    elif layer_sel == 'ConvM4':
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=k_size, stride=1, padding='same'), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, f_base, kernel_size=k_size, stride=1, padding='same'), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(f_base),
            nn.Conv2d(f_base, out_channels, kernel_size=k_size, stride=1, padding='same'), nn.Sigmoid()
        )
    elif layer_sel == 'ConvM5':
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=k_size, stride=1, padding='same'), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, max(1, f_base//2), kernel_size=k_size, stride=1, padding='same'), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(max(1, f_base//2)),
            nn.Conv2d(max(1, f_base//2), f_base, kernel_size=k_size, stride=1, padding='same'), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(f_base),
            nn.Conv2d(f_base, out_channels, kernel_size=k_size, stride=1, padding='same'), nn.Sigmoid()
        )
    elif layer_sel == 'ConvM6':
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=k_size, stride=1, padding='same'), nn.Sigmoid(),
            nn.Upsample(scale_factor=4, mode='nearest'), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, f_base, kernel_size=k_size, stride=1, padding='same'), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(f_base),
            nn.Conv2d(f_base, out_channels, kernel_size=k_size, stride=1, padding='same'), nn.Sigmoid()
        )
    else:
        raise ValueError(f'Unknown convolutional net name: {layer_sel}')


class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

class VectorQuantizer(nn.Module):
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

class ModelSavedDynamic():
    def __init__(self, modelSel, layerSel, imageDim, base_filters, kernel_size, latentDim=32, num_embeddings=32):
        self.modelName = modelSel
        self.layerSel = layerSel
        self.im_height, self.im_width, self.im_channel = imageDim
        self.latentDim = latentDim
        self.num_embeddings = num_embeddings

        self.base_encoder = get_encoder(self.layerSel, self.im_channel, base_filters, kernel_size)

        dummy_input = torch.zeros(1, self.im_channel, self.im_height, self.im_width)
        with torch.no_grad():
            dummy_output = self.base_encoder(dummy_input)

        _, self.filterCount, self.redEncHeight, self.redEncWidth = dummy_output.shape
        self.flatDim = self.filterCount * self.redEncHeight * self.redEncWidth

        self.base_decoder = get_decoder(self.layerSel, self.filterCount, self.im_channel, base_filters, kernel_size)

        if self.modelName == 'BAE1':
            self.model = BAE1_Net(self.base_encoder, self.base_decoder)
        elif self.modelName == 'BAE2':
            self.model = BAE2_Net(self.base_encoder, self.base_decoder, self.flatDim, self.latentDim, self.filterCount, self.redEncHeight, self.redEncWidth)
        elif self.modelName == 'VAE1':
            self.model = VAE1_Net(self.base_encoder, self.base_decoder, self.flatDim, self.latentDim, self.filterCount, self.redEncHeight, self.redEncWidth)
        elif self.modelName == 'VAE2':
            self.model = VAE2_Net(self.base_encoder, self.base_decoder, self.flatDim, self.latentDim, self.filterCount, self.redEncHeight, self.redEncWidth)
        elif self.modelName == 'VQVAE1':
            self.model = VQVAE_Net(self.base_encoder, self.base_decoder, self.latentDim, self.num_embeddings, self.filterCount)
        else:
            raise ValueError('Unknown model name: ' + self.modelName)
            
    def get_model(self):
        return self.model


def get_data(batch_size):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def vae_loss_function(recon_x, x, z_mean, z_log_var):
    MSE = nn.MSELoss()(recon_x, x)
    KLD = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return MSE + (KLD / x.size(0))

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_sel = trial.suggest_categorical("modelSel", ['BAE1', 'BAE2', 'VAE1', 'VAE2', 'VQVAE1'])
    layer_sel = trial.suggest_categorical("layerSel", ['ConvM1', 'ConvM2', 'ConvM3', 'ConvM4', 'ConvM5', 'ConvM6'])
    
    base_filters = trial.suggest_categorical("base_filters", [8, 16, 32, 64])
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
    latent_dim = trial.suggest_categorical("latentDim", [32, 64, 128, 256])
    
    num_emb = 32
    if model_sel == 'VQVAE1':
        num_emb = trial.suggest_categorical("num_embeddings", [16, 32, 64, 128])

    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    try:
        model_wrapper = ModelSavedDynamic(
            modelSel=model_sel, 
            layerSel=layer_sel, 
            imageDim=(IMG_SIZE, IMG_SIZE, 3), 
            base_filters=base_filters, 
            kernel_size=kernel_size,
            latentDim=latent_dim,
            num_embeddings=num_emb
        )
        model = model_wrapper.get_model().to(device)
    except Exception as e:
        raise optuna.exceptions.TrialPruned()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader, val_loader = get_data(batch_size)
    
    epochs = 5 
    for epoch in range(epochs):
        model.train()
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            
            if model_sel in ['VAE1', 'VAE2']:
                reconstructions, z_mean, z_log_var = model(images)
                loss = vae_loss_function(reconstructions, images, z_mean, z_log_var)
            elif model_sel == 'VQVAE1':
                reconstructions, vq_loss = model(images)
                loss = nn.MSELoss()(reconstructions, images) + vq_loss
            else: 
                reconstructions = model(images)
                loss = nn.MSELoss()(reconstructions, images)
                
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                if model_sel in ['VAE1', 'VAE2']:
                    reconstructions, z_mean, z_log_var = model(images)
                    v_loss = vae_loss_function(reconstructions, images, z_mean, z_log_var)
                elif model_sel == 'VQVAE1':
                    reconstructions, vq_loss = model(images)
                    v_loss = nn.MSELoss()(reconstructions, images) + vq_loss
                else:
                    reconstructions = model(images)
                    v_loss = nn.MSELoss()(reconstructions, images)
                    
                val_loss += v_loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20) 
    
    print("\n" + "="*50)
    print(f"Best reconstruction loss: {study.best_value:.5f}")
    print("Best configuration:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*50)