# -*- coding: utf-8 -*-
"""
Adapted to PyTorch
Visualización en tiempo real de reconstrucciones del Autoencoder.
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms

from ModelSaved import ModelSaved

allowedSuffixes = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

def parse_args():
    parser = argparse.ArgumentParser(description='Visualizar reconstrucciones en tiempo real (PyTorch)')
    
    parser.add_argument('--modelName', '-m', default='BAE1', type=str)
    parser.add_argument('--layerName', '-l', default='ConvM1', type=str)
    parser.add_argument('--imHeight', '--height', default=256, type=int)
    parser.add_argument('--imWidth', '--width', default=256, type=int)
    parser.add_argument('--imChannel', '--channel', default=3, type=int)
    parser.add_argument('--modelWeights', '-w', default='model.weights.pt', type=str)
    parser.add_argument('--images', '-i', default='./IndustryBiscuit_Folders/test/ok', type=str)
    parser.add_argument('--batchSize', '--batch', default=16, type=int)

    return parser.parse_args()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imageDim = (args.imHeight, args.imWidth, args.imChannel)
    
    modelObj = ModelSaved(
        model_sel=args.modelName, 
        layer_sel=args.layerName, 
        image_dim=imageDim, 
        data_variance=0.5, 
        intermediate_dim=64, 
        latent_dim=32, 
        num_embeddings=32
    )
    
    model = modelObj.get_model()
    model.load_state_dict(torch.load(args.modelWeights, map_location=device))
    model.to(device)
    model.eval()

    ax_cols = 2
    ax_rows = 4
    fig, axs = plt.subplots(ax_rows, ax_cols * 2, figsize=(12, 10))
    plt.ion()
    plt.show()

    transform = transforms.Compose([
        transforms.Resize((args.imHeight, args.imWidth)),
        transforms.ToTensor(), # Esto ya escala a [0, 1]
    ])

    imagesPath = Path(args.images)
    imageFileList = [f for f in imagesPath.iterdir() if f.suffix.lower() in allowedSuffixes]
    
    batch_tensors = []
    
    for imagePath in imageFileList:
        img = Image.open(imagePath).convert('RGB')
        img_t = transform(img)
        batch_tensors.append(img_t)

        if len(batch_tensors) >= args.batchSize:
            input_batch = torch.stack(batch_tensors).to(device)
            batch_tensors.clear()

            with torch.no_grad():
                output = model(input_batch)
                if isinstance(output, tuple):
                    output = output[0]

            input_np = input_batch.cpu().permute(0, 2, 3, 1).numpy()
            output_np = output.cpu().permute(0, 2, 3, 1).numpy()
            
            for c in range(ax_cols):
                for r in range(ax_rows):
                    idx = (c * ax_rows + r) % len(input_np)
                    
                    axs[r, 2 * c].imshow(input_np[idx])
                    axs[r, 2 * c].axis('off')
                    if r == 0: axs[r, 2 * c].set_title("Original")

                    axs[r, 2 * c + 1].imshow(np.clip(output_np[idx], 0, 1))
                    axs[r, 2 * c + 1].axis('off')
                    if r == 0: axs[r, 2 * c + 1].set_title("Recon")

            plt.pause(0.1)
            fig.canvas.draw()

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)