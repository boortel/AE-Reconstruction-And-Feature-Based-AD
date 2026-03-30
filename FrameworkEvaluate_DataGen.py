
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 11:50:00 2024

@author: Simon Bilik

This module is used to evaluate the AE Reconstruction and Feature Based AD framework.
It loads up weights of the selected combination and sort the unknown images of the given dataset to the YAML file.
Optionally, it copies and sort the dataset images to the OK and NOK folder structure.

Please select the desired model from the module ModelSaved.py as the model argument

"""

import os
import time
import yaml
import shutil
import argparse
import configparser

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
from typing import Dict, List, Type

from ModelSaved import ModelSaved
from ModelClassificationBase import ModelClassificationBase
from ModelClassificationEnc import ModelClassificationEnc
from ModelClassificationErrM import ModelClassificationErrM
from ModelClassificationSIFT import ModelClassificationSIFT
from ModelClassificationHardNet1 import ModelClassificationHardNet1
from ModelClassificationHardNet2 import ModelClassificationHardNet2
from ModelClassificationHardNet3 import ModelClassificationHardNet3
from ModelClassificationHardNet4 import ModelClassificationHardNet4

import Extract_logs
extractLogs = Extract_logs.main

import ProcessLogJSON
processLogs = ProcessLogJSON.main

class SimpleImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, path

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate models using PyTorch')
    parser.add_argument('--saveImgToFile', '-e', default=True, type=bool, help='Set True for model evaluation')
    return parser.parse_args()

def main():
    args = parse_args()
    saveImgToFile = args.saveImgToFile        
    iniBasePath = './init'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Usando dispositivo: {device} ---")

    cfg = configparser.ConfigParser()
    ext = ('.ini')

    for filename in os.listdir(iniBasePath):
        if not filename.endswith(ext):
            continue

        print(f"\n>> Procesando configuración: {filename}")
        cfg.read(os.path.join(iniBasePath, filename))

        experimentPath = cfg.get('General', 'modelBasePath', fallback='NaN')
        labelInfo = cfg.get('General', 'labelInfo', fallback='NaN')
        imageDim = (cfg.getint('General', 'imHeight', fallback=0), 
                    cfg.getint('General', 'imWidth', fallback=0),
                    cfg.getint('General', 'imChannel', fallback=3))

        predictionDataPath = cfg.get('Prediction', 'predictionDataPath', fallback='-')
        predictionResultPath = cfg.get('Prediction', 'predictionResultPath', fallback='-')
        predictionBatchSize = cfg.getint('Prediction', 'batchSize', fallback=32)

        modelName = cfg.get('Prediction', 'modelName', fallback='-')
        layerName = cfg.get('Prediction', 'layerName', fallback='-')
        featureExtractorName = cfg.get('Prediction', 'featureExtractorName', fallback='-')
        anomalyAlgorythmName = cfg.get('Prediction', 'anomalyAlgorythmName', fallback='Local Outlier Factor')
        
        basePath = os.path.join(experimentPath, f'{layerName}_{labelInfo}', modelName)
        aeWeightsPath = os.path.join(basePath, 'model.weights.pt') 

        modelObj = ModelSaved(
            model_sel=modelName,
            layer_sel=layerName,
            image_dim=imageDim,
            data_variance=0.5, 
            intermediate_dim=64,
            latent_dim=32,
            num_embeddings=32
        )

        model = modelObj.get_model()
        if os.path.exists(aeWeightsPath):
            model.load_state_dict(torch.load(aeWeightsPath, map_location=device))
        else:
            print(f"ADVERTENCIA: No se encontraron pesos en {aeWeightsPath}")
            continue
            
        model.to(device)
        model.eval()

        featureExtractorMap = {
            'ErrM': ModelClassificationErrM,
            'SIFT': ModelClassificationSIFT,
            'HardNet1': ModelClassificationHardNet1,
            'HardNet2': ModelClassificationHardNet2,
            'HardNet3': ModelClassificationHardNet3,
            'HardNet4': ModelClassificationHardNet4,
        }
        featureExtractorClass = featureExtractorMap[featureExtractorName]

        os.makedirs(predictionResultPath, exist_ok=True)
        okPath = os.path.join(predictionResultPath, 'OK')
        nokPath = os.path.join(predictionResultPath, 'NOK')
        os.makedirs(okPath, exist_ok=True)
        os.makedirs(nokPath, exist_ok=True)

        labelsPath = os.path.join(predictionResultPath, 'labels.yaml')
        labelsDict = {'OK': [], 'NOK': []}

        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        fileNames = [str(p) for p in Path(predictionDataPath).rglob('*') if p.suffix.lower() in valid_exts]
        fileNames.sort()

        transform = transforms.Compose([
            transforms.Resize((imageDim[0], imageDim[1]), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        dataset = SimpleImageDataset(fileNames, transform=transform)
        dataloader = DataLoader(dataset, batch_size=predictionBatchSize, shuffle=False)

        all_reconstructions = []
        all_originals = []
        startTime = time.time()

        with torch.no_grad():
            for batch_imgs, _ in dataloader:
                batch_imgs = batch_imgs.to(device)
                output = model(batch_imgs)
                
                recon = output[0] if isinstance(output, tuple) else output

                all_originals.append(batch_imgs.cpu().numpy().transpose(0, 2, 3, 1))
                all_reconstructions.append(recon.cpu().numpy().transpose(0, 2, 3, 1))

        orig_data = np.concatenate(all_originals, axis=0)
        dec_data = np.concatenate(all_reconstructions, axis=0)

        filterStrength = 0.5
        orig_data_filtered = np.array([
            (filterStrength * img + (1 - filterStrength) * cv.GaussianBlur(img, (25, 25), 0)) 
            for img in orig_data
        ])

        if orig_data_filtered.ndim == 3 and imageDim[2] == 1:
            orig_data_filtered = np.expand_dims(orig_data_filtered, axis=-1)

        prediction_data = {
            'Predict': {
                'Org': orig_data_filtered,
                'Dec': dec_data,
                'Lab': [None for _ in dec_data],
            }
        }
        
        feature_extractor = featureExtractorClass(
            os.path.join(basePath, 'modelData'),
            predictionResultPath, modelName, layerName, 'Classification test', imageDim,
            prediction_data,
            [anomalyAlgorythmName],
            False
        )
        
        labels = feature_extractor.predictedLabels
        
        for path, label in zip(fileNames, labels):
            target_folder = okPath if label else nokPath
            if saveImgToFile:
                shutil.copy(path, target_folder)
            
            key = 'OK' if label else 'NOK'
            labelsDict[key].append(os.path.abspath(path))

        print(f"--- {time.time() - startTime:.2f} segundos ---")

        with open(labelsPath, 'w') as labelsFile:
            yaml.safe_dump(labelsDict, labelsFile)

if __name__ == '__main__':
    main()