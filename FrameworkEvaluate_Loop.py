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
import re
import time
import yaml
import shutil
import random
import argparse
import configparser

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
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

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate models defined in the ini files of the init directory')
    parser.add_argument('--saveImgToFile', '-e', default=True, type=bool, help='Set True for model evaluation')
    args = parser.parse_args()
    return args

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def main():
    args = parse_args()
    saveImgToFile = args.saveImgToFile        
    iniBasePath = './init'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = configparser.ConfigParser()
    ext = ('.ini')

    for filename in os.listdir(iniBasePath):
        if not filename.endswith(ext):
            continue

        cfg.read(os.path.join(iniBasePath, filename))

        experimentPath = cfg.get('General', 'modelBasePath', fallback='NaN')
        labelInfo = cfg.get('General', 'labelInfo', fallback='NaN')
        imageDim = (cfg.getint('General', 'imHeight', fallback=0), 
                    cfg.getint('General', 'imWidth', fallback=0),
                    cfg.getint('General', 'imChannel', fallback=3))

        predictionDataPath = cfg.get('Prediction', 'predictionDataPath', fallback='.')
        predictionResultPath = cfg.get('Prediction', 'predictionResultPath', fallback='.')
        predictionBatchSize = cfg.getint('Prediction', 'batchSize', fallback=32)

        modelName = 'BAE1'
        layerName = 'ConvM1'
        featureExtractorName = 'HardNet2'
        anomalyAlgorythmName = 'Local Outlier Factor'
        
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

        images = []
        imagePaths = []
        allowedSuffixes = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

        fileList = sorted(Path(predictionDataPath).glob('*.*'))
        imageFileList = [file for file in fileList if file.suffix.lower() in allowedSuffixes]

        if not os.path.exists(predictionResultPath):
            os.mkdir(predictionResultPath)

        okPath, nokPath = [os.path.join(predictionResultPath, subfolder) for subfolder in ['OK', 'NOK']]
        for p in [okPath, nokPath]:
            if not os.path.exists(p): os.mkdir(p)

        labelsPath = os.path.join(predictionResultPath, 'labels.yaml')
        labelsDict = {'OK': [], 'NOK': []}

        batchCount = 0
        imageSize = (imageDim[0], imageDim[1])
        
        transform = transforms.Compose([
            transforms.Resize(imageSize),
            transforms.ToTensor(),
        ])

        for i, imageFile in enumerate(imageFileList):
            imagePath = Path(predictionDataPath) / imageFile
            
            img_pil = Image.open(imagePath).convert('RGB')
            img_tensor = transform(img_pil)

            imagePaths.append(imagePath)
            images.append(img_tensor)

            is_last_image = (i == len(imageFileList) - 1)
            if len(images) >= predictionBatchSize or is_last_image:
                startTime = time.time()
                batchCount += 1
                
                input_batch = torch.stack(images).to(device)
                
                with torch.no_grad():
                    output_batch = model(input_batch)
                    if isinstance(output_batch, tuple):
                        output_batch = output_batch[0]

                input_np = input_batch.cpu().numpy().transpose(0, 2, 3, 1)
                output_np = output_batch.cpu().numpy().transpose(0, 2, 3, 1)

                prediction_data = {
                    'Predict': {
                        'Org': input_np,
                        'Dec': output_np,
                        'Lab': [None for _ in input_np],
                    }
                }

                extractor_instance = featureExtractorClass(
                    os.path.join(basePath, 'modelData'), '', '', '', '', 
                    imageDim, prediction_data, [anomalyAlgorythmName], False
                )
                labels = extractor_instance.predictedLabels

                sortedLabels = {path: label for path, label in zip(imagePaths, labels)}
                OK = [path for path, label in sortedLabels.items() if label]
                NOK = [path for path, label in sortedLabels.items() if not label]

                if saveImgToFile:
                    for subDir, imgs_to_copy in zip((okPath, nokPath), (OK, NOK)):
                        for img_p in imgs_to_copy:
                            shutil.copy(img_p, subDir)

                for label_key, results in zip(('OK', 'NOK'), (OK, NOK)):
                    abs_paths = [os.path.abspath(r) for r in results]
                    if abs_paths:
                        labelsDict[label_key].append(abs_paths)

                imagePaths.clear()
                images.clear()
                print("--- %s seconds ---" % (time.time() - startTime))

        with open(labelsPath, 'w') as labelsFile:
            yaml.safe_dump(labelsDict, labelsFile)

if __name__ == '__main__':
    main()