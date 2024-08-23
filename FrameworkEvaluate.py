
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

import argparse
import configparser

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.utils import gen_batches

from typing import Dict, List, Type
from keras_preprocessing.image import img_to_array, load_img


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


## Parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(description = 'Train and evaluate models defined in the ini files of the init directory')
    
    parser.add_argument('--saveImgToFile', '-e', default = True, type = bool, help = 'Set True for model evaluation')

    args = parser.parse_args()

    return args


## Main function
def main():

    args = parse_args()

    # Get the arg values
    saveImgToFile = args.saveImgToFile        

    # Ini base path
    iniBasePath = './init'

    # Initialize the config parser and the extension filter
    cfg = configparser.ConfigParser()
    ext = ('.ini')

    # Loop through all ini files in the init directory
    for filename in os.listdir(iniBasePath):

        # Get only the .ini files
        if not filename.endswith(ext):
            continue

        ## Load the ini file and get the arguments
        cfg.read(os.path.join('init', filename))

        # General configuration
        experimentPath = cfg.get('General', 'modelBasePath', fallback = 'NaN')
        labelInfo = cfg.get('General', 'labelInfo', fallback = 'NaN')
        imageDim = (cfg.getint('General', 'imHeight', fallback = '0'), 
                    cfg.getint('General', 'imWidth', fallback = '0'),
                    cfg.getint('General', 'imChannel', fallback = '0'))

        # Prediction configuration
        predictionDataPath = cfg.get('Prediction', 'predictionDataPath', fallback = '.')
        predictionResultPath = cfg.get('Prediction', 'predictionResultPath', fallback = '.')
        predictionBatchSize = cfg.getint('Prediction', 'batchSize', fallback = '0')

        ## Find the optimal combination
        # extractLogs()
        # processLogs()

        # Select the optimal combination
        modelName = 'BAE2'
        layerName = 'ConvM4'
        featureExtractorName = 'HardNet1'
        anomalyAlgorythmName = 'Robust covariance'
        basePath = os.path.join(experimentPath, f'{layerName}_{labelInfo}', modelName)
        aeWeightsPath = os.path.join(basePath, 'model.weights.h5')

        ## Construct the model and load the weights
        modelObj = ModelSaved(
            modelName,
            layerName,
            imageDim,
            dataVariance = 0.5, 
            intermediateDim = 64,
            latentDim = 32,
            num_embeddings = 32
        )

        model = modelObj.model
        model.load_weights(aeWeightsPath)

        ## Load the selected feature extractor
        featureExtractor:Type[ModelClassificationBase] = {
            # 'Enc' : ModelClassificationEnc,
            'ErrM' : ModelClassificationErrM,
            'SIFT' : ModelClassificationSIFT,
            'HardNet1' : ModelClassificationHardNet1,
            'HardNet2' : ModelClassificationHardNet2,
            'HardNet3' : ModelClassificationHardNet3,
            # 'HardNet4' : ModelClassificationHardNet4,
        }[featureExtractorName]

        # Get the list of all image files
        images = []
        imagePaths = []
        allowedSuffixes = ['.jpg', '.jpeg','.png', '.bmp', '.gif']

        fileList = map(Path, os.listdir(predictionDataPath))
        imageFileList = [file for file in fileList if file.suffix.lower() in allowedSuffixes]

        ## Prepare the directory to store the predictions
        if not os.path.exists(predictionResultPath):
            os.mkdir(predictionResultPath)

        okPath, nokPath = [os.path.join(predictionResultPath, subfolder) for subfolder in ['OK', 'NOK']]

        if not os.path.exists(okPath):
            os.mkdir(okPath)
        if not os.path.exists(nokPath):
            os.mkdir(nokPath)

        labelsPath = os.path.join(predictionResultPath, 'labels.yaml')
        labelsDict = {'OK': [], 'NOK':[]}

        ## Set constants
        batchCount = 0
        imageSize = (imageDim[0], imageDim[1])
        
        # Loop through the image files
        for imageFile in imageFileList:
            imagePath = Path(predictionDataPath) / imageFile

            image = load_img(imagePath, target_size=imageSize)
            imageArray = img_to_array(image)

            imagePaths.append(imagePath)
            images.append(imageArray)

            # load images in batches (if the amount of loaded images is big enough for a batch,
            # or if the number ofremaining images is less than the size of a batch and all of them are loaded
            if len(images) >= predictionBatchSize or len(images) == len(imageFileList) - (predictionBatchSize * batchCount):

                startTime = time.time()

                batchCount += 1
                input = np.array(images) / 255
                output = model.predict(input)

                # Build prediction data
                prediction_data = {
                    'Predict':
                    {
                        'Org': input,
                        'Dec': output,
                        'Lab': [None for _ in input],
                    }
                }

                labels = featureExtractor(os.path.join(basePath, 'modelData'), '', '', '', '', imageDim, prediction_data, [anomalyAlgorythmName], False).predictedLabels

                # Get the OK-NOK labels
                sorted = {imagePath: label for imagePath, label in zip(imagePaths, labels)}
                OK = [imagePath for imagePath, label in sorted.items() if label]
                NOK = [imagePath for imagePath, label in sorted.items() if not label]

                # Save the sorted images if set
                if saveImgToFile:
                    for subDir, images in zip((okPath, nokPath), (OK, NOK)):
                        for image in images:
                            os.popen(f'cp {image} {subDir}')

                # Store the labels to the dictionary
                for label, results in zip(('OK', 'NOK'), (OK, NOK)):
                    paths = [f'{os.path.abspath(r)}' for r in results]

                    if paths != []:
                        labelsDict[label].append(paths)

                imagePaths.clear()
                images.clear()

                print("--- %s seconds ---" % (time.time() - startTime))

        with open(labelsPath, 'w') as labelsFile:
            yaml.safe_dump(labelsDict, labelsFile)

        return
        
if __name__ == '__main__':
    main()
    