
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

import cv2 as cv
import numpy as np
import tensorflow as tf

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


## Parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(description = 'Train and evaluate models defined in the ini files of the init directory')
    
    parser.add_argument('--saveImgToFile', '-e', default = True, type = bool, help = 'Set True for model evaluation')

    args = parser.parse_args()

    return args

## Normalize dataset
def changeInputsTest(images):
    
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    x_norm = tf.image.resize(normalization_layer(images),[images.shape[1], images.shape[2]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    return x_norm


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
        #modelName = 'BAE2'
        #layerName = 'ConvM4'
        modelName = 'BAE2'
        layerName = 'ConvM3'
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

        ## Prepare the directory to store the predictions
        if not os.path.exists(predictionResultPath):
            os.mkdir(predictionResultPath)

        okPath, nokPath = [os.path.join(predictionResultPath, subfolder) for subfolder in ['OK', 'NOK']]

        if not os.path.exists(okPath):
            os.mkdir(okPath)
        if not os.path.exists(nokPath):
            os.mkdir(nokPath)

        # Prepare the YAML file with predictions
        labelsPath = os.path.join(predictionResultPath, 'labels.yaml')
        labelsDict = {'OK': [], 'NOK':[]}

        # Set data generator
        ds = tf.keras.utils.image_dataset_from_directory(
            (predictionDataPath),
            image_size = (imageDim[0], imageDim[1]),
            color_mode = 'rgb' if imageDim[2] == 3 else 'grayscale',
            batch_size = predictionBatchSize,
            label_mode = None,
            shuffle = False)
        
        # Get the filenames
        fileNames = ds.file_paths
        
        startTime = time.time()

        # Normalize data and get the reconstruction
        ds = ds.map(changeInputsTest)
        output = model.predict(ds)

        # Normalize the decoded data
        for i in range(output.shape[0]):
            output[i] = np.atleast_3d(cv.normalize(output[i], None, 0, 1, cv.NORM_MINMAX, cv.CV_32F))

        # Build prediction data
        prediction_data = {
            'Predict':
            {
                'Org': np.concatenate([img for img in ds], axis=0),
                'Dec': output,
                'Lab': [None for _ in output],
            }
        }

        # Get the labels and sort the OK / NOK files
        labels = featureExtractor(os.path.join(basePath, 'modelData'), '', '', '', '', imageDim, prediction_data, [anomalyAlgorythmName], False).predictedLabels
        sorted = {imagePath: label for imagePath, label in zip(fileNames, labels)}

        OK = [imagePath for imagePath, label in sorted.items() if label]
        NOK = [imagePath for imagePath, label in sorted.items() if not label]

        print("--- %s seconds ---" % (time.time() - startTime))

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

        with open(labelsPath, 'w') as labelsFile:
            yaml.safe_dump(labelsDict, labelsFile)

        return
        
if __name__ == '__main__':
    main()
    