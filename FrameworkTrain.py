# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:23:05 2021

@author: Simon Bilik

This module is used to train the AE Reconstruction and Feature Based AD framework.
It saves the weights of the selected combinations and stores the results to the text log.

Please aware - text log has to be manually erased before new experiments, otherwise it got appended.

"""

import os
import logging
import argparse

import warnings
import traceback
import matplotlib
import configparser
import torch

from typing import Dict, List, Type

from ModelTrainAndEval import ModelTrainAndEval
from ModelDataGenerators import ModelDataGenerators
from ModelClassificationEnc import ModelClassificationEnc
from ModelClassificationErrM import ModelClassificationErrM
from ModelClassificationSIFT import ModelClassificationSIFT
from ModelClassificationHardNet1 import ModelClassificationHardNet1
from ModelClassificationHardNet2 import ModelClassificationHardNet2
from ModelClassificationHardNet3 import ModelClassificationHardNet3
from ModelClassificationHardNet4 import ModelClassificationHardNet4
from MetricsToJSON import extract_and_save

import Extract_logs
extractLogs = Extract_logs.main

import ProcessLogJSON
processLogs = ProcessLogJSON.main



## Parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(description = 'Train and evaluate models defined in the ini files of the init directory')
    
    parser.add_argument('--modelEval', '-e', default = True, type = bool, help = 'Set True for model evaluation')
    parser.add_argument('--logClear', '-l', default = False, type = bool, help = 'Set True to delete old log before operation')

    args = parser.parse_args()

    return args


def main():

    warnings.simplefilter(action = 'ignore', category = FutureWarning)

    iniBasePath = './init'

    args = parse_args()

    modelEval = args.modelEval
    logClear = args.logClear

    cfg = configparser.ConfigParser()
    ext = ('.ini')
    
    if logClear:
        if os.path.exists('./ProgramLog.txt'):
            os.remove('./ProgramLog.txt')

        if os.path.exists('./log'):
            for file in os.listdir('./log'):
                os.remove(os.path.join('./log', file))
            

    logging.basicConfig(filename='./ProgramLog.txt', level=logging.INFO, format='(%(asctime)s %(levelname)-7s) %(message)s')

    for filename in os.listdir(iniBasePath):

        if not filename.endswith(ext):
            continue

        cfg.read(os.path.join('init', filename))

        experimentPath = cfg.get('General', 'modelBasePath', fallback = 'NaN')
        labelInfo = cfg.get('General', 'labelInfo', fallback = 'NaN')
        npzSave = cfg.getboolean('General', 'npzSave', fallback = False)
        imageDim = (cfg.getint('General', 'imHeight', fallback = 0), 
                    cfg.getint('General', 'imWidth', fallback = 0),
                    cfg.getint('General', 'imChannel', fallback = 0))
        imIndxList = cfg.get('General', 'imIndxList', fallback = 'NaN')

        layerSel = cfg.get('Training', 'layerSel', fallback = 'NaN')
        modelSel = cfg.get('Training', 'modelSel', fallback = 'NaN')
        datasetPath = cfg.get('Training', 'datasetPath', fallback = 'NaN')
        batchSize = cfg.getint('Training', 'batchSize', fallback = 0)
        numEpoch = cfg.getint('Training', 'numEpoch', fallback = 0)
        
        imIndxList = (imIndxList.replace(" ", "")).split(",")
        imIndxList = list(map(int, imIndxList))
        layerSel = (layerSel.replace(" ", "")).split(",")
        modelSel = (modelSel.replace(" ", "")).split(",")
        
        # Separate the ini files in log
        logging.info('-' * 96)
        logging.info('                                                                                                ')
        logging.info(labelInfo)
        logging.info('                                                                                                ')
        logging.info('-' * 96)
        
        # Create experiment directory
        if not os.path.exists(experimentPath):
            os.makedirs(experimentPath)
        
        # Create the experiment related data generator object
        dataGenerator = ModelDataGenerators(experimentPath, datasetPath, labelInfo, imageDim, batchSize, npzSave)     
        
        # Loop through the selected convolutional layers
        for layer in layerSel:

            # Loop through the selected models
            for model in modelSel:
                
                # Set the model path
                modelPath = os.path.join(experimentPath, layer + '_' + labelInfo, model)

                # Create the model data directory
                modelDataPath = os.path.join(modelPath, 'modelData')
                
                # Create variable to store model data
                modelData = {}

                if not os.path.exists(modelDataPath):
                    os.makedirs(modelDataPath)

                # Train and evaluate the model
                try:
                    modelObj = ModelTrainAndEval(modelPath, model, layer, dataGenerator, labelInfo, imageDim, imIndxList, numEpoch, modelEval, npzSave)

                    modelData = modelObj.dataGenerator.processedData

                except Exception as e:
                    logging.error(f'An error occured during the training or evaluation of {model} model...') 
                    traceback.print_exc()
                else:
                    logging.info(f'Model {model} was trained successfully...')
                
                # Extract the features and classify the model results 
                extractors = [
                    #ModelClassificationEnc,
                    ModelClassificationErrM,
                    #ModelClassificationSIFT,
                    ModelClassificationHardNet1,
                    ModelClassificationHardNet2,
                    ModelClassificationHardNet3,
                    #ModelClassificationHardNet4,
                ]

                for extractor in extractors:
                    try:
                        extractor(modelDataPath, experimentPath, model, layer, labelInfo, imageDim, modelData)
                    except Exception as e:
                        logging.error(f'An error occured in extractor {extractor.__name__} for model {model}...')
                        traceback.print_exc()
                
                # Close the opened figures to spare memory
                matplotlib.pyplot.close('all')
                extract_and_save()
                

    return
        
if __name__ == '__main__':
    main()