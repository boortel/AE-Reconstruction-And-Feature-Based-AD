# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:23:05 2021

@author: Simon Bilik

This module provides the autoencoder training and evaluation based on the folder dataset structure.

Please select the desired model from the module ModelSaved.py as the model argument

"""

import os
import logging
import argparse
import warnings
import traceback
import configparser

from ModelTrainAndEval import ModelTrainAndEval
from ModelClassificationErrM import ModelClassificationErrM
from ModelClassificationSIFT import ModelClassificationSIFT


## Parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(description = 'Train and evaluate models defined in the ini files of the init directory')
    
    parser.add_argument('--modelTrain', default = True, type = bool, help = 'Set True for model training')
    parser.add_argument('--modelEval', default = True, type = bool, help = 'Set True for model evaluation')

    args = parser.parse_args()

    return args


## Main function
def main():

    # Supress future warnings
    warnings.simplefilter(action = 'ignore', category = FutureWarning)

    # Ini base path
    iniBasePath = './init'

    args = parse_args()

    # Get the arg values
    modelTrain = args.modelTrain
    modelEval = args.modelEval

    # Initialize the config parser and the extension filter
    cfg = configparser.ConfigParser()
    ext = ('.ini')
    
    # Initialize the logging
    logging.basicConfig(filename='./ProgramLog.txt', level=logging.INFO, format='(%(asctime)s %(levelname)-7s) %(message)s')

    # Loop through all ini files in the init directory
    for filename in os.listdir(iniBasePath):

        # Get only the .ini files
        if filename.endswith(ext):

            # Load the ini file and get the arguments
            cfg.read(os.path.join('init', filename))

            # General
            labelInfo = cfg.get('General', 'labelInfo', fallback = 'NaN')
            modelBasePath = cfg.get('General', 'modelBasePath', fallback = 'NaN')
            imageDim = (cfg.getint('General', 'imHeight', fallback = '0'), 
                        cfg.getint('General', 'imWidth', fallback = '0'),
                        cfg.getint('General', 'imChannel', fallback = '0'))

            # Training
            layerSel = cfg.get('Training', 'layerSel', fallback = 'NaN')
            modelSel = cfg.get('Training', 'modelSel', fallback = 'NaN')
            datasetPath = cfg.get('Training', 'datasetPath', fallback = 'NaN')
            batchSize = cfg.getint('Training', 'batchSize', fallback = '0')
            numEpoch = cfg.getint('Training', 'numEpoch', fallback = '0')
            
            # Parse the layers and models names
            layerSel = (layerSel.replace(" ", "")).split(",")
            modelSel = (modelSel.replace(" ", "")).split(",")
            
            # Separate the ini files in log
            logging.info('------------------------------------------------------------------------------------------------')
            logging.info('                                                                                                ')
            logging.info(labelInfo)
            logging.info('                                                                                                ')
            logging.info('------------------------------------------------------------------------------------------------')
            
            # Loop through the selected convolutional layers
            for layer in layerSel:

                # Loop through the selected models
                for model in modelSel:
                    
                    # Set the model path
                    modelPath = os.path.join(modelBasePath, layer + '_' + labelInfo, model)

                    # Create the model data directory
                    modelDataPath = os.path.join(modelPath, 'modelData')

                    if not os.path.exists(modelDataPath):
                        os.makedirs(modelDataPath)
                    
                    if modelTrain or modelEval:

                        # Train and evaluate the model
                        try:
                            ModelTrainAndEval(modelPath, datasetPath, model, layer, labelInfo, imageDim, batchSize, numEpoch, modelTrain, modelEval)
                        except:
                            logging.error('An error occured during the training or evaluation of ' + modelSel + ' model...')
                            traceback.print_exc()
                        else:
                            logging.info('Model ' + model + ' was trained succesfuly...')

                    if True:
                        
                        # Classify the model results 
                        try:
                            mClass = ModelClassificationErrM(modelDataPath, model, layer, labelInfo, imageDim)

                            mClass.procDataFromFile('Train')
                            mClass.procDataFromFile('Test')

                            mClass.dataClassify()
                        except:
                            pass
                        
                        try:
                            mClass = ModelClassificationSIFT(modelDataPath, model, layer, labelInfo, imageDim, 'Points')

                            mClass.procDataFromFile('Train')
                            mClass.procDataFromFile('Test')

                            mClass.dataClassify()
                        except:
                            pass

if __name__ == '__main__':
    main()
