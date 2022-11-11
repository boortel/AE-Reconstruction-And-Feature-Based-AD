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

    parser.add_argument('--iniBasePath', default = './init', type = str, help = 'Path to ini files')
    parser.add_argument('--modelTrain', default = True, type = bool, help = 'Set True for model training')
    parser.add_argument('--modelEval', default = True, type = bool, help = 'Set True for model evaluation')

    args = parser.parse_args()

    return args


## Main function
def main():

    # Supress future warnings
    warnings.simplefilter(action = 'ignore', category = FutureWarning)

    args = parse_args()

    # Get the arg values
    iniBasePath = args.iniBasePath
    modelTrain = args.modelTrain
    modelEval = args.modelEval

    # Initialize the logging
    logging.basicConfig(level=logging.INFO, format='(%(asctime)s %(threadName)-10s %(levelname)-7s) %(message)s')

    # Initialize the config parser and the extension filter
    cfg = configparser.ConfigParser()
    ext = ('.ini')

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
            modelSel = cfg.get('Training', 'modelSel', fallback = 'NaN')
            datasetPath = cfg.get('Training', 'datasetPathTr', fallback = 'NaN')
            batchSize = cfg.getint('Training', 'batchSizeTr', fallback = '0')
            numEpoch = cfg.getint('Training', 'numEpochTr', fallback = '0')

            if modelTrain or modelEval:

                # Train the model
                try:
                    ModelTrainAndEval(modelBasePath, datasetPath, modelSel, labelInfo, imageDim, batchSize, numEpoch, modelTrain, modelEval)
                except:
                    logging.error(': An error occured during the training or evaluation of ' + modelSel + ' model...')
                    traceback.print_exc()
                else:
                    logging.info(': Model ' + modelSel + ' was trained succesfuly...')
                    

            mClass = ModelClassificationErrM(modelBasePath, modelSel, labelInfo, imageDim)

            mClass.procDataFromFile('Train')
            mClass.procDataFromFile('Test')

            mClass.dataClassify()

            mClass = ModelClassificationSIFT(modelBasePath, modelSel, labelInfo, imageDim, 'Points')

            mClass.procDataFromFile('Train')
            mClass.procDataFromFile('Test')

            mClass.dataClassify()

if __name__ == '__main__':
    main()
