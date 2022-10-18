# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:23:05 2021

@author: Šimon Bilík

This module provides the autoencoder training and evaluation based on the folder dataset structure.

Please select the desired model from the module ModelSaved.py as the model argument

"""

import os
import logging
import argparse
import configparser

import ModelTrain
import ModelEvaluation


def parse_args():
    parser = argparse.ArgumentParser(description = 'Train and evaluate models defined in the ini files of the init directory')

    parser.add_argument('--iniBasePath', default = './init', type = str, help = 'Path to ini files')
    parser.add_argument('--modelTrain', default = 1  , type = int, help = 'Set to 1 if you want to train models')
    parser.add_argument('--modelEval', default = 1  , type = int, help = 'Set to 1 if you want to evaluate models')

    args = parser.parse_args()

    return args


def main():
    """main function"""
    args = parse_args()

    # Get the arg values
    iniBasePath = args.iniBasePath
    modelTrain = args.modelTrain
    modelEval = args.modelEval

    # Initialize the logging
    logging.basicConfig(filename = '/ProgramLog.txt', level=logging.DEBUG, format='(%(asctime)s %(threadName)-10s %(levelname)-7s) %(message)s')

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
            datasetPathTr = cfg.get('Training', 'datasetPathTr', fallback = 'NaN')
            batchSizeTr = cfg.getint('Training', 'batchSizeTr', fallback = '0')
            numEpochTr = cfg.getint('Training', 'numEpochTr', fallback = '0')

            # Evaluation
            datasetPathEv = cfg.get('Evaluation', 'datasetPathEv', fallback = 'NaN')
            labelsPath = cfg.get('Evaluation', 'labelsPath', fallback = 'NaN')
            batchSizeEv = cfg.getint('Evaluation', 'batchSizeEv', fallback = '0')
            numEpochEv = cfg.getint('Evaluation', 'numEpochEv', fallback = '0')

            if modelTrain == 1:

                # Train the model
                try:
                    ModelTrain(modelBasePath, datasetPathTr, modelSel, labelInfo, imageDim, batchSizeTr, numEpochTr)
                except:
                    logging.error(': An error occured during the training of ' + modelSel + ' model...')
                else:
                    logging.info(': Model ' + modelSel + ' was trained succesfuly...')
            
            if modelEval == 1:

                # Evaluate the model
                try:
                    ModelEvaluation(modelBasePath, datasetPathEv, labelsPath, modelSel, labelInfo, imageDim, batchSizeEv, numEpochEv)
                except:
                    logging.error(': An error occured during the evaluation of ' + modelSel + ' model...')
                else:
                    logging.info(': Model ' + modelSel + ' was evaluated succesfuly...')

if __name__ == '__main__':
    main()
