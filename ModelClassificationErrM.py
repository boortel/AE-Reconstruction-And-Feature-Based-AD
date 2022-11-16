# -*- coding: utf-8 -*-
"""
Created on Wed Oct  12 17:23:05 2022

@author: Simon Bilik

This class is used for the classification of the evaluated model

"""

import os
import logging
import imagehash

import numpy as np

from PIL import Image
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as SSIM

from ModelClassificationBase import ModelClassificationBase

class ModelClassificationErrM(ModelClassificationBase):

    ## Constructor
    def __init__(self, modelDataPath, modelSel, layerSel, labelInfo, imageDim):

        # Call the parent
        ModelClassificationBase.__init__(self, modelDataPath, modelSel, layerSel, labelInfo, imageDim)

        # Set the feature extractor name
        self.featExtName = 'ErrMetrics'

        # Print feature extractor identifier
        print('Feature extraction method: ' + self.featExtName)
        print('-----------------------------------------------')
        
        logging.info('Feature extraction method: ' + self.featExtName)
        logging.info('-----------------------------------------------')
        

    ## Get data from file
    def procDataFromFile(self, actStr):
        
        self.actStr = actStr

        # Build the path
        procDatasetPath = os.path.join(self.modelDataPath, 'Eval_' + actStr + '.npz')

        # Load the NPZ file
        procDataset = np.load(procDatasetPath)

        if self.actStr == 'Train':

            # Store the data and get the metrics
            labels = procDataset['labels']
            okIdx = np.where(labels == 0)
            labels[okIdx] = -1

            self.processedDataTr = {'Org': procDataset['orgData'], 'Dec': procDataset['decData'], 'Lab': labels}
            self.metricsTr, _ = self.computeMetrics(self.processedDataTr)

        elif self.actStr == 'Test':

            labels = procDataset['labels']
            okIdx = np.where(labels == 0)
            labels[okIdx] = -1

            # Store the data and get the metrics
            self.processedDataTs = {'Org': procDataset['orgData'], 'Dec': procDataset['decData'], 'Lab': labels}
            self.metricsTs, self.labelsTs = self.computeMetrics(self.processedDataTs)


    ## Compute the classification metrics
    def computeMetrics(self, processedData):

        # Get the data
        orgData = processedData.get('Org')
        decData = processedData.get('Dec')
        labels = processedData.get('Lab')

        # Define the metrics arrays
        valuesL2 = []
        valuesMSE = []
        valuesSSIM = []
        valuesAvgH = []

        # Define the color mode
        if self.imageDim[2] == 3:
            cMode = "RGB"
            chAx = 2
        else:
            cMode = "L"
            chAx = None

        # Compute the metrics for all images in dataset
        for i in range(len(orgData)):

            # TODO: predelat L2 tak, aby fungovala i pro cernobile
            valuesL2.append(np.sum(np.square(np.subtract(orgData[i], decData[i])), axis=None))
            valuesMSE.append(MSE(orgData[i], decData[i]))
            valuesSSIM.append(SSIM(orgData[i], decData[i], data_range=255, channel_axis = chAx))

            imgOrg = Image.fromarray(orgData[i], mode = cMode)
            imgDec = Image.fromarray(decData[i], mode = cMode)
            valuesAvgH.append(imagehash.average_hash(imgOrg) - imagehash.average_hash(imgDec))

        # Convert the lists into the np arrays
        valuesL2 = np.array(valuesL2)
        valuesMSE = np.array(valuesMSE)
        valuesSSIM = np.array(valuesSSIM)
        valuesAvgH = np.array(valuesAvgH)

        # Get metrics np array
        metrics = self.normalize2DData(np.column_stack((valuesL2, valuesMSE, valuesSSIM, valuesAvgH)))

        # Visualise the data
        self.fsVisualise(metrics, labels)

        return metrics, labels
