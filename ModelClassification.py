# -*- coding: utf-8 -*-
"""
Created on Wed Oct  12 17:23:05 2022

@author: Simon Bilik

This class is used for the classification of the evaluated model

"""

import os
import sys
import imagehash

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as SSIM

class ModelClassification():

    ## Set the constants and paths
    def __init__(self, modelName):

        self.modelName = modelName
    

    ## Get data from dictionary
    def procDataFromDict(self, processedData, actStr):

        if actStr == 'Train':
        
            # Store data and get the metrics
            self.processedDataTr = processedData
            self.metricsTr = self.computeMetrics(self.processedDataTr)

        elif actStr == 'Test':

            # Store data and get the metrics
            self.processedDataTs = processedData
            self.metricsTs = self.computeMetrics(self.processedDataTs)


    ## Get data from file
    def procDataFromFile(self, procDatasetPath, actStr):

        # Load the NPZ file
        procDataset = np.load(procDatasetPath)

        if actStr == 'Train':

            # Store the data and get the metrics
            self.processedDataTr = {'Org': procDataset['orgData'], 'Enc': procDataset['encData'], 'Dec': procDataset['decData'], 'Lab': procDataset['labels']}
            self.metricsTr = self.computeMetrics(self.processedDataTr)

        elif actStr == 'Test':

            # Store the data and get the metrics
            self.processedDataTs = {'Org': procDataset['orgData'], 'Enc': procDataset['encData'], 'Dec': procDataset['decData'], 'Lab': procDataset['labels']}
            self.metricsTs = self.computeMetrics(self.processedDataTs)


    ## Compute the metrics
    def computeMetrics(self, processedData):

        # Get the data
        orgData = processedData.get('Org')
        decData = processedData.get('Dec')

        # Define the metrics arrays
        valuesL2 = []
        valuesMSE = []
        valuesSSIM = []
        valuesAvgH = []

        # Compute the metrics for all images in dataset
        for i in range(len(orgData)):

            # TODO: pridat imDim
            valuesL2.append(np.sum(np.square(np.subtract(orgData[i], decData[i])), axis=None))
            valuesMSE.append(MSE(orgData[i], decData[i]))
            valuesSSIM.append(SSIM(orgData[i], decData[i], data_range=255, channel_axis = 2))

            imgOrg = Image.fromarray(orgData[i], mode="RGB")
            imgDec = Image.fromarray(decData[i], mode="RGB")
            valuesAvgH.append(imagehash.average_hash(imgOrg) - imagehash.average_hash(imgDec))

        # Convert the lists into the np arrays
        valuesL2 = np.array(valuesL2)
        valuesMSE = np.array(valuesMSE)
        valuesSSIM = np.array(valuesSSIM)
        valuesAvgH = np.array(valuesAvgH)

        # Get metrics np array
        metrics = np.array([valuesL2, valuesMSE, valuesSSIM, valuesAvgH])

        return metrics