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
    def __init__(self, procDatasetPathTrain, procDatasetPathTest):

        # Set paths
        self.procDatasetPathTrain = procDatasetPathTrain
        self.procDatasetPathTest = procDatasetPathTest

        # Set actions
        self.action = ('Train', 'Test')


    ## Load test and evaluated data
    def loadEvaluatedData(self):

        # Load the NPZ files
        procDatasetTrain = np.load(self.procDatasetPathTrain)
        procDatasetTest = np.load(self.procDatasetPathTest)

        # Load the train data
        self.orgDataTrn = procDatasetTrain['orgData']
        self.encDataTrn = procDatasetTrain['encData']
        self.decDataTrn = procDatasetTrain['decData']

        self.labelsTr = procDatasetTrain['labels']

        # Load the test data
        self.orgDataTst = procDatasetTest['orgData']
        self.encDataTst = procDatasetTest['encData']
        self.decDataTst = procDatasetTest['decData']

        self.labelsTst = procDatasetTest['labels']

    ## Compute the metrics
    def computeMetrics(self, orgData, decData):

        # Define the metrics arrays
        valuesMSE = []
        valuesSSIM = []
        valuesAvgH = []

        # Compute the metrics for all images in dataset
        for i in range(len(orgData)):

            valuesMSE.append(MSE(orgData[i], decData[i]))
            valuesSSIM.append(SSIM(orgData[i], decData[i], data_range=255))

            imgOrg = Image.fromarray(orgData[i])
            imgDec = Image.fromarray(decData[i])
            valuesAvgH.append(imagehash.average_hash(imgOrg) - imagehash.average_hash(imgDec))

        # Convert the lists into the np arrays
        valuesMSE = np.array(valuesMSE)
        valuesSSIM = np.array(valuesSSIM)
        valuesAvgH = np.array(valuesAvgH)

        return (valuesMSE, valuesSSIM, valuesAvgH)