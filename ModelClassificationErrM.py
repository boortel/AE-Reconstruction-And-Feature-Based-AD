# -*- coding: utf-8 -*-
"""
Created on Wed Oct  12 17:23:05 2022

@author: Simon Bilik

This class is used for the classification of the evaluated model

"""

import logging
import imagehash
import traceback

import numpy as np

from PIL import Image
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as SSIM

from ModelClassificationBase import ModelClassificationBase

class ModelClassificationErrM(ModelClassificationBase):

    ## Constructor
    def __init__(
            self,
            modelDataPath,
            experimentPath,
            modelSel,
            layerSel,
            labelInfo,
            imageDim,
            modelData,
            anomaly_algorithm_selection = ["Robust covariance", "One-Class SVM", "Isolation Forest", "Local Outlier Factor"],
            visualize = True
        ):

        # Call the parent
        ModelClassificationBase.__init__(
            self,
            modelDataPath,
            experimentPath,
            modelSel,
            layerSel,
            labelInfo,
            imageDim,
            modelData,
            'ErrMetrics',
            anomaly_algorithm_selection,
            visualize
        )
        
        # Get data, metrics and classify the data
        try:
            if self.modelData:
                self.procDataFromDict()
            else:
                self.procDataFromFile()
                
            self.dataClassify()
        except:
            logging.error('An error occured during classification using ' + self.featExtName + ' feature extraction method...')
            traceback.print_exc()
        pass


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

        # Compute the metrics for all images in dataset
        for i in range(len(orgData)):

            # TODO: predelat L2 tak, aby fungovala i pro cernobile
            if(orgData.shape[3] == 3):
                valuesL2.append(np.sum(np.square(np.subtract(orgData[i], decData[i])), axis = None))
                valuesSSIM.append(SSIM(orgData[i], decData[i], data_range = 1, channel_axis = 2))
            else:
                valuesL2.append(np.sum(np.square(np.subtract(np.squeeze(orgData[i]), np.squeeze(decData[i]))), axis=None))
                valuesSSIM.append(SSIM(np.squeeze(orgData[i]), np.squeeze(decData[i]), data_range = 1, channel_axis = None))

            valuesMSE.append(MSE(orgData[i], decData[i]))

            # Convert images to gray
            if(orgData.shape[3] == 3):
                imgOrg = Image.fromarray(orgData[i], mode = "RGB")
                imgDec = Image.fromarray(decData[i], mode = "RGB")
            else:
                imgOrg = Image.fromarray(np.squeeze(orgData[i]), mode = "L")
                imgDec = Image.fromarray(np.squeeze(decData[i]), mode = "L")
            
            valuesAvgH.append(imagehash.average_hash(imgOrg) - imagehash.average_hash(imgDec))

        # Convert the lists into the np arrays
        valuesL2 = np.array(valuesL2)
        valuesMSE = np.array(valuesMSE)
        valuesSSIM = np.array(valuesSSIM)
        valuesAvgH = np.array(valuesAvgH)

        # Get metrics np array
        metrics = self.normalize2DData(np.column_stack((valuesL2, valuesMSE, valuesSSIM, valuesAvgH)))

        return metrics, labels
