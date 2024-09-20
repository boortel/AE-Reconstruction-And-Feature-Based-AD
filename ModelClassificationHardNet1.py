# -*- coding: utf-8 -*-
"""
Created on Wed Oct  12 17:23:05 2022

@author: Simon Bilik

This class is used for the classification of the evaluated model

"""

import logging
import traceback

import cv2 as cv
import numpy as np

from HardNet import HardNet
from sklearn.utils import gen_batches

from ModelClassificationBase import ModelClassificationBase

class ModelClassificationHardNet1(ModelClassificationBase):

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
        
        # Disable tf2 behavior for tf1
        #tf1.disable_v2_behavior()

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
            'HardNet1',
            anomaly_algorithm_selection,
            visualize
        )
        
        # Create HardNet model object
        self.hardNet = HardNet()
        
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
    def computeMetrics(self, processedData, eps = 1e-6):

        # Get the data
        diffData = np.subtract(processedData.get('Org'), processedData.get('Dec'))
        labels = processedData.get('Lab')
        
        # Initialize the conversion and metric lists
        gsData = []
        
        # Convert the data to grayscale and to required shape (batch, height, width, channel)
        if(diffData.shape[3] == 3):
            for img in diffData:
                gsData.append(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (32, 32), interpolation = cv.INTER_AREA))
        else:
            for img in diffData:
                gsData.append(cv.resize(np.squeeze(img), (32, 32), interpolation = cv.INTER_AREA))
            
        gsData = np.array(gsData)
        gsData = np.expand_dims(gsData, axis=3)
        
        # Generate batches
        batches = gen_batches(gsData.shape[0], 20)
        fsRun = True
        
        # Get HardNet features  
        for batch in batches:
            temp = self.hardNet.forward(gsData[batch])
            
            if fsRun:
                metrics = temp
                fsRun = False
            else:
                metrics = np.concatenate((metrics, temp), axis=0)
                
        # Convert the metrics to np array and normalize them
        metrics = self.normalize2DData(metrics)

        return metrics, labels
