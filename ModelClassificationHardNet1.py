# -*- coding: utf-8 -*-
"""
Created on Wed Oct  12 17:23:05 2022

@author: Simon Bilik

This class is used for the classification of the evaluated model

"""

import os
import keras
import pickle
import logging
import traceback

import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

from sklearn.utils import gen_batches
from sklearn.decomposition import PCA
from fishervector import FisherVectorGMM

from HardNet import HardNet
from ModelClassificationBase import ModelClassificationBase

class ModelClassificationHardNet1(ModelClassificationBase):

    ## Constructor
    def __init__(self, modelDataPath, experimentPath, modelSel, layerSel, labelInfo, imageDim):
        
        # Disable tf2 behavior for tf1
        #tf1.disable_v2_behavior()

        # Call the parent
        ModelClassificationBase.__init__(self, modelDataPath, experimentPath, modelSel, layerSel, labelInfo, imageDim, 'HardNet1')
        
        # Create HardNet model object
        self.hardNet = HardNet()
        

    ## Compute the classification metrics
    def computeMetrics(self, processedData, eps = 1e-6):

        # Get the data
        diffData = np.subtract(processedData.get('Org'), processedData.get('Dec'))
        labels = processedData.get('Lab')
        
        # Initialize the conversion and metric lists
        gsData = []
        metrics = []
        
        # Convert the data to grayscale and to required shape (batch, height, width, channel)
        for img in diffData:
            gsData.append(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (32, 32), interpolation = cv.INTER_AREA))
            
        gsData = np.array(gsData)
        gsData = np.expand_dims(gsData, axis=3)
        
        # Generate batches
        batches = gen_batches(gsData.shape[0], 20)
        
        # Get HardNet features
        with tf1.Session() as sess:
            
            for batch in batches:
                metrics.append(sess.run(self.hardNet.forward(gsData[batch])))
        
        metrics = np.array(metrics)
        metrics = metrics.reshape(metrics.shape[0] * metrics.shape[1], metrics.shape[2])

        # Visualise the data
        self.fsVisualise(metrics, labels)

        return metrics, labels
