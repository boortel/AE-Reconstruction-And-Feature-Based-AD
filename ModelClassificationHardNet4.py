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
from keras.models import Model
import tensorflow.compat.v1 as tf1

from sklearn.utils import gen_batches
from sklearn.decomposition import PCA
from skimage.util import view_as_blocks

from fishervector import FisherVectorGMM

from HardNet import HardNet
from ModelClassificationBase import ModelClassificationBase


class ModelClassificationHardNet4(ModelClassificationBase):

    ## Constructor
    def __init__(self, modelDataPath, experimentPath, modelSel, layerSel, labelInfo, imageDim, modelData):
        
        # Call the parent
        ModelClassificationBase.__init__(self, modelDataPath, experimentPath, modelSel, layerSel, labelInfo, imageDim, modelData, 'HardNet4')
        
        # Create HardNet model object
        self.hardNet = HardNet()
        
        # Get data, metrics and classify the data
        try:
            if self.modelData:
                self.procDataFromDict()
            else:
                self.procDataFromFile()
                
            #self.procDataFromFile()
            self.dataClassify()
        except:
            logging.error('An error occured during classification using ' + self.featExtName + ' feature extraction method...')
            traceback.print_exc()
        pass
        

    ## Compute the classification metrics
    def computeMetrics(self, processedData, eps = 1e-6):

        # Get the data
        orgData = processedData.get('Org')
        decData = processedData.get('Dec')
        
        labels = processedData.get('Lab')
        
        # Initialize the conversion and metric lists
        gsData = []
        metrics = []
        
        # Get the sub-img parameters
        hSub = self.imageDim[0]/32
        wSub = self.imageDim[1]/32
            
        # Get HardNet features for each image
        for imgOrg, imgDec in zip(orgData, decData):
            
            # Convert images to gray
            imgOrg = cv.cvtColor(imgOrg, cv.COLOR_BGR2GRAY)
            imgDec = cv.cvtColor(imgDec, cv.COLOR_BGR2GRAY)
            
            # Split the image into 32x32 images
            batchOrg = view_as_blocks(imgOrg, (32, 32))
            batchOrg = batchOrg.reshape(batchOrg.shape[0]*batchOrg.shape[1], 32, 32, 1)
            
            batchDec = view_as_blocks(imgDec, (32, 32))
            batchDec = batchDec.reshape(batchDec.shape[0]*batchDec.shape[1], 32, 32, 1)
            
            # Get HardNet features
            desOrg = self.hardNet.forward(batchOrg)
            desDec = self.hardNet.forward(batchDec)
            
            # Match descriptors
            #dist = np.linalg.norm(desOrg-desDec, axis=1) ** 2
            dist = np.log(np.linalg.norm(desOrg-desDec, axis=1))
            metrics.append(dist)

        # Convert the metrics to np array
        metrics = np.array(metrics)
        
        # Visualise the data
        self.fsVisualise(metrics, labels)

        return metrics, labels
