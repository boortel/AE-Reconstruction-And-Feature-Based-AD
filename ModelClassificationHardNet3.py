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

from scipy import spatial
from HardNet import HardNet
from skimage.util import view_as_blocks

from ModelClassificationBase import ModelClassificationBase


class ModelClassificationHardNet3(ModelClassificationBase):

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
            'HardNet3',
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
        orgData = processedData.get('Org')
        decData = processedData.get('Dec')
        
        labels = processedData.get('Lab')
        
        # Initialize the conversion and metric lists
        metrics = []
            
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
            
            # Match descriptors.
            #dist = np.linalg.norm(desOrg-desDec, axis=1)
            #metrics.append(dist)
            
            dist = []
            
            # Match descriptors using cosine simillarity scaled between 0-1
            for desVecOrg, desVecDec in zip(desOrg, desDec):
                dist.append(np.abs(spatial.distance.cosine(desVecOrg, desVecDec)))
            
            metrics.append(np.array(dist))

        # Convert the metrics to np array and normalize them
        metrics = self.normalize2DData(np.array(metrics))

        return metrics, labels
