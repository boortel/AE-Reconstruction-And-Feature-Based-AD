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
from skimage.util import view_as_blocks

from ModelClassificationBase import ModelClassificationBase


class ModelClassificationHardNet2(ModelClassificationBase):

    ## Constructor
    def __init__(self, modelDataPath, experimentPath, modelSel, layerSel, labelInfo, imageDim, modelData):
        
        try:
            # Call the parent
            ModelClassificationBase.__init__(self, modelDataPath, experimentPath, modelSel, layerSel, labelInfo, imageDim, modelData, 'HardNet2')
        except:
            traceback.print_exc()
        
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
        metrics = []
            
        # Get HardNet features for each image
        for img in diffData:
            
            # Convert image to gray
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Split the image into 32x32 images
            batch = view_as_blocks(img, (32, 32))
            batch = batch.reshape(batch.shape[0]*batch.shape[1], 32, 32, 1)

            # Get HardNet features and its norm
            temp = self.hardNet.forward(batch)

            norms = np.linalg.norm(temp, None, axis=0)
            metrics.append(norms)

        # Convert the metrics to np array and normalize them
        metrics = self.normalize2DData(np.array(metrics))

        return metrics, labels
