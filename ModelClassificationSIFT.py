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

from sklearn.decomposition import PCA

from ModelClassificationBase import ModelClassificationBase

class ModelClassificationSIFT(ModelClassificationBase):

    ## Constructor
    def __init__(self, 
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
        ModelClassificationBase.__init__(self,
                                         modelDataPath,
                                         experimentPath, 
                                         modelSel,
                                         layerSel,
                                         labelInfo,
                                         imageDim,
                                         modelData,
                                         'SIFT',
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


    ## Preprocess the images
    def imgPreprocess(self, img):
        
        # Map the image to UINT8
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        imgP = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

        return imgP

    ## Compute the classification metrics
    def computeMetrics(self, processedData):

        # Get the data
        imgData = np.subtract(processedData.get('Org'), processedData.get('Dec'))
        #imgData = processedData.get('Dec')
        labels = processedData.get('Lab')

        # Initialize the metric array
        metricArray = []
        nFeatures = 5

        # Create SIFT detector object
        sift = cv.SIFT_create(nfeatures = nFeatures)
        
        # Loop through the images in imgData
        for i in range(np.size(labels)):
            
            # Initialize the metric arrays and safety counter
            valSizeSIFT = []
            valRespSIFT = []
            counter = 0

            # Get an image, preprocess it, convert it to UINT8 and to grayscale
            imgP = self.imgPreprocess(imgData[i][:][:])
            
            # Get the SIFT features
            kpSIFT, _ = sift.detectAndCompute(imgP, None)

            for i in range(5):
                try:
                    # Append the features to the array
                    valSizeSIFT.append(kpSIFT[i].size)
                    valRespSIFT.append(kpSIFT[i].response)

                except:
                    # Not enough feature points, fill the missing values with zeros
                    logging.warning('Missing ' + f'{float(nFeatures - counter):}' + ' feature points, appending zeros to fix the lenght.')

                    valSizeSIFT.append(0)
                    valRespSIFT.append(0)

            # Convert the lists into the np arrays
            metricArray.append(valSizeSIFT + valRespSIFT)

        # Get metrics np array
        metrics = self.normalize2DData(np.array(metricArray))

        # Reduce the dimensionality of metrics
        if metrics.shape[1] > 50:
            pca_red = PCA(n_components = 50)
            metrics = pca_red.fit_transform(metrics)

        return metrics, labels
