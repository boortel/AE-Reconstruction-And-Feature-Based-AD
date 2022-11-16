# -*- coding: utf-8 -*-
"""
Created on Wed Oct  12 17:23:05 2022

@author: Simon Bilik

This class is used for the classification of the evaluated model

"""

import os
import logging
import traceback

import cv2 as cv
import numpy as np

from sklearn.decomposition import PCA
from fishervector import FisherVectorGMM

from ModelClassificationBase import ModelClassificationBase

class ModelClassificationSIFT(ModelClassificationBase):

    ## Constructor
    def __init__(self, modelDataPath, modelSel, layerSel, labelInfo, imageDim, extractorType):

        # Call the parent
        ModelClassificationBase.__init__(self, modelDataPath, modelSel, layerSel, labelInfo, imageDim)

        # Set the feature extractor name
        self.featExtName = 'SIFTMetrics'

        # Set the feature extraction type
        if extractorType == 'Points' or extractorType == 'Features':
            self.extractorType = extractorType
        else:
            logging.error('Unknown type of extraction ' + extractorType)
            traceback.print_exc()
            return

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

            self.processedDataTr = {'Dif': procDataset['difData'], 'Lab': labels}
            self.metricsTr, _ = self.computeMetrics(self.processedDataTr)

        elif self.actStr == 'Test':

            labels = procDataset['labels']
            okIdx = np.where(labels == 0)
            labels[okIdx] = -1

            # Store the data and get the metrics
            self.processedDataTs = {'Dif': procDataset['difData'], 'Lab': labels}
            self.metricsTs, self.labelsTs = self.computeMetrics(self.processedDataTs)


    ## Preprocess the images
    def imgPreprocess(self, img):
        
        # Define the erosion and dilatation elements
        elementSize = 1
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * elementSize + 1, 2 * elementSize + 1), (elementSize, elementSize))

        # Define the gaussian filter parameters
        sigma = 6
        krSz = np.uint8(2*np.ceil(2*sigma) + 1)

        # Map the image to UINT8
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Blur the image and substract it from original to get high freq
        blur = cv.GaussianBlur(gray, (krSz,krSz), sigma)
        diff = gray - blur

        # Sharpen the image
        #imgP = 0.6*gray + 0.4*diff

        imgP = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        #imgP = cv.Canny(imgP, 100, 200)

        #cv.imwrite('sift_keypoints.jpg', cv.normalize(imgP, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))

        # Convert the image to GS and binarize it
        #diff = cv.normalize(diff, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        #th, imgB = cv.threshold(diff, 128, 255, cv.THRESH_BINARY)

        # Erode and dilate the image
        #imgEr = cv.erode(imgB, element)
        #imgP = cv.dilate(imgEr, element)

        return cv.normalize(imgP, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    ## Compute the classification metrics
    def computeMetrics(self, processedData):

        # Get the data
        diffData = processedData.get('Dif')
        labels = processedData.get('Lab')

        # Initialize the metric array
        metricArray = []
        nFeatures = 10

        # Create SIFT detector object
        sift = cv.SIFT_create(nfeatures = nFeatures)
        
        # Loop through the images in diffData
        for i in range(np.size(labels)):

            # Get an image, preprocess it, convert it to UINT8 and to grayscale
            imgP = self.imgPreprocess(diffData[i][:][:])
            
            # Get the SIFT features
            kpSIFT, desSIFT = sift.detectAndCompute(imgP, None)

            ## Compute the metrics from the key points
            if self.extractorType == 'Points':

                # Initialize the metric arrays and safety counter
                valSizeSIFT = []
                valRespSIFT = []
                counter = 0

                for point in kpSIFT:
                    # Append the features to the array
                    valSizeSIFT.append(point.size)
                    valRespSIFT.append(point.response)
                
                    if counter >= 4:
                        break
                    else:
                        counter += 1

                # Convert the lists into the np arrays
                metricArray.append(valSizeSIFT + valRespSIFT)

            ## Compute the metrics from features
            elif self.extractorType == 'Features':
                
                desSIFT = desSIFT[0:nFeatures, :]
                metricArray.append(desSIFT)

        # Get metrics np array
        if self.extractorType == 'Points':
            metrics = self.normalize2DData(np.array(metricArray))

        elif self.extractorType == 'Features':
            
            metrics = np.array(metricArray)

            # Fit the Fisher Vector with the training dataset
            if self.actStr == 'Train':
                # Fit the model in the case of training dataset
                self.fv_gmm = FisherVectorGMM().fit_by_bic(metrics, choices_n_kernels=[2,5,10,20])

            # Compute the Fisher Vector values and flat to the shape (nSamples, nPoints*nFeatures) them for the further processing
            metrics = self.fv_gmm.predict(metrics)
            metrics = metrics.reshape(metrics.shape[0], metrics.shape[1]*metrics.shape[2])

        # Reduce the dimensionality of metrics
        if metrics.shape[1] > 50:
            pca_red = PCA(n_components = 50)
            metrics = pca_red.fit_transform(metrics)

        # Visualise the data
        self.fsVisualise(metrics, labels)

        return metrics, labels
