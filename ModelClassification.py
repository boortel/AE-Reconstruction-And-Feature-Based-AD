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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as SSIM

class ModelClassification():

    ## Set the constants and paths
    def __init__(self, modelBasePath, modelSel, labelInfo, imageDim):

        # Set the base path
        self.modelName = modelSel
        self.outputPath = os.path.join(modelBasePath, modelSel + labelInfo)

        # Set the constants
        self.imageDim = imageDim
    

    ## Get data from dictionary
    def procDataFromDict(self, processedData, actStr):

        self.actStr = actStr

        if self.actStr == 'Train':
        
            # Store data and get the metrics
            self.processedDataTr = processedData
            self.metricsTr = self.computeMetrics(self.processedDataTr)

        elif self.actStr == 'Test':

            # Store data and get the metrics
            self.processedDataTs = processedData
            self.metricsTs = self.computeMetrics(self.processedDataTs)


    ## Get data from file
    def procDataFromFile(self, actStr):
        
        self.actStr = actStr

        # Build the path
        procDatasetPath = os.path.join(self.outputPath, 'modelData', 'Eval_' + actStr + '.npz')

        # Load the NPZ file
        procDataset = np.load(procDatasetPath)

        if self.actStr == 'Train':

            # Store the data and get the metrics
            self.processedDataTr = {'Org': procDataset['orgData'], 'Enc': procDataset['encData'], 'Dec': procDataset['decData'], 'Lab': procDataset['labels']}
            self.metricsTr = self.computeMetrics(self.processedDataTr)

        elif self.actStr == 'Test':

            # Store the data and get the metrics
            self.processedDataTs = {'Org': procDataset['orgData'], 'Enc': procDataset['encData'], 'Dec': procDataset['decData'], 'Lab': procDataset['labels']}
            self.metricsTs = self.computeMetrics(self.processedDataTs)


    ## Compute the metrics
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

        # Define the color mode
        if self.imageDim[2] == 3:
            cMode = "RGB"
            chAx = 2
        else:
            cMode = "L"
            chAx = None

        # Compute the metrics for all images in dataset
        for i in range(len(orgData)):

            # TODO: predelat L2 tak, aby fungovala i pro cernobile
            valuesL2.append(np.sum(np.square(np.subtract(orgData[i], decData[i])), axis=None))
            valuesMSE.append(MSE(orgData[i], decData[i]))
            valuesSSIM.append(SSIM(orgData[i], decData[i], data_range=255, channel_axis = chAx))

            imgOrg = Image.fromarray(orgData[i], mode = cMode)
            imgDec = Image.fromarray(decData[i], mode = cMode)
            valuesAvgH.append(imagehash.average_hash(imgOrg) - imagehash.average_hash(imgDec))

        # Convert the lists into the np arrays
        valuesL2 = np.array(valuesL2)
        valuesMSE = np.array(valuesMSE)
        valuesSSIM = np.array(valuesSSIM)
        valuesAvgH = np.array(valuesAvgH)

        # Get metrics np array
        metrics = np.column_stack((valuesL2, valuesMSE, valuesSSIM, valuesAvgH))

        # Visualise the data
        self.fsVisualise(metrics, labels)

        return metrics

    ## Visualise the feature space
    def fsVisualise(self, metrics, labels):

        # Perform the t-SNE and feature space visualisation
        tsne_metrics = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate=100, init='pca').fit_transform(metrics)

        # Perform the PCA and feature space visualisation
        pca = PCA(n_components=2)
        pca_metrics = pca.fit_transform(metrics)

        # Visualise the data
        fig, axarr = plt.subplots(2)
        tempTitle = " Feature space visualisations " + self.modelName + " model."
        fig.suptitle(tempTitle, fontsize=14, y=1.08)

        okIdx = np.where(labels == 1)
        nokIdx = np.where(labels == 0)

        axarr[0].scatter(tsne_metrics[okIdx, 0], tsne_metrics[okIdx, 1], s = 4)
        axarr[0].scatter(tsne_metrics[nokIdx, 0], tsne_metrics[nokIdx, 1], s = 4)
        axarr[0].set(xlabel = "t-SNE 1", ylabel = "t-SNE 2")
        axarr[0].set_title("t-SNE Feature space visualisation")
        axarr[0].legend(["OK", "NOK"], loc='upper right')
        
        axarr[1].scatter(pca_metrics[okIdx, 0], pca_metrics[okIdx, 1], s = 4)
        axarr[1].scatter(pca_metrics[nokIdx, 0], pca_metrics[nokIdx, 1], s = 4)
        axarr[1].set(xlabel = "PCA 1", ylabel = "PCA 2")
        axarr[1].set_title("PCA Feature space visualisation")
        axarr[1].legend(["OK", "NOK"], loc='upper right')
        
        fig.tight_layout()
        fig.savefig(os.path.join(self.outputPath, 'modelData', self.modelName + self.actStr +'_FeatureSpace.png'))
        