# -*- coding: utf-8 -*-
"""
Created on Wed Oct  12 17:23:05 2022

@author: Simon Bilik

This class is used for the classification of the evaluated model

"""

import dis
import os
import time
import imagehash

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

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
        self.labelInfo = labelInfo
    

    ## Get data from dictionary
    def procDataFromDict(self, processedData, actStr):

        self.actStr = actStr

        if self.actStr == 'Train':
        
            # Store data and get the metrics
            self.processedDataTr = processedData
            self.metricsTr, _ = self.computeMetrics(self.processedDataTr)

        elif self.actStr == 'Test':

            # Store data and get the metrics
            self.processedDataTs = processedData
            self.metricsTs, self.labelsTs = self.computeMetrics(self.processedDataTs)


    ## Get data from file
    def procDataFromFile(self, actStr):
        
        self.actStr = actStr

        # Build the path
        procDatasetPath = os.path.join(self.outputPath, 'modelData', 'Eval_' + actStr + '.npz')

        # Load the NPZ file
        procDataset = np.load(procDatasetPath)

        if self.actStr == 'Train':

            # Store the data and get the metrics
            labels = procDataset['labels']
            okIdx = np.where(labels == 0)
            labels[okIdx] = -1

            self.processedDataTr = {'Org': procDataset['orgData'], 'Enc': procDataset['encData'], 'Dec': procDataset['decData'], 'Lab': labels}
            self.metricsTr, _ = self.computeMetrics(self.processedDataTr)

        elif self.actStr == 'Test':

            labels = procDataset['labels']
            okIdx = np.where(labels == 0)
            labels[okIdx] = -1

            # Store the data and get the metrics
            self.processedDataTs = {'Org': procDataset['orgData'], 'Enc': procDataset['encData'], 'Dec': procDataset['decData'], 'Lab': labels}
            self.metricsTs, self.labelsTs = self.computeMetrics(self.processedDataTs)


    ## Normalize 1D data (minmax)
    def normalize1DData(self, data):

        return (data - np.min(data)) / (np.max(data) - np.min(data))


    ## Normalize data (Gaussian)
    def normalize2DData(self, data):

        transformer = RobustScaler().fit(data)
        dataSc = transformer.transform(data)

        #return (data - np.min(data)) / (np.max(data) - np.min(data))
        return dataSc

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
        metrics = self.normalize2DData(np.column_stack((valuesL2, valuesMSE, valuesSSIM, valuesAvgH)))
        #metrics = np.column_stack((valuesL2, valuesMSE, valuesSSIM, valuesAvgH))

        # Visualise the data
        self.fsVisualise(metrics, labels)

        return metrics, labels


    ## Calculate classification metrics
    def getEvaluationMetrics(self, scores, name, ax):

        #precision, recall, _ = precision_recall_curve(testy, scores)
        #prc_auc = auc(recall, precision)

        roc_auc = roc_auc_score(self.labelsTs, scores)
        fpr, tpr, _ = roc_curve(self.labelsTs, scores)

        ax.plot(fpr, tpr)
        ax.set_title(name + ', AUC: ' + f'{float(roc_auc):.2f}')
        ax.set(xlabel = "False positive rate", ylabel = "True positive rate")
    

    ## Get the OK and NOK counts
    def getConfusionMatrix(self, labelsPred, name, ax):

        #okIdx = np.where(labelsPred == -1)
        #labelsPred[okIdx] = 0

        cm = confusion_matrix(self.labelsTs, labelsPred)

        print('-----------------------')
        print("Model name: ", self.modelName + '_' + self.labelInfo)
        print("Algorithm: ", name)
        print("Confusion matrix")
        print(cm)

        ConfusionMatrixDisplay.from_predictions(self.labelsTs, labelsPred, ax = ax)

        #plt.savefig(os.path.join(self.outputPath, 'modelData', self.modelName + self.actStr + name + '_ConfusionMatrix.png'))


    ## Classify the results
    def dataClassify(self):
        
        # Compare AD detection algorithms
        outliers_fraction = 0.005
        anomaly_algorithms = [
            ("Robust covariance", EllipticEnvelope(contamination = outliers_fraction, support_fraction = 0.9)),
            ("One-Class SVM", svm.OneClassSVM(nu = outliers_fraction, kernel = "rbf", gamma = 0.0000001)),
            ("Isolation Forest", IsolationForest(contamination = outliers_fraction, random_state = 42)),
            ("Local Outlier Factor", LocalOutlierFactor(n_neighbors = 15, contamination = outliers_fraction))]
        
        # Visualise the data
        fig, axarr = plt.subplots(2, len(anomaly_algorithms))
        fig.set_size_inches(16, 8)

        tempTitle = " Feature space visualisations " + self.modelName + " model."
        fig.suptitle(tempTitle, fontsize=14, y=1.08)
        #fig.subplots_adjust(left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01)

        plotNum = 0

        for name, algorithm in anomaly_algorithms:

            # Fit the model
            t0 = time.time()
            algorithm.fit(self.metricsTr)
            t1 = time.time()

            print('Model fitting time: ' + f'{float(t1 - t0):.2f}' + 's')

            # Fit the data and tag outliers
            if name == "Local Outlier Factor":
                y_pred = algorithm.fit_predict(self.metricsTs)
                scores = y_pred.ravel() * (-1)
            else:
                y_pred = algorithm.fit(self.metricsTr).predict(self.metricsTs)
                scores = algorithm.decision_function(self.metricsTs).ravel() * (-1)

            # Calculate evaluation metrics
            self.getEvaluationMetrics(scores, name, axarr[0][plotNum])

            # Calculate confusion matrix
            self.getConfusionMatrix(y_pred, name, axarr[1][plotNum])

            plotNum +=1

        fig.tight_layout()
        fig.savefig(os.path.join(self.outputPath, 'modelData', self.modelName + self.actStr +'_ClassEval.png'))

    ## Visualise the feature space
    def fsVisualise(self, metrics, labels):

        # Perform the t-SNE and feature space visualisation
        tsne_metrics = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate=100, init='pca').fit_transform(metrics)

        # Perform the PCA and feature space visualisation
        pca = PCA(n_components=2)
        pca_metrics = pca.fit_transform(metrics)

        # Visualise the data
        fig, axarr = plt.subplots(2)
        fig.set_size_inches(8, 8)

        tempTitle = " Feature space visualisations " + self.modelName + " model."
        fig.suptitle(tempTitle, fontsize=14, y=1.08)

        okIdx = np.where(labels == 1)
        nokIdx = np.where(labels == -1)

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
        