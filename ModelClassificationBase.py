# -*- coding: utf-8 -*-
"""
Created on Wed Oct  12 17:23:05 2022

@author: Simon Bilik

This class is used for the classification of the evaluated model

"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

class ModelClassificationBase():

    ## Set the constants and paths
    def __init__(self, modelBasePath, modelSel, labelInfo, imageDim):

        # Set the base path
        self.modelName = modelSel
        self.outputPath = os.path.join(modelBasePath, modelSel + labelInfo)

        # Set the constants
        self.imageDim = imageDim
        self.labelInfo = labelInfo
        self.featExtName = ''

        # Print the separator
        print('-----------------------------------------------')
        print("Autoencoder architecture name: ", self.modelName)
        print('')

    
    ## Get data from file
    def procDataFromFile(self, actStr):
        pass


    ## Compute the classification metrics
    def computeMetrics(self, processedData):
        pass
    

    ## Get data from dictionary
    def procDataFromDict(self, processedData, actStr):

        self.actStr = actStr

        if self.actStr == 'Train':
        
            # Store data and get the metrics
            self.processedDataTr = processedData
            self.metricsTr, self.labelsTr = self.computeMetrics(self.processedDataTr)

        elif self.actStr == 'Test':

            # Store data and get the metrics
            self.processedDataTs = processedData
            self.metricsTs, self.labelsTs = self.computeMetrics(self.processedDataTs)


    ## Normalize data (Gaussian)
    def normalize2DData(self, data):

        transformer = RobustScaler().fit(data)
        dataSc = transformer.transform(data)

        #return (data - np.min(data)) / (np.max(data) - np.min(data))
        return dataSc


    ## Calculate classification metrics
    def getEvaluationMetrics(self, scores, name, ax):

        #precision, recall, _ = precision_recall_curve(testy, scores)
        #prc_auc = auc(recall, precision)

        roc_auc = roc_auc_score(self.labelsTs, scores)
        fpr, tpr, _ = roc_curve(self.labelsTs, scores)

        print('-----------------------')
        print("Model name: ", self.modelName + '_' + self.labelInfo)
        print("Algorithm: ", name)
        print("AUC: " + f'{float(roc_auc):.2f}')

        ax.plot(fpr, tpr)
        ax.set_title(name + ', AUC: ' + f'{float(roc_auc):.2f}')
        ax.set(xlabel = "False positive rate", ylabel = "True positive rate")
    

    ## Get the OK and NOK counts
    def getConfusionMatrix(self, labelsPred, name, ax):

        cm = confusion_matrix(self.labelsTs, labelsPred)

        print("Confusion matrix")
        print(cm)

        ConfusionMatrixDisplay.from_predictions(self.labelsTs, labelsPred, ax = ax)


    ## Classify the results
    def dataClassify(self):
        
        # Compare AD detection algorithms
        outliers_fraction = 0.5
        anomaly_algorithms = [
            ("Robust covariance", EllipticEnvelope(contamination = outliers_fraction, support_fraction = 0.9)),
            ("One-Class SVM", svm.OneClassSVM(nu = outliers_fraction, kernel = "rbf", gamma = 0.0000001)),
            ("Isolation Forest", IsolationForest(contamination = outliers_fraction, random_state = 42)),
            ("Local Outlier Factor", LocalOutlierFactor(n_neighbors = 15, contamination = outliers_fraction, novelty = True))]
        
        # Visualise the data
        fig, axarr = plt.subplots(2, len(anomaly_algorithms))
        fig.set_size_inches(16, 8)

        tempTitle = "Classification results of the " + self.modelName + " AE model with " + self.featExtName + " feature extraction."
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
            y_pred = algorithm.predict(self.metricsTs)
            scores = algorithm.decision_function(self.metricsTs)#.ravel()*(-1)

            # Calculate evaluation metrics
            self.getEvaluationMetrics(scores, name, axarr[0][plotNum])

            # Calculate confusion matrix
            self.getConfusionMatrix(y_pred, name, axarr[1][plotNum])

            plotNum +=1

        fig.tight_layout()
        fig.savefig(os.path.join(self.outputPath, 'modelData', self.modelName + self.actStr + self.featExtName +'_ClassEval.png'))


    ## Visualise the feature space
    def fsVisualise(self, metrics, labels):

        # Reduce the dimensionality of metrics for t-SNE transformation
        if metrics.shape[1] > 50:
            pca_red = PCA(n_components = 50)
            metrics = pca_red.fit_transform(metrics)

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
        fig.savefig(os.path.join(self.outputPath, 'modelData', self.modelName + self.actStr + self.featExtName +'_FeatureSpace.png'))
        