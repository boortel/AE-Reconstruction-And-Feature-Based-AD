# -*- coding: utf-8 -*-
"""
Created on Wed Oct  12 17:23:05 2022

@author: Simon Bilik

This class is used for the classification of the evaluated model

"""

import os
import time
import logging

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score


class ModelClassificationBase():

    ## Set the constants and paths
    def __init__(self, modelDataPath, experimentPath, modelSel, layerSel, labelInfo, imageDim, modelData, featExtName):

        # Set the base path
        self.layerSel = layerSel
        self.modelName = modelSel
        self.modelDataPath = modelDataPath
        self.experimentPath = experimentPath

        # Set the constants
        self.imageDim = imageDim
        self.labelInfo = labelInfo
        
        # Set the feature extractor name and print separator
        self.featExtName = featExtName
        
        # Save the modelData variable
        self.modelData = modelData
        
        logging.info('Feature extraction method: ' + self.featExtName)
        logging.info('------------------------------------------------------------------------------------------------')

    
    ## Get data from file
    def procDataFromFile(self):
        
        actStrs = ['Train', 'Test', 'Valid']
        
        for actStr in actStrs:

            self.actStr = actStr

            # Build the paths
            orgDatasetPath = os.path.join(self.experimentPath, 'Org_' + actStr + '.npz')
            procDatasetPath = os.path.join(self.modelDataPath, 'Eval_' + actStr + '.npz')

            # Load the NPZ files
            orgDataset = np.load(orgDatasetPath)
            procDataset = np.load(procDatasetPath)

            if self.actStr == 'Train':

                # Store the data and get the metrics
                self.processedData = {'Org': orgDataset['orgData'], 'Dec': procDataset['decData'], 'Lab': orgDataset['labels']}
                self.metricsTr, _ = self.computeMetrics(self.processedData)

            elif self.actStr == 'Test':

                # Store the data and get the metrics
                self.processedData = {'Org': orgDataset['orgData'], 'Dec': procDataset['decData'], 'Lab': orgDataset['labels']}
                self.metricsTs, self.labelsTs = self.computeMetrics(self.processedData)
    

    ## Get data from dictionary
    def procDataFromDict(self):
        
        actStrs = ['Train', 'Test', 'Valid']
        
        for actStr in actStrs:
            
            self.actStr = actStr

            if self.actStr == 'Train':

                # Store the data and get the metrics
                self.processedData = {'Org': self.modelData[actStr]['Org'], 'Enc': self.modelData[actStr]['Enc'], 'Dec': self.modelData[actStr]['Dec'], 'Lab': self.modelData[actStr]['Lab']}
                self.metricsTr, _ = self.computeMetrics(self.processedData)

            elif self.actStr == 'Test':

                # Store the data and get the metrics
                self.processedData = {'Org': self.modelData[actStr]['Org'], 'Enc': self.modelData[actStr]['Enc'], 'Dec': self.modelData[actStr]['Dec'], 'Lab': self.modelData[actStr]['Lab']}
                self.metricsTs, self.labelsTs = self.computeMetrics(self.processedData)
                
                # Visualise the data
                self.fsVisualise(self.metricsTs, self.labelsTs)
                
            elif self.actStr == 'Valid':

                # Store the data and get the metrics
                self.processedData = {'Org': self.modelData[actStr]['Org'], 'Enc': self.modelData[actStr]['Enc'], 'Dec': self.modelData[actStr]['Dec'], 'Lab': self.modelData[actStr]['Lab']}
                self.metricsVl, self.labelsVl = self.computeMetrics(self.processedData)
            
            
    ## Compute the classification metrics
    def computeMetrics(self, processedData):
        pass


    ## Normalize data (Gaussian)
    def normalize2DData(self, data):

        transformer = RobustScaler().fit(data)
        dataSc = transformer.transform(data)

        #return (data - np.min(data)) / (np.max(data) - np.min(data))
        return dataSc


    ## Calculate classification metrics
    def getEvaluationMetrics(self, scores, name, ax):
        
        # Get the ROC curve
        precision, recall, _ = precision_recall_curve(self.labelsTs, scores)
        
        prc_auc = auc(recall, precision)
        roc_auc = roc_auc_score(self.labelsTs, scores)
        
        fpr, tpr, trs = roc_curve(self.labelsTs, scores)
        
        # Log the information
        logging.info("Algorithm: " + name)
        logging.info("AUC-ROC: "   + f'{float(roc_auc):.2f}')
        logging.info("AUC-PRE: "   + f'{float(prc_auc):.2f}')
        
        # Plot the ROC
        ax.plot(fpr, tpr)
        ax.set_title(name + ', AUC: ' + f'{float(roc_auc):.2f}')
        ax.set(xlabel = "False positive rate", ylabel = "True positive rate")
        
        # Get and return optimal treshold using Youden’s J statistic
        #optimal_idx = np.argmax(tpr - fpr)
        #optimal_trs = trs[optimal_idx]
        
        # Get and return optimal treshold using equal error rate (EER)
        fnr = 1 - tpr
        optimal_trs = trs[np.nanargmin(np.absolute((fnr - fpr)))]
        
        return optimal_trs
    

    ## Get the OK and NOK counts together with TPR and TNR
    def getConfusionMatrix(self, labelsPred, name, ax):

        cm = confusion_matrix(self.labelsTs, labelsPred)
        
        precision = precision_score(self.labelsTs, labelsPred)
        recall = recall_score(self.labelsTs, labelsPred)
        f1score = f1_score(self.labelsTs, labelsPred)
        
        logging.info("Precision: " + f'{float(precision):.2f}')
        logging.info("Recall: "    + f'{float(recall):.2f}')
        logging.info("F1-score: "  + f'{float(f1score):.2f}')
        
        # Get the TPR and TNR
        tnr = cm[0, 0]/(cm[0, 0] + cm[0, 1])
        tpr = cm[1, 1]/(cm[1, 0] + cm[1, 1])
        
        logging.info("TPR: " + f'{float(tnr):.2f}')
        logging.info("TNR: " + f'{float(tpr):.2f}')
        
        # Get the balance ratio
        tnc = 1 if cm[0, 0] == 0 else cm[0, 0]
        tpc = 1 if cm[1, 1] == 0 else cm[1, 1]
        
        bRatio = tnc/tpc if tnc <= tpc else -tpc/tnc
        
        logging.info("Balance ratio: " + f'{float(bRatio):.2f}')
        
        # Display confusion matrix
        logging.info("Confusion matrix")
        logging.info(cm)

        ConfusionMatrixDisplay.from_predictions(self.labelsTs, labelsPred, ax = ax)


    ## Classify the results
    def dataClassify(self):
        
        # Compare AD detection algorithms
        outliers_fraction = 0.01
        anomaly_algorithms = [
            ("Robust covariance", EllipticEnvelope(contamination = outliers_fraction, support_fraction = 0.9)),
            ("One-Class SVM", svm.OneClassSVM(nu = outliers_fraction, kernel = "rbf", gamma = 'scale')),
            ("Isolation Forest", IsolationForest(contamination = outliers_fraction, random_state = 42)),
            ("Local Outlier Factor", LocalOutlierFactor(n_neighbors = 15, contamination = outliers_fraction, novelty = True))]
        
        # Visualise the data
        fig, axarr = plt.subplots(2, len(anomaly_algorithms))
        fig.set_size_inches(16, 8)

        tempTitle = "Classification results of the " + self.layerSel + '-' + self.modelName + '_' + self.labelInfo + " model using " + self.featExtName + " feature extraction"
        fig.suptitle(tempTitle, fontsize=16)
        #fig.subplots_adjust(left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01)

        plotNum = 0

        for name, algorithm in anomaly_algorithms:

            # Fit the model
            t0 = time.time()
            algorithm.fit(self.metricsTr)
            t1 = time.time()

            logging.info('Model fitting time: ' + f'{float(t1 - t0):.2f}' + 's')
            
            # Get the scores from the validation DS, calculate ROC evaluation metric and get optimal trsh
            #valScores = algorithm.decision_function(self.metricsVl)#.ravel()*(-1)
            #optimal_trs = self.getEvaluationMetrics(valScores, name, axarr[0][plotNum])
            
            # Get the scores from the test DS, calculate ROC evaluation metric and get optimal trsh
            tstScores = algorithm.decision_function(self.metricsTs)
            optimal_trs = self.getEvaluationMetrics(tstScores, name, axarr[0][plotNum])
            
            # Fit the data and tag outliers
            y_pred = tstScores
            
            okIdx = np.where(tstScores >= optimal_trs)
            nokIdx = np.where(tstScores < optimal_trs)
            
            y_pred[okIdx] = 1
            y_pred[nokIdx] = -1

            # Calculate confusion matrix
            self.getConfusionMatrix(y_pred, name, axarr[1][plotNum])

            plotNum +=1

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        
        fig.savefig(os.path.join(self.modelDataPath, self.layerSel + '-' + self.modelName  + '_' + self.labelInfo + '_' + self.featExtName + '_ClassEval_.png'))


    ## Visualise the feature space
    def fsVisualise(self, metrics, labels):
        
        # Reduce the dimensionality of metrics for t-SNE transformation
        if metrics.shape[0] > 50 and metrics.shape[1] > 50:
            pca_red = PCA(n_components = 50)
            metrics = pca_red.fit_transform(metrics)
        
        # Perform the t-SNE and feature space visualisation
        if metrics.shape[0] < 30:
            # Set perplexity as a half value to the number of samples for small datasets
            perp = int(metrics.shape[0]/2)
        else:
            perp = 30
        
        tsne_metrics = TSNE(n_components=2, perplexity=perp, n_iter=1000, learning_rate=100, init='pca').fit_transform(metrics)

        # Perform the PCA and feature space visualisation
        pca = PCA(n_components=2)
        pca_metrics = pca.fit_transform(metrics)

        # Visualise the data
        fig, axarr = plt.subplots(2)
        fig.set_size_inches(8, 8)

        tempTitle = "FS visualisations of the " + self.layerSel + '-' + self.modelName + '_' + self.labelInfo + " model using " + self.featExtName + " feature extraction"
        fig.suptitle(tempTitle, fontsize=12)

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
        fig.subplots_adjust(top=0.88)
        
        fig.savefig(os.path.join(self.modelDataPath, self.layerSel + '-' + self.modelName  + '_' + self.labelInfo + '_' + self.featExtName + '_FeatureSpace.png'))
        