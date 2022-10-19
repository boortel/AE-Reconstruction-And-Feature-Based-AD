# -*- coding: utf-8 -*-
"""
Created on Wed Oct  12 17:23:05 2022

@author: Simon Bilik

This class is used for the classification of the evaluated model

"""

import time
import imagehash

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn import svm

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as SSIM
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve

class ModelClassification():

    ## Set the constants and paths
    def __init__(self, modelName):

        self.modelName = modelName
    

    ## Get data from dictionary
    def procDataFromDict(self, processedData, actStr):

        if actStr == 'Train':
        
            # Store data and get the metrics
            self.processedDataTr = processedData
            self.metricsTr = self.computeMetrics(self.processedDataTr)

        elif actStr == 'Test':

            # Store data and get the metrics
            self.processedDataTs = processedData
            self.metricsTs = self.computeMetrics(self.processedDataTs)


    ## Get data from file
    def procDataFromFile(self, procDatasetPath, actStr):

        # Load the NPZ file
        procDataset = np.load(procDatasetPath)

        if actStr == 'Train':

            # Store the data and get the metrics
            self.processedDataTr = {'Org': procDataset['orgData'], 'Enc': procDataset['encData'], 'Dec': procDataset['decData'], 'Lab': procDataset['labels']}
            self.metricsTr = self.computeMetrics(self.processedDataTr)

        elif actStr == 'Test':

            # Store the data and get the metrics
            self.processedDataTs = {'Org': procDataset['orgData'], 'Enc': procDataset['encData'], 'Dec': procDataset['decData'], 'Lab': procDataset['labels']}
            self.metricsTs = self.computeMetrics(self.processedDataTs)


    ## Compute the metrics
    def computeMetrics(self, processedData):

        # Get the data
        orgData = processedData.get('Org')
        decData = processedData.get('Dec')

        # Define the metrics arrays
        valueL2 = []
        valuesMSE = []
        valuesSSIM = []
        valuesAvgH = []

        # Compute the metrics for all images in dataset
        for i in range(len(orgData)):

            valueL2.append(np.sum(np.square(np.subtract(orgData[i], decData[i])), axis=None))
            valuesMSE.append(MSE(orgData[i], decData[i]))
            valuesSSIM.append(SSIM(orgData[i], decData[i], data_range=255))

            imgOrg = Image.fromarray(orgData[i])
            imgDec = Image.fromarray(decData[i])
            valuesAvgH.append(imagehash.average_hash(imgOrg) - imagehash.average_hash(imgDec))

        # Convert the lists into the np arrays
        valueL2 = np.array(valueL2)
        valuesMSE = np.array(valuesMSE)
        valuesSSIM = np.array(valuesSSIM)
        valuesAvgH = np.array(valuesAvgH)

        # Get metrics np array
        metrics = np.array(valueL2, valuesMSE, valuesSSIM, valuesAvgH)

        return metrics


    def getAUC(self, testy, scores):
        precision, recall, _ = precision_recall_curve(testy, scores)
        roc_auc = roc_auc_score(testy, scores)
        prc_auc = auc(recall, precision)

        return roc_auc, prc_auc

    def count_OK_NOK(self, scores1, images_test_all_labels,name,metric_name):
        true_OK = 0
        false_OK = 0
        true_NOK = 0
        false_NOK = 0
        all_OK = 0
        all_NOK = 0
        for i in range(images_test_all_labels.shape[0]):
            if (scores1[i] == -1) and (images_test_all_labels[i] == 1):
                true_NOK +=1
            elif (scores1[i] == 1) and (images_test_all_labels[i] == 1):
                false_OK +=1
            elif (scores1[i] == -1) and (images_test_all_labels[i] == 0):
                false_NOK +=1
            elif (scores1[i] == 1) and (images_test_all_labels[i] == 0):
                true_OK +=1
            if (images_test_all_labels[i] == 0):
                all_OK += 1
            else:
                all_NOK += 1

        print('-----------------------')
        print("Metric name: ", metric_name)
        print("Algorithm: ", name)
        print("All OK: ", all_OK)
        print("True OK: ", true_OK)
        print("All NOK: ", all_NOK)
        print("True NOK: ", true_NOK)
        print("False OK: ", false_OK)
        print("False NOK: ", false_NOK)

    def algorithm_compare(self, train,test_all,images_test_all_labels,metric_name):
        #porovnání algoritmů pro detekci anomálií
        outliers_fraction = 0.07
        anomaly_algorithms = [
            ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
            ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                            gamma=0.0000001)),
            ("Isolation Forest", IsolationForest(contamination=outliers_fraction,
                                                random_state=42)),
            ("Local Outlier Factor", LocalOutlierFactor(
                n_neighbors=35, contamination=outliers_fraction))]
                
        plot_num = 1
        plt.figure(figsize=(15, 10))
        plt.suptitle(metric_name, fontsize=16)
        for name, algorithm in anomaly_algorithms:
            t0 = time.time()
            algorithm.fit(train)
            t1 = time.time()
            plt.subplot(2,len(anomaly_algorithms), plot_num)
            plt.title(name, size=18)

            # fit the data and tag outliers
            if name == "Local Outlier Factor":
                y_pred = algorithm.fit_predict(test_all)
                scores = y_pred.ravel() * (-1)
            else:
                y_pred = algorithm.fit(train).predict(test_all)
                scores = algorithm.decision_function(test_all).ravel() * (-1)

            colors = np.array(['#ff7f00','#377eb8'])

            plt.scatter(test_all[:, 0], test_all[:, 1], s=10, color=colors[(y_pred + 1) // 2])
            plt.xticks(())
            plt.yticks(())
            plt.subplot(2,len(anomaly_algorithms), plot_num+4)

            # calculate evaluation metrics
            roc_auc, prc_auc = self.getAUC(images_test_all_labels, scores)
            fpr,tpr, thresholds = roc_curve(images_test_all_labels, scores)

            #fig, ax = plt.subplots(figsize=(6, 6))
            plt.plot(fpr,tpr)
            plt.title("ROC curve")
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            self.count_OK_NOK(y_pred,images_test_all_labels,name,metric_name)

            plot_num += 1