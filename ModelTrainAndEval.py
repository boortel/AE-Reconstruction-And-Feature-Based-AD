# -*- coding: utf-8 -*-
"""
Created on Thurs Oct 13 2022

@author: Simon Bilik

This class is used for training and evaluation of the selected model

"""

import os
import logging
import traceback
import keras.models

import cv2 as cv
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from scipy import stats

from keras import callbacks

from skimage.util import view_as_blocks
from skimage.metrics import structural_similarity as SSIM

from ModelSaved import ModelSaved

AUTOTUNE = tf.data.experimental.AUTOTUNE

class ModelTrainAndEval():

    ## Set the constants and paths
    def __init__(self, modelPath, model, layer, dataGenerator, labelInfo, imageDim, imIndxList, numEpoch, evalFlag, npzSave):

        # Set model and training parameters
        self.labelInfo = labelInfo
        self.numEpoch = numEpoch
        
        # Set image dimensions and plot indices
        self.imageDim = imageDim
        self.imIndxList = imIndxList
          
        # Set model name and save path
        self.layerName = layer
        self.modelName = model
        self.modelPath = modelPath
        self.npzSave = npzSave
        
        # Set data generator
        self.dataGenerator = dataGenerator

        # Get the model and its type
        modelObj = ModelSaved(self.modelName, self.layerName, self.imageDim, dataVariance = 0.5, intermediateDim = 64, latentDim = 32, num_embeddings = 32)

        self.model = modelObj.model
        self.typeAE = modelObj.typeAE
        
        # Print the separator
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info("Autoencoder architecture name: " + self.layerName + '-' + self.modelName  + '_' + self.labelInfo)
        logging.info('')

        # Set callbacks, train model and visualise the results
        self.setCallbacks()
        self.modelTrain()

        if evalFlag:
            # Encode, decode and visualise the training data
            self.dataEncodeDecode()


    ## Set callbacks
    def setCallbacks(self):

        try:
            # Configure the early stopping callback
            if self.typeAE == 'VAE1' or self.typeAE == 'VAE2' or self.typeAE == 'VQVAE1':

                self.esCallBack = callbacks.EarlyStopping(
                monitor = 'total_loss', 
                patience = 10)
                
            else:
                
                self.esCallBack = callbacks.EarlyStopping(
                    monitor = 'loss', 
                    patience = 10)

        except:
            logging.error('Callback initialization of the ' + self.layerName + '-' + self.modelName + ' model failed...')
            traceback.print_exc()
            return

        else:
            logging.info('Callback of the ' + self.layerName + '-' + self.modelName + ' model initialized...')
            

    ## Compile and train the model
    def modelTrain(self):

        try:
            # Set the validation DS
            if self.typeAE == 'VAE1' or self.typeAE == 'VAE2' or self.typeAE == 'VQVAE1':
                valDS = None
            else:
                valDS = self.dataGenerator.dsValid
            
            # Train the model
            self.trainHistory = self.model.fit(
                x = self.dataGenerator.dsTrain,
                epochs = self.numEpoch,   
                validation_data = valDS, 
                verbose = 1,
                callbacks = [self.esCallBack])
            
            #self.model.save(self.modelPath, save_format="hdf5", save_traces = True)
            #self.model.save(os.path.join(self.modelPath, 'model.keras'))
            self.model.save_weights(os.path.join(self.modelPath, 'model.weights.h5'))

        except:
            logging.error('Training of the ' + self.layerName + '-' + self.modelName + ' model failed...')
            traceback.print_exc()
            return

        else:
            logging.info('Training of the ' + self.layerName + '-' + self.modelName + ' model was finished...')
            self.visualiseTrainResults()
            
            
    ## Get the encoded and decoded data from selected model and dataset
    def dataEncodeDecode(self):
        
        actStrs = ['Train', 'Test', 'Valid']
        dataGens = [self.dataGenerator.dsTrain, self.dataGenerator.dsTest, self.dataGenerator.dsValid]
        
        for actStr, dataGen in zip(actStrs, dataGens):

            try:
                # Get the encoded data
                encoder = keras.models.Model(inputs = self.model.input, outputs = self.model.get_layer('enc').output)

                if self.typeAE == 'VAE1' or self.typeAE == 'VAE2':
                    z_mean, z_log_var, _ = encoder.predict(dataGen)
                    enc_out = np.dstack((z_mean, z_log_var))
                elif self.typeAE == 'VQVAE1':
                    quantizer = self.model.get_layer("vector_quantizer")
                    encoded_outputs = encoder.predict(dataGen)

                    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
                    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
                    enc_out = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
                else:
                    enc_out = encoder.predict(dataGen)

                # Get the decoded data
                dec_out = self.model.predict(dataGen)
                
                # Normalize the decoded data
                for i in range(dec_out.shape[0]):
                    dec_out[i] = np.atleast_3d(cv.normalize(dec_out[i], None, 0, 1, cv.NORM_MINMAX, cv.CV_32F))
                
                # Save the data for visualisation
                self.dataGenerator.processedData[actStr]['Enc'] = enc_out
                self.dataGenerator.processedData[actStr]['Dec'] = dec_out

                # Save the obtained data to NPZ
                if self.npzSave:
                    outputPath = os.path.join(self.modelPath, 'modelData', 'Eval_' + actStr)
                    np.savez_compressed(outputPath, encData = enc_out, decData = dec_out)

                # Visualise the obtained data
                if actStr == 'Test':
                    self.visualiseEncDecResults(actStr)

                # Get the Pearson correlation coeff.
                if actStr == 'Test':
                    self.getSimilarityCoeff(dec_out)

            except:
                logging.error('Data encode and decode for the model ' + self.layerName + '-' + self.modelName + ' failed...')
                traceback.print_exc()
                return

        else:
            logging.info('Data encode and decode for the model ' + self.layerName + '-' + self.modelName + ' was succesful...')
            
            
    ## Get Pearson correlation coefficient and SSIM metric
    def getSimilarityCoeff(self, decData):
        
        classIDs = [-1, 1]
        classLab = ['NOK', 'OK']
        
        pAvg = []
        ssimAvg = []
        
        # Get the original data
        tempData = self.dataGenerator.processedData.get('Test')
        orgData = tempData.get('Org')
        labels = tempData.get('Lab')
        
        for classID, classLb in zip(classIDs, classLab):
            # Get IDs and data
            idx = np.where(labels == classID)
            
            orgDataSel = orgData[idx[0], :, :, :]
            decDataSel = decData[idx[0], :, :, :]
            
            pVal = []
            ssimVal = []
            
            # Loop through the test images
            for imgOrg, imgDec in zip(orgDataSel, decDataSel):
                
                # Compute the SSIM metric
                ssimVal.append(SSIM(imgOrg, imgDec, data_range = 1, channel_axis = 2))
                
                # Convert images to gray
                if(imgOrg.shape[2] == 3):
                    imgOrg = cv.cvtColor(imgOrg, cv.COLOR_BGR2GRAY)
                    imgDec = cv.cvtColor(imgDec, cv.COLOR_BGR2GRAY)
                else:
                    imgOrg = np.squeeze(imgOrg)
                    imgDec = np.squeeze(imgDec)
                
                # Split the image into 32x32 images
                batchOrg = view_as_blocks(imgOrg, (32, 32))
                batchOrg = batchOrg.reshape(batchOrg.shape[0]*batchOrg.shape[1], 32, 32)
                
                batchDec = view_as_blocks(imgDec, (32, 32))
                batchDec = batchDec.reshape(batchDec.shape[0]*batchDec.shape[1], 32, 32)
                
                temppVal = []
                
                for subImgOrg, subImgDec in zip(batchOrg, batchDec):
                    
                    # Compute Pearson Coefficient
                    pears = stats.pearsonr(subImgOrg.flatten(), subImgDec.flatten())
                    if not np.isnan(pears.statistic):
                        temppVal.append(np.abs(pears.statistic))
                    
                pVal.append(np.median(np.array(temppVal)))
            
            # Compute median SSIM
            ssimVal = np.median(np.array(ssimVal))
            ssimAvg.append(ssimVal)
            
            # Compute median p-value
            pVal = np.median(np.array(pVal))
            pAvg.append(pVal)
            
            logging.info('Median Pearson Coefficient: ' + f'{float(pVal):.2f}' + ' for class ' + classLb)
            logging.info('Median SSIM value: ' + f'{float(ssimVal):.2f}' + ' for class ' + classLb)
        
        # Compute the ratio between the Pearson coeffs and SSIM by the OK and NOK data
        if pAvg[1] == 0:
            pRatio = 0
        else:
            pRatio = pAvg[0]/pAvg[1]
            
        if ssimAvg[1] == 0:
            ssimRatio = 0
        else:
            ssimRatio = ssimAvg[0]/ssimAvg[1]

        logging.info('Pearson Coefficient ratio: ' + f'{float(pRatio):.2f}' + ' for model ' + self.layerName + '-' + self.modelName)
        logging.info('SSIM ratio: ' + f'{float(ssimRatio):.2f}' + ' for model ' + self.layerName + '-' + self.modelName)
        
    
    ## Return original, encoded and decoded data with labels
    def returnProcessedData(self):
        return self.dataGenerator.processedData
            
    
    ## Visualise the results
    def visualiseTrainResults(self):

        try:
            # Plot the history and save the curves
            
            if self.typeAE == 'VAE1' or self.typeAE == 'VAE2':
                train_loss = self.trainHistory.history['total_loss']
                val_loss = self.trainHistory.history['kl_loss']
                plotLabel = 'KL loss [-]'
                tempTitle = 'Training and KL Loss of ' + self.layerName + '-' + self.modelName + '_' + self.labelInfo + ' model'
            elif self.typeAE == 'VQVAE1':
                train_loss = self.trainHistory.history['total_loss']
                val_loss = self.trainHistory.history['vqvae_loss']
                plotLabel = 'VQ-VAE loss [-]'
                tempTitle = 'Training and VQ-VAE Loss of ' + self.layerName + '-' + self.modelName + '_' + self.labelInfo + ' model'
            else:
                val_loss = self.trainHistory.history['val_loss']
                train_loss = self.trainHistory.history['loss']
                plotLabel = 'Validation loss [-]'
                tempTitle = 'Training and Validation Loss of ' + self.layerName + '-' + self.modelName + '_' + self.labelInfo + ' model'
            
            fig, axarr = plt.subplots(2)
            fig.suptitle(tempTitle, fontsize=14)
            
            axarr[0].plot(train_loss)
            axarr[0].set(xlabel = 'Number of Epochs', ylabel = 'Training Loss [-]')
            
            axarr[1].plot(val_loss)
            axarr[1].set(xlabel = 'Number of Epochs', ylabel = plotLabel)
            
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            
            fig.savefig(os.path.join(self.modelPath, 'modelData', self.layerName + '-' + self.modelName + '_' + self.labelInfo + '_TrainLosses.png'))
        
        except:
            logging.error('Visualisation of the ' + self.modelName + ' model training results failed...')
            traceback.print_exc()

        else:
            logging.info('Visualisation of the ' + self.modelName + ' model training results was finished...')
            
            
    ## Visualise the results
    def visualiseEncDecResults(self, actStr):

        # TODO: pridat parametr, ktery urci vykreslovane vzorky
        try:
            # Set the train or test data
            if actStr == 'Train':
                label = 'during training'
            else:
                label = 'during testing'
            
            # Get the original, encoded and decoded data
            tempData = self.dataGenerator.processedData.get(actStr)
            
            orgData = tempData.get('Org')
            encData = tempData.get('Enc')
            decData = tempData.get('Dec')
            
            # Compute the difference images (org - dec)
            diffData = np.subtract(orgData, decData)
            
            # Define the image source and title lists
            imgSourceList = [orgData, encData, decData, diffData]
            imgTitleList = ['Original', 'Encoded', 'Decoded', 'Difference']

            # Plot the encoded samples from all classes
            fig, axarr = plt.subplots(len(self.imIndxList), 4)
            tempTitle = 'Visualisations of the ' + self.layerName + '-' + self.modelName + '_' + self.labelInfo + ' model'
            
            fig.suptitle(tempTitle, fontsize=18)
            fig.set_size_inches(4 * len(self.imIndxList), 16)
            
            vIdx = 0
            
            # Loop through selected images for plot
            for imgIndx in self.imIndxList:
                
                hIdx = 0
                
                # Loop through org, enc, dec and diff data
                for imgTitle, imgSource in zip(imgTitleList, imgSourceList):
                    
                    axarr[vIdx, hIdx].set_title(imgTitle)
                    
                    # Plot the encoded images
                    if imgTitle == 'Encoded':
                        
                        if self.typeAE == 'VAE1' or self.typeAE == 'VAE2':
                            axarr[vIdx, hIdx].scatter(imgSource[imgIndx, :, 0], imgSource[imgIndx, :, 1], s = 4)
                            axarr[vIdx, hIdx].set(xlabel = "Mean", ylabel = "Variance", xlim = (-10, 10), ylim = (-10, 10))
                            
                        elif self.typeAE == 'BAE1' or self.typeAE == 'BAE2':
                            axarr[vIdx, hIdx].imshow(cv.normalize(imgSource[imgIndx].mean(axis=2), None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
                            axarr[vIdx, hIdx].axis('off')
                            
                        elif self.typeAE == 'VQVAE1':
                            axarr[vIdx, hIdx].imshow(cv.normalize(imgSource[imgIndx], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
                            axarr[vIdx, hIdx].axis('off')
                    
                    # Plot the original, decoded and diff images
                    else:
                        axarr[vIdx, hIdx].imshow(cv.normalize(imgSource[imgIndx], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
                        axarr[vIdx, hIdx].axis('off')
                        
                    hIdx += 1
                    
                vIdx += 1

            # Save the illustration figure
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            
            fig.savefig(os.path.join(self.modelPath, 'modelData', self.layerName + '-' + self.modelName + '_' + self.labelInfo + '_' + actStr + '_AEResults.png'))
        
        except:
            logging.error('Data visualisation of the model ' + self.layerName + '-' + self.modelName + '_' + self.labelInfo + ' and its ' + actStr + ' dataset failed...')
            traceback.print_exc()

        else:
            logging.info('Data visualisation of the model ' + self.layerName + '-' + self.modelName + ' and its ' + actStr + ' dataset was succesful...')