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
from scipy.io import savemat
from functools import partial
from ModelSaved import ModelSaved
from skimage.metrics import structural_similarity as SSIM

AUTOTUNE = tf.data.experimental.AUTOTUNE

class ModelTrainAndEval():

    ## Set the constants and paths
    def __init__(self, modelPath, model, layer, dataGenerator, labelInfo, imageDim, numEpoch, trainFlag, evalFlag):

        # Set model and training parameters
        self.labelInfo = labelInfo
        self.numEpoch = numEpoch
        
        # Set image dimensions
        self.imageDim = imageDim
          
        # Set model name and save path
        self.layerName = layer
        self.modelName = model
        self.modelPath = modelPath
        
        # Set data generator
        self.dataGenerator = dataGenerator

        # Get the model and its type
        modelObj = ModelSaved(self.modelName, self.layerName, self.imageDim, dataVariance = 0.5, intermediateDim = 64, latentDim = 50, num_embeddings = 32)

        self.model = modelObj.model
        self.typeAE = modelObj.typeAE
        
        # Print the separator
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info("Autoencoder architecture name: " + self.layerName + '-' + self.modelName  + '_' + self.labelInfo)
        logging.info('')

        if trainFlag:
            # Set callbacks, train model and visualise the results
            self.setCallbacks()
            self.modelTrain()
        else:
            try:
                # Load the model
                self.model = keras.models.load_model(self.modelPath)
            except:
                logging.error('Desired model: ' + self.layerName + '-' + self.modelName + ' cannot be loaded...')
                traceback.print_exc()
                return

        if evalFlag:
            # Encode, decode and visualise the training data
            self.dataEncodeDecode()


    ## Set callbacks
    def setCallbacks(self):

        try:
            # Configure the early stopping callback
            self.esCallBack = callbacks.EarlyStopping(
                monitor = 'loss', 
                patience = 5)

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
            
            self.model.save(self.modelPath, save_format="hdf5", save_traces = True)

        except:
            logging.error('Training of the ' + self.layerName + '-' + self.modelName + ' model failed...')
            traceback.print_exc()
            return

        else:
            logging.info('Training of the ' + self.layerName + '-' + self.modelName + ' model was finished...')
            self.visualiseTrainResults()
            
            
    ## Get the encoded and decoded data from selected model and dataset
    def dataEncodeDecode(self):

        processedData = {}
        
        actStrs = ['Train', 'Test']
        dataGens = [self.dataGenerator.dsTrain, self.dataGenerator.dsTest]
        
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
                
                # Save the data for visualisation
                processedData = {'Enc': enc_out, 'Dec': dec_out}

                # Save the obtained data to NPZ
                outputPath = os.path.join(self.modelPath, 'modelData', 'Eval_' + actStr)
                np.savez_compressed(outputPath, encData = enc_out, decData = dec_out)

                # Visualise the obtained data
                self.visualiseEncDecResults(actStr, processedData)

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
                
                # Resize the images
                h = imgOrg.shape[0]
                w = imgOrg.shape[1]
                
                # Compute the SSIM metric
                ssimVal.append(SSIM(imgOrg, imgDec, data_range = 1, channel_axis = 2))
                
                # Convert the images to grayscale for the correlation computation
                imgOrg = cv.resize(cv.cvtColor(imgOrg, cv.COLOR_BGR2GRAY), (int(h/8), int(w/8)))
                imgDec = cv.resize(cv.cvtColor(imgDec, cv.COLOR_BGR2GRAY), (int(h/8), int(w/8)))
                
                # Compute Pearson Coefficient
                res = stats.pearsonr(imgOrg.flatten(), imgDec.flatten())
                pVal.append(res.statistic)
            
            # Compute average SSIM
            ssimVal = np.average(np.array(ssimVal))
            ssimAvg.append(ssimVal)
            
            # Compute average p-value
            pVal = np.average(np.array(pVal))
            pAvg.append(pVal)
            
            logging.info('Average Pearson Coefficient: ' + f'{float(pVal):.2f}' + ' for class ' + classLb)
            logging.info('Average SSIM value: ' + f'{float(ssimVal):.2f}' + ' for class ' + classLb)
        
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
            
    
    ## Visualise the results
    def visualiseTrainResults(self):

        try:
            # Plot the history and save the curves
            train_loss = self.trainHistory.history['loss']
            
            if self.typeAE == 'VAE1' or self.typeAE == 'VAE2':
                val_loss = self.trainHistory.history['kl_loss']
                plotLabel = 'KL loss [-]'
            elif self.typeAE == 'VQVAE1':
                val_loss = self.trainHistory.history['vqvae_loss']
                plotLabel = 'VQ-VAE loss [-]'
            else:
                val_loss = self.trainHistory.history['val_loss']
                plotLabel = 'Validation loss [-]'
            
            fig, axarr = plt.subplots(2)
            tempTitle = " Training and Validation Loss of " + self.layerName + '-' + self.modelName + '_' + self.labelInfo + " model."
            fig.suptitle(tempTitle, fontsize=14, y=1.08)
            
            axarr[0].plot(train_loss)
            axarr[0].set(xlabel = 'Number of Epochs', ylabel = 'Training Loss [-]')
            
            axarr[1].plot(val_loss)
            axarr[1].set(xlabel = 'Number of Epochs', ylabel = plotLabel)
            
            fig.tight_layout()
            fig.savefig(os.path.join(self.modelPath, 'modelData', self.layerName + '-' + self.modelName + '_' + self.labelInfo + '_TrainLosses.png'))
        
        except:
            logging.error('Visualisation of the ' + self.modelName + ' model training results failed...')
            traceback.print_exc()

        else:
            logging.info('Visualisation of the ' + self.modelName + ' model training results was finished...')
            
            
    ## Visualise the results
    def visualiseEncDecResults(self, actStr, processedData):

        # TODO: pridat parametr, ktery urci vykreslovane vzorky
        try:
            # Set the train or test data
            if actStr == 'Train':
                label = ' during training.'
            else:
                label = ' during testing.'
                
            # Get the original data
            tempData = self.dataGenerator.processedData.get(actStr)
            orig_data = tempData.get('Org')
            
            # Get the decoded data
            enc_out = processedData.get('Enc')
            dec_out = processedData.get('Dec')
            
            # Compute the difference images (org - dec)
            diff_data = np.subtract(orig_data, dec_out)

            # Plot the encoded samples from all classes
            fig, axarr = plt.subplots(4,4)
            tempTitle = ' Original, encoded and decoded images of the ' + self.layerName + '-' + self.modelName + '_' + self.labelInfo + ' autoencoder model, ' + label + '.'
            
            fig.suptitle(tempTitle, fontsize=14, y=1.08)
            fig.set_size_inches(16, 16)

            # Cookie: 27, 35, 205
            
            axarr[0,0].set_title("Original")
            axarr[0,0].imshow(cv.normalize(orig_data[0], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
            axarr[0,0].axis('off')
            axarr[1,0].imshow(cv.normalize(orig_data[10], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
            axarr[1,0].axis('off')
            axarr[2,0].imshow(cv.normalize(orig_data[25], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
            axarr[2,0].axis('off')
            axarr[3,0].imshow(cv.normalize(orig_data[35], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
            axarr[3,0].axis('off')
            
            axarr[0,1].set_title("Encoded")
            
            if self.typeAE == 'VAE1' or self.typeAE == 'VAE2':
                axarr[0,1].scatter(enc_out[0, :, 0], enc_out[0, :, 1], s = 4)
                axarr[0,1].set(xlabel = "Mean", ylabel = "Variance")
                axarr[1,1].scatter(enc_out[10, :, 0], enc_out[10, :, 1], s = 4)
                axarr[1,1].set(xlabel = "Mean", ylabel = "Variance")
                axarr[2,1].scatter(enc_out[25, :, 0], enc_out[25, :, 1], s = 4)
                axarr[2,1].set(xlabel = "Mean", ylabel = "Variance")
                axarr[3,1].scatter(enc_out[35, :, 0], enc_out[35, :, 1], s = 4)
                axarr[3,1].set(xlabel = "Mean", ylabel = "Variance")
            elif self.typeAE == 'BAE1' or self.typeAE == 'BAE2':
                pass
            else:
                axarr[0,1].imshow(cv.normalize(enc_out[0], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
                axarr[0,1].axis('off')
                axarr[1,1].imshow(cv.normalize(enc_out[10], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
                axarr[1,1].axis('off')
                axarr[2,1].imshow(cv.normalize(enc_out[25], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
                axarr[2,1].axis('off')
                axarr[3,1].imshow(cv.normalize(enc_out[35], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
                axarr[3,1].axis('off')

            axarr[0,2].set_title("Decoded")
            axarr[0,2].imshow(cv.normalize(dec_out[0], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
            axarr[0,2].axis('off')
            axarr[1,2].imshow(cv.normalize(dec_out[10], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
            axarr[1,2].axis('off')
            axarr[2,2].imshow(cv.normalize(dec_out[25], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
            axarr[2,2].axis('off')
            axarr[3,2].imshow(cv.normalize(dec_out[35], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
            axarr[3,2].axis('off')

            axarr[0,3].set_title("Diff image")
            axarr[0,3].imshow(cv.normalize(diff_data[0], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
            axarr[0,3].axis('off')
            axarr[1,3].imshow(cv.normalize(diff_data[10], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
            axarr[1,3].axis('off')
            axarr[2,3].imshow(cv.normalize(diff_data[25], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
            axarr[2,3].axis('off')
            axarr[3,3].imshow(cv.normalize(diff_data[35], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
            axarr[3,3].axis('off')

            # Save the illustration figure
            fig.savefig(os.path.join(self.modelPath, 'modelData', self.layerName + '-' + self.modelName + '_' + self.labelInfo + '_' + actStr + '_AEResults.png'))
        
        except:
            logging.error('Data visualisation of the model ' + self.layerName + '-' + self.modelName + '_' + self.labelInfo + ' and its ' + actStr + ' dataset failed...')
            traceback.print_exc()

        else:
            logging.info('Data visualisation of the model ' + self.layerName + '-' + self.modelName + ' and its ' + actStr + ' dataset was succesful...')