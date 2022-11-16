# -*- coding: utf-8 -*-
"""
Created on Thurs Oct 13 2022

@author: Simon Bilik

This class is used for training and evaluation of the selected model

"""

import os
import logging
import traceback

import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import stats
from keras import callbacks
from scipy.io import savemat
from functools import partial
from ModelSaved import ModelSaved
from keras.models import Model, load_model
from albumentations import Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip, Rotate

AUTOTUNE = tf.data.experimental.AUTOTUNE

class ModelTrainAndEval():

    ## Set the constants and paths
    def __init__(self, modelPath, datasetPath, model, layer, labelInfo, imageDim, batchSize, numEpoch, trainFlag, evalFlag):
        
        # Set paths
        self.datasetPath = datasetPath

        # Set model and training parameters
        self.labelInfo = labelInfo
        self.numEpoch = numEpoch
        self.batchSize = batchSize
        
        # Set image dimensions
        self.imageDim = imageDim
          
        # Set model name and save path
        self.layerName = layer
        self.modelName = model
        self.modelPath = modelPath

        # Get the model and its type
        modelObj = ModelSaved(self.modelName, self.layerName, self.imageDim, dataVariance = 0.5, intermediateDim = 64, latentDim = 50, num_embeddings = 32)

        self.model = modelObj.model
        self.typeAE = modelObj.typeAE

        # Set generators
        self.getGenerators()
        
        # Print the separator
        print('-----------------------------------------------')
        print("Autoencoder architecture name: " + self.layerName + '-' + self.modelName  + '_' + self.labelInfo)
        print('')
        
        logging.info('-----------------------------------------------')
        logging.info("Autoencoder architecture name: " + self.layerName + '-' + self.modelName  + '_' + self.labelInfo)
        logging.info('')

        if trainFlag:
            # Set callbacks, train model and visualise the results
            self.setCallbacks()
            self.modelTrain()
        else:
            try:
                # Load the model
                self.model = load_model(self.modelPath)
            except:
                logging.error('Desired model: ' + self.layerName + '-' + self.modelName + ' cannot be loaded...')
                traceback.print_exc()
                return

        if evalFlag:
            # Encode, decode and visualise the training data
            self.dataEncodeDecode('Train')
            self.dataEncodeDecode('Test')
        
    
    ## Normalize the input data (Evaluation)
    def NormalizeData(self, data):

        return (data - np.min(data)) / (np.max(data) - np.min(data))

    
    ## Normalize dataset with no labels
    def changeInputsVAE(self, images):
        
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        x = tf.image.resize(normalization_layer(images),[self.imageDim[0], self.imageDim[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        return x
    
    
    ## Normalize dataset with labels
    def changeInputsAE(self, images, labels):
        
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        x = tf.image.resize(normalization_layer(images),[self.imageDim[0], self.imageDim[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        return x, x
    
    
    ## Normalize dataset with labels
    def changeInputs(self, images, labels):
        
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        x = tf.image.resize(normalization_layer(images),[self.imageDim[0], self.imageDim[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        return x, labels
    
    
    ## Set the data generator
    def setGenerator(self, mode):
        
        if self.typeAE == 'VAE1' or self.typeAE == 'VAE2' or self.typeAE == 'VQVAE1' or mode == 'test':
            labelMode = None
        else:
            labelMode = 'int'
            
        # Returns generator with and without the labels
        ds = tf.keras.utils.image_dataset_from_directory(
            (self.datasetPath + mode),
            image_size = self.tSize,
            color_mode = self.cMode,
            batch_size = self.batchSize,
            crop_to_aspect_ratio = True,
            label_mode = labelMode,
            shuffle = False)

        dsL = tf.keras.utils.image_dataset_from_directory(
            (self.datasetPath + mode),
            image_size = self.tSize,
            color_mode = self.cMode,
            batch_size = self.batchSize,
            crop_to_aspect_ratio = True,
            label_mode = 'int',
            shuffle = False)
        
        dsL = dsL.map(self.changeInputs)
        
        if self.typeAE == 'VAE1' or self.typeAE == 'VAE2' or self.typeAE == 'VQVAE1' or mode == 'test':
            ds = ds.map(self.changeInputsVAE)
        else:
            ds = ds.map(self.changeInputsAE)
        
        return ds, dsL
    
    
    ## Get the data generators
    def getGenerators(self):

        try:
            # Set image dimensions
            self.tSize = (self.imageDim[0], self.imageDim[1])

            if self.imageDim[2] == 3:
                self.cMode = 'rgb'
            else:
                self.cMode = 'grayscale'

            # Get train DS
            self.dsTrain, self.dsTrainL =  self.setGenerator('train')
            
            # Get validation DS
            self.dsValid, self.dsValidL =  self.setGenerator('valid')
            
            # Get test DS
            self.dsTest, self.dsTestL =  self.setGenerator('test')
            
        except:
            logging.error('Data generators initialization of the ' + self.layerName + '-' + self.modelName + ' model failed...')
            traceback.print_exc()
            return

        else:
            logging.info('Data generators of the ' + self.layerName + '-' + self.modelName + ' model initialized...')


    ## Set callbacks
    def setCallbacks(self):

        try:
            # Configure the early stopping callback
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
                valDS = self.dsValid
            
            # Train the model
            self.trainHistory = self.model.fit(
                x = self.dsTrain,
                epochs = self.numEpoch,   
                validation_data = valDS, 
                verbose = 1,
                callbacks = [self.esCallBack])
            
            self.model.save(self.modelPath)

        except:
            logging.error('Training of the ' + self.layerName + '-' + self.modelName + ' model failed...')
            traceback.print_exc()
            return

        else:
            logging.info('Training of the ' + self.layerName + '-' + self.modelName + ' model was finished...')
            self.visualiseTrainResults()
            
            
    ## Get the encoded and decoded data from selected model and dataset
    def dataEncodeDecode(self, actStr):

        processedData = {}

        try:
            # Set the train or test data
            if actStr == 'Train':
                dataset = self.dsTrain
                datasetL = self.dsTrainL
            else:
                dataset = self.dsTest
                datasetL = self.dsTestL
            
            # Get the encoded data
            encoder = Model(inputs = self.model.input, outputs = self.model.get_layer('enc').output)
            
            if self.typeAE == 'VAE1' or self.typeAE == 'VAE2':
                z_mean, z_log_var, _ = encoder.predict(dataset)
                enc_out = np.dstack((z_mean, z_log_var))
            elif self.typeAE == 'VQVAE1':
                quantizer = self.model.get_layer("vector_quantizer")
                encoded_outputs = encoder.predict(dataset)
                
                flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
                codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
                enc_out = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
            else:
                enc_out = self.NormalizeData(encoder.predict(dataset))

            # Get the decoded data
            dec_out = self.NormalizeData(self.model.predict(dataset))

            # Get the original data to be saved in npz
            orig_data = self.NormalizeData(np.concatenate([img for img, _ in datasetL], axis=0))

            # TODO: presunout k error mtr
            filterStrength = 0.5
            orig_data = np.array([(filterStrength*img + (1 - filterStrength)*cv.GaussianBlur(img, (25,25), 0)) for img in orig_data])

            # Get the difference images (org - dec)
            diff_data = np.subtract(orig_data, dec_out)

            # Get labels and transform them to format to -1: NOK and 1:OK
            labels = np.concatenate([labels for _, labels in datasetL], axis=0)
            nokIdx = np.where(labels == 0)
            labels[nokIdx] = -1

            processedData = {'Org': orig_data, 'Enc': enc_out, 'Dec': dec_out, 'Dif': diff_data, 'Lab': labels}
            
            # Save the obtained data to NPZ and MAT
            outputPath = os.path.join(self.modelPath, 'modelData', 'Eval_' + actStr)
            np.savez_compressed(outputPath, orgData = orig_data, encData = enc_out, decData = dec_out, difData = diff_data, labels = labels)

            # Visualise the obtained data
            self.visualiseEncDecResults(actStr, processedData)
            
            # Get the Pearson correlation coeff.
            if actStr == 'Test':
                self.getPearsonCoeff(orig_data, dec_out, labels)
        
        except:
            logging.error('Data encode and decode for the model ' + self.layerName + '-' + self.modelName + ' failed...')
            traceback.print_exc()
            return

        else:
            logging.info('Data encode and decode for the model ' + self.layerName + '-' + self.modelName + ' was succesful...')
            
            
    ## Get Pearson correlation coefficient
    def getPearsonCoeff(self, orgData, decData, labels):
        
        classIDs = [-1, 1]
        classLab = ['NOK', 'OK']
        
        for classID, classLb in zip(classIDs, classLab):
            # Get IDs and data
            idx = np.where(labels == classID)
            
            orgDataSel = orgData[idx]
            decDataSel = decData[idx]
            
            pVal = []
            
            # Loop through the test images
            for imgOrg, imgDec in zip(orgDataSel, decDataSel):
                
                # Resize the images
                (h, w) = imgOrg.shape[:2]
                
                imgOrg = cv.resize(cv.cvtColor(imgOrg, cv.COLOR_BGR2GRAY), (int(h/8), int(w/8)))
                imgDec = cv.resize(cv.cvtColor(imgDec, cv.COLOR_BGR2GRAY), (int(h/8), int(w/8)))
                
                # Compute Pearson Coefficient
                res = stats.pearsonr(imgOrg.flatten(), imgDec.flatten())
                pVal.append(res.pvalue)
                
            # Compute average p-value
            pVal = np.average(np.array(pVal))
            
            print('Average Pearson Coefficient: ' + f'{float(pVal):.2f}' + ' for class ' + classLb)
            
            logging.info('Average Pearson Coefficient: ' + f'{float(pVal):.2f}' + ' for class ' + classLb)
            
    
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

            orig_data = processedData.get('Org')
            enc_out = processedData.get('Enc')
            dec_out = processedData.get('Dec')
            diff_data = processedData.get('Dif')

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