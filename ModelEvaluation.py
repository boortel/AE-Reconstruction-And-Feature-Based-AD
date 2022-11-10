# -*- coding: utf-8 -*-
"""
Created on Wed Oct  12 17:23:05 2022

@author: Simon Bilik

This class is used for the evaluation of the trained model

"""

import os
import logging
import traceback

import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.io import savemat

from keras.models import Model, load_model

class ModelEvaluation():

    ## Set the constants and paths
    def __init__(self, modelBasePath, datasetPathEv, modelSel, labelInfo, imageDim, batchSizeEv, numEpochEv):

        # Set paths
        self.modelPath = os.path.join(modelBasePath, modelSel + labelInfo)
        self.datasetPath = datasetPathEv

        # Set image dimensions (height, width, channels)
        self.imageDim = imageDim

        # Set batch size and number of epochs
        self.batchSize = batchSizeEv
        self.numEpochEv = numEpochEv

        # Set model name
        self.modelName = modelSel

        # Set image generator and get labels to save in npz
        self.setGenerators()

        # Encode, decode and visualise the training data
        _ = self.dataEncodeDecode('Train')
        _ = self.dataEncodeDecode('Test')


    ## Normalize the input data
    def NormalizeData(self, data):

        return (data - np.min(data)) / (np.max(data) - np.min(data))


    ## Set the image generators and its parameters
    def setGenerators(self):
        
        try:
            # Set the image size
            tSize = (self.imageDim[0], self.imageDim[1])

            if self.imageDim[2] == 3:
                cMode = 'rgb'
            else:
                cMode = 'gray'

            # Define normalization layer
            normalization_layer = tf.keras.layers.Rescaling(1./255)

            # Train DS
            dsTrain = tf.keras.utils.image_dataset_from_directory(
                (self.datasetPath + 'train'),
                image_size = tSize,
                color_mode = cMode,
                batch_size = self.batchSize,
                crop_to_aspect_ratio = True,
                label_mode = None,
                shuffle = False)

            self.dsTrain = dsTrain.map(lambda x: (normalization_layer(x)))

            self.dsTrainL = tf.keras.utils.image_dataset_from_directory(
                (self.datasetPath + 'train'),
                image_size = tSize,
                color_mode = cMode,
                batch_size = self.batchSize,
                crop_to_aspect_ratio = True,
                label_mode = 'int',
                shuffle = False)

            # Test DS
            dsTest = tf.keras.utils.image_dataset_from_directory(
                (self.datasetPath + 'test'),
                image_size = tSize,
                color_mode = cMode,
                batch_size = self.batchSize,
                crop_to_aspect_ratio = True,
                label_mode = None,
                shuffle = False)

            self.dsTest = dsTest.map(lambda x: (normalization_layer(x)))

            self.dsTestL = tf.keras.utils.image_dataset_from_directory(
                (self.datasetPath + 'test'),
                image_size = tSize,
                color_mode = cMode,
                batch_size = self.batchSize,
                crop_to_aspect_ratio = True,
                label_mode = 'int',
                shuffle = False)

        except:
            logging.error(': Loading of the datasets failed...')
            traceback.print_exc()
            return

        else:
            logging.info(': Loading of the datasets was succesful...')


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

            outputPath = os.path.join(self.modelPath, 'modelData', 'Eval_' + actStr)

            # Load the model
            autoencoder = load_model(self.modelPath)
            
            # Get the encoded data
            encoder = Model(inputs = autoencoder.input, outputs = autoencoder.get_layer('enc').output)
            enc_out = self.NormalizeData(encoder.predict(dataset))

            # Get the decoded data
            dec_out = self.NormalizeData(autoencoder.predict(dataset))

            # Get the original data to be saved in npz
            orig_data = np.concatenate([img for img in dataset], axis=0)
            labels = np.concatenate([labels for _, labels in datasetL], axis=0)

            # Get the difference images (org - dec)
            diff_data = np.subtract(orig_data, dec_out)

            # Transform the labels format to -1: NOK and 1:OK
            nokIdx = np.where(labels == 0)
            labels[nokIdx] = -1

            processedData = {'Org': orig_data, 'Enc': enc_out, 'Dec': dec_out, 'Dif': diff_data, 'Lab': labels}
            
            # Save the obtained data to NPZ and MAT
            np.savez_compressed(outputPath, orgData = orig_data, encData = enc_out, decData = dec_out, difData = diff_data, labels = labels)
            #savemat(outputPath + '.mat', processedData, do_compression = True)

            # Visualise the obtained data
            self.visualiseResults(actStr, processedData)
        
        except:
            logging.error(': Data encode and decode for the model ' + self.modelName + ' failed...')
            traceback.print_exc()
            return

        else:
            logging.info(': Data encode and decode for the model ' + self.modelName + ' was succesful...')
        
        return processedData


    ## Visualise the results
    def visualiseResults(self, actStr, processedData):

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
            fig.set_size_inches(16, 16)

            fig.suptitle('Original, encoded and decoded images of the ' + self.modelName + ' autoencoder model' + label, fontsize=14, y=1.08)

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
            
            #axarr[0,1].set_title("Encoded")
            #axarr[0,1].imshow(cv.normalize(enc_out[0], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
            #axarr[0,1].axis('off')
            #axarr[1,1].imshow(cv.normalize(enc_out[10], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
            #axarr[1,1].axis('off')
            #axarr[2,1].imshow(cv.normalize(enc_out[25], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
            #axarr[2,1].axis('off')
            #axarr[3,1].imshow(cv.normalize(enc_out[35], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
            #axarr[3,1].axis('off')

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
            fig.savefig(os.path.join(self.modelPath, 'modelData', self.modelName + actStr + '_AEResults.png'))
        
        except:
            logging.error(': Data visualisation of the model ' + self.modelName + ' and its ' + actStr + ' dataset failed...')
            traceback.print_exc()

        else:
            logging.info(': Data visualisation of the model ' + self.modelName + ' and its ' + actStr + ' dataset was succesful...')