# -*- coding: utf-8 -*-
"""
Created on Wed Oct  12 17:23:05 2022

@author: Simon Bilik

This class is used for the evaluation of the trained model

"""

import os
import sys
import logging
import traceback
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import savemat

from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator

class ModelEvaluation():

    ## Set the constants and paths
    def __init__(self, modelBasePath, datasetPathEv, labelsPath, modelSel, labelInfo, imageDim, batchSizeEv, numEpochEv):

        # Set paths
        self.labelsPath = labelsPath
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
        self.getLabels()

        # Encode, decode and visualise the training data
        _ = self.dataEncodeDecode('Train')
        _ = self.dataEncodeDecode('Test')

    ## Normalize the input data
    def NormalizeData(self, data):

        return (data - np.min(data)) / (np.max(data) - np.min(data))
    

    ## Read all labels from the npz file
    def getLabels(self):
        try:
            # Load npz data
            data = np.load(self.labelsPath)

            try:
                self.train_label = data['train']
                self.valid_label = data['valid']
                self.test_label = data['test']
            except:
                print('Loading data should be numpy array and has "images" and "labels" keys.')
                sys.exit(1)
        except:
            logging.error(': Loading the labels failed...')
            traceback.print_exc()

        else:
            logging.info(': Loading the labels was succesful...')


    ## Set the image generators and its parameters
    def setGenerators(self):
        
        try:
            tSize = (self.imageDim[0], self.imageDim[1])

            if self.imageDim[2] == 3:
                cMode = 'rgb'
            else:
                cMode = 'gray'

            train_datagen = ImageDataGenerator(
                rescale = 1./255)
                # samplewise_center = True,
                # samplewise_std_normalization = True)
            
            test_datagen = ImageDataGenerator(
                rescale = 1./255)
                # samplewise_center = True,
                # samplewise_std_normalization = True)
            
            # Data generator for train data
            self.train_generator = train_datagen.flow_from_directory(
                (self.datasetPath + 'train'), 
                target_size = tSize, 
                batch_size = self.batchSize,
                color_mode = cMode,
                class_mode = 'input',
                shuffle = False)

            self.train_generatorL = train_datagen.flow_from_directory(
                (self.datasetPath + 'train'), 
                target_size = tSize, 
                batch_size = self.batchSize,
                color_mode = cMode,
                class_mode = 'binary',
                shuffle = False)
            
            # Data generator for test data
            self.test_generator = test_datagen.flow_from_directory(
                (self.datasetPath + 'test'), 
                target_size = tSize, 
                batch_size = self.batchSize,
                color_mode = cMode,
                class_mode = 'input',
                shuffle = False)

            self.test_generatorL = test_datagen.flow_from_directory(
                (self.datasetPath + 'test'), 
                target_size = tSize, 
                batch_size = self.batchSize,
                color_mode = cMode,
                class_mode = 'binary',
                shuffle = False)

        except:
            logging.error(': Setting up the data generators failed...')
            traceback.print_exc()

        else:
            logging.info(': Setting up the data generators was succesful...')


    ## Get the encoded and decoded data from selected model and dataset
    def dataEncodeDecode(self, actStr):

        processedData = {}

        try:
            # Set the train or test data
            if actStr == 'Train':
                data_generator = self.train_generator
                data_generatorL = self.train_generatorL
                #labels = self.train_label
            else:
                data_generator = self.test_generator
                data_generatorL = self.test_generatorL
                #labels = self.test_label

            outputPath = os.path.join(self.modelPath, 'modelData', 'Eval_' + actStr)

            # Load the model
            autoencoder = load_model(self.modelPath)
            
            # Get the encoded data
            encoder = Model(inputs = autoencoder.input, outputs = autoencoder.get_layer('enc').output)
            enc_out = self.NormalizeData(encoder.predict(data_generator))

            # Get the decoded data
            dec_out = self.NormalizeData(autoencoder.predict(data_generator))

            # Get the original data to be saved in npz
            data_generator.reset()
            orig_data = np.concatenate([data_generator.next()[0] for i in range(data_generator.__len__())])
            labels = np.concatenate([data_generatorL.next()[1] for i in range(data_generatorL.__len__())])

            # Get the difference images (org - dec)
            diff_data = np.subtract(orig_data, dec_out)

            # Transform the labels format to -1: NOK and 1:OK
            nokIdx = np.where(labels == 0)
            labels[nokIdx] = -1

            processedData = {'Org': orig_data, 'Enc': enc_out, 'Dec': dec_out, 'Dif': diff_data, 'Lab': labels}
            
            # Save the obtained data to NPZ and MAT
            np.savez(outputPath, orgData = orig_data, encData = enc_out, decData = dec_out, difData = diff_data, labels = labels)
            savemat(outputPath + '.mat', processedData, do_compression = True)

            # Visualise the obtained data
            self.visualiseResults(actStr, processedData)
        
        except:
            logging.error(': Data encode and decode for the model ' + self.modelName + ' failed...')
            traceback.print_exc()

        else:
            logging.info(': Data encode and decode for the model ' + self.modelName + ' was succesful...')
        
        return processedData


    ## Visualise the results
    def visualiseResults(self, actStr, processedData):
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

            fig.suptitle('Original, encoded and decoded images of the ' + self.modelName + ' autoencoder model' + label)
            
            axarr[0,0].set_title("Original")
            axarr[0,0].imshow(orig_data[0])
            axarr[0,0].axis('off')
            axarr[1,0].imshow(orig_data[27])
            axarr[1,0].axis('off')
            axarr[2,0].imshow(orig_data[35])
            axarr[2,0].axis('off')
            axarr[3,0].imshow(orig_data[205])
            axarr[3,0].axis('off')
            
            axarr[0,1].set_title("Encoded")
            axarr[0,1].imshow(enc_out[0])
            axarr[0,1].axis('off')
            axarr[1,1].imshow(enc_out[27])
            axarr[1,1].axis('off')
            axarr[2,1].imshow(enc_out[35])
            axarr[2,1].axis('off')
            axarr[3,1].imshow(enc_out[205])
            axarr[3,1].axis('off')

            axarr[0,2].set_title("Decoded")
            axarr[0,2].imshow(dec_out[0])
            axarr[0,2].axis('off')
            axarr[1,2].imshow(dec_out[27])
            axarr[1,2].axis('off')
            axarr[2,2].imshow(dec_out[35])
            axarr[2,2].axis('off')
            axarr[3,2].imshow(dec_out[205])
            axarr[3,2].axis('off')

            axarr[0,3].set_title("Diff image")
            axarr[0,3].imshow(diff_data[0])
            axarr[0,3].axis('off')
            axarr[1,3].imshow(diff_data[27])
            axarr[1,3].axis('off')
            axarr[2,3].imshow(diff_data[35])
            axarr[2,3].axis('off')
            axarr[3,3].imshow(diff_data[205])
            axarr[3,3].axis('off')

            # Save the illustration figure
            fig.savefig(os.path.join(self.modelPath, 'modelData', self.modelName + actStr + '_AEResults.png'))
        
        except:
            logging.error(': Data visualisation of the model ' + self.modelName + ' and its ' + actStr + ' dataset failed...')
            traceback.print_exc()

        else:
            logging.info(': Data visualisation of the model ' + self.modelName + ' and its ' + actStr + ' dataset was succesful...')