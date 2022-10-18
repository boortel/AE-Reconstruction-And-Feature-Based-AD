# -*- coding: utf-8 -*-
"""
Created on Wed Oct  12 17:23:05 2022

@author: Simon Bilik

This class is used for the evaluation of the trained model

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as ssim

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

        # Set actions
        self.action = ('Train', 'Test')

        # Set model name
        self.modelName = modelSel

        # Set image generator and get labels to save in npz
        self.setGenerators()
        self.getLabels()

        # Encode, decode and visualise the data
        self.dataEncodeDecode()
        self.visualiseResults()


    ## Normalize the input data
    def NormalizeData(data):

        return (data - np.min(data)) / (np.max(data) - np.min(data))
    

    ## Read all labels from the npz file
    def getLabels(self):
        # Load npz data
        data = np.load(self.labelsPath)

        try:
            self.train_label = data['train']
            self.valid_label = data['valid']
            self.test_label = data['test']
        except:
            print('Loading data should be numpy array and has "images" and "labels" keys.')
            sys.exit(1)


    ## Set the image generators and its parameters
    def setGenerators(self):

        tSize = (self.imageDim(0), self.imageDim(1))

        if self.imageDim(2) == 3:
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
        
        # Data generator for test data
        self.test_generator = test_datagen.flow_from_directory(
            (self.datasetPath + 'test'), 
            target_size = tSize, 
            batch_size = self.batchSize,
            color_mode = cMode,
            class_mode = 'input',
            shuffle = False)


    ## Get the encoded and decoded data from selected model and dataset
    def dataEncodeDecode(self):
        
        for actStr in self.action:

            # Set the train or test data
            if actStr == 'Train':
                data_generator = self.train_generator
                labels = self.train_label
            else:
                data_generator = self.test_generator
                labels = self.test_label

            outputPath = os.path.join(self.modelPath, 'Eval_' + actStr)

            # Load the model
            autoencoder = load_model(self.modelPath)
            
            # Get the encoded data
            encoder = Model(inputs = autoencoder.input, outputs = autoencoder.get_layer('enc').output)
            self.enc_out = self.NormalizeData(encoder.predict(data_generator))

            # Get the decoded data
            self.dec_out = self.NormalizeData(autoencoder.predict(data_generator))

            # Get the original data to be saved in npz
            data_generator.reset()
            self.orig_data = np.concatenate([data_generator.next()[0] for i in range(data_generator.__len__())])
            
            # Save the obtained data
            np.savez(outputPath, orgData = self.orig_data, encData = self.enc_out, decData = self.dec_out, labels = labels)


    ## Visualise the results
    def visualiseResults(self):
        
        for actStr in self.action:

            # Set the train or test data
            if actStr == 'Train':
                label = ' during training.'
            else:
                label = ' during testing.'

            # Plot the encoded samples from all classes
            fig, axarr = plt.subplots(4,3)
            fig.suptitle('Original, encoded and decoded images of the ' + self.modelName + ' autoencoder model' + label)
            
            axarr[0,0].set_title("Original")
            axarr[0,0].imshow(self.orig_data[0])
            axarr[0,0].axis('off')
            axarr[1,0].imshow(self.orig_data[27])
            axarr[1,0].axis('off')
            axarr[2,0].imshow(self.orig_data[35])
            axarr[2,0].axis('off')
            axarr[3,0].imshow(self.orig_data[200])
            axarr[3,0].axis('off')
            
            axarr[0,1].set_title("Encoded")
            axarr[0,1].imshow(self.enc_out[0])
            axarr[0,1].axis('off')
            axarr[1,1].imshow(self.enc_out[27])
            axarr[1,1].axis('off')
            axarr[2,1].imshow(self.enc_out[35])
            axarr[2,1].axis('off')
            axarr[3,1].imshow(self.enc_out[200])
            axarr[3,1].axis('off')

            axarr[0,2].set_title("Decoded")
            axarr[0,2].imshow(self.dec_out[0])
            axarr[0,2].axis('off')
            axarr[1,2].imshow(self.dec_out[27])
            axarr[1,2].axis('off')
            axarr[2,2].imshow(self.dec_out[35])
            axarr[2,2].axis('off')
            axarr[3,2].imshow(self.dec_out[200])
            axarr[3,2].axis('off')

            # Save the illustration figure
            fig.savefig(os.path.join(self.modelPath, self.modelName + actStr + 'AEResults.png'))
