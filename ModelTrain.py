# -*- coding: utf-8 -*-
"""
Created on Thurs Oct  13 17:23:05 2022

@author: Simon Bilik

This class is used for training the selected model

"""

import os
import logging
import traceback
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import callbacks
from functools import partial
from ModelSaved import ModelSaved
from albumentations import Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip, Rotate

AUTOTUNE = tf.data.experimental.AUTOTUNE

class ModelTrain():

    ## Set the constants and paths
    def __init__(self, modelBasePath, datasetPathTr, modelSel, labelInfo, imageDim, batchSize, numEpoch):
        
        # Set paths
        self.outputPath = os.path.join(modelBasePath, modelSel + labelInfo)
        self.datasetPathTr = datasetPathTr

        # Create the model directory
        mPath = os.path.join(self.outputPath, 'modelData')

        if not os.path.exists(mPath):
            os.makedirs(mPath)

        # Set image dimensions
        self.imageDim = imageDim

        # Set model and training parameters
        self.modelName = modelSel
        self.numEpoch = numEpoch
        self.batchSize = batchSize
        
        # Set generators and callback
        self.setGenerators()
        self.setCallbacks()

        # Set model
        modelObj = ModelSaved(self.modelName, self.imageDim, self.dsTrain, intermediateDim = 64, latentDim = 100)

        self.model = modelObj.model
        self.VAE = modelObj.VAE

        # Set the augmentations
        self.transforms = Compose([
            Rotate(),
            RandomBrightness(),
            RandomContrast(),
            HorizontalFlip(),
        ])

        # Train the model and visualise the results
        self.modelTrain()
        self.visualiseResults()

    
    ## Help function to map IO
    def change_inputs(self, images):
        
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        x = tf.image.resize(normalization_layer(images),[self.imageDim[0], self.imageDim[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        return x
    
    
    ## Augment the image using TF library
    def process_image(self, image):
        
        # Cast and normalize image
        image = tf.image.convert_image_dtype(image, tf.float32)
        
        # Apply simple augmentations
        image = tf.image.random_flip_left_right(image)
        image = tf.image.resize(image,[self.imageDim[0], self.imageDim[1]])
        
        return image
    
    
    ## Set the data generator
    def setGenerators(self):

        try:
            # Set image dimensions
            tSize = (self.imageDim[0], self.imageDim[1])

            if self.imageDim[2] == 3:
                cMode = 'rgb'
            else:
                cMode = 'grayscale'

            # Train DS
            dsTrain = tf.keras.utils.image_dataset_from_directory(
                (self.datasetPathTr + 'train'),
                image_size = tSize,
                color_mode = cMode,
                batch_size = self.batchSize,
                crop_to_aspect_ratio = True,
                label_mode = None,
                shuffle = False)

            # Augment the dataset (with normalization) and reset its shapes
            self.dsTrain = dsTrain.map(self.change_inputs)
            #self.dsTrain = dsTrain.map(self.process_image, num_parallel_calls=AUTOTUNE)

            # Validation DS
            dsValid = tf.keras.utils.image_dataset_from_directory(
                (self.datasetPathTr + 'valid'),
                image_size = tSize,
                color_mode = cMode,
                batch_size = self.batchSize,
                crop_to_aspect_ratio = True,
                label_mode = None,
                shuffle = False)

            # Normalize the dataset
            self.dsValid = dsValid.map(self.change_inputs)
            
        except:
            logging.error(': Data generators initialization of the ' + self.modelName + ' model failed...')
            traceback.print_exc()

        else:
            logging.info(': Data generators of the ' + self.modelName + ' model initialized...')


    ## Set callbacks
    def setCallbacks(self):

        try:
            # Configure the tensorboard callback
            # self.tbCallBack = tensorflow.keras.callbacks.TensorBoard(
            #     log_dir = './data/', 
            #     histogram_freq = 0,
            #     update_freq = 'batch',
            #     write_graph = True, 
            #     write_images = True)
            
            # Configure the early stopping callback
            self.esCallBack = callbacks.EarlyStopping(
                monitor = 'loss', 
                patience = 20)

        except:
            logging.error(': Callback initialization of the ' + self.modelName + ' model failed...')
            traceback.print_exc()

        else:
            logging.info(': Callback of the ' + self.modelName + ' model initialized...')
            

    ## Compile and train the model
    def modelTrain(self):

        try:
            # Set the validation DS
            if self.VAE:
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
                # callbacks = [tbCallBack, esCallBack])
            
            self.model.save(self.outputPath)

        except:
            logging.error(': Training of the ' + self.modelName + ' model failed...')
            traceback.print_exc()

        else:
            logging.info(': Training of the ' + self.modelName + ' model was finished...')

    
    ## Visualise the results
    def visualiseResults(self):

        try:
            # Plot the history and save the curves
            if self.VAE:
                train_loss = self.trainHistory.history['loss']
                val_loss = self.trainHistory.history['kl_loss']
            else:
                train_loss = self.trainHistory.history['loss']
                val_loss = self.trainHistory.history['val_loss']
            
            fig, axarr = plt.subplots(2)
            tempTitle = " Training and Validation Loss of " + self.modelName + " model."
            fig.suptitle(tempTitle, fontsize=14, y=1.08)
            
            axarr[0].plot(train_loss)
            axarr[0].set(xlabel = "Number of Epochs", ylabel = "Training Loss [-]")
            
            axarr[1].plot(val_loss)
            axarr[1].set(xlabel = "Number of Epochs", ylabel = "Validation Loss [-]")
            
            fig.tight_layout()
            fig.savefig(os.path.join(self.outputPath, 'modelData', self.modelName + '_Losses.png'))
        
        except:
            logging.error(': Visualisation of the ' + self.modelName + ' model results failed...')
            traceback.print_exc()

        else:
            logging.info(': Visualisation of the ' + self.modelName + ' model results was finished...')