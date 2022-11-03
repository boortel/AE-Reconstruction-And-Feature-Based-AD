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

        # Set training parameters
        self.numEpoch = numEpoch
        self.batchSize = batchSize

        # Set model
        self.modelName = modelSel
        modelObj = ModelSaved(self.modelName, self.imageDim, intermediateDim = 64)

        self.model = modelObj.model

        # Set the augmentations
        self.transforms = Compose([
            Rotate(limit=40),
            RandomBrightness(limit=0.1),
            JpegCompression(quality_lower=85, quality_upper=100, p=0.5),
            HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            RandomContrast(limit=0.2, p=0.5),
            HorizontalFlip(),
        ])
        
        # Set generators and callback
        self.setGenerators()
        self.setCallbacks()

        # Train the model and visualise the results
        self.modelTrain()
        self.visualiseResults()

    
    ## Set the data generator
    def setGenerators(self):

        try:
            # Set image dimensions
            tSize = (self.imageDim[0], self.imageDim[1])

            if self.imageDim[2] == 3:
                cMode = 'rgb'
            else:
                cMode = 'gray'

            # Define normalization layer
            normalization_layer = tf.keras.layers.Rescaling(1./255)

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
            dsTrain = dsTrain.map(partial(self.process_data), num_parallel_calls = AUTOTUNE).prefetch(AUTOTUNE)
            self.dsTrain = dsTrain.map(self.set_shapes, num_parallel_calls = AUTOTUNE).batch(32).prefetch(AUTOTUNE)

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
            self.dsValid = dsValid.map(lambda x: (normalization_layer(x)))
            
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
                patience = 6)

        except:
            logging.error(': Callback initialization of the ' + self.modelName + ' model failed...')
            traceback.print_exc()

        else:
            logging.info(': Callback of the ' + self.modelName + ' model initialized...')


    ## Help function for albumentation - aug_fn
    def aug_fn(self, image):
        data = {"image":image}
        aug_data = self.transforms(**data)
        aug_img = aug_data["image"]
        aug_img = tf.cast(aug_img/255.0, tf.float32)
        aug_img = tf.image.resize(aug_img, size=[self.imageDim[0], self.imageDim[1]])

        return aug_img


    ## Help function for albumentation - process data
    def process_data(self, image, label):
        aug_img = tf.numpy_function(func = self.aug_fn, inp = image, Tout = tf.float32)

        return aug_img, label


    ## Help function for albumentation - set shapes
    def set_shapes(self, img, label):
        img.set_shape(self.imageDim)
        label.set_shape([])

        return img, label


    ## Compile and train the model
    def modelTrain(self):

        try:
            # Define the class weights
            class_weight = {0: 1., 
                            1: 50.}

            # Train the model
            self.trainHistory = self.model.fit(
                self.dsTrain,
                epochs = self.numEpoch,   
                validation_data = self.dsValid, 
                verbose = 1,
                class_weight = class_weight,
                callbacks = [self.esCallBack])
                # callbacks = [tbCallBack, esCallBack])
            
            # Save the model
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