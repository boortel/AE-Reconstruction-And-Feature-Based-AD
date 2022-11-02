# -*- coding: utf-8 -*-
"""
Created on Thurs Oct  13 17:23:05 2022

@author: Simon Bilik

This class is used for training the selected model

"""

import os
import logging
import traceback
import matplotlib.pyplot as plt

from ModelSaved import ModelSaved

from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator

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

            # Path constants
            train_dir = self.datasetPathTr + 'train'
            validation_dir = self.datasetPathTr + 'valid'
            
            # Load the normalized images
            train_datagen = ImageDataGenerator(
                rescale = 1./255,
                # samplewise_center = True,
                # samplewise_std_normalization = True,
                rotation_range = 360,
                width_shift_range = 0.05,
                height_shift_range = 0.05,
                horizontal_flip = True)
            
            validation_datagen = ImageDataGenerator(
                rescale = 1./255)
                # samplewise_center = True,
                # samplewise_std_normalization = True)
                
            # Data generator for training data
            self.train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size = tSize, 
                batch_size = self.batchSize,
                color_mode = cMode,
                class_mode = 'input',
                shuffle = True)
            
            # Data generator for validation data
            self.validation_generator = validation_datagen.flow_from_directory(
                validation_dir, 
                target_size = tSize, 
                batch_size = self.batchSize,
                color_mode = cMode,
                class_mode = 'input', 
                shuffle = False)
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
                patience = 3)

        except:
            logging.error(': Callback initialization of the ' + self.modelName + ' model failed...')
            traceback.print_exc()

        else:
            logging.info(': Callback of the ' + self.modelName + ' model initialized...')


    ## Compile and train the model
    def modelTrain(self):

        try:
            # Train the model
            self.trainHistory = self.model.fit(
                self.train_generator, 
                steps_per_epoch = self.train_generator.samples/self.train_generator.batch_size, 
                epochs = self.numEpoch,   
                validation_data = self.validation_generator, 
                validation_steps = self.validation_generator.samples/self.validation_generator.batch_size, 
                verbose = 1,
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