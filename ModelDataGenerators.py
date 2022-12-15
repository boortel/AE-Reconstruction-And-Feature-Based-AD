# -*- coding: utf-8 -*-
"""
Created on Thurs Oct 13 2022

@author: Simon Bilik

This class is used to set data generators and to save the original data to the npz file

"""

import os
import logging
import cv2 as cv
import numpy as np
import tensorflow as tf

class ModelDataGenerators():
    
    ## Set the constants and paths
    def __init__(self, experimentPath, datasetPath, labelInfo, imageDim, batchSize):
        
        # Set the paths and constants
        self.experimentPath = experimentPath
        self.datasetPath = datasetPath
        self.labelInfo = labelInfo
        self.batchSize = batchSize
        self.imageDim = imageDim
        
        # Create datasets
        self.getGenerators()
        
        # Save the original data to NPZ
        self.saveOrgData()
        
    
    ## Normalize and augment training dataset (image as label)
    def changeInputsTrain(self, images, labels):
        
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        x_norm = tf.image.resize(normalization_layer(images),[self.imageDim[0], self.imageDim[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        # Augment the image
        x_norm = tf.image.random_flip_left_right(x_norm)
        x_norm = tf.image.random_contrast(x_norm, lower=0.0, upper=1.0)
        
        # Random image inversion and saturation
        x_norm = (1-x_norm) if tf.random.uniform([]) < 0.5 else x_norm
        x_norm = tf.image.adjust_saturation(x_norm, 3)
        
        # Add salt and pepper noise
        random_values = tf.random.uniform(shape=x_norm[0, ..., -1:].shape)
        x_noise = tf.where(random_values < 0.1, 1., x_norm)
        x_noise = tf.where(1 - random_values < 0.1, 0., x_noise)
        
        return x_noise, x_norm
    
    
    ## Normalize test and validation datasets (image as label)
    def changeInputsTest(self, images, labels):
        
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        x_norm = tf.image.resize(normalization_layer(images),[self.imageDim[0], self.imageDim[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        return x_norm, x_norm
    
    
    ## Normalize dataset (text as label)
    def changeInputs(self, images, labels):
        
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        x = tf.image.resize(normalization_layer(images),[self.imageDim[0], self.imageDim[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        return x, labels
    
    
    ## Set the data generator
    def setGenerator(self, mode):
            
        # Returns generator with and without the labels
        ds = tf.keras.utils.image_dataset_from_directory(
            (self.datasetPath + mode),
            image_size = self.tSize,
            color_mode = self.cMode,
            batch_size = self.batchSize,
            label_mode = 'int',
            shuffle = False)

        dsL = tf.keras.utils.image_dataset_from_directory(
            (self.datasetPath + mode),
            image_size = self.tSize,
            color_mode = self.cMode,
            batch_size = self.batchSize,
            label_mode = 'binary',
            shuffle = False)
        
        # Map the datasets (and augment only Train)
        if mode == 'Train':  
            ds = ds.map(self.changeInputsTrain)
        else:
            ds = ds.map(self.changeInputsTest)
            
        # Map the labeled dataset
        dsL = dsL.map(self.changeInputs)
        
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
            logging.error('Data generators initialization for the ' + self.labelInfo + ' experiment failed...')
            traceback.print_exc()
            return

        else:
            logging.info('Data generators of the ' + self.labelInfo + ' experiment initialized...')
            
    
    ## Save the original data and labels to NPZ file
    def saveOrgData(self):
        
        self.processedData = {}
        
        actStrs = ['Train', 'Test']
        dataGens = [self.dsTrainL, self.dsTestL] 
        
        for actStr, dataGen in zip(actStrs, dataGens):
            
            tempDict = {}
        
            # Get the original data to be saved in npz
            orig_data = np.concatenate([img for img, _ in dataGen], axis=0)

            # Get labels and transform them to format to -1: NOK and 1:OK
            labels = np.concatenate([labels for _, labels in dataGen], axis=0)
            nokIdx = np.where(labels == 0)
            labels[nokIdx] = -1
            
            # Filter the original data
            filterStrength = 0.5
            orig_data = np.array([(filterStrength*img + (1 - filterStrength)*cv.GaussianBlur(img, (25,25), 0)) for img in orig_data])

            # Save the obtained data to NPZ
            outputPath = os.path.join(self.experimentPath, 'Org_' + actStr)
            np.savez_compressed(outputPath, orgData = orig_data, labels = labels)
            
            # Save the processed data to dictionary for a later acces
            tempDict = {'Org': orig_data, 'Lab': labels}
            self.processedData[actStr] = tempDict