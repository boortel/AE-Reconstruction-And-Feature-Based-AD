# -*- coding: utf-8 -*-
"""
Created on Tues Nov 15 2022

@author: Simon Bilik

This class returns convolutional layers later used in the ModelSaved.py script. Feel free to define any new layers if necessary.

"""

import logging
import numpy as np

from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, BatchNormalization, LeakyReLU


## Class with the saved models
class ModelLayers():

    ## Set the constants and paths
    def __init__(self, layerSel, imageDim):

        # Global parameters
        self.layerSel = layerSel
        
        # Image dimensions
        self.imHeight = imageDim[0]
        self.imWidth = imageDim[1]
        self.imChannel = imageDim[2]
        
    ## Get encoder convolutional layers
    def getEncoder(self):
        
        input_img = Input(shape = (self.imHeight, self.imWidth, self.imChannel), name='input_layer')
        
        # Structure from https://keras.io/examples/generative/vae/
        if self.layerSel == 'ConvM1':

            # Block-1
            netEnc = Conv2D(32, kernel_size=3, strides= 2, padding='same', name='conv_E1')(input_img)
            netEnc = BatchNormalization(name='bn_E1')(netEnc)
            netEnc = LeakyReLU(name='lrelu_E1')(netEnc)

            # Block-2
            netEnc = Conv2D(64, kernel_size=3, strides= 2, padding='same', name='conv_E2')(netEnc)
            netEnc = BatchNormalization(name='bn_E2')(netEnc)
            netEnc = LeakyReLU(name='lrelu_E2')(netEnc)

            # Block-3
            netEnc = Conv2D(64, 3, 2, padding='same', name='conv_E3')(netEnc)
            netEnc = BatchNormalization(name='bn_E3')(netEnc)
            netEnc = LeakyReLU(name='lrelu_E3')(netEnc)

            # Block-4
            netEnc = Conv2D(64, 3, 2, padding='same', name='conv_E4')(netEnc)
            netEnc = BatchNormalization(name='bn_E4')(netEnc)
            netEnc = LeakyReLU(name='lrelu_E4')(netEnc)

            # Block-5
            netEnc = Conv2D(64, 3, 2, padding='same', name='conv_E5')(netEnc)
            netEnc = BatchNormalization(name='bn_E5')(netEnc)
            netEnc = LeakyReLU(name='out_E')(netEnc)

            ## Encoding reduction parameters
            redEncHeight = np.int32(self.imHeight / 32)
            redEncWidth = np.int32(self.imWidth / 32)
            filterCount = np.int32(64)
            
        # Structure from MVT
        elif self.layerSel == 'ConvM2':
            
            netEnc = Conv2D(32, (4, 4), strides=2, activation='relu', padding='same', name='conv_E1')(input_img)
            netEnc = BatchNormalization(name='bn_E1')(netEnc)
            
            netEnc = Conv2D(32, (4, 4), strides=2, activation='relu', padding='same', name='conv_E2')(netEnc)
            netEnc = BatchNormalization(name='bn_E2')(netEnc)
            
            netEnc = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same', name='conv_E3')(netEnc)
            netEnc = BatchNormalization(name='bn_E3')(netEnc)
            
            netEnc = Conv2D(64, (4, 4), strides=2, activation='relu', padding='same', name='conv_E4')(netEnc)
            netEnc = BatchNormalization(name='bn_E4')(netEnc)
            
            netEnc = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', name='conv_E5')(netEnc)
            netEnc = BatchNormalization(name='bn_E5')(netEnc)
            
            netEnc = Conv2D(128, (4, 4), strides=2, activation='relu', padding='same', name='conv_E6')(netEnc)
            netEnc = BatchNormalization(name='bn_E6')(netEnc)
            
            netEnc = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', name='conv_E7')(netEnc)
            netEnc = BatchNormalization(name='bn_E7')(netEnc)
            
            netEnc = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same', name='conv_E8')(netEnc)
            netEnc = BatchNormalization(name='bn_E8')(netEnc)
            
            netEnc = Conv2D(1, (8, 8), strides=1, padding='same', name='out_E')(netEnc)
            netEnc = BatchNormalization(name='bn_E9')(netEnc)
            
            ## Encoding reduction parameters
            redEncHeight = np.int32(self.imHeight / 16)
            redEncWidth = np.int32(self.imWidth / 16)
            filterCount = np.int32(16)
        
        # Basic structure 1
        elif self.layerSel == 'ConvM3':
            
            netEnc = Conv2D(32, 3, activation="relu", strides=2, padding="same", name='conv_E1')(input_img)
            netEnc = BatchNormalization(name='bn_E1')(netEnc)
            
            netEnc = Conv2D(64, 3, activation="relu", strides=2, padding="same", name='out_E')(netEnc)
            netEnc = BatchNormalization(name='bn_E2')(netEnc)
            
            ## Encoding reduction parameters
            redEncHeight = np.int32(self.imHeight / 4)
            redEncWidth = np.int32(self.imWidth / 4)
            filterCount = np.int32(64)
        
        # Basic structure 2
        elif self.layerSel == 'ConvM4':
            
            netEnc = Conv2D(8, (5, 5), activation='relu', padding='same', name='conv_E1')(input_img)
            netEnc = MaxPooling2D((2, 2), padding='same', name='mpool_E1')(netEnc)
            netEnc = BatchNormalization(name='bn_E1')(netEnc)
            
            netEnc = Conv2D(4, (3, 3), activation='relu', padding='same', name='conv_E2')(netEnc)
            netEnc = MaxPooling2D((2, 2), padding='same', name='out_E')(netEnc)
            netEnc = BatchNormalization(name='bn_E2')(netEnc)
            
            ## Encoding reduction parameters
            redEncHeight = np.int32(self.imHeight / 4)
            redEncWidth = np.int32(self.imWidth / 4)
            filterCount = np.int32(4)
            
        # Basic structure 3 from https://blog.keras.io/building-autoencoders-in-keras.html
        elif self.layerSel == 'ConvM5':
            
            netEnc = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_E1')(input_img)
            netEnc = MaxPooling2D((2, 2), padding='same', name='mpool_E1')(netEnc)
            netEnc = BatchNormalization(name='bn_E1')(netEnc)
            
            netEnc = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_E2')(netEnc)
            netEnc = MaxPooling2D((2, 2), padding='same', name='mpool_E2')(netEnc)
            netEnc = BatchNormalization(name='bn_E2')(netEnc)
            
            netEnc = Conv2D(4, (3, 3), activation='relu', padding='same', name='conv_E3')(netEnc)
            netEnc = MaxPooling2D((2, 2), padding='same', name='out_E')(netEnc)
            netEnc = BatchNormalization(name='bn_E3')(netEnc)
            
            ## Encoding reduction parameters
            redEncHeight = np.int32(self.imHeight / 8)
            redEncWidth = np.int32(self.imWidth / 8)
            filterCount = np.int32(4)
            
        # Asymetric encoder and decoder (ConvM5 vs ConvM4)
        elif self.layerSel == 'ConvM6':
            
            netEnc = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_E1')(input_img)
            netEnc = MaxPooling2D((2, 2), padding='same', name='mpool_E1')(netEnc)
            netEnc = BatchNormalization(name='bn_E1')(netEnc)
            
            netEnc = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_E2')(netEnc)
            netEnc = MaxPooling2D((2, 2), padding='same', name='mpool_E2')(netEnc)
            netEnc = BatchNormalization(name='bn_E2')(netEnc)
            
            netEnc = Conv2D(4, (3, 3), activation='relu', padding='same', name='conv_E3')(netEnc)
            netEnc = MaxPooling2D((2, 2), padding='same', name='out_E')(netEnc)
            netEnc = BatchNormalization(name='bn_E3')(netEnc)
            
            ## Encoding reduction parameters
            redEncHeight = np.int32(self.imHeight / 8)
            redEncWidth = np.int32(self.imWidth / 8)
            filterCount = np.int32(4)
            
            
        # TODO: Define and add other models in the same way

        else:
            logging.error('Unknown convolutional net name: ' + self.layerSel)
            raise ValueError('Unknown convolutional net name: ' + self.layerSel)
            return
        
        return netEnc, input_img, redEncHeight, redEncWidth, filterCount
    
    
    ## Get decoder convolutional layers
    def getDecoder(self, inputNet, filterCount):
        
        # Structure from https://keras.io/examples/generative/vae/
        if self.layerSel == 'ConvM1':
        
            # Block-1
            netDec = Conv2DTranspose(filterCount, 3, strides= 2, padding='same', name='convT_D1')(inputNet)
            netDec = BatchNormalization(name='bn_D1')(netDec)
            netDec = LeakyReLU(name='lrelu_D1')(netDec)

            # Block-2
            netDec = Conv2DTranspose(64, 3, strides= 2, padding='same', name='convT_D2')(netDec)
            netDec = BatchNormalization(name='bn_D2')(netDec)
            netDec = LeakyReLU(name='lrelu_D2')(netDec)

            # Block-3
            netDec = Conv2DTranspose(64, 3, 2, padding='same', name='convT_D3')(netDec)
            netDec = BatchNormalization(name='bn_D3')(netDec)
            netDec = LeakyReLU(name='lrelu_D3')(netDec)

            # Block-4
            netDec = Conv2DTranspose(32, 3, 2, padding='same', name='convT_D4')(netDec)
            netDec = BatchNormalization(name='bn_D4')(netDec)
            netDec = LeakyReLU(name='lrelu_D4')(netDec)

            # Block-5
            output_img = Conv2DTranspose(self.imChannel, 3, 2,padding='same', activation='sigmoid', name='convT_D5')(netDec)
        
        # Structure from MVT
        elif self.layerSel == 'ConvM2':
            
            netDec = Conv2D(filterCount, (3, 3), strides=1, activation='relu', padding='same', name='conv_D1')(inputNet)
            netDec = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', name='conv_D2')(netDec)
            netDec = UpSampling2D((2, 2), name='upsmp_D1')(netDec)
            netDec = BatchNormalization(name='bn_D1')(netDec)
            
            netDec = Conv2D(128, (4, 4), strides=2, activation='relu', padding='same', name='conv_D3')(netDec)
            netDec = UpSampling2D((2, 2), name='upsmp_D2')(netDec)
            netDec = BatchNormalization(name='bn_D2')(netDec)
            
            netDec = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', name='conv_D4')(netDec)
            netDec = UpSampling2D((2, 2), name='upsmp_D3')(netDec)
            netDec = BatchNormalization(name='bn_D3')(netDec)
            
            netDec = Conv2D(64, (4, 4), strides=2, activation='relu', padding='same', name='conv_D5')(netDec)
            netDec = UpSampling2D((2, 2), name='upsmp_D4')(netDec)
            netDec = BatchNormalization(name='bn_D4')(netDec)
            
            netDec = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same', name='conv_D6')(netDec)
            netDec = UpSampling2D((2, 2), name='upsmp_D5')(netDec)
            netDec = BatchNormalization(name='bn_D5')(netDec)
            
            netDec = Conv2D(32, (4, 4), strides=2, activation='relu', padding='same', name='conv_D7')(netDec)
            netDec = UpSampling2D((4, 4), name='upsmp_D6')(netDec)
            netDec = BatchNormalization(name='bn_D6')(netDec)
            
            netDec = Conv2D(32, (4, 4), strides=2, activation='relu', padding='same', name='conv_D8')(netDec)
            netDec = UpSampling2D((2, 2), name='upsmp_D7')(netDec)
            netDec = BatchNormalization(name='bn_D7')(netDec)
            
            output_img = Conv2D(self.imChannel, (8, 8), activation='sigmoid', padding='same', name='conv_D9')(netDec)
            
        # Basic structure 1
        elif self.layerSel == 'ConvM3':
            
            netDec = Conv2DTranspose(filterCount, (3, 3), activation="relu", strides = 2, padding="same", name='convT_D1')(inputNet)
            netDec = Conv2DTranspose(32, (3, 3), activation="relu", strides = 2, padding="same", name='convT_D2')(netDec)
            netDec = BatchNormalization(name='bn_D1')(netDec)

            output_img = Conv2DTranspose(self.imChannel, (3, 3), activation="sigmoid", padding="same", name='convT_D3')(netDec)
        
        # Basic structure 2
        elif self.layerSel == 'ConvM4':
            
            netDec = Conv2D(filterCount, (3, 3), activation='relu', padding='same', name='conv_D1')(inputNet)
            netDec = UpSampling2D((2, 2), name='upsmp_D1')(netDec)
            netDec = BatchNormalization(name='bn_D1')(netDec)
            
            netDec = Conv2D(8, (5, 5), activation='relu', padding='same', name='conv_D2')(netDec)
            netDec = UpSampling2D((2, 2), name='upsmp_D2')(netDec)
            netDec = BatchNormalization(name='bn_D2')(netDec)
            
            output_img = Conv2D(self.imChannel, (3, 3), activation='sigmoid', padding='same', name='conv_D3')(netDec)
            
        # Basic structure 3
        elif self.layerSel == 'ConvM5':
            
            netDec = Conv2D(filterCount, (3, 3), activation='relu', padding='same', name='conv_D1')(inputNet)
            netDec = UpSampling2D((2, 2), name='upsmp_D1')(netDec)
            netDec = BatchNormalization(name='bn_D1')(netDec)
            
            netDec = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_D2')(netDec)
            netDec = UpSampling2D((2, 2), name='upsmp_D2')(netDec)
            netDec = BatchNormalization(name='bn_D2')(netDec)
            
            netDec = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_D3')(netDec)
            netDec = UpSampling2D((2, 2), name='upsmp_D3')(netDec)
            netDec = BatchNormalization(name='bn_D3')(netDec)
            
            output_img = Conv2D(self.imChannel, (3, 3), activation='sigmoid', padding='same', name='conv_D4')(netDec)
            
            
        # Asymetric encoder and decoder (ConvM5 vs ConvM4)
        elif self.layerSel == 'ConvM6':
            
            netDec = Conv2D(filterCount, (3, 3), activation='relu', padding='same', name='conv_D1')(inputNet)
            netDec = UpSampling2D((4, 4), name='upsmp_D1')(netDec)
            netDec = BatchNormalization(name='bn_D1')(netDec)
            
            netDec = Conv2D(8, (5, 5), activation='relu', padding='same', name='conv_D2')(netDec)
            netDec = UpSampling2D((2, 2), name='upsmp_D2')(netDec)
            netDec = BatchNormalization(name='bn_D2')(netDec)
            
            output_img = Conv2D(self.imChannel, (3, 3), activation='sigmoid', padding='same', name='conv_D3')(netDec)
        
        
        # TODO: Define and add other models in the same way

        else:
            logging.error('Unknown convolutional net name: ' + self.layerSel)
            raise ValueError('Unknown convolutional net name: ' + self.layerSel)
            return
        
        return output_img
    