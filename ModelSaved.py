# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:23:05 2021

@author: Simon Bilik

This class returns compiled autoencoder model later used in the AE_train.py script. Feel free to define any new models if necessary.

"""

import keras
import logging
import traceback
import numpy as np
import tensorflow as tf

from keras import optimizers
from keras.models import Model
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Lambda, Reshape, BatchNormalization, LeakyReLU

from ModelHelperVAE import VAE, Sampling, VQVAETrainer


## Class with the saved models
class ModelSaved():

    ## Set the constants and paths
    def __init__(self, modelSel, imageDim, trainDataset, dataVariance = 0.5, intermediateDim = 64, latentDim = 100):

        # Global parameters
        self.modelName = modelSel
        self.VAE = False

        # Image dimensions
        self.imHeight = imageDim[0]
        self.imWidth = imageDim[1]
        self.imChannel = imageDim[2]

        # TODO: channel??
        self.imgSize = self.imHeight * self.imWidth

        # VAE parameters
        self.intermediateDim = intermediateDim
        self.dataVariance = dataVariance
        self.latentDim = latentDim
        
        self.trainDataset = trainDataset

        # Initialize and return the selected model
        try:
            if self.modelName == 'VAE_C1':
                self.model = self.build_vaeC1_model()
                self.VAE = True

            elif self.modelName == 'VAE_F1':
                self.model = self.build_vaeF1_model()
                self.VAE = True

            elif self.modelName == 'BAE1':
                self.model = self.build_bae1_model()

            elif self.modelName == 'BAE2':
                self.model = self.build_bae2_model()

            elif self.modelName == 'MVT':
                self.model = self.build_mvt_model()

            # TODO: Define and add other models in the same way

            else:
                logging.error(': Unknown model name: ' + self.modelName)
                raise ValueError('Unknown model name: ' + self.modelName)

            self.model.summary()

        except:
            logging.error(': Initialization of the selected model: ' + self.modelName + ' failed....')
            traceback.print_exc()
    
    
    ## Convolutional VAE1
    def build_vaeC1_model(self):
        
        input_img = Input(shape = (self.imHeight, self.imWidth, self.imChannel), name='input_layer')
        
        # Encode-----------------------------------------------------------
        
        # Block-1
        x = Conv2D(32, kernel_size=3, strides= 2, padding='same', name='conv_1')(input_img)
        x = BatchNormalization(name='bn_1')(x)
        x = LeakyReLU(name='lrelu_1')(x)

        # Block-2
        x = Conv2D(64, kernel_size=3, strides= 2, padding='same', name='conv_2')(x)
        x = BatchNormalization(name='bn_2')(x)
        x = LeakyReLU(name='lrelu_2')(x)

        # Block-3
        x = Conv2D(64, 3, 2, padding='same', name='conv_3')(x)
        x = BatchNormalization(name='bn_3')(x)
        x = LeakyReLU(name='lrelu_3')(x)

        # Block-4
        x = Conv2D(64, 3, 2, padding='same', name='conv_4')(x)
        x = BatchNormalization(name='bn_4')(x)
        x = LeakyReLU(name='lrelu_4')(x)

        # Block-5
        x = Conv2D(64, 3, 2, padding='same', name='conv_5')(x)
        x = BatchNormalization(name='bn_5')(x)
        x = LeakyReLU(name='lrelu_5')(x)
        
        # Final Block
        flatten = Flatten()(x)
        z_mean = Dense(self.latentDim, name = 'mean')(flatten)
        z_log_var = Dense(self.latentDim, name='log_var')(flatten)
        
        z = Sampling()([z_mean, z_log_var])
        
        encoder = Model(inputs = input_img, outputs = [z_mean, z_log_var, z], name="enc")
        
        # Decode-----------------------------------------------------------
     
        latent_inputs = Input(shape=(self.latentDim,))
        
        x = Dense(4096, activation="relu")(latent_inputs)
        x = Reshape((8,8,64))(x)

        # Block-1
        x = Conv2DTranspose(64, 3, strides= 2, padding='same',name='conv_transpose_1')(x)
        x = BatchNormalization(name='bn_1')(x)
        x = LeakyReLU(name='lrelu_6')(x)

        # Block-2
        x = Conv2DTranspose(64, 3, strides= 2, padding='same', name='conv_transpose_2')(x)
        x = BatchNormalization(name='bn_2')(x)
        x = LeakyReLU(name='lrelu_7')(x)

        # Block-3
        x = Conv2DTranspose(64, 3, 2, padding='same', name='conv_transpose_3')(x)
        x = BatchNormalization(name='bn_3')(x)
        x = LeakyReLU(name='lrelu_8')(x)

        # Block-4
        x = Conv2DTranspose(32, 3, 2, padding='same', name='conv_transpose_4')(x)
        x = BatchNormalization(name='bn_4')(x)
        x = LeakyReLU(name='lrelu_9')(x)
        
        # Block-5
        decoder_outputs = Conv2DTranspose(3, 3, 2,padding='same', activation='sigmoid', name='conv_transpose_5')(x)
        decoder = Model(inputs = latent_inputs, outputs = decoder_outputs, name="dec")
        
        z_mean, z_log_var, z = encoder(input_img)
        
        reconstructions = decoder(z)
        
        vae = VAE(input_img, reconstructions, encoder, decoder, self.modelName)
        vae.compile(optimizer = keras.optimizers.Adam())
        
        return vae
    
    
    ## Convolutional VAE2
    def build_vaeC2_model(self):

        input_img = Input(shape = (self.imHeight, self.imWidth, self.imChannel))
        
        # Encode-----------------------------------------------------------
        
        x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(input_img)
        x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = Flatten()(x)
        x = Dense(16, activation="relu")(x)
        
        z_mean = Dense(self.latentDim, name="z_mean")(x)
        z_log_var = Dense(self.latentDim, name="z_log_var")(x)
        
        z = Sampling()([z_mean, z_log_var])
        encoder = Model(input_img, [z_mean, z_log_var, z], name="enc")
        
        # Decode-----------------------------------------------------------
        latent_inputs = Input(shape=(self.latentDim,))
        
        x = Dense(7*7*64, activation="relu")(latent_inputs)
        x = Reshape((7,7,64))(x)
        x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        
        decoder_outputs = Conv2DTranspose(self.imChannel, 3, activation="sigmoid", padding="same")(x)
        decoder = Model(latent_inputs, decoder_outputs, name="dec")
        
        z_mean, z_log_var, z = encoder(input_img)
        
        reconstructions = decoder(z)
        
        vae = VAE(input_img, reconstructions, encoder, decoder)
        vae.compile(optimizer = keras.optimizers.Adam())

        return vae


    ## Fully-connected VAE2
    def build_vaeF1_model(self):

        input_img = Input(shape = (self.imHeight, self.imWidth, self.imChannel))
        
        # Encode-----------------------------------------------------------
        x = Dense(self.intermediateDim, activation='relu')(input_img)

        z_mean = Dense(self.latentDim, name="z_mean")(x)
        z_log_var = Dense(self.latentDim, name="z_log_var")(x)
        
        z = Sampling()([z_mean, z_log_var])
        encoder = Model(input_img, [z_mean, z_log_var, z], name="enc")

        # Decode-----------------------------------------------------------
        latent_inputs = Input(shape=(self.latentDim,))

        x = Dense(self.intermediateDim, activation='relu')(latent_inputs)
        outputs = Dense(self.imgSize, activation='sigmoid')(x)

        # Decoder
        decoder = Model(latent_inputs, outputs, name = 'Decoder')

        # Instantiate VAE model
        z_mean, z_log_var, z = encoder(input_img)
        
        reconstructions = decoder(z)
        
        vae = VAE(input_img, reconstructions, encoder, decoder)
        vae.compile(optimizer = keras.optimizers.Adam())

        return vae


    ## Convolutional VQ-VAE
    def build_vqvaeC1_model(self):
        input_img = Input(shape = (self.imHeight, self.imWidth, self.imChannel))
        
        # Encode-----------------------------------------------------------
        x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(input_img)
        x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        encoder_outputs = Conv2D(self.latentDim, 1, padding="same")(x)

        encoder = keras.Model(input_img, encoder_outputs, name="enc")

        # Decode-----------------------------------------------------------
        latent_inputs = keras.Input(shape = encoder(self.latentDim).output.shape[1:])

        x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(latent_inputs)
        x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        reconstruction = Conv2DTranspose(1, 3, padding="same")(x)

        decoder = keras.Model(latent_inputs, reconstruction, name="dec")

        vqvae = VQVAETrainer(input_img, reconstruction, encoder, decoder, self.modelName, self.dataVariance, self.latentDim)
        vqvae.compile(optimizer=keras.optimizers.Adam())

        return vqvae


    ## Basic AE model
    def build_bae_model_old(self):
        """
        build basic autoencoder model
        """
        input_img = Input(shape=(self.imHeight, self.imWidth, self.imChannel))

        # Encode-----------------------------------------------------------
        net = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        net = MaxPooling2D((2, 2), padding='same')(net)
        net = Conv2D(8, (3, 3), activation='relu', padding='same')(net)
        net = MaxPooling2D((2, 2), padding='same')(net) 
        net = Conv2D(4, (3, 3), activation='relu', padding='same')(net)
        encoded = MaxPooling2D((2, 2), padding='same', name='enc')(net)

        # Decode---------------------------------------------------------------------
        net = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
        net = UpSampling2D((2, 2))(net)
        net = Conv2D(8, (3, 3), activation='relu', padding='same')(net)
        net = UpSampling2D((2, 2))(net)
        net = Conv2D(16, (3, 3), activation='relu', padding='same')(net)
        net = UpSampling2D((2, 2))(net)
        decoded = Conv2D(self.imChannel, (3, 3), activation='sigmoid', padding='same')(net)
        # ---------------------------------------------------------------------

        model = Model(input_img, decoded, name = self.modelName)
        
        # Configure the model for training
        model.compile(loss = 'mean_squared_error', optimizer = optimizers.Adam()) 
            # metrics = [tensorflow.keras.metrics.AUC(), tensorflow.keras.metrics.Accuracy()])

        return model


    ## Basic AE model
    def build_bae1_model(self):
        """
        build basic autoencoder model
        """
        input_img = Input(shape=(self.imHeight, self.imWidth, self.imChannel))

        # Encode-----------------------------------------------------------
        net = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        net = MaxPooling2D((2, 2), padding='same')(net)
        net = Conv2D(8, (3, 3), activation='relu', padding='same')(net)
        net = MaxPooling2D((2, 2), padding='same')(net) 
        net = Conv2D(4, (3, 3), activation='relu', padding='same')(net)
        net = MaxPooling2D((2, 2), padding='same', name='enc')(net)
        net = Dense(10, activation='relu')(net)
        encoded = Dense(1000, activation='relu')(net)

        # Decode---------------------------------------------------------------------
        net = Dense(10, activation='relu')(encoded)
        net = Conv2D(4, (3, 3), activation='relu', padding='same')(net)
        net = UpSampling2D((2, 2))(net)
        net = Conv2D(8, (3, 3), activation='relu', padding='same')(net)
        net = UpSampling2D((2, 2))(net)
        net = Conv2D(16, (3, 3), activation='relu', padding='same')(net)
        net = UpSampling2D((2, 2))(net)
        decoded = Conv2D(self.imChannel, (3, 3), activation='sigmoid', padding='same')(net)
        # ---------------------------------------------------------------------


        model = Model(input_img, decoded, name = self.modelName)
        
        # Configure the model for training
        model.compile(loss = 'mean_squared_error', optimizer = optimizers.Adam()) 
            # metrics = [tensorflow.keras.metrics.AUC(), tensorflow.keras.metrics.Accuracy()])

        return model


    ## Basic AE model
    def build_bae2_model(self):
        """
        build basic autoencoder model
        """
        input_img = Input(shape=(self.imHeight, self.imWidth, self.imChannel))

        # Encode-----------------------------------------------------------
        net = Conv2D(8, (5, 5), activation='relu', padding='same')(input_img)
        net = MaxPooling2D((2, 2), padding='same')(net) 
        net = Conv2D(4, (3, 3), activation='relu', padding='same')(net)
        encoded = MaxPooling2D((2, 2), padding='same', name='enc')(net)

        # Decode---------------------------------------------------------------------
        net = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
        net = UpSampling2D((2, 2))(net)
        net = Conv2D(8, (5, 5), activation='relu', padding='same')(net)
        net = UpSampling2D((2, 2))(net)
        decoded = Conv2D(self.imChannel, (3, 3), activation='sigmoid', padding='same')(net)
        # ---------------------------------------------------------------------

        model = Model(input_img, decoded, name = self.modelName)
        
        # Configure the model for training
        model.compile(loss = 'mean_squared_error', optimizer = optimizers.Adam()) 
            # metrics = [tensorflow.keras.metrics.AUC(), tensorflow.keras.metrics.Accuracy()])

        return model


    ## MVTec
    def build_mvt_model(self):

        input_img = Input(shape=(self.imHeight, self.imWidth, self.imChannel))  # adapt this if using `channels_first` image data format
        
        # Encode-----------------------------------------------------------
        x = Conv2D(32, (4, 4), strides=2 , activation='relu', padding='same')(input_img)
        x = Conv2D(32, (4, 4), strides=2, activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)
        x = Conv2D(64, (4, 4), strides=2, activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
        x = Conv2D(128, (4, 4), strides=2, activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)
        encoded = Conv2D(1, (8, 8), strides=1, padding='same', name='enc')(x)
        
        # Decode---------------------------------------------------------------------
        x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(encoded)
        x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(128, (4, 4), strides=2, activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (4, 4), strides=2, activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (4, 4), strides=2, activation='relu', padding='same')(x)
        x = UpSampling2D((4, 4))(x)
        x = Conv2D(32, (4, 4), strides=2, activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(self.imChannel, (8, 8), activation='sigmoid', padding='same')(x)
        # ---------------------------------------------------------------------
        
        model = Model(input_img, decoded, name = self.modelName)
        
        # Configure the model for training
        model.compile(loss = 'mean_squared_error', optimizer = optimizers.Adam()) 
            # metrics = [tensorflow.keras.metrics.AUC(), tensorflow.keras.metrics.Accuracy()])

        return model
