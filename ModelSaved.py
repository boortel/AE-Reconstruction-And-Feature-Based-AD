# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:23:05 2021

@author: Šimon Bilík

This class returns compiled autoencoder model later used in the AE_train.py script. Feel free to define any new models if necessary.

"""

import logging
import traceback
import numpy as np

from keras.losses import binary_crossentropy

from keras import optimizers
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Lambda, Reshape

class ModelSaved():

    ## Set the constants and paths
    def __init__(self, modelSel, imageDim, intermediateDim = 64, latentDim = 2):

        # Global parameters
        self.modelName = modelSel

        # Image dimensions
        self.imHeight = imageDim[0]
        self.imWidth = imageDim[1]
        self.imChannel = imageDim[2]

        # TODO: channel??
        self.imgSize = self.imHeight * self.imWidth

        # VAE parameters
        self.intermediateDim = intermediateDim
        self.latentDim = latentDim

        # Initialize and return the selected model
        try:
            if self.modelName == 'VAE_C1':
                self.model = self.build_vaeC1_model()

            elif self.modelName == 'VAE_F1':
                self.model = self.build_vaeF1_model()

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
        

    ## Sampling function for the VAEs
    def sampling(self, args):
        #z_mean, z_log_sigma = args
        z_mean = args[0]
        z_log_sigma = args[1]
        epsilon = K.random_normal(shape = (K.shape(z_mean)[0], self.latentDim), mean = 0., stddev = 0.1)

        return z_mean + K.exp(z_log_sigma) * epsilon


    ## Convolutional VAE1
    def build_vaeC1_model(self):

        input_img = Input(shape = (self.imHeight, self.imWidth, self.imChannel))

        # Encode-----------------------------------------------------------
        x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(input_img)
        x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)

        shape_before_flatten = K.int_shape(x)[1:]

        encoder_flatten = Flatten()(x)
        h = Dense(self.intermediateDim, activation='relu',name="h")(encoder_flatten)
        
        z_log_sigma = Dense(self.latentDim,name="z_log_sigma")(h)
        z_mean = Dense(self.latentDim,name="zmean")(h)
        z = Lambda(self.sampling)([z_mean, z_log_sigma])

        encoder = Model(input_img, [z_mean, z_log_sigma, z], name = 'Encoder')

        # Decode-----------------------------------------------------------
        latent_inputs = Input(shape=(self.latentDim,), name='z_sampling')
        x = Dense(self.intermediateDim, activation='relu', name="x")(latent_inputs)
        x = Dense(units=np.prod(shape_before_flatten), activation='relu')(x)
        x = Reshape(target_shape=shape_before_flatten)(x)

        x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)

        outputs = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

        decoder = Model(latent_inputs, outputs, name = 'Decoder')

        # Instantiate VAE model
        outputs = decoder(encoder(input_img)[2])

        reconstruction_loss = binary_crossentropy(input_img, outputs)
        reconstruction_loss *= self.imgSize

        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        vae_loss = K.mean(reconstruction_loss + kl_loss)

        vae = Model(input_img, outputs, name = self.modelName)

        vae.add_loss(vae_loss)
        vae.compile(optimizer = 'adam')

        return vae


    ## Fully-connected VAE2
    def build_vaeF1_model(self):

        input_img = Input(shape = (self.imHeight, self.imWidth, self.imChannel))
        
        # Encode-----------------------------------------------------------
        h = Dense(self.intermediateDim, activation='relu')(input_img)

        z_log_sigma = Dense(self.latentDim)(h)
        z_mean = Dense(self.latentDim)(h)

        z = Lambda(self.sampling)([z_mean, z_log_sigma])

        # Encoder
        encoder = Model(input_img, [z_mean, z_log_sigma, z], name = 'Encoder')

        # Decode-----------------------------------------------------------
        latent_inputs = Input(shape=(self.latentDim,), name='z_sampling')

        x = Dense(self.intermediateDim, activation='relu')(latent_inputs)
        outputs = Dense(self.imgSize, activation='sigmoid')(x)

        # Decoder
        decoder = Model(latent_inputs, outputs, name = 'Decoder')

        # Instantiate VAE model
        outputs = decoder(encoder(input_img)[2])

        reconstruction_loss = binary_crossentropy(input_img, outputs)
        reconstruction_loss *= self.imgSize

        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        vae_loss = K.mean(reconstruction_loss + kl_loss)

        vae = Model(input_img, outputs, name = self.modelName)

        vae.add_loss(vae_loss)
        vae.compile(optimizer = 'adam')

        return vae


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
