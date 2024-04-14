# -*- coding: utf-8 -*-
"""
Created on Wed Jan 6 2021

@author: Simon Bilik

This class returns compiled autoencoder model later used in the ModelTrainAndEval.py script. Feel free to define any new models if necessary.

"""

import keras
import logging
import traceback

from keras import optimizers
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten, Reshape

from ModelLayers import ModelLayers
from ModelHelperVAE import VAE, Sampling, VQVAETrainer, VectorQuantizer


## Class with the saved models
class ModelSaved():

    ## Set the constants and paths
    def __init__(self, modelSel, layerSel, imageDim, dataVariance = 0.5, intermediateDim = 64, latentDim = 16, num_embeddings = 32):

        # Global parameters
        self.modelName = modelSel
        self.layerSel = ModelLayers(layerSel, imageDim)

        # VAE parameters
        self.intermediateDim = intermediateDim
        self.num_embeddings = num_embeddings
        self.dataVariance = dataVariance
        self.latentDim = latentDim

        # Initialize and return the selected model
        try:
            if self.modelName == 'VAE1':
                self.model = self.build_vae1_model()

            elif self.modelName == 'VAE2':
                self.model = self.build_vae2_model()
                
            elif self.modelName == 'VQVAE1':
                self.model = self.build_vqvaeC1_model()

            elif self.modelName == 'BAE1':
                self.model = self.build_bae1_model()

            elif self.modelName == 'BAE2':
                self.model = self.build_bae2_model()

            # TODO: Define and add other models in the same way

            else:
                logging.error('Unknown model name: ' + self.modelName)
                raise ValueError('Unknown model name: ' + self.modelName)
                return

            self.model.summary()

        except:
            logging.error('Initialization of the selected model: ' + self.modelName + ' failed....')
            traceback.print_exc()
    
    
    ## Variational autoencoder 1
    def build_vae1_model(self):
        
        # Set the model type
        self.typeAE = 'VAE1'
        
        # Encoder -----------------------------------------------------------
        netEnc, input_img, redEncHeight, redEncWidth, filterCount = self.layerSel.getEncoder()
        
        flatten = Flatten()(netEnc)
        z_mean = Dense(self.latentDim, name = 'mean')(flatten)
        z_log_var = Dense(self.latentDim, name='log_var')(flatten)
        
        z = Sampling()([z_mean, z_log_var])
        
        encoder = Model(inputs = input_img, outputs = [z_mean, z_log_var, z], name = "enc")
        
        # Decoder -----------------------------------------------------------
        latent_inputs = Input(shape=(self.latentDim,))
        
        x = Dense(redEncHeight * redEncWidth * filterCount, activation="relu")(latent_inputs)
        x = Reshape((redEncHeight, redEncWidth, filterCount))(x)
        
        output_img = self.layerSel.getDecoder(x, filterCount)
        
        decoder = Model(inputs = latent_inputs, outputs = output_img, name = "dec")
        
        z_mean, z_log_var, z = encoder(input_img)
        
        reconstructions = decoder(z)
        
        vae = VAE(input_img, reconstructions, encoder, decoder, self.modelName)
        vae.compile(optimizer = keras.optimizers.Adam())
        
        return vae
    
    
    ## Variational autoencoder 2 (with fully connected layers before z-parameters computation)
    def build_vae2_model(self):

        # Set the model type
        self.typeAE = 'VAE2'
        
        # Encoder -----------------------------------------------------------
        netEnc, input_img, redEncHeight, redEncWidth, filterCount = self.layerSel.getEncoder()
        
        x = Flatten()(netEnc)
        x = Dense(16, activation="relu")(x)
        
        z_mean = Dense(self.latentDim, name="z_mean")(x)
        z_log_var = Dense(self.latentDim, name="z_log_var")(x)
        
        z = Sampling()([z_mean, z_log_var])
        encoder = Model(input_img, [z_mean, z_log_var, z], name = "enc")
        
        # Decode-----------------------------------------------------------
        latent_inputs = Input(shape=(self.latentDim,))
        
        x = Dense(redEncHeight * redEncWidth * filterCount, activation="relu")(latent_inputs)
        x = Reshape((redEncHeight, redEncWidth, filterCount))(x)
        
        output_img = self.layerSel.getDecoder(x, filterCount)
        
        decoder = Model(latent_inputs, output_img, name = "dec")
        
        z_mean, z_log_var, z = encoder(input_img)
        
        reconstructions = decoder(z)
        
        vae = VAE(input_img, reconstructions, encoder, decoder, self.modelName)
        vae.compile(optimizer = keras.optimizers.Adam())

        return vae


    ## Convolutional VQ-VAE
    def build_vqvaeC1_model(self):
        
        # Set the model type
        self.typeAE = 'VQVAE1'
        
        # Encoder -----------------------------------------------------------
        netEnc, input_img, _, _, filterCount = self.layerSel.getEncoder()
        
        encoder_outputs = Conv2D(self.latentDim, 1, padding="same")(netEnc)

        encoder = keras.Model(input_img, encoder_outputs, name="enc")

        # Decode-----------------------------------------------------------
        latent_inputs = keras.Input(shape = encoder.output.shape[1:])
        
        output_img = self.layerSel.getDecoder(latent_inputs, filterCount)

        decoder = keras.Model(latent_inputs, output_img, name="dec")
        
        # Instantiate VQ-VAE model
        vq_layer = VectorQuantizer(self.num_embeddings, self.latentDim, name="vector_quantizer")
        
        encoder_outputs = encoder(input_img)
        quantized_latents = vq_layer(encoder_outputs)
        reconstructions = decoder(quantized_latents)
        
        vqvae = keras.Model(input_img, reconstructions, name="vq_vae")

        vqvae = VQVAETrainer(input_img, reconstructions, vqvae, self.modelName, self.dataVariance, self.latentDim)
        vqvae.compile(optimizer=keras.optimizers.Adam())

        return vqvae

    
    ## Basic autoencoder model
    def build_bae1_model(self):
        
        # Set the model type
        self.typeAE = 'BAE1'
        
        # Encoder -----------------------------------------------------------
        netEnc, input_img, _, _, filterCount = self.layerSel.getEncoder()
        
        # Decoder -----------------------------------------------------------
        output_img = self.layerSel.getDecoder(netEnc, filterCount)
        
        model = Model(input_img, output_img, name = self.modelName)
        
        # Rename the out_E layer for enc
        for i, layer in enumerate(model.layers):
            if layer.name == 'out_E':
                layer.name = 'enc'
        
        # Configure the model for training
        model.compile(loss = 'mean_squared_error', optimizer = optimizers.Adam()) 

        return model
    

    ## Basic autoencoder model with fully connected layers before encoding
    def build_bae2_model(self):
        
        # Set the model type
        self.typeAE = 'BAE2'

        # Encode-----------------------------------------------------------
        netEnc, input_img, _, _, filterCount = self.layerSel.getEncoder()
        
        x = Dense(10, activation='relu', name='dense_EM1')(netEnc)
        encoded = Dense(self.latentDim, activation='relu', name='enc')(x)

        # Decode---------------------------------------------------------------------
        x = Dense(10, activation='relu', name='dense_DM1')(encoded)
        
        output_img = self.layerSel.getDecoder(x, filterCount)
        
        model = Model(input_img, output_img, name = self.modelName)
        
        # Configure the model for training
        model.compile(loss = 'mean_squared_error', optimizer = optimizers.Adam()) 

        return model
    