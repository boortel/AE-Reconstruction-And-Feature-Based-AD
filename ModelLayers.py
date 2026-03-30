
# -*- coding: utf-8 -*-
"""
Created on Tues Nov 15 2022

@author: Simon Bilik

This class returns convolutional layers later used in the ModelSaved.py script. Feel free to define any new layers if necessary.

"""

import logging
import torch
import torch.nn as nn
from dataclasses import dataclass



@dataclass
class LayerConfig:
    """Configuration for encoder-decoder pair."""

    name: str
    reduction_factor: int  # How much the spatial dimensions are reduced
    bottleneck_channels: int


LAYER_CONFIGS: dict[str, LayerConfig] = {
    "ConvM1": LayerConfig("ConvM1", 32, 64),
    "ConvM2": LayerConfig("ConvM2", 16, 1),
    "ConvM3": LayerConfig("ConvM3", 4, 64),
    "ConvM4": LayerConfig("ConvM4", 4, 4),
    "ConvM5": LayerConfig("ConvM5", 8, 4),
    "ConvM6": LayerConfig("ConvM6", 8, 4),
}


def get_encoder(layer_sel, in_channels):
    """ Returns the Encoder model based on the selected layer configuration. """
    
    if layer_sel == 'ConvM1':
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU()
        )

    elif layer_sel == 'ConvM2':
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32), nn.Sigmoid(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32), nn.Sigmoid(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same'), nn.BatchNorm2d(32), nn.Sigmoid(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.Sigmoid(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'), nn.BatchNorm2d(64), nn.Sigmoid(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.Sigmoid(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding='same'), nn.BatchNorm2d(64), nn.Sigmoid(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding='same'), nn.BatchNorm2d(32), nn.Sigmoid(),
            nn.Conv2d(32, 1, kernel_size=8, stride=1, padding='same'), nn.BatchNorm2d(1)
        )

    elif layer_sel == 'ConvM3':
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.Sigmoid(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.Sigmoid()
        )

    elif layer_sel == 'ConvM4':
        return nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=5, stride=1, padding='same'), nn.MaxPool2d(2, 2), nn.BatchNorm2d(8), nn.Sigmoid(),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding='same'), nn.MaxPool2d(2, 2), nn.BatchNorm2d(4), nn.Sigmoid()
        )

    elif layer_sel == 'ConvM5':
        return nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding='same'), nn.MaxPool2d(2, 2), nn.BatchNorm2d(16), nn.Sigmoid(),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding='same'), nn.MaxPool2d(2, 2), nn.BatchNorm2d(8), nn.Sigmoid(),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding='same'), nn.MaxPool2d(2, 2), nn.BatchNorm2d(4), nn.Sigmoid()
        )

    elif layer_sel == 'ConvM6':
        return nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding='same'), nn.MaxPool2d(2, 2), nn.BatchNorm2d(16), nn.Sigmoid(),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding='same'), nn.MaxPool2d(2, 2), nn.BatchNorm2d(8), nn.Sigmoid(),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding='same'), nn.MaxPool2d(2, 2), nn.BatchNorm2d(4), nn.Sigmoid()
        )

    else:
        logging.error(f'Unknown convolutional net name: {layer_sel}')
        raise ValueError(f'Unknown convolutional net name: {layer_sel}')


def get_decoder(layer_sel, in_channels, out_channels):
    """ Returns the Decoder model based on the selected layer configuration. """
    
    if layer_sel == 'ConvM1':
        # Added output_padding=1 to match Keras shape multiplication
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(in_channels), nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    elif layer_sel == 'ConvM2':
        # Replaced Keras UpSampling2D with PyTorch Upsample
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding='same'), nn.Sigmoid(),
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding='same'), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding='same'), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding='same'), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), nn.Sigmoid(),
            nn.Upsample(scale_factor=4, mode='nearest'), nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(32),
            
            nn.Conv2d(32, out_channels, kernel_size=8, stride=1, padding='same'), nn.Sigmoid()
        )

    elif layer_sel == 'ConvM3':
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
            nn.ConvTranspose2d(in_channels, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=1, padding=1), nn.Sigmoid() 
        )

    elif layer_sel == 'ConvM4':
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding='same'), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(in_channels),
            
            nn.Conv2d(in_channels, 8, kernel_size=5, stride=1, padding='same'), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(8),
            
            nn.Conv2d(8, out_channels, kernel_size=3, stride=1, padding='same'), nn.Sigmoid()
        )

    elif layer_sel == 'ConvM5':
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding='same'), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(in_channels),
            
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding='same'), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(8),
            
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding='same'), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(16),
            
            nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding='same'), nn.Sigmoid()
        )

    elif layer_sel == 'ConvM6':
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding='same'), nn.Sigmoid(),
            nn.Upsample(scale_factor=4, mode='nearest'), nn.BatchNorm2d(in_channels),
            
            nn.Conv2d(in_channels, 8, kernel_size=5, stride=1, padding='same'), nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm2d(8),
            
            nn.Conv2d(8, out_channels, kernel_size=3, stride=1, padding='same'), nn.Sigmoid()
        )

    else:
        logging.error(f'Unknown convolutional net name: {layer_sel}')
        raise ValueError(f'Unknown convolutional net name: {layer_sel}')