# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:23:05 2021

@author: Šimon Bilík

This script loads the data from the OK x NOK folder structure
and saves them to the two npz files - one for the training and the second for validation.

data_dir: Set this variable with the path to your dataset

"""

import os
import argparse
import numpy as np

from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Create custom NPZ data for the model evaluation')
    parser.add_argument('--data_path', default='../IndustryBiscuit_Folders/', type=str, help='path to dataset')

    args = parser.parse_args()

    return args

def NormalizeData(data):
    """Performs the data normalization"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def main():
    """main function"""
    args = parse_args()

    # The custom dataset directory
    data_dir = args.data_path

    # label
    # 0:NOK, 1:OK

    train_image = []
    train_label = []

    test_image = []
    test_label = []

    valid_image = []
    valid_label = []

    print('Loading the input data...')

    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        
        temp = os.path.basename(dirpath)
        
        for im_path in filenames:
            # Load and resize the input images
            img = Image.open(os.path.join(dirpath, im_path))            
            img = img.resize((256, 256))
            
            # Convert the input image to numpy array, normalize it and center it
            pixels = np.asarray(img)
            pixels = pixels.astype('float32')
            pixels /= 255.0
            # pixels = np.expand_dims(pixels, -1)
            
            # mean = pixels.mean()
            # pixels = pixels - mean

            # Train images
            if (dirpath == data_dir + 'train/ok'):
                train_image.append(pixels)
                train_label.append(1)
                
            elif (dirpath == data_dir + 'train/nok'):
                train_image.append(pixels)
                train_label.append(0)

            # Test images
            if (dirpath == data_dir + 'test/ok'):
                test_image.append(pixels)
                test_label.append(1)
                
            elif (dirpath == data_dir + 'test/nok'):
                test_image.append(pixels)
                test_label.append(0)

            # Validation images
            if (dirpath == data_dir + 'valid/ok'):
                valid_image.append(pixels)
                valid_label.append(1)
                
            elif (dirpath == data_dir + 'valid/nok'):
                valid_image.append(pixels)
                valid_label.append(0)

    if train_image:
        train_images = NormalizeData(np.array(train_image))
    else:
        train_images = np.array([])

    train_labels = np.array(train_label)

    if test_image:
        test_images = NormalizeData(np.array(test_image))
    else:
        test_images = np.array([])

    test_labels = np.array(test_label)

    if valid_image:
        valid_images = NormalizeData(np.array(valid_image))
    else:
        valid_images = np.array([])

    valid_labels = np.array(valid_label)

    print('Train images array shape: ', train_images.shape)
    print('Test images array shape: ', test_images.shape)
    print('Validation images array shape: ', valid_images.shape)

    ## Save the files
    print('Saving the input data...')

    # Save the train and test data
    np.savez(os.path.join('./data', 'cookie_train.npz'), images = train_images, labels = train_labels)
    np.savez(os.path.join('./data', 'cookie_test.npz'), images = test_images, labels = test_labels)
    np.savez(os.path.join('./data', 'cookie_valid.npz'), images = valid_images, labels = valid_labels)

    # Save the labels to the separate file
    np.savez(os.path.join('./data', 'cookie_labels.npz'), train = train_labels, valid = valid_labels, test = test_labels)

    print('Input data saved...')

if __name__ == '__main__':
    main()
