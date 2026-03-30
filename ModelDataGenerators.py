# -*- coding: utf-8 -*-
"""
Created on Thurs Oct 13 2022

@author: Simon Bilik

This class is used to set data generators and to save the original data to the npz file

"""
# -*- coding: utf-8 -*-
"""
PyTorch Implementation of ModelDataGenerators

This class is used to set data generators and to save the original data to the npz file
"""

import os
import logging
import traceback
import cv2 as cv
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ModelDataGenerators():
    
    ## Set the constants and paths
    def __init__(self, experimentPath, datasetPath, labelInfo, imageDim, batchSize, npzSave):
        
        # Set the paths and constants
        self.experimentPath = experimentPath
        self.datasetPath = datasetPath
        self.labelInfo = labelInfo
        self.batchSize = batchSize
        self.imageDim = imageDim
        self.npzSave = npzSave
        
        # Create datasets
        self.getGenerators()
        
        # Save the original data to NPZ
        self.saveOrgData()
        
    
    ## Define augmentation pipeline for training dataset
    def getTrainTransforms(self):

        transform_list = [
            transforms.Resize((self.imageDim[0], self.imageDim[1]), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.5, hue=0.2), 
            transforms.ToTensor()
        ]
        
        class AddSaltPepperNoise(object):
            def __init__(self, prob=0.1):
                self.prob = prob
                
            def __call__(self, tensor):
                noise_tensor = torch.rand(tensor.size())
            
                tensor = torch.where(noise_tensor < (self.prob / 2), torch.ones_like(tensor), tensor)
                
                tensor = torch.where(noise_tensor > 1 - (self.prob / 2), torch.zeros_like(tensor), tensor)
                return tensor

        transform_list.append(AddSaltPepperNoise(prob=0.1))
        
        class RandomInvert(object):
            def __call__(self, tensor):
                if torch.rand(1).item() < 0.5:
                    return 1.0 - tensor
                return tensor
                
        transform_list.append(RandomInvert())
        
        if self.imageDim[2] == 1:
            transform_list.insert(0, transforms.Grayscale(num_output_channels=1))

        return transforms.Compose(transform_list)

    
    def getTestTransforms(self):
        transform_list = [
            transforms.Resize((self.imageDim[0], self.imageDim[1]), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ]
        
        if self.imageDim[2] == 1:
            transform_list.insert(0, transforms.Grayscale(num_output_channels=1))

        return transforms.Compose(transform_list)

    
    ## Set the data generator (DataLoader)
    def setGenerator(self, mode):
        folder_path = os.path.join(self.datasetPath, mode)
        
        if mode == 'train':
            transform = self.getTrainTransforms()
            shuffle = True
        else:
            transform = self.getTestTransforms()
            shuffle = False
        
        class AutoencoderDataset(torch.utils.data.Dataset):
            def __init__(self, folder_path, mode, transform, imageDim):
                self.base_dataset = datasets.ImageFolder(root=folder_path, transform=transform)
                self.mode = mode
                
                clean_transform_list = [
                    transforms.Resize((imageDim[0], imageDim[1]), interpolation=transforms.InterpolationMode.NEAREST),
                    transforms.ToTensor()
                ]
                if imageDim[2] == 1:
                    clean_transform_list.insert(0, transforms.Grayscale(num_output_channels=1))
                self.clean_transform = transforms.Compose(clean_transform_list)
                
                self.clean_base = datasets.ImageFolder(root=folder_path, transform=self.clean_transform)

            def __len__(self):
                return len(self.base_dataset)

            def __getitem__(self, idx):
                img_augmented, _ = self.base_dataset[idx]
                img_clean, label = self.clean_base[idx]
                
                return img_augmented, img_clean, label

        dataset = AutoencoderDataset(folder_path, mode, transform, self.imageDim)

        
        data_loader = DataLoader(
            dataset, 
            batch_size=self.batchSize, 
            shuffle=shuffle, 
            num_workers=4, 
            pin_memory=True if torch.cuda.is_available() else False 
        )
        

        return data_loader, data_loader 

    
    ## Get the data generators
    def getGenerators(self):
        try:
            self.tSize = (self.imageDim[0], self.imageDim[1])

            # Get train DS
            self.dsTrain, self.dsTrainL =  self.setGenerator('train')
            
            # Get validation DS
            self.dsValid, self.dsValidL =  self.setGenerator('valid')
            
            # Get test DS
            self.dsTest, self.dsTestL =  self.setGenerator('test')
            
        except Exception as e:
            logging.error(f'Data generators initialization for the {self.labelInfo} experiment failed...')
            traceback.print_exc()
            return

        else:
            logging.info(f'Data generators of the {self.labelInfo} experiment initialized...')
            
    
    ## Save the original data and labels to NPZ file
    def saveOrgData(self):
        self.processedData = {}
        
        actStrs = ['Train', 'Test', 'Valid']
        dataGens = [self.dsTrainL, self.dsTestL, self.dsValidL]
        
        for actStr, dataLoader in zip(actStrs, dataGens):
            
            orig_data_list = []
            labels_list = []
            
            for _, clean_imgs, labels in dataLoader:
                clean_imgs_np = clean_imgs.permute(0, 2, 3, 1).numpy()
                labels_np = labels.numpy()
                
                orig_data_list.append(clean_imgs_np)
                labels_list.append(labels_np)
                
            orig_data = np.concatenate(orig_data_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)

            # Get labels and transform them to format to -1: NOK and 1:OK
            nokIdx = np.where(labels == 0)
            labels[nokIdx] = -1
            
            filterStrength = 0.5
            filtered_data = []
            for img in orig_data:

                blurred = cv.GaussianBlur(img, (25,25), 0)
                filtered = filterStrength * img + (1 - filterStrength) * np.atleast_3d(blurred)
                filtered_data.append(filtered)
                
            orig_data = np.array(filtered_data)

            # Save the obtained data to NPZ
            if self.npzSave:
                outputPath = os.path.join(self.experimentPath, f'Org_{actStr}')
                os.makedirs(os.path.dirname(outputPath), exist_ok=True)
                np.savez_compressed(outputPath, orgData=orig_data, labels=labels)
            
            # Save the processed data to dictionary for a later access
            self.processedData[actStr] = {'Org': orig_data, 'Lab': labels}