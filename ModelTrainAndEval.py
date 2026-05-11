# -*- coding: utf-8 -*-
"""
Created on Thurs Oct 13 2022

@author: Simon Bilik

This class is used for training and evaluation of the selected model

"""

import os
import random
import logging
import traceback

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from scipy import stats
from skimage.util import view_as_blocks
from skimage.metrics import structural_similarity as SSIM

from ModelSaved import ModelSaved 

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class ModelTrainAndEval():
    """Class used for training and evaluation of the selected PyTorch model."""

    def __init__(self, modelPath, model, layer, dataGenerator, labelInfo, imageDim, imIndxList, numEpoch, evalFlag, npzSave):
        
        self.labelInfo = labelInfo
        self.numEpoch = numEpoch
        
        self.imageDim = imageDim
        self.imIndxList = imIndxList
          
        self.layerName = layer
        self.modelName = model
        self.modelPath = modelPath
      
        self.imageDim = imageDim
        self.npzSave = npzSave
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        self.dataGenerator = dataGenerator

        modelObj = ModelSaved(self.modelName, self.layerName, self.imageDim)
        
        self.model = modelObj.get_model().to(self.device)
        self.typeAE = self.modelName 
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        self.trainHistory = {'loss': [], 'val_loss': [], 'kl_loss': [], 'total_loss': []}

        logging.info('-' * 96)
        logging.info(f"Autoencoder architecture name: {self.layerName}-{self.modelName}_{self.labelInfo}")
        logging.info('')

        if self.numEpoch > 0:
            self.modelTrain()

        if evalFlag:
            self.dataEncodeDecode()


    def calculate_loss(self, x, output):
        if self.typeAE in ['BAE1', 'BAE2', 'DAE', 'AttnAE']:
            recon = output
            loss = nn.MSELoss()(recon, x)
            return loss, {'loss': loss.item()}
            
        elif self.typeAE == 'SAE':
            recon = output
            mse_loss = nn.MSELoss()(recon, x)
            sparsity_weight = 1e-4 
            total_loss = mse_loss + (sparsity_weight * self.model.sparsity_loss)
            return total_loss, {'total_loss': total_loss.item(), 'mse_loss': mse_loss.item(), 'sparsity_loss': self.model.sparsity_loss.item()}
            
        elif self.typeAE in ['VAE1', 'VAE2']:
            recon, mu, log_var = output
            recon_loss = nn.MSELoss(reduction='sum')(recon, x)
            
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            total_loss = recon_loss + kl_loss
            return total_loss, {'total_loss': total_loss.item(), 'kl_loss': kl_loss.item()}

        elif self.typeAE == 'VQVAE1':
            reconstructions, vq_loss = output
            
            recon_loss = torch.nn.functional.mse_loss(reconstructions, x)
            
            total_loss = recon_loss + vq_loss
            
            loss_dict = {
                'loss': total_loss.item(),
                'recon_loss': recon_loss.item(),
                'vq_loss': vq_loss.item()}  
            return total_loss, loss_dict      
        else:
            raise NotImplementedError(f"Loss for {self.typeAE} not implemented.")


    def modelTrain(self):
        """Custom PyTorch Training Loop."""
        try:
            valDS = None if self.typeAE in ['VAE1', 'VAE2', 'VQVAE1'] else self.dataGenerator.dsValid
            
            #Early Stopping
            patience = 20
            best_loss = float('inf')
            epochs_no_improve = 0
            
            for epoch in range(self.numEpoch):
                self.model.train()
                epoch_losses = []
                

                for batch in self.dataGenerator.dsTrain:
                    x = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                    
                    self.optimizer.zero_grad()
                    output = self.model(x)
                    
                    loss, loss_dict = self.calculate_loss(x, output)
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_losses.append(loss_dict)
                

                avg_train_loss = np.mean([d.get('total_loss', d.get('loss')) for d in epoch_losses])
                self.trainHistory['loss'].append(avg_train_loss)
                if self.typeAE in ['VAE1', 'VAE2']:
                    self.trainHistory['total_loss'].append(avg_train_loss)
                    self.trainHistory['kl_loss'].append(np.mean([d['kl_loss'] for d in epoch_losses]))

                logging.info(f"Epoch {epoch+1}/{self.numEpoch} - Train Loss: {avg_train_loss:.4f}")

                val_loss_val = avg_train_loss 
                if valDS is not None:
                    self.model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for batch in valDS:
                            x_val = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                            output_val = self.model(x_val)
                            v_loss, _ = self.calculate_loss(x_val, output_val)
                            val_losses.append(v_loss.item())
                    
                    val_loss_val = np.mean(val_losses)
                    self.trainHistory['val_loss'].append(val_loss_val)
                    logging.info(f"Epoch {epoch+1} - Val Loss: {val_loss_val:.4f}")

                #Early Stopping
                if val_loss_val < best_loss:
                    best_loss = val_loss_val
                    epochs_no_improve = 0
                    # Save best model
                    torch.save(self.model.state_dict(), os.path.join(self.modelPath, 'model.weights.pt'))
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        logging.info("Early stopping triggered.")
                        break

        except Exception as e:
            logging.error(f'Training of the {self.layerName}-{self.modelName} model failed...')
            traceback.print_exc()
            return
        else:
            logging.info(f'Training of the {self.layerName}-{self.modelName} model was finished...')
            self.visualiseTrainResults()


    def dataEncodeDecode(self):
        actStrs = ['Train', 'Test', 'Valid']
        dataGens = [self.dataGenerator.dsTrain, self.dataGenerator.dsTest, self.dataGenerator.dsValid]
        
        self.model.eval() 

        for actStr, dataGen in zip(actStrs, dataGens):
            try:
                enc_list, dec_list, org_list = [], [], []
                
                with torch.no_grad():
                    for batch in dataGen:
                        x = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                        
                        if self.typeAE in ['VAE1', 'VAE2']:
                            dec_out, z_mean, z_log_var = self.model(x)
                            enc_out = torch.stack((z_mean, z_log_var), dim=-1)
                            
                        elif self.typeAE == 'VQVAE1':
                                vqvae_out = self.model(x)
                                dec_out = vqvae_out[0] if isinstance(vqvae_out, tuple) else vqvae_out
                                
                                encoder_out = self.model.encoder(x)
                                enc_out = encoder_out[0] if isinstance(encoder_out, tuple) else encoder_out
                        else:
                            enc_out = self.model.encoder(x)
                            dec_out = self.model(x)

                        x_np = x.detach().cpu().permute(0, 2, 3, 1).numpy()
                        dec_np = dec_out.detach().cpu().permute(0, 2, 3, 1).numpy()
                        enc_np = enc_out.detach().cpu().numpy() 

                        org_list.append(x_np)
                        dec_list.append(dec_np)
                        enc_list.append(enc_np)
                org_full = np.concatenate(org_list, axis=0)
                enc_full = np.concatenate(enc_list, axis=0)
                dec_full = np.concatenate(dec_list, axis=0)

                if not hasattr(self.dataGenerator, 'processedData'):
                    self.dataGenerator.processedData = {k: {} for k in actStrs}

                self.dataGenerator.processedData[actStr]['Org'] = org_full
                self.dataGenerator.processedData[actStr]['Enc'] = enc_full
                self.dataGenerator.processedData[actStr]['Dec'] = dec_full

                if self.npzSave:
                    outputPath = os.path.join(self.modelPath, 'modelData', f'Eval_{actStr}')
                    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
                    np.savez_compressed(outputPath, encData=enc_full, decData=dec_full)

                if actStr == 'Test':
                    self.visualiseEncDecResults(actStr) 
                    self.getSimilarityCoeff(dec_full)

            except Exception as e:
                logging.error(f'Data encode/decode for {self.layerName}-{self.modelName} failed...')
                traceback.print_exc()
                return
        else:
            logging.info(f'Data encode/decode for {self.layerName}-{self.modelName} successful...')

    def getSimilarityCoeff(self, decData):
        classIDs = [-1, 1]
        classLab = ['NOK', 'OK']
        pAvg, ssimAvg = [], []
        
        tempData = self.dataGenerator.processedData.get('Test')
        orgData = tempData.get('Org')
        labels = tempData.get('Lab')
        
        for classID, classLb in zip(classIDs, classLab):
            idx = np.where(labels == classID)[0]
            if len(idx) == 0:
                pAvg.append(0); ssimAvg.append(0); continue

            orgDataSel = orgData[idx]
            decDataSel = decData[idx]
            pVal, ssimVal = [], []
            
            for imgOrg, imgDec in zip(orgDataSel, decDataSel):
                if imgOrg.shape[0] < imgOrg.shape[2]:
                    imgOrg = imgOrg.transpose(1, 2, 0)
                    imgDec = imgDec.transpose(1, 2, 0)

                s = SSIM(imgOrg, imgDec, data_range=1.0, channel_axis=2)
                ssimVal.append(s)
                
                imgOrg_u8 = (np.clip(imgOrg, 0, 1) * 255).astype(np.uint8)
                imgDec_u8 = (np.clip(imgDec, 0, 1) * 255).astype(np.uint8)

                if imgOrg_u8.shape[2] == 3:
                    imgOrgGray = (0.299 * imgOrg_u8[:,:,0] + 0.587 * imgOrg_u8[:,:,1] + 0.114 * imgOrg_u8[:,:,2]).astype(np.uint8)
                    imgDecGray = (0.299 * imgDec_u8[:,:,0] + 0.587 * imgDec_u8[:,:,1] + 0.114 * imgDec_u8[:,:,2]).astype(np.uint8)
                else:
                    imgOrgGray = np.squeeze(imgOrg_u8)
                    imgDecGray = np.squeeze(imgDec_u8)
                
                try:
                    h, w = imgOrgGray.shape
                    if h % 32 == 0 and w % 32 == 0:
                        batchOrg = view_as_blocks(imgOrgGray, (32, 32)).reshape(-1, 32, 32)
                        batchDec = view_as_blocks(imgDecGray, (32, 32)).reshape(-1, 32, 32)
                        
                        temppVal = [np.abs(stats.pearsonr(o.flatten(), d.flatten()).statistic) 
                                    for o, d in zip(batchOrg, batchDec)]
                        pVal.append(np.median(np.array(temppVal)))
                except Exception: continue

            ssimAvg.append(np.median(ssimVal) if ssimVal else 0)
            pAvg.append(np.median(pVal) if pVal else 0)

    def visualiseTrainResults(self):
        try:
            history = self.trainHistory
            tempTitle = f'Loss de {self.layerName}-{self.modelName}_{self.labelInfo}'
            
            if self.typeAE in ['VAE1', 'VAE2']:
                train_loss = history.get('total_loss', [])
                val_loss = history.get('kl_loss', [])
                plotLabel = 'KL Loss'
            elif self.typeAE == 'VQVAE1':
                train_loss = history.get('total_loss', [])
                val_loss = history.get('vqvae_loss', [])
                plotLabel = 'VQ-VAE Loss'
            else:
                train_loss = history.get('train_loss', [])
                val_loss = history.get('val_loss', [])
                plotLabel = 'Validation Loss'
            
            fig, axarr = plt.subplots(2, figsize=(10, 8))
            fig.suptitle(tempTitle, fontsize=14)
            
            axarr[0].plot(train_loss, label='Train')
            axarr[0].set(xlabel='Epochs', ylabel='Loss')
            axarr[0].legend()
            
            axarr[1].plot(val_loss, color='orange', label=plotLabel)
            axarr[1].set(xlabel='Epochs', ylabel=plotLabel)
            axarr[1].legend()
            
            fig.tight_layout(rect=(0, 0.03, 1, 0.95))
            
            save_dir = os.path.join(self.modelPath, 'modelData')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{self.layerName}-{self.modelName}_{self.labelInfo}_TrainLosses.png')
            
            fig.savefig(save_path)
            plt.close(fig)
        except Exception:
            logging.error("Error visualizing")
            traceback.print_exc()

    def visualiseEncDecResults(self, actStr):

        try:
            # Set the train or test data
            if actStr == 'Train':
                label = 'during training'
            else:
                label = 'during testing'
            
            # Get the original, encoded and decoded data
            tempData = self.dataGenerator.processedData.get(actStr)
            
            def ensure_hwc(arr):
                if arr is None: return arr
                if arr.ndim == 4:
                    if arr.shape[2] == arr.shape[3] and arr.shape[1] != arr.shape[2]:
                        return np.transpose(arr, (0, 2, 3, 1))
                    elif arr.shape[1] in [1, 3] and arr.shape[-1] not in [1, 3]:
                        return np.transpose(arr, (0, 2, 3, 1))
                return arr

            orgData = ensure_hwc(tempData.get('Org'))
            encData = ensure_hwc(tempData.get('Enc'))
            decData = ensure_hwc(tempData.get('Dec'))
            
            diffData = np.subtract(orgData, decData)
            
            imgSourceList = [orgData, encData, decData, diffData]
            imgTitleList = ['Original', 'Encoded', 'Decoded', 'Difference']

            fig, axarr = plt.subplots(len(self.imIndxList), 4)
            tempTitle = 'Visualisations of the ' + self.layerName + '-' + self.modelName + '_' + self.labelInfo + ' model'
            
            fig.suptitle(tempTitle, fontsize=18)
            fig.set_size_inches(16, 4 * len(self.imIndxList)) 
            
            vIdx = 0
            
            def normalize_to_uint8(arr):
                arr_min = np.min(arr)
                arr_max = np.max(arr)
                if arr_max == arr_min:
                    return np.zeros_like(arr, dtype=np.uint8)
                return (255.0 * (arr - arr_min) / (arr_max - arr_min)).astype(np.uint8)

            def prep_for_imshow(img):
                if img.ndim == 3 and img.shape[-1] == 1:
                    return img.squeeze(-1)
                return img

            for imgIndx in self.imIndxList:
                if imgIndx >= len(orgData): continue 
                
                hIdx = 0
                
                for imgTitle, imgSource in zip(imgTitleList, imgSourceList):
                    
                    ax = axarr[vIdx, hIdx] if len(self.imIndxList) > 1 else axarr[hIdx]
                    ax.set_title(imgTitle)
                    
                    img_single = imgSource[imgIndx]
                    
                    # Plot the encoded images
                    if imgTitle == 'Encoded':
                        if self.typeAE in ['VAE1', 'VAE2']:
                            ax.scatter(img_single[:, 0], img_single[:, 1], s = 4)
                            ax.set(xlabel = "Mean", ylabel = "Variance", xlim = (-10, 10), ylim = (-10, 10))
                            
                        elif self.typeAE in ['BAE1', 'BAE2', 'DAE', 'SAE', 'AttnAE']:
                            if img_single.ndim == 3:
                                ax.imshow(normalize_to_uint8(img_single.mean(axis=-1)))
                            else:
                                ax.imshow(normalize_to_uint8(img_single))
                            ax.axis('off')
                            
                        elif self.typeAE == 'VQVAE1':
                            ax.imshow(normalize_to_uint8(prep_for_imshow(img_single)))
                            ax.axis('off')
                    
                    else:
                        ax.imshow(normalize_to_uint8(prep_for_imshow(img_single)))
                        ax.axis('off')
                        
                    hIdx += 1
                    
                vIdx += 1

            # Save the illustration figure
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            
            save_dir = os.path.join(self.modelPath, 'modelData')
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, self.layerName + '-' + self.modelName + '_' + self.labelInfo + '_' + actStr + '_AEResults.png')
            fig.savefig(save_path)
            plt.close(fig)
        
        except Exception:
            logging.error('Data visualisation of the model ' + self.layerName + '-' + self.modelName + '_' + self.labelInfo + ' and its ' + actStr + ' dataset failed...')
            traceback.print_exc()

        else:
            logging.info('Data visualisation of the model ' + self.layerName + '-' + self.modelName + ' and its ' + actStr + ' dataset was succesful...')