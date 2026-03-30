import os
import time
import yaml
import shutil
import argparse
import configparser
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image

from ModelSaved import ModelSaved
from ModelTrainAndEval import ModelTrainAndEval

from ModelClassificationErrM import ModelClassificationErrM
from ModelClassificationSIFT import ModelClassificationSIFT
from ModelClassificationHardNet1 import ModelClassificationHardNet1
from ModelClassificationHardNet2 import ModelClassificationHardNet2
from ModelClassificationHardNet3 import ModelClassificationHardNet3
from ModelClassificationHardNet4 import ModelClassificationHardNet4
from EvaluationsToJSON import extract_eval_and_save

class SimpleImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, path

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate PyTorch AD Models')
    parser.add_argument('--saveImgToFile', '-e', default=True, type=bool)
    return parser.parse_args()

def main():
    args = parse_args()
    saveImgToFile = args.saveImgToFile        
    iniBasePath = './init'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on device: {device} ---")

    cfg = configparser.ConfigParser()

    for filename in os.listdir(iniBasePath):
        if not filename.endswith('.ini'):
            continue

        cfg.read(os.path.join(iniBasePath, filename))

        experimentPath = cfg.get('General', 'modelBasePath', fallback='NaN')
        labelInfo = cfg.get('General', 'labelInfo', fallback='NaN')
        imageDim = (cfg.getint('General', 'imHeight', fallback=0), 
                    cfg.getint('General', 'imWidth', fallback=0),
                    cfg.getint('General', 'imChannel', fallback=3))
        
        imIndxList_str = cfg.get('General', 'imIndxList', fallback='0,1,2,3')
        imIndxList = [int(i.strip()) for i in imIndxList_str.split(',')]

        predictionDataPath = cfg.get('Prediction', 'predictionDataPath', fallback='-')
        predictionResultPath = cfg.get('Prediction', 'predictionResultPath', fallback='-')
        predictionBatchSize = cfg.getint('Prediction', 'batchSize', fallback=32)
        modelName = cfg.get('Prediction', 'modelName', fallback='-')
        layerName = cfg.get('Prediction', 'layerName', fallback='-')
        featureExtractorName = cfg.get('Prediction', 'featureExtractorName', fallback='-')
        anomalyAlgorithmName = cfg.get('Prediction', 'anomalyAlgorithmName', fallback='-')

        basePath = os.path.join(experimentPath, f'{layerName}_{labelInfo}', modelName)
        aeWeightsPath = os.path.join(basePath, 'model.weights.pt')

        if not os.path.exists(aeWeightsPath):
            print(f"Error: Weights not found in {aeWeightsPath}")
            continue

        try:
            modelObj = ModelSaved(
                modelSel=modelName,         
                layerSel=layerName,         
                imageDim=imageDim,         
                dataVariance=0.5,          
                intermediateDim=64,         
                latentDim=32,               
                num_embeddings=32
            )
            model = modelObj.get_model()
            model.load_state_dict(torch.load(aeWeightsPath, map_location=device))
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"Error initializing model: {e}")
            continue

        os.makedirs(predictionResultPath, exist_ok=True)
        okPath = os.path.join(predictionResultPath, 'OK')
        nokPath = os.path.join(predictionResultPath, 'NOK')
        os.makedirs(okPath, exist_ok=True)
        os.makedirs(nokPath, exist_ok=True)

        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        img_list = [str(p) for p in Path(predictionDataPath).rglob('*') if p.suffix.lower() in extensions]
        img_list.sort()
        
        if not img_list:
            print(f"Not images found in {predictionDataPath}")
            continue

        transform = transforms.Compose([
            transforms.Resize((imageDim[0], imageDim[1]), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        dataset = SimpleImageDataset(img_list, transform=transform)
        dataloader = DataLoader(dataset, batch_size=predictionBatchSize, shuffle=False)

        all_reconstructions = []
        all_originals = []
        all_encoded = []
        
        startTime = time.time()
        with torch.no_grad():
            for batch_imgs, _ in dataloader:
                batch_imgs = batch_imgs.to(device)
                
                output = model(batch_imgs)
                enc_out = model.encoder(batch_imgs)
                
                recon = output[0] if isinstance(output, tuple) else output
                
                all_originals.append(batch_imgs.cpu().numpy().transpose(0, 2, 3, 1))
                all_reconstructions.append(recon.cpu().numpy().transpose(0, 2, 3, 1))
                
                enc_np = enc_out.cpu().numpy()
                
                enc_np = enc_out.cpu().numpy()
                
                if enc_np.ndim == 4:
                    enc_np = np.max(enc_np, axis=1) 
                    enc_np = np.expand_dims(enc_np, axis=-1)
                    
                elif enc_np.ndim == 2:
                    dim = enc_np.shape[1]
                    grid_size = int(np.sqrt(dim))
                    if grid_size * grid_size == dim:
                        enc_np = enc_np.reshape(-1, grid_size, grid_size, 1)
                    else:
                        enc_np = enc_np.reshape(-1, 1, dim, 1)

                all_encoded.append(enc_np)

        orig_data = np.concatenate(all_originals, axis=0)
        dec_data = np.concatenate(all_reconstructions, axis=0)
        enc_data = np.concatenate(all_encoded, axis=0)

        filterStrength = 0.5
        orig_data_filtered = np.array([
            (filterStrength * img + (1 - filterStrength) * cv.GaussianBlur(img, (25, 25), 0)) 
            for img in orig_data
        ])

        if orig_data_filtered.ndim == 3 and imageDim[2] == 1:
            orig_data_filtered = np.expand_dims(orig_data_filtered, axis=-1)

        prediction_data = {
            'Test': {
                'Org': orig_data_filtered,
                'Dec': dec_data,
                'Enc': enc_data,
                'Lab': np.array([-1] * len(dec_data))
            }
        }

        model_master = ModelTrainAndEval(
            modelPath=basePath,
            model=modelName,
            layer=layerName,
            dataGenerator=type('obj', (object,), {'processedData': prediction_data}),
            labelInfo=labelInfo,
            imageDim=imageDim,
            imIndxList=imIndxList,
            numEpoch=0,
            evalFlag=False,
            npzSave=False
        )

        model_master.visualiseEncDecResults('Test')
        model_master.getSimilarityCoeff(dec_data)

        extractors = {
            'ErrM': ModelClassificationErrM,
            'SIFT': ModelClassificationSIFT,
            'HardNet1': ModelClassificationHardNet1,
            'HardNet2': ModelClassificationHardNet2,
            'HardNet3': ModelClassificationHardNet3,
            'HardNet4': ModelClassificationHardNet4,
        }

        if featureExtractorName not in extractors:
            print(f"Error: Extractor {featureExtractorName} not defined.")
            continue

        feature_extractor = extractors[featureExtractorName](
            os.path.join(basePath, 'modelData'),
            predictionResultPath, modelName, layerName, 'Classification test', imageDim,
            {'Predict': prediction_data['Test']},
            [anomalyAlgorithmName],
            False
        )
        
        labels = feature_extractor.predictedLabels 
        
        labelsDict = {'OK': [], 'NOK': []}
        OK_results = []
        NOK_results = []

        for path, label in zip(img_list, labels):
            abs_path = os.path.abspath(path)
            if label:
                OK_results.append(abs_path)
                if saveImgToFile: shutil.copy(path, okPath)
            else:
                NOK_results.append(abs_path)
                if saveImgToFile: shutil.copy(path, nokPath)

        if OK_results: labelsDict['OK'].append(OK_results)
        if NOK_results: labelsDict['NOK'].append(NOK_results)

        print(f"--- Processed: {len(img_list)} images in {time.time() - startTime:.2f}s ---")
        print(f"Results -> OK: {len(OK_results)} | NOK: {len(NOK_results)}")

        with open(os.path.join(predictionResultPath, 'labels.yaml'), 'w') as f:
            yaml.safe_dump(labelsDict, f)
            
    extract_eval_and_save()

if __name__ == '__main__':
    main()