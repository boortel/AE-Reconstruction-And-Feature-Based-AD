import os
import argparse
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img
# from tensorflow.keras.utils.image_dataset_from_directory(

from ModelSaved import ModelSaved

allowedSuffixes = [
    '.jpg', '.jpeg',
    '.png', '.bmp', '.gif'
]

def parse_args():
    parser = argparse.ArgumentParser(description = 'Train and evaluate models defined in the ini files of the init directory')
    
    parser.add_argument('--modelName', '-m', default = 'BAE2', type = str, help = '')
    parser.add_argument('--layerName', '-l', default = 'ConvM1', type = str, help = '')
    parser.add_argument('--imHeight', '--height', default = 256, type = int, help = '')
    parser.add_argument('--imWidth', '--width', default = 256, type = int, help = '')
    parser.add_argument('--imChannel', '--channel', default = 3, type = int, help = '')
    parser.add_argument('--modelWeights', '-w', default = 'data/Cookie_OCC/ConvM1_Cookie_OCC/BAE2/model.weights.h5', type = str, help = '')
    parser.add_argument('--images', '-i', default = '../Datasets/IndustryBiscuit/Images', type = str, help = '')
    parser.add_argument('--output', '-o', default='../ClassifiedData/IndustryBiscuit', type = str, help = '')
    parser.add_argument('--batchSize', '--batch', default = 512, type = int, help = '')

    args = parser.parse_args()

    return args

def main(args):
    imageDim = (args.imHeight, args.imWidth, args.imChannel)
    imageSize = (args.imHeight, args.imWidth)
    modelObj = ModelSaved(args.modelName, args.layerName, imageDim, dataVariance = 0.5, intermediateDim = 64, latentDim = 32, num_embeddings = 32)
    modelWeightsPath = args.modelWeights
    imagesPath = args.images
    batchSize = args.batchSize

    model = modelObj.model
    model.load_weights(modelWeightsPath)

    ax_cols = 2
    ax_rows = 4
    fig, axs = plt.subplots(ax_rows, ax_cols * 2, figsize=(10, 10))
    plt.ion()
    plt.show()

    fileList = map(Path, os.listdir(imagesPath))
    imageFileList = [file for file in fileList if file.suffix.lower() in allowedSuffixes]
    images = []
    for imageFile in imageFileList:
        imagePath = Path(imagesPath) / imageFile
        image = load_img(imagePath, target_size=imageSize)
        image_array = img_to_array(image)
        images.append(image_array)

        if len(images) >= batchSize:
            input = np.array(images) / 255
            images.clear()
            output = model.predict(input)
            
            for c in range(ax_cols):
                for r in range(ax_rows):
                    i = int(c * r / ax_cols / ax_rows * len(input))
                    axs[r, 2 * c].imshow(input[i])
                    axs[r, 2 * c + 1].imshow(output[i])

                    axs[r, 2 * c].axis('off')
                    axs[r, 2 * c + 1].axis('off')

            plt.pause(.1)
            plt.draw()

    return

if __name__ == '__main__':
    args = parse_args()
    main(args)