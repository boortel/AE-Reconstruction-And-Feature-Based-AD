import os
import json
import shutil

from math import floor

## Main function
def main():

    # Set the parameters
    # TODO: argparser
    datasetNames = ['Aphanizomenon_flosaquae', 'Centrales_sp', 'Dolichospermum_an', 'Chaetoceros_sp', 'Nodularia_spumigena', 'Pauliella_taeniata', 'Peridiniella_catenata_chain', 'Peridiniella_catenata_single', 'Skeletonema_marinoi']
    #datasetNames = ['Dolichospermum_an']
    
    tstRatioNOK = 0.6
    valRatioNOK = 1 - tstRatioNOK

    for datasetName in datasetNames:

        ## Set the dataset path and create directory structure for OCC

        basePath = './' + datasetName + '/'

        imPath = basePath + datasetName + '_YOLO/images'
        lbPath = basePath + datasetName + '_YOLO/labels'
        occPath = basePath + datasetName + '_OCC'

        # Create the directories for the image storage
        if not os.path.exists(occPath):
            
            # Create the folder structure
            os.makedirs(occPath + '/train/ok')
            os.makedirs(occPath + '/train/nok')
            os.makedirs(occPath + '/valid/ok')
            os.makedirs(occPath + '/valid/nok')
            os.makedirs(occPath + '/test/ok')
            os.makedirs(occPath + '/test/nok')

        else:
            print("Folder structure with the OCC dataset already exists...")
            continue


        ## Get the OK, NOK and Parasite class IDs

        # Open JSON file as dictionary
        f = open(basePath + '/' + datasetName + '_YOLO/notes.json')

        notes = json.load(f)

        # Define the empty lists
        idsOK = []
        idsNOK = []
        idPAR = -1

        for i in notes['categories']:

            if  i['name'] == 'Parasite':
                idPAR = i['id']

            elif '_Clean' in i['name']:
                idsOK.append(i['id'])

            else:
                idsNOK.append(i['id'])


        ## Split the OK and NOK images

        imgsOK = []
        imgsNOK = []

        # Loop through the label files
        for labelFile in os.listdir(lbPath):

            # Open the annotations files
            with open(os.path.join(lbPath, labelFile), 'r') as f:

                # Read the annotations
                annotations = f.readlines()

                # Get the image name
                imgFile = labelFile.replace('txt', 'png')

                lsAnnot = []

                # Get the single annotations
                for annotation in annotations:

                    annotation = annotation.strip()
                    annotation = annotation.split()

                    # Append annotation ID if it exists
                    if annotation:
                        lsAnnot.append(int(annotation[0]))

                # Check if the annotations contain NOK or Parasite class
                if any(x in lsAnnot for x in idsNOK) or idPAR in lsAnnot:
                    imgsNOK.append(imgFile)
                else:
                    imgsOK.append(imgFile)

        # Get total counts of the OK and NOK images
        cntOK = len(imgsOK)
        cntNOK = len(imgsNOK)

        # Print out the dataset statistics
        print('Dataset name: ', datasetName)
        print('OK samples count: ', cntOK)
        print('NOK samples count: ', cntNOK)
        print('')
        
        # Set the train-test-val split based on the available data (tst and val dataset will be always balanced)
        cntOK_maxTstVal = cntOK * 0.2 + cntOK * 0.1
        cntOK_desTstVal = cntNOK * tstRatioNOK + cntNOK * valRatioNOK

        if cntOK_maxTstVal >= cntOK_desTstVal:
            cntOK_setTst = int(floor(cntOK_desTstVal * tstRatioNOK))
            cntOK_setVal = int(floor(cntOK_desTstVal * valRatioNOK))

        else:
            cntOK_setTst = int(floor(cntOK_maxTstVal * tstRatioNOK))
            cntOK_setVal = int(floor(cntOK_maxTstVal * valRatioNOK))

        # Copy the images to the OCC folder structure

        # Test data
        for i in range(cntOK_setTst):
            shutil.copy(os.path.join(imPath, imgsOK[i]), (occPath + '/test/ok/'))
            shutil.copy(os.path.join(imPath, imgsNOK[i]), (occPath + '/test/nok/'))

        # Valid data
        for i in range(cntOK_setVal):
            shutil.copy(os.path.join(imPath, imgsOK[i + cntOK_setTst]), (occPath + '/valid/ok/'))
            shutil.copy(os.path.join(imPath, imgsNOK[i + cntOK_setTst]), (occPath + '/valid/nok/'))

        # Train data
        for i in range(cntOK - cntOK_setTst - cntOK_setVal):
            shutil.copy(os.path.join(imPath, imgsOK[i + cntOK_setTst + cntOK_setVal]), (occPath + '/train/ok/'))

        # Save everything to zip folder
        #shutil.make_archive(datasetName, 'zip', basePath)
        


if __name__ == '__main__':
    main()
