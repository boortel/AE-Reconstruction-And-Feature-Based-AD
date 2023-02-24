import json

## Main function
def main():

    # Open JSON files as dictionary
    f = open('parsed-all-log_PlanktonAll.json')
    dataAll = json.load(f)

    PlanktonAll_OCC = dataAll['PlanktonAll_OCC']

    f = open('parsed-all-log_PlanktonSpec.json')
    dataPln = json.load(f)


    ## Find results of the model-extractor-classifier (MEC) with the highest AUC
    bestMEC = {'model': '', 'f_ext': '', 'class': '', 'AUC': 0, 'PREC': 0, 'REC': 0, 'F1': 0}
    bestAUC = 0

    # Loop through all models
    for model in PlanktonAll_OCC['models']:

        # Loop through all feature extractors
        for f_ext in PlanktonAll_OCC['models'][model]['f_exts']:

            # Loop through all classifiers
            for clas in PlanktonAll_OCC['models'][model]['f_exts'][f_ext]['classifiers']:
                
                if bestAUC <= PlanktonAll_OCC['models'][model]['f_exts'][f_ext]['classifiers'][clas]['alg_metrics']['f1']:
                    
                    # Assign the best AUC and MEC parameters
                    bestAUC = PlanktonAll_OCC['models'][model]['f_exts'][f_ext]['classifiers'][clas]['alg_metrics']['f1']

                    AUC = PlanktonAll_OCC['models'][model]['f_exts'][f_ext]['classifiers'][clas]['alg_metrics']['auc-roc']

                    PREC = PlanktonAll_OCC['models'][model]['f_exts'][f_ext]['classifiers'][clas]['alg_metrics']['precision']
                    REC = PlanktonAll_OCC['models'][model]['f_exts'][f_ext]['classifiers'][clas]['alg_metrics']['recall']
                    F1 = PlanktonAll_OCC['models'][model]['f_exts'][f_ext]['classifiers'][clas]['alg_metrics']['f1']

                    bestMEC['model'] = model
                    bestMEC['f_ext'] = f_ext
                    bestMEC['class'] = clas

                    bestMEC['PREC'] = PREC
                    bestMEC['AUC'] = AUC
                    bestMEC['REC'] = REC
                    bestMEC['F1'] = F1


    ## Get per-species results for the selected MEC (Tab1)
    speciesRes = {}

    # Loop through the plankton species
    for species in dataPln:

        AUC = dataPln[species]['models'][bestMEC['model']]['f_exts'][bestMEC['f_ext']]['classifiers'][bestMEC['class']]['alg_metrics']['auc-roc']

        PREC = dataPln[species]['models'][bestMEC['model']]['f_exts'][bestMEC['f_ext']]['classifiers'][bestMEC['class']]['alg_metrics']['precision']
        REC = dataPln[species]['models'][bestMEC['model']]['f_exts'][bestMEC['f_ext']]['classifiers'][bestMEC['class']]['alg_metrics']['recall']
        F1 = dataPln[species]['models'][bestMEC['model']]['f_exts'][bestMEC['f_ext']]['classifiers'][bestMEC['class']]['alg_metrics']['f1']

        speciesRes[species] = {'AUC': AUC, 'PREC': PREC, 'REC': REC, 'F1': F1}

    ## Fix feature extractor and classifier, get results for all models (Tab2)
    fixedEC = {}

    # Loop through all models
    for model in PlanktonAll_OCC['models']:

        AUC = PlanktonAll_OCC['models'][model]['f_exts'][bestMEC['f_ext']]['classifiers'][bestMEC['class']]['alg_metrics']['auc-roc']

        PREC = PlanktonAll_OCC['models'][model]['f_exts'][bestMEC['f_ext']]['classifiers'][bestMEC['class']]['alg_metrics']['precision']
        REC = PlanktonAll_OCC['models'][model]['f_exts'][bestMEC['f_ext']]['classifiers'][bestMEC['class']]['alg_metrics']['recall']
        F1 = PlanktonAll_OCC['models'][model]['f_exts'][bestMEC['f_ext']]['classifiers'][bestMEC['class']]['alg_metrics']['f1']

        fixedEC[model] = {'AUC': AUC, 'PREC': PREC, 'REC': REC, 'F1': F1}


    ## Fix model and classifier, get results for all feature extractors (Tab3)
    fixedMC = {}

    # Loop through all feature extractors
    for f_ext in PlanktonAll_OCC['models'][bestMEC['model']]['f_exts']:

        AUC = PlanktonAll_OCC['models'][bestMEC['model']]['f_exts'][f_ext]['classifiers'][bestMEC['class']]['alg_metrics']['auc-roc']

        PREC = PlanktonAll_OCC['models'][bestMEC['model']]['f_exts'][f_ext]['classifiers'][bestMEC['class']]['alg_metrics']['precision']
        REC = PlanktonAll_OCC['models'][bestMEC['model']]['f_exts'][f_ext]['classifiers'][bestMEC['class']]['alg_metrics']['recall']
        F1 = PlanktonAll_OCC['models'][bestMEC['model']]['f_exts'][f_ext]['classifiers'][bestMEC['class']]['alg_metrics']['f1']

        fixedMC[f_ext] = {'AUC': AUC, 'PREC': PREC, 'REC': REC, 'F1': F1}


    ## Fix model and feature extractor, get results for all feature classifiers (Tab4)
    fixedME = {}

    # Loop through all classifiers
    for clas in PlanktonAll_OCC['models'][bestMEC['model']]['f_exts'][bestMEC['f_ext']]['classifiers']:

        AUC = PlanktonAll_OCC['models'][bestMEC['model']]['f_exts'][bestMEC['f_ext']]['classifiers'][clas]['alg_metrics']['auc-roc']

        PREC = PlanktonAll_OCC['models'][bestMEC['model']]['f_exts'][bestMEC['f_ext']]['classifiers'][clas]['alg_metrics']['precision']
        REC = PlanktonAll_OCC['models'][bestMEC['model']]['f_exts'][bestMEC['f_ext']]['classifiers'][clas]['alg_metrics']['recall']
        F1 = PlanktonAll_OCC['models'][bestMEC['model']]['f_exts'][bestMEC['f_ext']]['classifiers'][clas]['alg_metrics']['f1']

        fixedME[clas] = {'AUC': AUC, 'PREC': PREC, 'REC': REC, 'F1': F1}


    ## Get the optimal results for all plankton species
    optimalRes = {}
    bestAUC = 0

    # Loop through the plankton species
    for species in dataPln:

        # Loop through all models
        for model in dataPln[species]['models']:

            # Loop through all feature extractors
            for f_ext in dataPln[species]['models'][model]['f_exts']:

                # Loop through all classifiers
                for clas in dataPln[species]['models'][model]['f_exts'][f_ext]['classifiers']:
                    
                    if bestAUC <= dataPln[species]['models'][model]['f_exts'][f_ext]['classifiers'][clas]['alg_metrics']['f1']:

                        # Assign the best AUC and MEC parameters
                        bestAUC = dataPln[species]['models'][model]['f_exts'][f_ext]['classifiers'][clas]['alg_metrics']['f1']

                        AUC = dataPln[species]['models'][model]['f_exts'][f_ext]['classifiers'][clas]['alg_metrics']['auc-roc']

                        PREC = dataPln[species]['models'][model]['f_exts'][f_ext]['classifiers'][clas]['alg_metrics']['precision']
                        REC = dataPln[species]['models'][model]['f_exts'][f_ext]['classifiers'][clas]['alg_metrics']['recall']
                        F1 = dataPln[species]['models'][model]['f_exts'][f_ext]['classifiers'][clas]['alg_metrics']['f1']

                        temp = {'model': model, 'f_ext': f_ext, 'class': clas, 'AUC': AUC, 'PREC': PREC, 'REC': REC, 'F1': F1}

        optimalRes[species] = temp
        bestAUC = 0


    ## Generate LateX tables

    # Table 1
    with open('table1.txt', 'w+') as f:

        f.write('Plankton species       & AUC score & F1 score  & Prec  & Rec   \\\ \n')

        for spec in speciesRes:

            AUC = speciesRes[spec]['AUC']

            PREC = speciesRes[spec]['PREC']
            REC = speciesRes[spec]['REC']
            F1 = speciesRes[spec]['F1']

            f.write(spec + '\t\t\t& ' + f'{float(AUC):.2f}' + '\t\t\t& ' + f'{float(F1):.2f}' + '\t& ' + f'{float(PREC):.2f}' + '\t& ' + f'{float(REC):.2f}' + '    \\\ \n')

    # Table 2
    with open('table2.txt', 'w+') as f:

        f.write('Model name     & AUC score & F1 score  & Prec  & Rec   \\\ \n')

        for model in fixedEC:

            AUC = fixedEC[model]['AUC']

            PREC = fixedEC[model]['PREC']
            REC = fixedEC[model]['REC']
            F1 = fixedEC[model]['F1']

            f.write(model + '\t\t& ' + f'{float(AUC):.2f}' + '\t\t\t& ' + f'{float(F1):.2f}' + '\t& ' + f'{float(PREC):.2f}' + '\t& ' + f'{float(REC):.2f}' + ' \\\ \n')
        

    # Table 3
    with open('table3.txt', 'w+') as f:

        f.write('Feature extractor      & AUC score & F1 score  & Prec  & Rec   \\\ \n')

        for f_ext in fixedMC:

            AUC = fixedMC[f_ext]['AUC']

            PREC = fixedMC[f_ext]['PREC']
            REC = fixedMC[f_ext]['REC']
            F1 = fixedMC[f_ext]['F1']

            f.write(f_ext + '\t\t\t\t& ' + f'{float(AUC):.2f}' + '\t\t\t& ' + f'{float(F1):.2f}' + '\t& ' + f'{float(PREC):.2f}' + '\t& ' + f'{float(REC):.2f}' + ' \\\ \n')
        

    # Table 4
    with open('table4.txt', 'w+') as f:

        f.write('Classifier         & AUC score & F1 score  & Prec  & Rec   \\\ \n')

        for classifier in fixedME:

            AUC = fixedME[classifier]['AUC']

            PREC = fixedME[classifier]['PREC']
            REC = fixedME[classifier]['REC']
            F1 = fixedME[classifier]['F1']

            f.write(classifier + '\t\t& ' + f'{float(AUC):.2f}' + '\t\t& ' + f'{float(F1):.2f}' + '\t& ' + f'{float(PREC):.2f}' + '\t& ' + f'{float(REC):.2f}' + '    \\\ \n')
        

    # Table 5
    with open('table5.txt', 'w+') as f:

        f.write('Plankton species         & Model name     & Feature extractor      & Classifier         & AUC score & F1 score  & Prec  & Rec  \\\ \n')

        for species in optimalRes:

            model = optimalRes[species]['model']
            f_ext = optimalRes[species]['f_ext']
            classifier = optimalRes[species]['class']

            AUC = optimalRes[species]['AUC']

            PREC = optimalRes[species]['PREC']
            REC = optimalRes[species]['REC']
            F1 = optimalRes[species]['F1']

            f.write(species + '\t\t& ' + model + '\t\t\t& ' + f_ext + '\t\t\t& '  + classifier + '\t\t& ' + f'{float(AUC):.2f}' + '\t\t& ' + f'{float(F1):.2f}' + '\t& ' + f'{float(PREC):.2f}' + '\t& ' + f'{float(REC):.2f}' + '  \\\ \n')
        

    print('Processing of JSON files was done...')

if __name__ == '__main__':
    main()