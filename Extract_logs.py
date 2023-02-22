import os
import json
import argparse

def main():
# input parameters
    parser = argparse.ArgumentParser(description='ArgParser')
    parser.add_argument('--input', type=str, default='./ProgramLog.txt', help='input file path with file name')
    parser.add_argument('--output', type=str, default='./parsed-log.json', help='output file path with file name')
    args = parser.parse_args()
    str_input = args.input
    str_output = args.output
    str_output_all = os.path.join(os.path.dirname(args.input),"parsed-all-log.json")
    dictionary = {}
    with open(str_input) as f:
        txt = f.read()
        datasets_txt = txt.split('Data generators of the ')
        datasets_txt=datasets_txt[1:]
        for dataset_txt in datasets_txt:
            dataset_name = dataset_txt.split('\n', 1)[0].split(' ')[0]
            dictionary[dataset_name]={'models':{}}
            models_txt = dataset_txt.split('Training of the ')
            models_txt = models_txt[1:]
            for model_txt in models_txt:
                model_name = model_txt.split('\n', 1)[0].split(' ')[0]
                sep = False
                try:
                    pearson_nok = model_txt.split('Median Pearson Coefficient: ', 1)[1].split(' ')[0]
                    pearson_ok = model_txt.split('Median Pearson Coefficient: ', 2)[2].split(' ')[0]
                    pearson_c = model_txt.split('Coefficient ratio: ', 1)[1].split(' ')[0]
                    ssim_nok = model_txt.split('Median SSIM value: ', 1)[1].split(' ')[0]
                    ssim_ok = model_txt.split('Median SSIM value: ', 2)[2].split(' ')[0]
                    ssim_ratio = model_txt.split('SSIM ratio: ', 1)[1].split(' ')[0]
                    dictionary[dataset_name]['models'][model_name] = { 
                                               "model_metrics": {
                                               "p_nok": float(pearson_nok),
                                               "p_ok": float(pearson_ok),
                                               "pearson_r": float(pearson_c),
                                               "ssim_nok": float(ssim_nok),
                                               "ssim_ok": float(ssim_ok),
                                               "ssim_r": float(ssim_ratio)},
                                               "f_exts":{}
                                              }
                except:
                    print(model_name, "failed evaluation at the dataset", dataset_name, "...skipping")
                    pearson_nok = 100
                    pearson_ok = 100
                    pearson_c = 100
                    ssim_nok = 100
                    ssim_ok = 100
                    ssim_ratio = 100
                    dictionary[dataset_name]['models'][model_name] = { 
                                       "model_metrics": {
                                       "p_nok": float(pearson_nok),
                                       "p_ok": float(pearson_ok),
                                       "pearson_r": float(pearson_c),
                                       "ssim_nok": float(ssim_nok),
                                       "ssim_ok": float(ssim_ok),
                                       "ssim_r": float(ssim_ratio)},
                                       "f_exts":{}
                                      }
                if sep == False:
                    f_exts_txt = model_txt.split('Feature extraction method: ')
                    f_exts_txt = f_exts_txt[1:]
                    sep = True
                for f_ext_txt in f_exts_txt:
                    sep1 = False
                    f_ext_name = f_ext_txt.split('\n', 1)[0].split(' ')[0]
                    dictionary[dataset_name]['models'][model_name]["f_exts"][f_ext_name] = {"classifiers":{}}
                    if sep1 == False:
                        algos_txt = f_ext_txt.split('Algorithm: ')
                        algos_txt = algos_txt[1:5]
                        sep1 == True

                    for algo_txt in algos_txt:
                        algo_name, metrics = algo_txt.split('\n', 1)
                        metrics = metrics.split('\n')[0:8]

                        for i in range(len(metrics)):
                            metrics[i] = float(metrics[i].split(' ')[-1])
                            # metrics[i] = metrics[i].split(' ')[-1:]
                        dictionary[dataset_name]['models'][model_name]["f_exts"][f_ext_name]['classifiers'][algo_name] = { "alg_metrics":{"auc-roc": metrics[0],
                                                                                       "auc-pre": metrics[1],
                                                                                       "precision": metrics[2],
                                                                                       "recall": metrics[3],
                                                                                       "f1": metrics[4],
                                                                                       "tpr": metrics[5],
                                                                                       "tnr": metrics[6],
                                                                                       "balance_r": metrics[7]}}

    json_object = json.dumps(dictionary, indent = 1) 
    with open(str_output_all, "w") as outfile:
        outfile.write(json_object)

    dictionary_top = {}
    p_r1 = 100
    ssim_r2 = 100
    with open(str_output_all) as f:
        log = json.load(f)
        for dataset_name, models in log.items():
            best_c_roc_roc = -1
            best_c_roc_pre = -1
            best_c_pre_roc = -1
            best_c_pre_pre = -1
            p_r1 = 100
            ssim_r2 = 100
            dictionary_top[dataset_name]={'best_model_pearson':[],
                                         'best_model_ssim':[],
                                         'best_classifier_roc':[],
                                         'best_classifier_pre':[]}


            # dictionary_top[dataset_name]['best_model_pearson']=defaultdict(list)
            for model_name, f_exts in models['models'].items():
                if f_exts['model_metrics']['pearson_r'] < p_r1 :
                    p_r1 = f_exts['model_metrics']['pearson_r']
                    ssim_r1 = f_exts['model_metrics']['ssim_r']
                    auc_roc1 = -1
                    auc_pre1 = -1
                    pre1 = 0
                    f1_1 = 0
                    recall1 = 0
                    for f_ext_name, f_ext in f_exts['f_exts'].items():
                        for classifier_name, classifiers in f_ext['classifiers'].items():
                            if classifiers['alg_metrics']['auc-roc'] > auc_roc1 and classifiers['alg_metrics']['auc-pre'] > auc_pre1:
                                auc_roc1 = classifiers['alg_metrics']['auc-roc']
                                auc_pre1 = classifiers['alg_metrics']['auc-pre']
                                pre1 = classifiers['alg_metrics']['precision']
                                f1_1 = classifiers['alg_metrics']['f1']
                                recall1 = classifiers['alg_metrics']['recall']
                                f_ext_name1 = f_ext_name
                                model_name1 = model_name
                                classifier_name1 = classifier_name


                if f_exts['model_metrics']['ssim_r'] < ssim_r2:
                    p_r2 = f_exts['model_metrics']['pearson_r']
                    ssim_r2 = f_exts['model_metrics']['ssim_r']
                    auc_roc2 = -1
                    auc_pre2 = -1
                    pre2 = 0
                    f1_2 = 0
                    recall2 = 0
                    for f_ext_name, f_ext in f_exts['f_exts'].items():
                        for classifier_name, classifiers in f_ext['classifiers'].items():
                            if classifiers['alg_metrics']['auc-roc'] > auc_roc2 and classifiers['alg_metrics']['auc-pre'] > auc_pre2:
                                auc_roc2 = classifiers['alg_metrics']['auc-roc']
                                auc_pre2 = classifiers['alg_metrics']['auc-pre']
                                pre2 = classifiers['alg_metrics']['precision']
                                f1_2 = classifiers['alg_metrics']['f1']
                                recall2 = classifiers['alg_metrics']['recall']
                                model_name2 = model_name
                                f_ext_name2 = f_ext_name
                                classifier_name2 = classifier_name


            dictionary_top[dataset_name]['best_model_ssim'].append({
                                       'model':model_name2,
                                            'model_metrics': {
                                            "pearson_r": float(p_r2),
                                            "ssim_r": float(ssim_r2)},
                                                'f_ext':f_ext_name2,
                                            'classifier': classifier_name2,
                                            'classifier_metrics': {'auc-rc': auc_roc2,
                                                                   'auc-pre': auc_pre2,
                                                                   'f1': f1_2,
                                                                   'recall': recall2,
                                                                   'precision': pre2 }})
            dictionary_top[dataset_name]['best_model_pearson'].append({
                'model':model_name1,
                'model_metrics': {
                "pearson_r": float(p_r1),
                "ssim_r": float(ssim_r1)},
                    'f_ext':f_ext_name1,
                'classifier': classifier_name1,
                'classifier_metrics': { 'auc-roc': auc_roc1,
                                       'auc-pre': auc_pre1,
                                       'f1': f1_1,
                                       'recall': recall1,
                                       'precision': pre1 }})

            for model_name, f_exts in models['models'].items():
                if f_exts['model_metrics']['pearson_r'] == p_r1 and model_name not in [item['model'] for item in dictionary_top[dataset_name]['best_model_pearson']]:
                    p_r1 = f_exts['model_metrics']['pearson_r']
                    ssim_r1 = f_exts['model_metrics']['ssim_r']
                    for f_ext_name, f_ext in f_exts['f_exts'].items():
                        for classifier_name, classifiers in f_ext['classifiers'].items():
                            if classifiers['alg_metrics']['auc-roc'] >= auc_roc1 and classifiers['alg_metrics']['auc-pre'] >= auc_pre1:
                                auc_roc1 = classifiers['alg_metrics']['auc-roc']
                                auc_pre1 = classifiers['alg_metrics']['auc-pre']
                                pre1 = classifiers['alg_metrics']['precision']
                                f1_1 = classifiers['alg_metrics']['f1']
                                recall1 = classifiers['alg_metrics']['recall']
                                f_ext_name1 = f_ext_name
                                model_name1 = model_name
                                classifier_name1 = classifier_name

                                dictionary_top[dataset_name]['best_model_pearson'].append({
                                            'model':model_name1,
                                            'model_metrics': {
                                            "pearson_r": float(p_r1),
                                            "ssim_r": float(ssim_r1)},
                                                'f_ext':f_ext_name1,
                                            'classifier': classifier_name1,
                                            'classifier_metrics': { 'auc-roc': auc_roc1,
                                                                    'auc-pre': auc_pre1,
                                                                   'f1': f1_2,
                                                                   'recall': recall2,
                                                                   'precision': pre2 }})

                if f_exts['model_metrics']['ssim_r'] == ssim_r2 and model_name not in [item['model'] for item in dictionary_top[dataset_name]['best_model_ssim']]:
                    p_r2 = f_exts['model_metrics']['pearson_r']
                    ssim_r2 = f_exts['model_metrics']['ssim_r']
                    for f_ext_name, f_ext in f_exts['f_exts'].items():
                        for classifier_name, classifiers in f_ext['classifiers'].items():
                            if classifiers['alg_metrics']['auc-roc'] >= auc_roc2 and classifiers['alg_metrics']['auc-pre'] >= auc_pre2:
                                auc_roc2 = classifiers['alg_metrics']['auc-roc']
                                auc_pre2 = classifiers['alg_metrics']['auc-pre']
                                pre2 = classifiers['alg_metrics']['precision']
                                f1_2 = classifiers['alg_metrics']['f1']
                                recall2 = classifiers['alg_metrics']['recall']
                                model_name2 = model_name
                                f_ext_name2 = f_ext_name
                                classifier_name2 = classifier_name
                                dictionary_top[dataset_name]['best_model_ssim'].append({
                                                   'model':model_name2,
                                                        'model_metrics': {
                                                        "pearson_r": float(p_r2),
                                                        "ssim_r": float(ssim_r2)},
                                                            'f_ext':f_ext_name2,
                                                        'classifier': classifier_name2,
                                                        'classifier_metrics': {'auc-rc': auc_roc2,
                                                                               'auc-pre': auc_pre2,
                                                                               'f1': f1_2,
                                                                               'recall': recall2,
                                                                               'precision': pre2 }})    
            best_c_roc_roc = -1
            best_c_roc_pre = -1
            best_c_pre_roc = -1
            best_c_pre_pre = -1
            for model_name, f_exts in models['models'].items():
                for f_ext_name, f_ext in f_exts['f_exts'].items():
                    for classifier_name, classifiers in f_ext['classifiers'].items():
                        if classifiers['alg_metrics']['auc-roc'] > best_c_roc_roc:
                            best_c_roc_roc = classifiers['alg_metrics']['auc-roc']
                            best_c_roc_pre = classifiers['alg_metrics']['auc-pre']
                            best_pre2 = classifiers['alg_metrics']['precision']
                            best_f1_2 = classifiers['alg_metrics']['f1']
                            best_recall2 = classifiers['alg_metrics']['recall']
                            best_c_model_name1 = model_name
                            best_c_p_r2 = f_exts['model_metrics']['pearson_r']
                            best_c_ssim_r2 = f_exts['model_metrics']['ssim_r']
                            best_c_f_ext_name1 = f_ext_name
                            best_c_classifier_name1 = classifier_name

                        if classifiers['alg_metrics']['auc-pre'] > best_c_pre_pre:
                            best_c_pre_roc = classifiers['alg_metrics']['auc-roc']
                            best_c_pre_pre = classifiers['alg_metrics']['auc-pre']
                            best_pre1 = classifiers['alg_metrics']['precision']
                            best_f1_1 = classifiers['alg_metrics']['f1']
                            best_recall1 = classifiers['alg_metrics']['recall']
                            best_c_model_name2 = model_name
                            best_c_p_r1 = f_exts['model_metrics']['pearson_r']
                            best_c_ssim_r1 = f_exts['model_metrics']['ssim_r']
                            best_c_f_ext_name2 = f_ext_name
                            best_c_classifier_name2 = classifier_name

            dictionary_top[dataset_name]['best_classifier_roc'].append({
            'model':best_c_model_name1,
            'model_metrics': {
                "pearson_r": float(best_c_p_r1),
                "ssim_r": float(best_c_ssim_r1)},
            'f_ext':best_c_f_ext_name1,
            'classifier': best_c_classifier_name1,
                    'classifier_metrics': { 'auc-roc': best_c_roc_roc,
                                            'auc-pre': best_c_roc_pre,
                                            'f1': best_f1_1,
                                            'recall': best_recall1,
                                            'precision': best_pre1 }})

            dictionary_top[dataset_name]['best_classifier_pre'].append({
                                        'model':best_c_model_name2,
                                        'model_metrics': {
                                        "pearson_r": float(best_c_p_r2),
                                        "ssim_r": float(best_c_ssim_r2)},
                                            'f_ext':best_c_f_ext_name2,
                                        'classifier': best_c_classifier_name2,
                                        'classifier_metrics': { 'auc-roc': best_c_pre_roc,
                                                                'auc-pre': best_c_pre_pre,
                                                                'f1': best_f1_2,
                                                                'recall': best_recall2,
                                                                'precision': best_pre2 }})


            for model_name, f_exts in models['models'].items():
                for f_ext_name, f_ext in f_exts['f_exts'].items():
                    for classifier_name, classifiers in f_ext['classifiers'].items():
                        if classifiers['alg_metrics']['auc-roc'] == best_c_roc_roc and classifier_name not in [item['classifier'] for item in dictionary_top[dataset_name]['best_classifier_roc']]:
                            best_c_roc_roc = classifiers['alg_metrics']['auc-roc']
                            best_c_roc_pre = classifiers['alg_metrics']['auc-pre']
                            best_pre1 = classifiers['alg_metrics']['precision']
                            best_f1_1 = classifiers['alg_metrics']['f1']
                            best_recall1 = classifiers['alg_metrics']['recall']
                            best_c_p_r1 = f_exts['model_metrics']['pearson_r']
                            best_c_ssim_r1 = f_exts['model_metrics']['ssim_r']
                            best_c_model_name1 = model_name
                            best_c_p_r1 = best_c_p_r1
                            best_c_ssim_r1 = best_c_ssim_r1
                            best_c_f_ext_name1 = f_ext_name
                            best_c_classifier_name1 = classifier_name

                            dictionary_top[dataset_name]['best_classifier_roc'].append({
                                        'model':best_c_model_name1,
                                        'model_metrics': {
                                            "pearson_r": float(best_c_p_r1),
                                            "ssim_r": float(best_c_ssim_r1)},
                                        'f_ext':best_c_f_ext_name1,
                                        'classifier': best_c_classifier_name1,
                                                'classifier_metrics': { 'auc-roc': best_c_roc_roc,
                                                                        'auc-pre': best_c_roc_pre,
                                                                        'f1': best_f1_1,
                                                                        'recall': best_recall1,
                                                                        'precision': best_pre1 }})

                        if classifiers['alg_metrics']['auc-pre'] == best_c_pre_pre and classifier_name not in [item['classifier'] for item in dictionary_top[dataset_name]['best_classifier_pre']]:
                            best_c_pre_roc = classifiers['alg_metrics']['auc-roc']
                            best_c_pre_pre = classifiers['alg_metrics']['auc-pre']
                            best_pre2 = classifiers['alg_metrics']['precision']
                            best_f1_2 = classifiers['alg_metrics']['f1']
                            best_recall2 = classifiers['alg_metrics']['recall']
                            best_c_model_name2 = model_name
                            best_c_p_r2 = f_exts['model_metrics']['pearson_r']
                            best_c_ssim_r2 = f_exts['model_metrics']['ssim_r']
                            best_c_f_ext_name2 = f_ext_name
                            best_c_classifier_name2 = classifier_name

                            dictionary_top[dataset_name]['best_classifier_pre'].append({
                                    'model':best_c_model_name2,
                                    'model_metrics': {
                                        "pearson_r": float(best_c_p_r2),
                                        "ssim_r": float(best_c_ssim_r2)},
                                    'f_ext':best_c_f_ext_name2,
                                    'classifier': best_c_classifier_name2,
                                    'classifier_metrics': { 'auc-roc': best_c_pre_roc,
                                                                'auc-pre': best_c_pre_pre,
                                                                'f1': best_f1_2,
                                                                'recall': best_recall2,
                                                                'precision': best_pre2 }})

    json_object = json.dumps(dictionary_top, indent = 1) 
    with open(str_output, "w") as outfile:
        outfile.write(json_object)

                        
if __name__ == '__main__':
    main()

    
    