import os
import json
import argparse
import re

def extract_and_save():
    parser = argparse.ArgumentParser(description='Universal metric extraction to JSON')
    parser.add_argument('--input', type=str, default='./ProgramLog.txt', help='Input log file path')
    args = parser.parse_args()

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"Error: File {args.input} not found.")
        return

    model_blocks = re.split(r'Autoencoder architecture name:\s*', log_content)[1:]

    for block in model_blocks:
        first_line = block.split('\n', 1)[0].strip()
        
        arch_match = re.match(r'([^-]+)-([^_]+)_(.*)', first_line)
        if not arch_match:
            continue 
            
        arch_type = arch_match.group(1)   
        model_name = arch_match.group(2)  
        dataset = arch_match.group(3)     
        
        model_dir = os.path.join("data", dataset, f"{arch_type}_{dataset}", model_name)
        output_file = os.path.join(model_dir, f'{model_name}_metrics.json')

        model_data = {
            "model_metrics": {},
            "features_extracted": {}
        }

        p_nok = re.search(r'Median Pearson Coefficient:.*?([0-9.]+)', block)
        p_ok = re.search(r'Median Pearson Coefficient:.*?Median Pearson Coefficient:.*?([0-9.]+)', block, re.DOTALL)
        p_ratio = re.search(r'Coefficient ratio:.*?([0-9.]+)', block)
        
        s_nok = re.search(r'Median SSIM value:.*?([0-9.]+)', block)
        s_ok = re.search(r'Median SSIM value:.*?Median SSIM value:.*?([0-9.]+)', block, re.DOTALL)
        s_ratio = re.search(r'SSIM ratio:.*?([0-9.]+)', block)


        f_ext_blocks = re.split(r'Feature extraction method:\s*', block)[1:]
        
        for f_block in f_ext_blocks:
            f_ext_name = f_block.split('\n', 1)[0].strip()
            model_data["features_extracted"][f_ext_name] = {}

            algo_blocks = re.split(r'Algorithm:\s*', f_block)[1:]
            
            for a_block in algo_blocks:
                algo_name = a_block.split('\n', 1)[0].strip()
                
                metrics_dict = {}
                metric_keys = ["auc-roc", "auc-pre", "precision", "recall", "f1-score", "tpr", "tnr", "balance ratio"]
                
                for key in metric_keys:
                    match = re.search(rf'{key}[:\s]+([0-9.]+)', a_block, re.IGNORECASE)
                    
                    # Normalize key names for JSON (e.g., "f1-score" -> "f1", "balance ratio" -> "balance_r")
                    json_key = key.replace("-score", "").replace(" ratio", "_r")
                    metrics_dict[json_key] = float(match.group(1)) if match else None

                model_data["features_extracted"][f_ext_name][algo_name] = metrics_dict

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=4)
            
