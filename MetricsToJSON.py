import os
import json
import argparse
import re

def extract_and_save():
    # Input parameter configuration
    parser = argparse.ArgumentParser(description='Universal metric extraction to JSON')
    parser.add_argument('--input', type=str, default='./ProgramLog.txt', help='Input log file path')
    args = parser.parse_args()

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"Error: File {args.input} not found.")
        return

    # 1. Split the log by models using the architecture declaration line
    # This acts as the perfect separator for different models in the same log
    model_blocks = re.split(r'Autoencoder architecture name:\s*', log_content)[1:]

    for block in model_blocks:
        # The first line of the block will be something like: "ConvM1-BAE1_Cookie_OCC"
        first_line = block.split('\n', 1)[0].strip()
        
        # Extract Architecture, Model, and Dataset using regex
        arch_match = re.match(r'([^-]+)-([^_]+)_(.*)', first_line)
        if not arch_match:
            continue # Skip if it doesn't match the expected format
            
        arch_type = arch_match.group(1)   # e.g., ConvM1
        model_name = arch_match.group(2)  # e.g., BAE1
        dataset = arch_match.group(3)     # e.g., Cookie_OCC
        
        # --- DYNAMIC DIRECTORY CREATION ---
        # Builds: data/Cookie_OCC/ConvM1_Cookie_OCC/BAE1
        model_dir = os.path.join("data", dataset, f"{arch_type}_{dataset}", model_name)
        output_file = os.path.join(model_dir, f'{model_name}_metrics.json')

        model_data = {
            "model_metrics": {},
            "features_extracted": {}
        }

        # --- MODEL METRICS EXTRACTION ---
        # (These will safely return null/None if the metrics aren't in this specific log run)
        p_nok = re.search(r'Median Pearson Coefficient:.*?([0-9.]+)', block)
        p_ok = re.search(r'Median Pearson Coefficient:.*?Median Pearson Coefficient:.*?([0-9.]+)', block, re.DOTALL)
        p_ratio = re.search(r'Coefficient ratio:.*?([0-9.]+)', block)
        
        s_nok = re.search(r'Median SSIM value:.*?([0-9.]+)', block)
        s_ok = re.search(r'Median SSIM value:.*?Median SSIM value:.*?([0-9.]+)', block, re.DOTALL)
        s_ratio = re.search(r'SSIM ratio:.*?([0-9.]+)', block)


        # --- FEATURES AND CLASSIFIERS EXTRACTION ---
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
                    # Ignore timestamps and search for "Key: Value"
                    match = re.search(rf'{key}[:\s]+([0-9.]+)', a_block, re.IGNORECASE)
                    
                    # Normalize key names for JSON (e.g., "f1-score" -> "f1", "balance ratio" -> "balance_r")
                    json_key = key.replace("-score", "").replace(" ratio", "_r")
                    metrics_dict[json_key] = float(match.group(1)) if match else None

                model_data["features_extracted"][f_ext_name][algo_name] = metrics_dict

        # Write to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=4)
            
        print(f"Extracted metrics for {model_name} ({dataset}). Saved to: {output_file}")
