import os
import json
import argparse
import re

def extract_eval_and_save(log_path='./ProgramLog.txt'):
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"Error: File {log_path} not found.")
        return

   
    eval_blocks = re.split(r'Evaluating architecture:\s*', log_content)[1:]

    for block in eval_blocks:
        first_line = block.split('\n', 1)[0].strip()
        arch_match = re.match(r'([^-]+)-([^_]+)_(.*)', first_line)
        if not arch_match:
            continue 
            
        arch_type = arch_match.group(1)   
        model_name = arch_match.group(2)  
        dataset = arch_match.group(3)     
        
        model_dir = os.path.join("data", dataset, f"{arch_type}_{dataset}", model_name)
        os.makedirs(model_dir, exist_ok=True)
        output_file = os.path.join(model_dir, f'{model_name}_eval_metrics.json')

        eval_data = {
            "dataset_evaluated": dataset,
            "evaluation_metrics": {},
            "classifiers_performance": {}
        }


        test_loss = re.search(r'Test Loss:.*?([0-9.]+)', block)
        inference_time = re.search(r'Inference time:.*?([0-9.]+)', block)

        eval_data["evaluation_metrics"] = {
            "test_loss": float(test_loss.group(1)) if test_loss else None,
            "inference_time_seconds": float(inference_time.group(1)) if inference_time else None
        }


        f_ext_blocks = re.split(r'Feature extraction method:\s*', block)[1:]
        
        for f_block in f_ext_blocks:
            f_ext_name = f_block.split('\n', 1)[0].strip()
            eval_data["classifiers_performance"][f_ext_name] = {}

            algo_blocks = re.split(r'Algorithm:\s*', f_block)[1:]
            
            for a_block in algo_blocks:
                algo_name = a_block.split('\n', 1)[0].strip()
                
                metrics_dict = {}
                metric_keys = ["auc-roc", "precision", "recall", "f1-score", "accuracy"]
                
                for key in metric_keys:
                    match = re.search(rf'{key}[:\s]+([0-9.]+)', a_block, re.IGNORECASE)
                    json_key = key.replace("-score", "")
                    metrics_dict[json_key] = float(match.group(1)) if match else None

                eval_data["classifiers_performance"][f_ext_name][algo_name] = metrics_dict

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, indent=4)
            
        print(f"Extracted metrics for {model_name}. Saved to: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Evaluation Metrics to JSON')
    parser.add_argument('--input', type=str, default='./ProgramLog.txt')
    args = parser.parse_args()
    extract_eval_and_save(args.input)