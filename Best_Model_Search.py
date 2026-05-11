import sys
import json
from pathlib import Path

def find_best_model(target_dir, target_metric="auc-roc"):
    base_path = Path(target_dir)
    
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    best_score = -1.0
    best_models = []

    for json_path in base_path.rglob("*_metrics.json"):
        if json_path.name == "best_model.json":
            continue

        try:
            ae_model = json_path.parent.name
            layer_folder = json_path.parent.parent.name
            layer_model = layer_folder.split("_")[0]
        except IndexError:
            continue

        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
        
        features = data.get("features_extracted", {})
        
        for feature_name, classifiers in features.items():
            for clf_name, metrics in classifiers.items():
                if metrics and target_metric in metrics and metrics[target_metric] is not None:
                    score = metrics[target_metric]
                    
                    model_info = {
                        'layer_model': layer_model,
                        'ae_model': ae_model,
                        'feature_extractor': feature_name,
                        'classifier': clf_name,
                        'metrics': metrics
                    }

                    if score > best_score:
                        best_score = score
                        best_models = [model_info]
                    elif score == best_score:
                        best_models.append(model_info)

    if best_score == -1.0:
        print("No valid metrics found. No file was created.")
        return

    output_data = {
        "directory_analyzed": target_dir,
        "metric_optimized": target_metric,
        "best_score": best_score,
        "top_models": best_models
    }

    output_file = base_path / "best_model.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Success! Results saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_best_models.py <path_to_directory>")
        sys.exit(1)
        
    target_directory = sys.argv[1]
    find_best_model(target_dir=target_directory)