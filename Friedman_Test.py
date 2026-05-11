import os
import json
import glob
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

base_dir = os.path.abspath(os.getcwd())
data_rows = []

print(f"Scanning directory: {base_dir} ...")

search_pattern = os.path.join(base_dir, "data", "*_OCC", "*_*_OCC", "*", "*_metrics.json")
json_files = glob.glob(search_pattern)

if not json_files:
    print(f"Error: Could not find any files matching the pattern: {search_pattern}")
    exit()

ignored_metrics = {'precision', 'recall', 'tnr', 'tpr'}

for json_path in json_files:
    parts = os.path.normpath(json_path).split(os.sep)
    ae_name = parts[-2]       
    folder_name = parts[-3]   
    
    try:
        convm, dataset, _ = folder_name.split('_')
    except ValueError:
        continue

    with open(json_path, 'r') as file:
        try:
            data = json.load(file)
            features = data.get("features_extracted", {})
            for feat_space, classifiers in features.items():
                for clf_name, metrics in classifiers.items():
                    for metric_name, metric_val in metrics.items():
                        if metric_name.lower() in ignored_metrics:
                            continue
                            
                        if isinstance(metric_val, (int, float)):
                            data_rows.append({
                                'Metric': metric_name,
                                'Scenario': dataset, 
                                'Treatment': f"{convm}_{ae_name}_{feat_space}_{clf_name}", 
                                'Value': float(metric_val)
                            })
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON in file: {json_path}")

if not data_rows:
    print("Error: JSON files were found, but no valid numeric data could be extracted.")
    exit()

df = pd.DataFrame(data_rows)

num_scenarios = df['Scenario'].nunique()
lista_scenarios = df['Scenario'].unique().tolist()
num_models = df['Treatment'].nunique()

print("\n" + "="*50)
print(f"DATA EXTRACTION SUMMARY:")
print(f" -> Total Scenarios: {num_scenarios}")
print(f"    {lista_scenarios}")
print(f" -> Unique Treatments (Models): {num_models}")
print("="*50 + "\n")

all_results = {}

TOP_N_MODELS_FOR_TEST = 10 

print(f"Extracted metrics: {df['Metric'].unique().tolist()}")

for metric_name, metric_df in df.groupby('Metric'):
    metric_df = metric_df.copy()
    
    metric_df['Rank'] = metric_df.groupby('Scenario')['Value'].rank(ascending=False, method='average')
    avg_ranks = metric_df.groupby('Treatment')['Rank'].mean().sort_values()

    top_n_treatments = avg_ranks.head(TOP_N_MODELS_FOR_TEST).index.tolist()
    
    filtered_df = metric_df[metric_df['Treatment'].isin(top_n_treatments)].copy()
    
    filtered_df['Rank_Subset'] = filtered_df.groupby('Scenario')['Value'].rank(ascending=False, method='average')
    avg_ranks_subset = filtered_df.groupby('Treatment')['Rank_Subset'].mean().sort_values()

    pivot_df = filtered_df.pivot_table(index='Scenario', columns='Treatment', values='Value').dropna()

    if pivot_df.empty or pivot_df.shape[0] < 2 or pivot_df.shape[1] < 2:
        print(f"Warning: Skipping metric '{metric_name}': Not enough complete scenarios for testing.")
        continue

    stat, p_value = friedmanchisquare(*[pivot_df[col] for col in pivot_df.columns])

    metric_results = {
        "average_ranks_top_10_subset": avg_ranks_subset.to_dict(),
        "friedman_test": {
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05)
        },
        "nemenyi_test": {}
    }

    if p_value < 0.05:
        nemenyi_p_values = sp.posthoc_nemenyi_friedman(pivot_df)
        best_model = avg_ranks_subset.index[0]
        top_models_subset = avg_ranks_subset.index.tolist()
        
        comparisons_dict = {}
        for model in top_models_subset:
            if model != best_model:
                p_val = float(nemenyi_p_values.at[best_model, model])  # type: ignore
                
                comparisons_dict[model] = {
                    "p_value": p_val,
                    "statistical_tie": bool(p_val >= 0.05)
                }
        
        metric_results["nemenyi_test"] = {
            "best_model": best_model,
            "comparisons_vs_best": comparisons_dict
        }
        
    all_results[metric_name] = metric_results

output_path = os.path.join(base_dir, 'data', 'statistical_results_all_metrics.json')

with open(output_path, 'w') as out_file:
    json.dump(all_results, out_file, indent=4)

print(f"Successfully processed {len(all_results)} different metrics.")
print(f"Results successfully saved to: {output_path}")