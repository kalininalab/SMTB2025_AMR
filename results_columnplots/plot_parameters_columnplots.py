import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

folder = Path("/srv/scratch/AMR/ESMOutputOptimization")
species_list = [
    "Klebsiella_pneumoniae", "Acinetobacter_baumannii", "Escherichia_coli", 
    "Campylobacter_jejuni", "Neisseria_gonorrhoeae", "Pseudomonas_aeruginosa", 
    "Salmonella_enterica", "Staphylococcus_aureus", "Streptococcus_pneumoniae"
]
types = ["variance", "esm", "feature_importance"]
metrics = ["acc", "mcc", "loss"]

# Color mapping
color_map = {
    "variance": "#FFB6C1",             # pastel pink
    "esm": "#ADD8E6",                  # pastel blue
    "feature_importance": "#D8BFD8"    # pastel purple
}

results_dictionary = {}
for t in types:
    results_dictionary[t] = {}
    for s in species_list:
        species_key = f"{t}_{s}"
        file_path = os.path.join(folder, species_key, "version_0", "metrics.csv")
        if not os.path.exists(file_path):
            print(f"[Warning] Missing file: {file_path}")
            continue
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                print(f"[Warning] Not enough data in: {file_path}")
                continue
            values = lines[-1].strip().split(',')
            try:
                acc = float(values[2])
                loss = float(values[3])
                mcc = float(values[5])
            except (IndexError, ValueError):
                print(f"[Error] Malformed data in: {file_path}")
                continue

            results_dictionary[t][s] = {
                "acc": acc,
                "loss": loss,
                "mcc": mcc
            }

output_dir = Path("/home/marta/SMTB2025_AMR/plots")
output_dir.mkdir(parents=True, exist_ok=True)

for t in types:
    for metric in metrics:
        x_labels = []
        values = []
        for s in species_list:
            if s in results_dictionary[t]:
                x_labels.append(s)
                values.append(results_dictionary[t][s][metric])
        
        if not values:
            continue 

        plt.figure(figsize=(10, 6))
        plt.bar(x_labels, values, color=color_map[t])
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(metric.upper())
        plt.title(f"{t.replace('_', ' ').title()} - {metric.upper()}")
        plt.tight_layout()

        plot_path = output_dir / f"{t}_{metric}.png"
        plt.savefig(plot_path)
        plt.close()
