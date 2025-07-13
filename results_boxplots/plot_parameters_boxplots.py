import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

color_map = {
    "variance": "#FFB6C1",
    "esm": "#ADD8E6",
    "feature_importance": "#D8BFD8"
}

output_dir = Path("/home/marta/SMTB2025_AMR/boxplots")
output_dir.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid")

# Store best MCCs across all types
combined_best_mcc = []

for t in types:
    for metric in metrics:
        all_data = []
        best_mcc_per_species = {}

        for s in species_list:
            model_species_folder = folder / f"{t}_{s}"
            if not model_species_folder.exists():
                continue

            for i in range(21):
                version_folder = model_species_folder / f"version_{i}"
                file_path = version_folder / "metrics.csv"
                if not file_path.exists():
                    continue

                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        if len(lines) < 2:
                            continue
                        values = lines[-1].strip().split(',')
                        metric_value = None
                        if metric == "acc":
                            metric_value = float(values[2])
                        elif metric == "loss":
                            metric_value = float(values[3])
                        elif metric == "mcc":
                            metric_value = float(values[5])
                        if metric_value is not None:
                            all_data.append({
                                "species": s,
                                "value": metric_value
                            })

                            # Track best MCC per species
                            if metric == "mcc":
                                current_best = best_mcc_per_species.get(s, {"mcc": -1, "variant": None})
                                if metric_value > current_best["mcc"]:
                                    best_mcc_per_species[s] = {"mcc": metric_value, "variant": i}
                except Exception as e:
                    print(f"[Error] {file_path}: {e}")
                    continue

        if not all_data:
            continue

        # Collect best MCCs across types
        if metric == "mcc":
            for species, data in best_mcc_per_species.items():
                combined_best_mcc.append({
                    "species": species,
                    "variant": f"version_{data['variant']}",
                    "mcc": data['mcc'],
                    "type": t
                })

        # Plotting
        df = pd.DataFrame(all_data)

        plt.figure(figsize=(12, 6))
        sns.boxplot(x="species", y="value", data=df, color=color_map[t], showfliers=False)
        sns.stripplot(x="species", y="value", data=df, jitter=True, color='black', size=3, alpha=0.6)

        plt.xticks(rotation=45, ha='right')
        plt.ylabel(metric.upper())
        plt.title(f"{t.replace('_', ' ').title()} - {metric.upper()} (Box + Scatter)")
        plt.tight_layout()

        plot_path = output_dir / f"{t}_{metric}_box_scatter.png"
        plt.savefig(plot_path)
        plt.close()

# Save all best MCCs in one file
combined_file = output_dir / "best_mcc_all_types.txt"
with open(combined_file, 'w') as f:
    f.write("species\tbest_variant\tbest_mcc\ttype\n")
    for entry in combined_best_mcc:
        f.write(f"{entry['species']}\t{entry['variant']}\t{entry['mcc']:.4f}\t{entry['type']}\n")
