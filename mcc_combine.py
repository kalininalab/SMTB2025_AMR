import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
import os
import sys
folder = Path("/srv/scratch/AMR/ESMOutputDefault")
species_list = ["Klebsiella_pneumoniae", "Acinetobacter_baumannii","Escherichia_coli", "Campylobacter_jejuni", "Neisseria_gonorrhoeae", "Pseudomonas_aeruginosa", "Salmonella_enterica", "Staphylococcus_aureus", "Streptococcus_pneumoniae"]
types = ["variance", "esm", "feature_importance"]

results_dictionary = {}
for t in types:
    results_dictionary[t] = {}
    for s in species_list:
        results_dictionary[t][s] = {}
        species = f"{t}_{s}"
        for i in range(1):
            results_dictionary[t][s][i] = {}
            version = f"version_{i}"
            file_path = os.path.join(folder, species, version, "metrics.csv")
            df = file_path
            df = pd.read_csv(df)
            metrics = ["acc", "mcc", "loss_epoch"]
            for m in metrics:
                results_dictionary[t][s][i] = {}
                with open(file_path, 'r') as in_file:
                    lines = in_file.readlines()
                    results_line = lines[-1]
                    suplit = results_line.split(",")
                    acc = suplit[2]
                    loss = suplit[3]
                    mcc = suplit[5]
                    results_dictionary[t][s][i]["acc"] = acc 
                    results_dictionary[t][s][i]["loss"] = loss
                    results_dictionary[t][s][i]["mcc"] = mcc


                print(f"{t}, {s}, {i}, {m}")

import matplotlib.pyplot as plt

data = []
for t in results_dictionary:
    for s in results_dictionary[t]:
        for i in results_dictionary[t][s]:
            metrics = results_dictionary[t][s][i]
            entry = {
                't': t,
                's': s,
                'i': i,
                'mcc': metrics.get('mcc'),
                'acc': metrics.get('acc'),
                'loss': metrics.get('loss')
            }
            data.append(entry)

metrics = ['mcc', 'acc', 'loss']
x_labels = [f"{d['t']}-{d['s']}-{d['i']}" for d in data]
x = range(len(data))

fig, ax = plt.subplots(figsize=(10, 6))

width = 0.2
for idx, metric in enumerate(metrics):
    values = [d[metric] for d in data]
    ax.bar([p + width*idx for p in x], values, width=width, label=metric)

ax.set_xticks([p + width for p in x])
ax.set_xticklabels(x_labels, rotation=45, ha='right')
ax.set_ylabel('Metric Value')
ax.set_title('Performance Metrics per (t, s, i)')
ax.legend()
plt.tight_layout()
plt.show()
plt.savefig("/home/marta/SMTB2025_AMR/mcc_combine_plot.png")