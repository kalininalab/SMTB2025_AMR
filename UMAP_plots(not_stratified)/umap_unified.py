import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import argparse

parser = argparse.ArgumentParser(description="Run UMAP visualization for genotype data.")
parser.add_argument(
    "--dir",
    type=str,
    required=True,
    choices=["esm", "variance", "feature_importance", "GenomeOcean"],
    help="Directory to load genotype data from"
)
args = parser.parse_args()
here_parse_dir = args.dir

umap_min_dist = 0.5

if here_parse_dir == "esm" or "GenomeOcean":
    umap_metric = "cosine"
else:
    umap_metric = "hamming"

bacteria_names = [
    'Acinetobacter_baumannii', 'Campylobacter_jejuni', 'Escherichia_coli',
    'Neisseria_gonorrhoeae', 'Pseudomonas_aeruginosa', 'Salmonella_enterica',
    'Staphylococcus_aureus', 'Streptococcus_pneumoniae', 'Klebsiella_pneumoniae'
]

for bacteria_file in bacteria_names:
    genotype_path = f"/srv/scratch/AMR/data/{here_parse_dir}/{bacteria_file}/genotype.tsv"
    genotype_df = pd.read_csv(genotype_path, sep="\t", index_col=0)

    phenotype_path = f"/srv/scratch/AMR/IR_phenotype/{bacteria_file}/phenotype.txt"
    phenotype_df = pd.read_csv(phenotype_path, sep="\t", index_col=0)

    shared_strains = genotype_df.index.intersection(phenotype_df.index)
    genotype_df = genotype_df.loc[shared_strains]
    phenotype_df = phenotype_df.loc[shared_strains]

    antibiotic_index = 0
    valid_mask = phenotype_df.iloc[:, antibiotic_index] != 2
    genotype_df = genotype_df[valid_mask]
    phenotype_df = phenotype_df[valid_mask]

    genotype_array = genotype_df.to_numpy()
    phenotype_array = phenotype_df.iloc[:, antibiotic_index].astype(int).to_numpy()

    def load_split(file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f]

    train = load_split(f"/home/marta/SMTB2025_AMR/AMR-DL/Data/Splits/{bacteria_file}/train.txt")
    test = load_split(f"/home/marta/SMTB2025_AMR/AMR-DL/Data/Splits/{bacteria_file}/test_validation.txt")
    validation = load_split(f"/home/marta/SMTB2025_AMR/AMR-DL/Data/Splits/{bacteria_file}/validation.txt")

    strain_to_index = {strain: idx for idx, strain in enumerate(genotype_df.index)}

    def extract_data(strains):
        X, y = [], []
        for strain in strains:
            if strain in strain_to_index:
                idx = strain_to_index[strain]
                X.append(genotype_array[idx])
                y.append(phenotype_array[idx])
        return np.array(X), np.array(y)

    X_train, y_train = extract_data(train)
    X_test, y_test = extract_data(test)
    X_validation, y_validation = extract_data(validation)

    X_combined = np.vstack([X_train, X_test, X_validation])
    y_combined = np.hstack([y_train, y_test, y_validation])
    dataset_labels = (
        ['Train'] * len(X_train)
        + ['Test'] * len(X_test)
        + ['Validation'] * len(X_validation)
    )

    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined)

    print(f"Applying UMAP for {bacteria_file}...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=umap_min_dist,
        n_components=2,
        random_state=42,
        metric=umap_metric
    )
    embedding = reducer.fit_transform(X_combined_scaled)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for i, label in enumerate(['Susceptible', 'Resistant']):
        mask = y_combined == i
        plt.scatter(embedding[mask, 0], embedding[mask, 1],
                    c=['blue', 'red'][i], label=label, alpha=0.7, s=30)
    plt.title(f'{bacteria_file}\nUMAP {here_parse_dir} Phenotype')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for i, label in enumerate(['Train', 'Test', 'Validation']):
        mask = np.array(dataset_labels) == label
        plt.scatter(embedding[mask, 0], embedding[mask, 1],
                    c=['green', 'orange', 'purple'][i], label=label, alpha=0.7, s=30)
    plt.title(f'{bacteria_file}\nUMAP {here_parse_dir} Dataset Split')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = f'/home/marta/SMTB2025_AMR/UMAP_plots_(stratified)/{here_parse_dir} output plots'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{bacteria_file}_umap_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"UMAP completed for {bacteria_file}")
    print(f"  Training samples:   {len(X_train)} (Susceptible: {np.sum(y_train == 0)}, Resistant: {np.sum(y_train == 1)})")
    print(f"  Test samples:       {len(X_test)} (Susceptible: {np.sum(y_test == 0)}, Resistant: {np.sum(y_test == 1)})")
    print(f"  Validation samples: {len(X_validation)} (Susceptible: {np.sum(y_validation == 0)}, Resistant: {np.sum(y_validation == 1)})")
    print(f"  Total features:     {X_combined.shape[1]}")
    print(f"  Plot saved to:      {output_dir}/{bacteria_file}_umap_visualization.png")
    print("-" * 80)
