import numpy as np
import csv
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

bacteria_names = [
    'Acinetobacter_baumannii', 'Campylobacter_jejuni', 'Escherichia_coli',
    'Neisseria_gonorrhoeae', 'Pseudomonas_aeruginosa', 'Salmonella_enterica',
    'Staphylococcus_aureus', 'Streptococcus_pneumoniae', 'Klebsiella_pneumoniae'
]

all_genotypes = []
all_bacteria_labels = []

for bacteria_file in bacteria_names:
    with open(f"/srv/scratch/AMR/data/esm/{bacteria_file}/genotype.tsv", 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        headers = next(reader)
        genotype_data = {rows[0]: rows[1:] for rows in reader}

    with open(f"/srv/scratch/AMR/IR_phenotype/{bacteria_file}/phenotype.txt", 'r') as file2:
        reader = csv.reader(file2, delimiter='\t')
        headers = next(reader)
        phenotype_data = {rows[0]: rows[1:] for rows in reader}

    antibiotic_index = 0
    phenotype_data = {
        strain: phenotypes for strain, phenotypes in phenotype_data.items()
        if strain in genotype_data and phenotypes[antibiotic_index] != "2"
    }
    genotype_data = {strain: genotype_data[strain] for strain in phenotype_data}

    genotype_array = np.array(
        [list(map(float, genotypes)) for genotypes in genotype_data.values()]
    )

    all_genotypes.append(genotype_array)
    all_bacteria_labels.extend([bacteria_file] * len(genotype_array))

X_all = np.vstack(all_genotypes)
bacteria_labels = np.array(all_bacteria_labels)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

print("Applying UMAP on all bacteria data...")
embedding = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
).fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
unique_bacteria = sorted(set(bacteria_labels))
palette = sns.color_palette("tab10", len(unique_bacteria))
color_dict = {bacteria: palette[i] for i, bacteria in enumerate(unique_bacteria)}

for bacteria in unique_bacteria:
    mask = bacteria_labels == bacteria
    plt.scatter(
        embedding[mask, 0], embedding[mask, 1],
        label=bacteria.replace("_", " "), c=[color_dict[bacteria]],
        alpha=0.7, s=30
    )

plt.title("UMAP All Bacteria (ESM Features)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

output_path = "/home/marta/SMTB2025_AMR/UMAP_plots/ESM_combined_u.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Combined UMAP saved to {output_path}")
