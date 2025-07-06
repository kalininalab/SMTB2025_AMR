from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import csv
import numpy as np
import sklearn
import sys

bacteria_names = ['Acinetobacter_baumannii', 'Campylobacter_jejuni', 'Escherichia_coli', 'Neisseria_gonorrhoeae', 'Pseudomonas_aeruginosa', 'Salmonella_enterica','Staphylococcus_aureus', 'Streptococcus_pneumoniae', 'Klebsiella_pneumoniae']
for bacteria_file in bacteria_names:
  file = open(f"/srv/scratch/AMR/Reduced_genotype/{bacteria_file}_reduced_genotype.tsv", 'r')
  reader = csv.reader(file, delimiter='\t')

  headers = next(reader)
  genotype_data = {rows[0]: rows[1:] for rows in reader}
  feature_names = headers[1:]

  # Use StringIO to simulate a file object
  file2 = open(f"/srv/scratch/AMR/IR_phenotype/{bacteria_file}/phenotype.txt", 'r')

  reader = csv.reader(file2, delimiter='\t')
  headers = next(reader)
  phenotype_data = {rows[0]: rows[1:] for rows in reader}

  phenotype_data = {strain: phenotypes for strain, phenotypes in phenotype_data.items() if strain in genotype_data}

  # Filter strains based on antibiotic resistance
  antibiotic_index = 0
  strains_to_be_skipped = [strain for strain, phenotypes in phenotype_data.items() if len(phenotypes) > antibiotic_index and phenotypes[antibiotic_index] == "2"]
  genotype_data = {strain: genotypes for strain, genotypes in genotype_data.items() if strain not in strains_to_be_skipped}
  phenotype_data = {strain: phenotypes for strain, phenotypes in phenotype_data.items() if strain not in strains_to_be_skipped}

  # Reorder phenotype_data according to the order of keys in genotype_data
  ordered_phenotype_data = {strain: phenotype_data[strain] for strain in genotype_data if strain in phenotype_data}
  phenotype_data = ordered_phenotype_data

  genotype_array = np.array([list(map(int, genotypes)) for genotypes in genotype_data.values()])
  phenotype_array = np.array([int(phenotypes[antibiotic_index]) for phenotypes in phenotype_data.values()])

  X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(genotype_array, phenotype_array, random_state=42, test_size=float(0.2))

  lr_model = LinearRegression()

  lr_model.fit(X_train, y_train)
  y_pred = lr_model.predict(X_test)

  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)

  print(bacteria_file)
  print(f"Mean Squared Error: {mse:.4f}")
  print(f"R^2 Score: {r2:.4f}")