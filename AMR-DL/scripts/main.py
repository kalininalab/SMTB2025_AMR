import csv
import numpy as np
import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

file = open("/scratch/AMR/Variant_GPA_Kmer_uncompressed/Pseudomonas_aeruginosa_train_genotype_vargpakmer.tsv", 'r')

reader = csv.reader(file, delimiter='\t')

headers = next(reader)
genotype_data = {rows[0]: rows[1:] for rows in reader}
feature_names = headers[1:]

# Use StringIO to simulate a file object
file2 = open("/home/user/SMTB2025_AMR/AMR-DL/Data/Phenotype/IR_phenotype/Pseudomonas_aeruginosa/phenotype.txt", 'r')

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

rf_cls = RandomForestClassifier(class_weight={0: sum(y_train), 1: len(y_train) - sum(y_train)}, n_estimators=5, max_depth=2, min_samples_leaf=1, min_samples_split=2)
rf_cls.fit(X_train, y_train)

y_hat = rf_cls.predict(X_test)

model_mcc_score = sklearn.metrics.matthews_corrcoef(y_test, y_hat)

feature_importance = rf_cls.feature_importances_ 

gini_importances = pd.Series(feature_importance, index=feature_names)
importances_dict = gini_importances.to_dict()
sorted_importances = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)

for i in sorted_importances:
    if i[1] <= 0:
        break
    else:
        print(i)