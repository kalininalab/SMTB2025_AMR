import csv
import numpy as np
import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

file = open("/srv/scratch/AMR/Variant_GPA_Kmer_uncompressed/Pseudomonas_aeruginosa_train_genotype_vargpakmer.tsv", 'r')

reader = csv.reader(file, delimiter='\t')

headers = next(reader)
genotype_data = {rows[0]: rows[1:] for rows in reader}
feature_names = headers[1:]

# Use StringIO to simulate a file object
file2 = open("/srv/scratch/AMR/IR_phenotype/Pseudomonas_aeruginosa/phenotype.txt", 'r')

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

train = []
test = []
validation = []

train_file = open("/home/marta/SMTB2025_AMR/AMR-DL/Data/Splits/Pseudomonas_aeruginosa/train.txt", 'r')
test_file = open("/home/marta/SMTB2025_AMR/AMR-DL/Data/Splits/Pseudomonas_aeruginosa/test.txt", 'r')
validation_file = open("/home/marta/SMTB2025_AMR/AMR-DL/Data/Splits/Pseudomonas_aeruginosa/validation.txt", 'r')

for line in train_file:
    line = line.strip()
    train.append(line)

for line in test_file:
    line = line.strip()
    train.append(line)

for line in validation_file:
    line = line.strip()
    train.append(line) #tu skonczylam

X_train = []
y_train = []
X_test = []
y_test = []
X_validation = []
y_validation = []

train_strains_to_be_used = []
test_strains_to_be_used = []
validation_strains_to_be_used = []

for train_strain in train:
    if train_strain in strain_to_index:
        train_strains_to_be_used.append(train_strain)
        idx = strain_to_index[train_strain]
        X_train.append(genotype_array[idx])  # Append the list of genotypes using the index
        y_train.append(phenotype_array[idx])  # Append the phenotype value using the index

for test_strain in test:
    if test_strain in strain_to_index:
        test_strains_to_be_used.append(test_strain)
        idx = strain_to_index[test_strain]
        X_test.append(genotype_array[idx])  # Append the list of genotypes using the index
        y_test.append(phenotype_array[idx])  # Append the phenotype value using the index

for validation_strain in validation:
    if validation_strain in strain_to_index:
        validation_strains_to_be_used.append(validation_strain)
        idx = strain_to_index[validation_strain]
        X_validation.append(genotype_array[idx])  # Append the list of genotypes using the index
        y_validation.append(phenotype_array[idx])  # Append the phenotype value using the index

# Convert lists to numpy arrays
X_train = np.array(X_train, dtype=int)
y_train = np.array(y_train, dtype=int)
X_test = np.array(X_test, dtype=int)
y_test = np.array(y_test, dtype=int)
X_validation = np.array(X_validation, dtype=int)
y_validation = np.array(y_validation, dtype=int)

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

