import csv
import numpy as np
import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import xgboost as xgb

g_file = "/srv/scratch/AMR/Variant_GPA_Kmer_uncompressed/Campylobacter_jejuni_train_genotype_vargpakmer.tsv"
with open(g_file, 'r') as file:
    reader = csv.reader(file, delimiter='\t')

    headers = next(reader)
    genotype_data = {rows[0]: rows[1:] for rows in reader}
    feature_names = headers[1:]

# Use StringIO to simulate a file object
file2 = open("/srv/scratch/AMR/IR_phenotype/Campylobacter_jejuni/phenotype.txt", 'r')

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

with open("/srv/scratch/AMR/XGB_Cleaned_KMER_IR/Campylobacter_jejuni/seed_42_testsize_0.2_resampling_holdout_XGBOOST_model.sav", 'rb') as trained_model_file:
    model = pickle.load(trained_model_file)

train = []
test = []
validation = []

train_file = open("/home/alper/SMTB2025_AMR/AMR-DL/Data/Splits/Campylobacter_jejuni/train.txt", 'r')
test_file = open("/home/alper/SMTB2025_AMR/AMR-DL/Data/Splits/Campylobacter_jejuni/test_validation.txt", 'r')
validation_file = open("/home/alper/SMTB2025_AMR/AMR-DL/Data/Splits/Campylobacter_jejuni/validation.txt", 'r')

for line in train_file:
    line = line.strip()
    train.append(line)

for line in test_file:
    line = line.strip()
    test.append(line)

for line in validation_file:
    line = line.strip()
    validation.append(line) 

X_train = []
y_train = []
X_test = []
y_test = []
X_validation = []
y_validation = []

train_strains_to_be_used = []
test_strains_to_be_used = []
validation_strains_to_be_used = []

strain_to_index = {strain: idx for idx, strain in enumerate(genotype_data.keys())}

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

dtest = xgb.DMatrix(X_test)
y_hat = model.predict(dtest)
y_pred = (y_hat >= 0.5).astype(int)

model_mcc_score = sklearn.metrics.matthews_corrcoef(y_test, y_pred)

feature_importance_dict = model.get_score(importance_type ='weight')

feature_importance_dict_with_corrected_names = {}

for feature in feature_importance_dict.keys():
    feature_index = int(feature[1:])  # Convert feature index to integer
    curr_feature_name = feature_names[feature_index]  # Get the feature name using the index
    feature_importance_dict_with_corrected_names[curr_feature_name] = feature_importance_dict[feature]

sorted_importances = sorted(feature_importance_dict_with_corrected_names.items(), key=lambda x: x[1], reverse=True)

print(model_mcc_score)
print(sorted_importances[:10])

for i in sorted_importances:
    if i[1] <= 0:
        break
    else:
        print(i)
