from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import csv
import numpy as np
import sklearn
import sys
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)

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

    train = []
    test = []
    validation = []

    train_file = open(f"/home/marta/SMTB2025_AMR/AMR-DL/Data/Splits/{bacteria_file}/train.txt", 'r')
    test_file = open(f"/home/marta/SMTB2025_AMR/AMR-DL/Data/Splits/{bacteria_file}/test_validation.txt", 'r')
    validation_file = open(f"/home/marta/SMTB2025_AMR/AMR-DL/Data/Splits/{bacteria_file}/validation.txt", 'r')

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

    param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    }

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = float("nan")

    with open("XGBoost_nogrid_results.txt", "a") as answers:
        print(bacteria_file, file=answers)
        print(f"Accuracy: {accuracy:.3f}", file=answers)
        print(f"Precision: {precision:.3f}", file=answers)
        print(f"Recall: {recall:.3f}", file=answers)
        print(f"F1-score: {f1:.3f}", file=answers)
        print(f"MCC: {mcc:.3f}", file=answers)
        print(f"AUC-ROC: {auc:.3f}", file=answers)
        print(" ", file=answers)