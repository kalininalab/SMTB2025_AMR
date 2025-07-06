import csv
import numpy as np
import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import xgboost as xgb
import os
import ast

#bacteria_list = os.listdir(f"/srv/scratch/AMR/IR_phenotype/")
bacteria_list = ["Klebsiella_pneumoniae"]
for bac in bacteria_list:
    
    with open(f"/srv/scratch/AMR/Variant_GPA_Kmer_uncompressed/{bac}_train_genotype_vargpakmer.tsv", 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        headers = next(reader)
        header_indices = {name: index for index, name in enumerate(headers[1:], start=1)}
        binary_table_dict = {}
        for row in reader:
            strain = row[0]
            mutations = row[1:]
            binary_table_dict[strain] = {mutation_name: mutations[header_indices[mutation_name]-1] for mutation_name in headers[1:]}

    new_btd = {}

    features_to_keep = []

    with open(f"/home/alper/SMTB2025_AMR/Script/alper/feature_importance_{bac}.txt") as fif:
        lines = fif.readlines()
        feature_importance = {}
        for line in lines:
            line = line.strip()
            line_tuple = ast.literal_eval(line)
            feature = line_tuple[0]
            features_to_keep.append(feature)
    
    for strain, mutations in binary_table_dict.items():
        new_mutations = {mutation: mutations[mutation] for mutation in features_to_keep if mutation in mutations}
        new_btd[strain] = new_mutations

    pd.DataFrame.from_dict(new_btd, orient='index').to_csv(f"/home/alper/SMTB2025_AMR/Script/alper/{bac}_reduced_genotype.tsv", sep='\t')
