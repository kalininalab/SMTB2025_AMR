import csv
from pathlib import Path
from tqdm import tqdm
import os

strains_to_keep = []
bacteria_names = ['Acinetobacter_baumannii', 'Campylobacter_jejuni', 'Escherichia_coli', 'Neisseria_gonorrhoeae', 'Pseudomonas_aeruginosa', 'Salmonella_enterica','Staphylococcus_aureus', 'Streptococcus_pneumoniae', 'Klebsiella_pneumoniae']
for bacteria_file in bacteria_names:
    file2 = open(f"/srv/scratch/AMR/IR_phenotype/{bacteria_file}/phenotype.txt", 'r')
    reader = csv.reader(file2, delimiter='\t')
    next(reader)
    for row in reader:
        if row:
            strains_to_keep.append(row[0])

    rgi_dir = Path("/srv/scratch/AMR/rgi_tsv")
    tsv_files = list(rgi_dir.glob("*.tsv"))
    id_protein = open("/home/marta/SMTB2025_AMR/rgi_protein.txt", "w")

    for tsv_file in tqdm(tsv_files):
        with open(tsv_file, 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            headers = next(reader)
            file_id = tsv_file.stem 
            count = 0
            if file_id in strains_to_keep:
                os.makedirs(f"/srv/scratch/AMR/rgi_per_bac_per_strain/{bacteria_file}/", exist_ok=True)
                with open(f"/srv/scratch/AMR/rgi_per_bac_per_strain/{bacteria_file}/{file_id}.fasta","w") as out_file:
                    for row in reader:
                        protein_seq = row[18]
                        out_file.write(f">{file_id}_{count}\n{protein_seq}\n")
                        count +=1

    strains_to_keep = []


