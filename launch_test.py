import subprocess
from pathlib import Path

models = [
    "/srv/scratch/AMR/NNOutputHPO_StratifiedDatasailSplit/esm_Acinetobacter_baumannii/version_5/checkpoints/best_model.ckpt",
    "/srv/scratch/AMR/NNOutputHPO_StratifiedDatasailSplit/esm_Neisseria_gonorrhoeae/version_4/checkpoints/best_model.ckpt",
    "/srv/scratch/AMR/NNOutputHPO_StratifiedDatasailSplit/esm_Klebsiella_pneumoniae/version_2/checkpoints/best_model.ckpt",
    "/srv/scratch/AMR/NNOutputHPO_StratifiedDatasailSplit/esm_Streptococcus_pneumoniae/version_2/checkpoints/best_model.ckpt",
    "/srv/scratch/AMR/NNOutputHPO_StratifiedDatasailSplit/esm_Salmonella_enterica/version_1/checkpoints/best_model.ckpt",
    "/srv/scratch/AMR/NNOutputHPO_StratifiedDatasailSplit/esm_Staphylococcus_aureus/version_0/checkpoints/best_model.ckpt",
    "/srv/scratch/AMR/NNOutputHPO_StratifiedDatasailSplit/esm_Campylobacter_jejuni/version_5/checkpoints/best_model.ckpt",
    "/srv/scratch/AMR/NNOutputHPO_StratifiedDatasailSplit/esm_Pseudomonas_aeruginosa/version_5/checkpoints/best_model.ckpt",
    "/srv/scratch/AMR/NNOutputHPO_StratifiedDatasailSplit/esm_Escherichia_coli/version_4/checkpoints/best_model.ckpt",
]

for model_path in models:
    for dataset_folder in Path("/srv/scratch/AMR/data_datasail_stratify_split/esm/").glob("*/"):
        print(f"Testing model: {model_path} on data: {dataset_folder}")
        subprocess.run(["python", "scripts/test_model.py", "--data", dataset_folder, "--checkpoints", model_path])
