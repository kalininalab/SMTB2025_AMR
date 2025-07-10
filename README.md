# AMR project SMTB 2025

## Installation


1. Create conda environment with `conda create -f conda_environment.yaml`
2. Activate the environment with `conda activate amr`
3. Install the package with `pip install -e .`

## Training

Run the training script with:

```bash
python scripts/train_model.py --data /src/scratch/AMR/data/<dataset_type>/<bacteria_name>
```

For training on all of the bacteria, you can use:

```bash
for i in /src/scratch/AMR/data/*/*; do
    python scripts/train_model.py --data $i
done
```
