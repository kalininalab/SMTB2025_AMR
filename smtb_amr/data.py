import pickle
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset
from tqdm import tqdm


class AMRDataset(Dataset):
    def __init__(self, genotype_file: str, phenotype_file: str, split_file: str):
        self.x = pd.read_csv(genotype_file, sep="\t", index_col=0)
        self.y = pd.read_csv(phenotype_file, sep="\t", index_col=0)
        with open(split_file) as f:
            split_indices = f.read().splitlines()
            split_indices = list(set(split_indices).intersection(self.x.index))
        self.x = self.x.loc[split_indices]
        self.y = self.y.loc[split_indices]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_item = self.x.iloc[idx].values.astype("float32")
        y_item = self.y.iloc[idx].values.astype("float32")
        return x_item, y_item

    @property
    def n_feats(self):
        return self.x.shape[1]


class AMRDataModule(LightningDataModule):
    def __init__(
        self,
        folder: str,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        super().__init__()
        self.folder = folder
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train = AMRDataset(
            genotype_file=f"{self.folder}/genotype.tsv",
            phenotype_file=f"{self.folder}/phenotype.tsv",
            split_file=f"{self.folder}/train.txt",
        )
        self.val = AMRDataset(
            genotype_file=f"{self.folder}/genotype.tsv",
            phenotype_file=f"{self.folder}/phenotype.tsv",
            split_file=f"{self.folder}/validation.txt",
        )
        self.test = AMRDataset(
            genotype_file=f"{self.folder}/genotype.tsv",
            phenotype_file=f"{self.folder}/phenotype.tsv",
            split_file=f"{self.folder}/test.txt",
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def n_feats(self):
        return self.train.n_feats


class ESMDataset(Dataset):
    def __init__(self, embeddings_folder: str, csv_file: str, split: Literal["train", "valid", "test"]):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df["split"] == split]
        self.embeddings_folder = Path(embeddings_folder)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with open(self.embeddings_folder / f"{row['ID']}.pkl", "rb") as f:
            x = pickle.load(f).detach().cpu().float()
        y = torch.tensor(row["label"], dtype=torch.float32).view(1)
        return x, y


class ESMDataModule(LightningDataModule):
    def __init__(
        self,
        embeddings_folder: str,
        csv_file: str,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        super().__init__()
        self.embeddings_folder = embeddings_folder
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train = ESMDataset(self.embeddings_folder, self.csv_file, "train")
        self.val = ESMDataset(self.embeddings_folder, self.csv_file, "valid")
        self.test = ESMDataset(self.embeddings_folder, self.csv_file, "test")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def n_feats(self):
        if len(self.train) > 0:
            return self.train[0][0].shape[0]
        return 0
