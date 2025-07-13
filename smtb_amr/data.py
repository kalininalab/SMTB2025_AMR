import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset


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
