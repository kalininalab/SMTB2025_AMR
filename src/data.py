import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset


class BinaryDataset(Dataset):
    def __init__(self, genotype_file: str, phenotype_file: str):
        self.x = pd.read_csv(genotype_file, sep="\t", index_col=0)
        self.y = pd.read_csv(phenotype_file, sep="\t", index_col=0)
        self.y = self.y.loc[self.x.index]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_item = self.x.iloc[idx].values.astype(float)
        y_item = self.y.iloc[idx].values.astype(float)
        return x_item, y_item


class DataModule(LightningDataModule):
    def __init__(
        self,
        ds_class: Dataset,
        genotype_file: str,
        phenotype_file: str,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        self.dataset = ds_class(genotype_file, phenotype_file)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
