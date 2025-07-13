import pandas as pd
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torchmetrics as M
from sklearn.model_selection import train_test_split
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import numpy as np
import argparse

#here parse {species}

genotype_file = f"/srv/scratch/AMR/Reduced_genotype/{species}_reduced_genotype.tsv"
phenotype_file = f"/srv/scratch/AMR/IR_phenotype/{species}/phenotype.txt"
x = pd.read_csv(genotype_file, sep="\t", index_col=0)
y = pd.read_csv(phenotype_file, sep="\t", index_col=0)
y = y.loc[x.index]

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

def get_dataloaders(batch_size):
    train = torch.utils.data.DataLoader(
        list(zip(
            torch.tensor(x_train.values, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.float32),
        )),
        batch_size=batch_size,
        shuffle=True,
    )
    val = torch.utils.data.DataLoader(
        list(zip(
            torch.tensor(x_val.values, dtype=torch.float32),
            torch.tensor(y_val.values, dtype=torch.float32),
        )),
        batch_size=batch_size,
        shuffle=False,
    )
    return train, val

class MyModel(L.LightningModule):
    def __init__(self, n_feats, dropout, hidden_dim, lr):
        super().__init__()
        self.save_hyperparameters()
        self.mlp = nn.Sequential(
            nn.Linear(n_feats, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.mlp(x)

    def shared_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        acc = M.functional.accuracy(y_hat.sigmoid() > 0.5, y.int(), task="binary", num_classes=2)
        return {"loss": loss, "acc": acc}

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch)
        self.log("train_loss", out["loss"])
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch)
        self.log("val_loss", out["loss"], prog_bar=True)
        self.log("val_acc", out["acc"], prog_bar=True)
        return out["loss"]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def train_model(config):
    train_loader, val_loader = get_dataloaders(config["batch_size"])
    
    model = MyModel(
        n_feats=x.shape[1],
        dropout=config["dropout"],
        hidden_dim=config["hidden_dim"],
        lr=config["lr"]
    )

    trainer = L.Trainer(
        max_epochs=20,
        logger=False,
        enable_progress_bar=False,
        callbacks=[TuneReportCallback(metrics={"val_loss": "val_loss"}, on="validation_end")],
    )

    trainer.fit(model, train_loader, val_loader)

search_space = {
    "dropout": tune.uniform(0.1, 0.6),
    "hidden_dim": tune.choice([32, 64, 128, 256]),
    "lr": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.choice([16, 32, 64, 128]),
}

scheduler = ASHAScheduler(
    max_t=20,
    grace_period=5,
    reduction_factor=2,
)

tuner = tune.Tuner(
    train_model,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        metric="val_loss",
        mode="min",
        scheduler=scheduler,
        num_samples=20, 
    ),
)

results = tuner.fit()
best_config = results.get_best_result().config
print("Best hyperparameters found:")
print(best_config)

#save answers in a separate file