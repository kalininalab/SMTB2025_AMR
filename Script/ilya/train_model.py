import pandas as pd
import lightning as L
from lightning.pytorch.loggers import CSVLogger
import torchmetrics as M
import torch
from sklearn.model_selection import train_test_split
import argparse


# Acinetobacter_baumannii Escherichia_coli Neisseria_gonorrhoeae Salmonella_enterica Streptococcus_pneumoniae Campylobacter_jejuni Klebsiella_pneumoniae Pseudomonas_aeruginosa Staphylococcus_aureus


parser = argparse.ArgumentParser(description="Train a model on AMR data")
parser.add_argument(
    "--species", type=str, required=True, help="Species name for the model"
)
parser.add_argument(
    "--dropout", type=float, default=0.5, help="Dropout rate for the model"
)
parser.add_argument(
    "--max_epochs", type=int, default=50, help="Number of epochs to train"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training and evaluation"
)
parser.add_argument(
    "--hidden_dim", type=int, default=64, help="Hidden dimension for the model"
)

args = parser.parse_args()

genotype_file = f"/srv/scratch/AMR/Reduced_genotype/{args.species}_reduced_genotype.tsv"
phenotype_file = f"/srv/scratch/AMR/IR_phenotype/{args.species}/phenotype.txt"
x = pd.read_csv(genotype_file, sep="\t", index_col=0)
y = pd.read_csv(phenotype_file, sep="\t", index_col=0)
y = y.loc[x.index]

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(
    x_val, y_val, test_size=0.5, random_state=42
)


train_dataloader = torch.utils.data.DataLoader(
    list(
        zip(
            torch.tensor(x_train.values, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.float32),
        )
    ),
    batch_size=args.batch_size,
    shuffle=True,
)
val_dataloader = torch.utils.data.DataLoader(
    list(
        zip(
            torch.tensor(x_val.values, dtype=torch.float32),
            torch.tensor(y_val.values, dtype=torch.float32),
        )
    ),
    batch_size=args.batch_size,
    shuffle=False,
)
test_dataloader = torch.utils.data.DataLoader(
    list(
        zip(
            torch.tensor(x_test.values, dtype=torch.float32),
            torch.tensor(y_test.values, dtype=torch.float32),
        )
    ),
    batch_size=args.batch_size,
    shuffle=False,
)


class MyModel(L.LightningModule):
    def __init__(
        self,
        n_feats: int,
        dropout: float = 0.5,
        hidden_dim: int = 64,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_feats, hidden_dim),  # layer 1
            torch.nn.ReLU(),  # activation function
            torch.nn.Dropout(dropout),  # dropout for regularization
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1),
        )
        self.lr = lr

    def forward(self, x: torch.Tensor):
        return self.mlp(x)

    def shared_step(self, batch, step: str = "train"):
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        mcc = M.functional.matthews_corrcoef(
            y_hat.sigmoid() > 0.5,
            y.int(),
            num_classes=2,
            task="binary",
        )
        acc = M.functional.accuracy(
            y_hat.sigmoid() > 0.5,
            y.int(),
            num_classes=2,
            task="binary",
        )
        self.log(f"{step}_loss", loss, on_step=True, on_epoch=True)
        self.log(f"{step}_mcc", mcc, on_step=False, on_epoch=True)
        self.log(f"{step}_acc", acc, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, step="train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, step="val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, step="test")

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=0.1, patience=5, verbose=True
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }


model = MyModel(n_feats=x.shape[1], dropout=args.dropout)
csv_logger = CSVLogger("logs", name=args.species)
checkpointer = L.pytorch.callbacks.ModelCheckpoint(
    # dirpath="checkpoints",
    filename="best_model",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)
trainer = L.Trainer(
    max_epochs=50,
    accelerator="cpu",
    logger=csv_logger,
    callbacks=[checkpointer],
    enable_progress_bar=True,
    log_every_n_steps=10,
)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.test(model, dataloaders=test_dataloader)
