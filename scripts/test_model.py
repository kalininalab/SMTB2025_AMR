import argparse

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from smtb_amr.data import AMRDataModule
from smtb_amr.model import MyModel

torch.set_float32_matmul_precision("medium")

parser = argparse.ArgumentParser(description="Train a model on AMR data")
parser.add_argument("--data", type=str, required=True, help="Data folder")
parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for the model")
parser.add_argument("--max_epochs", type=int, default=50, help="Number of epochs to train")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for the model")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
parser.add_argument("--checkpoints")

args = parser.parse_args()


datamodule = AMRDataModule(
    folder=args.data,
    batch_size=args.batch_size,
    num_workers=args.num_workers,  # Adjust as needed for your environment
)
datamodule.setup()

model = MyModel.load_from_checkpoint(args.checkpoints, n_feats=datamodule.n_feats)
csv_logger = CSVLogger("logs", name=f"{args.checkpoints.split('/')[-4][4:]}_{args.data.split('/')[-1]}")

trainer = Trainer(
    max_epochs=args.max_epochs,
    accelerator="auto",
    devices="1",
    logger=csv_logger,
    enable_progress_bar=True,
    log_every_n_steps=10,
)
trainer.test(model, datamodule=datamodule)
