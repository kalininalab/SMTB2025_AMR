import lightning as L
import torch
import torchmetrics as M
from lightning.pytorch.loggers import CSVLogger


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

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.1, patience=5, verbose=True)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
