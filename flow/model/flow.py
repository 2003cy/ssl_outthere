"""Conditional NSF, following LowResPT's LightningModule + YAML configuration.

From a notebook in flow/:
    !python model/flow.py fit --config z_1_3.yaml
    !python model/flow.py fit --config full.yaml

forward/sample/log_prob use unscaled CLIP latents and labels in model units.
For the supplied configs these are log10(1+z), log M, log SFR100, log Zgas,
and log10(A_V / mag). They do not automatically convert redshift back to z.
"""

import sys
from pathlib import Path
from typing import Optional, Sequence

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lightning.pytorch as L
import torch
import torch.nn as nn
import zuko
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import LightningCLI
from torch.distributions import AffineTransform, TransformedDistribution


class Flow(L.LightningModule):
    """MLP context trunk + conditional neural spline flow trained with NLL."""

    def __init__(
        self,
        input_dim: int = 512,
        features: int = 5,
        trunk_hidden_dim: int = 256,
        context_dim: int = 32,
        dropout: float = 0.2,
        transforms: int = 4,
        bins: int = 8,
        hidden_features: Sequence[int] = (128, 128),
        passes: Optional[int] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        target_names: Optional[list[str]] = None,
        data_config: Optional[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, trunk_hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(trunk_hidden_dim, context_dim), nn.GELU(),
        )
        self.flow = zuko.flows.NSF(features=features, context=context_dim,
                                   transforms=transforms, bins=bins,
                                   hidden_features=list(hidden_features), passes=passes)
        # Save train-only normalization with the model, so checkpoint inference
        # needs neither the original sklearn objects nor the training notebook.
        self.register_buffer("x_mean", torch.zeros(input_dim))
        self.register_buffer("x_scale", torch.ones(input_dim))
        self.register_buffer("y_mean", torch.zeros(features))
        self.register_buffer("y_scale", torch.ones(features))

    def on_fit_start(self):
        data = self.trainer.datamodule
        if data.X.shape[1] != self.hparams.input_dim or data.Y.shape[1] != self.hparams.features:
            raise ValueError(f"Set model.input_dim={data.X.shape[1]} and model.features={data.Y.shape[1]} "
                             "to match the prepared data")
        for name, values in (("x_mean", data.xs.mean_), ("x_scale", data.xs.scale_),
                             ("y_mean", data.ys.mean_), ("y_scale", data.ys.scale_)):
            getattr(self, name).copy_(torch.as_tensor(values, device=self.device, dtype=self.dtype))
        self.hparams.target_names = list(data.hparams.target_columns)
        # LightningCLI injects private metadata such as _instantiator; these
        # are not arguments to JADESDataModule when rebuilding it in a notebook.
        self.hparams.data_config = {k: v for k, v in data.hparams.items()
                                   if not k.startswith("_")}
        self.hparams.data_config["fusion_checkpoint"] = data.fusion_checkpoint

    def _distribution(self, x):
        x = torch.as_tensor(x, device=self.device, dtype=self.dtype)
        return self.flow(self.trunk((x - self.x_mean) / self.x_scale))

    def forward(self, x):
        """Return the joint distribution in unstandardized model-label units."""
        return TransformedDistribution(
            self._distribution(x),
            AffineTransform(loc=self.y_mean, scale=self.y_scale, event_dim=1),
        )

    def _nll(self, batch):
        x, y = batch
        # Match the notebooks' training objective in standardized label space.
        return -self._distribution(x).log_prob((y - self.y_mean) / self.y_scale).mean()

    def training_step(self, batch, batch_idx):
        loss = self._nll(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False, batch_size=len(batch[0]))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._nll(batch)
        self.log("val_nll", loss, on_step=False, on_epoch=True, batch_size=len(batch[0]),
                 sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._nll(batch)
        self.log("test_nll", loss, batch_size=len(batch[0]), sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay)

    @torch.no_grad()
    def sample(self, x, n: int = 3000, batch_size: int = 128):
        """Draw (n, n_objects, features) samples; batch objects and return on CPU."""
        if n < 1 or batch_size < 1 or len(x) == 0:
            raise ValueError("n, batch_size and the number of input objects must be positive")
        self.eval()
        return torch.cat([self(x[start:start + batch_size]).sample((n,)).cpu()
                          for start in range(0, len(x), batch_size)], dim=1)

    def log_prob(self, x, y):
        """Joint log density, including the label-standardization Jacobian."""
        y = torch.as_tensor(y, device=self.device, dtype=self.dtype)
        return self(x).log_prob(y)


class EpochPrinter(Callback):
    """One plain line per epoch, like LowResPT's callback, for notebook ! commands."""

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        metrics = trainer.callback_metrics
        train = metrics.get("train_loss", float("nan"))
        val = metrics.get("val_nll", float("nan"))
        print(f"epoch {trainer.current_epoch:3d} | train_loss={float(train):.4f} | "
              f"val_nll={float(val):.4f}", flush=True)


if __name__ == "__main__":
    from flow.data.datamodule_jades import JADESDataModule

    LightningCLI(Flow, JADESDataModule, seed_everything_default=0,
                 save_config_callback=None, auto_configure_optimizers=False)
