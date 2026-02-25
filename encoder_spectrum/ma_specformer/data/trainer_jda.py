#!/u/yacheng/conda-envs/astrodino/bin/python
"""LightningCLI setup for ma_specformer training on JDA spectra."""

import sys
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.cli import LightningCLI

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import MASpecFormer
from data.datamodule import JDASpectrumDataModule


def main():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    cli = LightningCLI(
        MASpecFormer,
        JDASpectrumDataModule,
        save_config_callback=None,
    )


if __name__ == "__main__":
    main()
