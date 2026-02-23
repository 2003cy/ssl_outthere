#!/u/yacheng/conda-envs/astrodino/bin/python
"""LightningCLI setup for ma_specformer training."""

import sys
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.cli import LightningCLI

# Add current directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from model import MASpecFormer
from data.datamodule import MASpectrumDataModule


def main():
    """Entry point for LightningCLI-based training."""
    # Enable TensorCore optimization for NVIDIA GPUs with Tensor Cores
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    cli = LightningCLI(
        MASpecFormer,
        MASpectrumDataModule,
        save_config_callback=None,  # We use RunManager to save config
    )


if __name__ == "__main__":
    main()
