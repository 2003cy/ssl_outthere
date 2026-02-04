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
    
    # Create CLI with seed_everything support
    # LightningCLI automatically handles:
    # - Model initialization from config
    # - DataModule initialization from config
    # - Trainer initialization from config
    # - Training/validation/testing workflow
    # - Seed control via trainer.seed_everything
    cli = LightningCLI(
        MASpecFormer,
        MASpectrumDataModule,
        seed_everything_default=42,  # Default seed value
    )


if __name__ == "__main__":
    main()
