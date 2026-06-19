"""LightningCLI entrypoint for JWST_DINO pre-training.

Usage:
    python trainer.py fit --config jwst_dino.yaml
    python trainer.py fit --config jwst_dino.yaml --trainer.devices=4 --trainer.num_nodes=2
"""

import os
import sys

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# Make local model/ and data/ importable when run from this directory.
sys.path.insert(0, os.path.dirname(__file__))

import torch
from lightning.pytorch.cli import LightningCLI

from data.datamodule import JWSTDINODataModule
from model.jwst_dino import JWST_DINO


class _CLI(LightningCLI):
    """Define crop/patch geometry once under `data:`; the model reuses it.

    Same pattern as LowResPT's wl_ref_{min,max} linking.
    """

    def add_arguments_to_parser(self, parser):
        for key in ("patch_size", "patch_stride", "global_crops_size",
                    "local_crops_size", "local_crops_number"):
            parser.link_arguments(f"data.{key}", f"model.{key}")


def main():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    _CLI(
        model_class=JWST_DINO,
        datamodule_class=JWSTDINODataModule,
        save_config_callback=None,
        seed_everything_default=0,
    )


if __name__ == "__main__":
    main()
