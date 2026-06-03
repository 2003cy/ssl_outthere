"""LightningCLI entrypoint for LowResPT training.

Usage:
    python trainer.py fit --config low_res_pt.yaml
    python trainer.py fit --config low_res_pt.yaml --trainer.devices=[0,1]
"""

import sys
import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# Ensure local model/ and data/ are importable when running from this directory
sys.path.insert(0, os.path.dirname(__file__))

import torch
from lightning.pytorch.cli import LightningCLI

from data.datamodule import LowResDataModule
from model.low_res_pt import LowResPT


class _CLI(LightningCLI):
    """Link wl_ref_{min,max} from data config to model config.

    User sets them once under `data:` in the yaml; both DataModule and model
    receive the same values automatically.
    """

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.wl_ref_min", "model.wl_ref_min")
        parser.link_arguments("data.wl_ref_max", "model.wl_ref_max")


def main():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    _CLI(
        model_class=LowResPT,
        datamodule_class=LowResDataModule,
        save_config_callback=None,
        seed_everything_default=42,
    )


if __name__ == "__main__":
    main()
