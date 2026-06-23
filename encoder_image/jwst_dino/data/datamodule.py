"""DataModule for JWST_DINO pre-training (mirrors LowResPT/data/datamodule.py).

Builds the multi-crop DINO augmentation, the iBOT block-mask generator, and the
collate that casts crops + draws masks. DDP sharding is handled by Lightning's
auto-injected DistributedSampler (shuffle=True), so no custom sampler is needed.
"""

from functools import partial
from typing import Sequence, Union

import lightning as L
import torch
from torch.utils.data import DataLoader

from .augmentations import DataAugmentationJWSTDINO
from .dataset import JWST
from .masking import MaskingGenerator, collate_data_and_cast


class JWSTDINODataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        filter: str = "f150w",
        survey: Union[str, Sequence[str]] = "cosmos",
        batch_size: int = 96,
        num_workers: int = 16,
        # crop / patch geometry (linked to the model via LightningCLI)
        patch_size: int = 6,
        patch_stride: int = 3,
        global_crops_size: int = 72,
        local_crops_size: int = 36,
        local_crops_number: int = 8,
        center_crop_size: int = 100,
        # noise augmentation (per-tile relative half-normal noise)
        noise_w: float = 1.5,
        noise_s_max: float | None = None,
        # iBOT masking
        ibot_mask_ratio_min_max: Sequence[float] = (0.1, 0.3),
        ibot_mask_sample_probability: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = None
        self.val_dataset = None

        side = (global_crops_size - patch_size) // patch_stride + 1
        self.n_tokens = side * side
        self.mask_generator = MaskingGenerator(
            input_size=(side, side), max_num_patches=0.5 * self.n_tokens,
        )

    def _transform(self) -> DataAugmentationJWSTDINO:
        h = self.hparams
        return DataAugmentationJWSTDINO(
            h.local_crops_number,
            global_crops_size=h.global_crops_size,
            local_crops_size=h.local_crops_size,
            center_crop_size=h.center_crop_size,
            noise_w=h.noise_w,
            noise_s_max=h.noise_s_max,
        )

    def setup(self, stage: str = None) -> None:
        if self.train_dataset is None:
            h = self.hparams
            common = dict(root=h.root, filter=h.filter, survey=h.survey,
                          transform=self._transform())
            self.train_dataset = JWST(split="train", **common)
            self.val_dataset = JWST(split="val", **common)

    def _collate(self):
        return partial(
            collate_data_and_cast,
            mask_ratio_tuple=tuple(self.hparams.ibot_mask_ratio_min_max),
            mask_probability=self.hparams.ibot_mask_sample_probability,
            n_tokens=self.n_tokens,
            mask_generator=self.mask_generator,
            dtype=torch.float32,  # bf16-mixed autocast handles precision in the model
        )

    def _loader(self, dataset, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=self._collate(),
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False, drop_last=True)
