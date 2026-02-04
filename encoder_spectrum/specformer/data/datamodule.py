from typing import Callable, Dict, List
import numpy as np
import datasets
import lightning as L
import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import CenterCrop


class AstroClipDataloader(L.LightningDataModule):
    def __init__(
        self,
        path: str,
        columns: List[str] = ["image", "spectrum"],
        batch_size: int = 512,
        num_workers: int = 10,
        collate_fn: Callable[[Dict[str, Tensor]], Dict[str, Tensor]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        self.dataset = datasets.load_from_disk(self.hparams.path)
        self.dataset.set_format(type="torch", columns=self.hparams.columns)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["train"],
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,  # NOTE: disable for debugging
            drop_last=True,
            collate_fn=self.hparams.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["test"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,  # NOTE: disable for debugging
            drop_last=True,
            collate_fn=self.hparams.collate_fn,
        )


class AstroClipCollator:
    def __init__(
        self,
        center_crop: int = 144,
        bands: List[str] = ["g", "r", "z"],
        m: float = 0.03,
        Q: int = 20,
    ):
        self.center_crop = CenterCrop(center_crop)
        self.to_rgb = ToRGB(bands=bands, m=m, Q=Q)

    def _process_images(self, images):
        # convert to rgb
        img_outs = []
        for img in images:
            rgb_img = torch.tensor(self.to_rgb(img)[None, :, :, :])
            img_outs.append(rgb_img)
        images = torch.concatenate(img_outs)

        images = self.center_crop(images.permute(0, 3, 2, 1))
        return images

    def __call__(self, samples):
        # collate and handle dimensions
        samples = default_collate(samples)
        # process images
        samples["image"] = self._process_images(samples["image"])
        return samples



class ToRGB:
    """
    Stretch a single-band cutout and broadcast it to RGB by cloning the channel
    after applying a simple arcsinh stretch (mirrors ToRGB logic but mono-band).
    """

    def __init__(self, scale: float = 1.0, m: float = 0.03, Q: float = 20.0, return_channel_pos: int = 0):
        self.scale = scale
        self.m = m
        self.Q = Q
        self._return_channel_pos = return_channel_pos
        
    def __call__(self, imgs: np.ndarray) -> np.ndarray:
        return_channel_pos = self._return_channel_pos
        # Accept HxW, 1xHxW, or 3xHxW inputs (already channel-first)
        arr = np.asarray(imgs, dtype=np.float32)
        if arr.ndim == 3:
            base = arr[0]
        elif arr.ndim == 2:
            base = arr
        else:
            base = arr.squeeze()

        base = np.maximum(0.0, base * self.scale + self.m)
        I = base
        I += (I == 0.0) * 1e-6
        fI = np.arcsinh(self.Q * I) / np.sqrt(self.Q + 1e-8)
        stretched = np.clip(base * fI / I, 0.0, 1.0).astype(np.float32)

        # Repeat into three identical channels; DataAugmentation converts to CHW later.
        rgb = np.stack((stretched, stretched, stretched), axis=-1)
        if return_channel_pos == 0:
            rgb = np.transpose(rgb, (2, 0, 1))  # C x H x W
            return rgb
        elif return_channel_pos == 2:
            return rgb
