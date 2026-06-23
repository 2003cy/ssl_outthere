"""Cross-survey helpers for the jwst_dino benchmark.

`crossmatch_indices` RA/DEC-matches the cutouts shared by two surveys (e.g. the OutThere
`sex-*` fields overlap the COSMOS field, ~8.5k common objects). `CutoutDataset` is a
minimal image-only loader (same center-crop + asinh preprocessing as CosmosMorphDataset)
used to extract embeddings for an arbitrary set of (rel_path, local_idx) cutouts.
"""

from __future__ import annotations

import os

import astropy.units as u
import numpy as np
import torch
from astropy.coordinates import SkyCoord
from astropy.table import Table
from torch.utils.data import Dataset
from torchvision import transforms

from dataset import AsinhStretch  # re-exported from data.augmentations via dataset.py


def crossmatch_indices(root, survey_a="cosmos", survey_b="outthere",
                       filter="f150w", tol_arcsec=0.5):
    """Return (tab_a, tab_b), row-aligned so row i is the SAME object seen in both surveys.

    Each survey_b cutout is matched to its nearest survey_a cutout; pairs farther than
    ``tol_arcsec`` are dropped. ``tab_a``/``tab_b`` carry the original index columns
    (id, ra, dec, tile, rel_path, local_idx) plus a ``sep_arcsec`` column.
    """
    root = os.path.expandvars(os.path.expanduser(root))
    a = Table.read(os.path.join(root, f"image_index_{survey_a}_{filter}.fits"))
    b = Table.read(os.path.join(root, f"image_index_{survey_b}_{filter}.fits"))
    ca = SkyCoord(a["ra"] * u.deg, a["dec"] * u.deg)
    cb = SkyCoord(b["ra"] * u.deg, b["dec"] * u.deg)
    idx, sep, _ = cb.match_to_catalog_sky(ca)          # nearest a for each b
    keep = sep < tol_arcsec * u.arcsec
    tab_a, tab_b = a[idx[keep]], b[keep]
    tab_a["sep_arcsec"] = tab_b["sep_arcsec"] = sep[keep].arcsec
    return tab_a, tab_b


class CutoutDataset(Dataset):
    """Image-only loader for an index Table (or rel_path/local_idx arrays).

    Returns the preprocessed cutout (center-crop + asinh, channel-first) ready for the
    teacher backbone — the same clean preprocessing as CosmosMorphDataset, no labels.
    """

    def __init__(self, root, table=None, rel_paths=None, local_idxs=None,
                 crop_size=72, Q=20.0, scale=1.0):
        self.root = os.path.expandvars(os.path.expanduser(root))
        if table is not None:
            rel_paths = np.asarray(table["rel_path"]).astype(str)
            local_idxs = np.asarray(table["local_idx"], dtype=np.int64)
        self.rel = list(rel_paths)
        self.loc = list(local_idxs)
        self.center_crop = transforms.CenterCrop(crop_size)
        self.stretch = AsinhStretch(scale=scale, Q=Q, return_channel_pos=0)
        self.shards: dict[str, np.ndarray] = {}

    def _shard(self, rel_path):
        if rel_path not in self.shards:
            self.shards[rel_path] = np.load(os.path.join(self.root, rel_path), mmap_mode="r")
        return self.shards[rel_path]

    def __len__(self):
        return len(self.rel)

    def __getitem__(self, i):
        cutout = self._shard(self.rel[i])[self.loc[i]]
        img = np.nan_to_num(cutout.astype(np.float32))[None]
        img = self.center_crop(torch.from_numpy(img)).numpy()
        return torch.from_numpy(self.stretch(img))


@torch.no_grad()
def extract_embeddings(net, dataset, device, batch_size=256, num_workers=8):
    """CLS embeddings (N, D) for every cutout in ``dataset``."""
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    out = []
    for imgs in tqdm(loader, desc="embeddings"):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            out.append(net(imgs.to(device))["cls"].float().cpu().numpy())
    return np.concatenate(out)
