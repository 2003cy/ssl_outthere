"""Extract CLS embeddings from image and spectrum encoders for fusion training.

Data source: images/cosmosf150w_jda_low/jda_cosmos_matched.h5
  - 1772 aligned (image, spectrum) pairs
  - image group: image["image"] (1772, 128, 128)
  - jda   group: jda["flux"]   (1772, 56), jda["wave"] (1772, 56)

Outputs (saved to encoder_fusion/data/embeddings/):
  - image.npy    (1772, 512)  — AstroDINO CLS tokens
  - spectrum.npy (1772, 128)  — MASpecFormer CLS tokens

Usage (from any directory):
    cd /home/yacheng/ssl_outthere/encoder_image/astrodino
    XFORMERS_DISABLED=1 python ../../encoder_fusion/extract_embeddings.py
"""

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "encoder_image" / "astrodino" / "benchmark"))

# ── Config ──────────────────────────────────────────────────────────────────
MATCHED_H5 = PROJECT_ROOT / "images" / "cosmosf150w_jda_low" / "jda_cosmos_matched.h5"
OUTPUT_DIR  = PROJECT_ROOT / "encoder_fusion" / "data" / "embeddings"

IMG_CONFIG  = (PROJECT_ROOT
               / "encoder_image/astrodino/model"
               / "astrodino_f150w_vitb_ps6_bs128/config.yaml")
IMG_WEIGHTS = (PROJECT_ROOT
               / "encoder_image/astrodino/model"
               / "astrodino_f150w_vitb_ps6_bs128/eval/training_112999/teacher_checkpoint.pth")
SPEC_CKPT   = (PROJECT_ROOT
               / "encoder_spectrum/ma_specformer/outputs"
               / "jda_prism_f100lp_20260228_151905/ckpt-epoch=143-val_loss=0.3413.ckpt")

BATCH_SIZE = 128
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CROP_SIZE  = 72   # cfg.crops.global_crops_size for astrodino_f150w_vitb_ps6_bs128


# ── Datasets ─────────────────────────────────────────────────────────────────
class MatchedImageDataset(Dataset):
    def __init__(self, h5_path: Path, crop_size: int, to_rgb):
        self.h5_path = h5_path
        self.crop    = transforms.CenterCrop(crop_size)
        self.to_rgb  = to_rgb
        with h5py.File(h5_path, "r") as f:
            self.n = f["image"]["image"].shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            img = f["image"]["image"][idx].astype("float32")  # (H, W)
        img_t = torch.from_numpy(img[np.newaxis])             # (1, H, W)
        img_t = self.crop(img_t)                              # (1, 72, 72)
        img_t = torch.from_numpy(self.to_rgb(img_t.numpy())) # (1, 72, 72)
        return img_t


class MatchedSpectrumDataset(Dataset):
    def __init__(self, h5_path: Path):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.n = f["jda"]["flux"].shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            flux = f["jda"]["flux"][idx].astype("float32")
            wave = f["jda"]["wave"][idx].astype("float32")
        valid_mask = (flux != 0.0) & np.isfinite(flux) & np.isfinite(wave)
        flux = np.where(np.isfinite(flux), flux, 0.0)
        return {
            "flux":       torch.from_numpy(flux),
            "wavelength": torch.from_numpy(wave),
            "valid_mask": torch.from_numpy(valid_mask),
        }


def spec_collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ── Normalization (matches MASpecFormer training) ────────────────────────────
def normalize_flux(flux: torch.Tensor, valid_mask: torch.Tensor, min_std: float = 0.1):
    valid_flux  = flux.clone()
    valid_flux[~valid_mask] = 0.0
    valid_count = valid_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
    mean        = valid_flux.sum(dim=1, keepdim=True) / valid_count
    sq_diff     = ((flux - mean) ** 2) * valid_mask.float()
    std         = (sq_diff.sum(dim=1, keepdim=True) / valid_count).sqrt().clamp(min=min_std)
    flux_norm   = (flux - mean) / std * valid_mask.float()
    return flux_norm, mean, std


# ── Extraction ───────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_image_embeddings(model, loader):
    model.eval()
    parts = []
    for imgs in tqdm(loader, desc="Image embeddings"):
        emb = model(imgs.to(DEVICE))
        if isinstance(emb, tuple):
            emb = emb[0]
        parts.append(emb.cpu().numpy())
    return np.concatenate(parts, axis=0)


@torch.no_grad()
def extract_spectrum_embeddings(model, loader):
    model.eval()
    min_std = float(model.hparams.get("min_std", 0.1))
    parts = []
    for batch in tqdm(loader, desc="Spectrum embeddings"):
        flux  = batch["flux"].to(DEVICE).float()
        wave  = batch["wavelength"].to(DEVICE).float()
        vmask = batch["valid_mask"].to(DEVICE).bool()
        flux_norm, mean, std = normalize_flux(flux, vmask, min_std=min_std)
        stats = torch.cat([mean, std], dim=-1)
        cls   = model.encode(
            flux=flux_norm,
            wavelengths=wave,
            valid_mask=vmask,
            stats=stats,
            return_cls_only=True,
        )
        parts.append(cls.cpu().numpy())
    return np.concatenate(parts, axis=0)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device : {DEVICE}")
    print(f"H5 file: {MATCHED_H5}")
    print()

    # ── Image encoder ────────────────────────────────────────────────────────
    print("[1/4] Loading image encoder …")
    from omegaconf import OmegaConf
    from dinov2.eval.setup import build_model_for_eval
    from preprocessing import get_torgb

    cfg       = OmegaConf.load(IMG_CONFIG)
    img_model = build_model_for_eval(cfg, pretrained_weights=str(IMG_WEIGHTS))
    img_model = img_model.eval().to(DEVICE)
    to_rgb, in_chans = get_torgb(cfg)
    print(f"  Loaded — in_chans={in_chans}, embed_dim={img_model.embed_dim}")

    # ── Spectrum encoder ──────────────────────────────────────────────────────
    print("[2/4] Loading spectrum encoder …")
    from encoder_spectrum.ma_specformer.model.ma_specformer import MASpecFormer
    spec_model = MASpecFormer.load_from_checkpoint(str(SPEC_CKPT), map_location=DEVICE)
    spec_model = spec_model.eval().to(DEVICE)
    print(f"  Loaded — embed_dim={spec_model.embed_dim}")

    # ── Extract image embeddings ──────────────────────────────────────────────
    print("[3/4] Extracting image embeddings …")
    img_ds     = MatchedImageDataset(MATCHED_H5, CROP_SIZE, to_rgb)
    img_loader = DataLoader(img_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    img_embs   = extract_image_embeddings(img_model, img_loader)
    out_img    = OUTPUT_DIR / "image.npy"
    np.save(out_img, img_embs)
    print(f"  Saved {img_embs.shape} → {out_img}")

    # ── Extract spectrum embeddings ───────────────────────────────────────────
    print("[4/4] Extracting spectrum embeddings …")
    spec_ds     = MatchedSpectrumDataset(MATCHED_H5)
    spec_loader = DataLoader(spec_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, collate_fn=spec_collate)
    spec_embs   = extract_spectrum_embeddings(spec_model, spec_loader)
    out_spec    = OUTPUT_DIR / "spectrum.npy"
    np.save(out_spec, spec_embs)
    print(f"  Saved {spec_embs.shape} → {out_spec}")

    print("\nDone.")


if __name__ == "__main__":
    main()
