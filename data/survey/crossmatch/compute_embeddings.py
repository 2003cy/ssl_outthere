"""Compute paired image + spectrum embeddings for the DJA×image crossmatch.

Reads the slim join table from build_crossmatch.py (one row per matched
(spectrum, cutout) pair) and, for every row, runs each encoder's public
raw→embedding API — no preprocessing is re-implemented here:

    image    JWST_DINO.compute_embedding_from_raw_image(flux)
                 → {"cls": (N,512), "patch": (N,P,512)}   (crop + asinh inside)
    spectrum LowResPT.compute_embedding_from_raw_spectrum(flux, wavelength, valid_mask)
                 → {"patch_token": (N,Nspec,D), "token_valid_mask", "stats", "cls_token"}
                   (data_stretch + patchify + encode inside)

Both encoders are pointed at the SAME object via the crossmatch table:
    image    cutout : root/rel_path (.npy shard) row local_idx  (raw pretraining flux)
    spectrum        : dja_id -> row of the DJA CATALOG          (f_nu, windowed)

Output: a standalone HDF5 in the fusion-embedding schema plus per-row provenance
(survey, image_id, tile, sep_arcsec). Rows sharing a dja_id (a spectrum matched in
>1 survey) recompute the same spectrum embedding so the arrays stay row-aligned.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import h5py
from astropy.io import fits
from astropy.table import Table
from tqdm.auto import tqdm

PROJECT_ROOT = Path(os.path.expanduser("~/ssl_outthere"))
IMAGE_ROOT   = PROJECT_ROOT / "data/image"
DJA_FITS     = PROJECT_ROOT / "data/spectrum/DJA_spectra_v4.5.fits"
XMATCH_FITS  = PROJECT_ROOT / "data/crossmatched/dja_x_f150w.fits"
OUT_H5       = PROJECT_ROOT / "data/crossmatched/embeddings_f150w.h5"

IMG_CKPT  = PROJECT_ROOT / "encoder_image/jwst_dino/outputs/jwst_dino_ps6_st3/version_6/checkpoints/last.ckpt"
SPEC_CKPT = (PROJECT_ROOT / "encoder_spectrum/LowResPT/outputs/"
             "low_res_pt_1_2_micron_noz_cut_no_lineweight_no_vistoken/version_0/"
             "checkpoints/epoch=epoch=267-val_hid_loss=val_hid_loss=0.2073.ckpt")

BATCH_SIZE = 256
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_JANSKY = False   # dataset-level unit: f_nu→f_lambda (÷λ²) before the encoder


class RawCutoutReader:
    """Batches of raw cutouts from .npy shards, addressed by (rel_path, local_idx)."""

    def __init__(self, root, rel_paths, local_idxs):
        self.root = str(root)
        self.rel  = rel_paths
        self.loc  = local_idxs
        self.shards = {}

    def _shard(self, rel_path):
        if rel_path not in self.shards:
            self.shards[rel_path] = np.load(os.path.join(self.root, rel_path), mmap_mode="r")
        return self.shards[rel_path]

    def batch(self, i0, i1):
        return np.stack([np.asarray(self._shard(self.rel[i])[self.loc[i]], dtype=np.float32)
                         for i in range(i0, i1)])


def raw_stats(values, valid):
    """Per-row [mean, std] of `values` over its `valid` entries → (n, 2) float32.

    Measured on the raw flux, before any encoder-side normalisation, so the pair
    carries the absolute scale that the encoders normalise away. Rows with no
    valid entry yield NaN, which the fusion StatsEncoder imputes to 0.
    """
    v = np.where(valid, values, np.nan).reshape(len(values), -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows
        mean = np.nanmean(v, axis=1)
        std  = np.nanstd(v, axis=1)
    return np.stack([mean, std], axis=1).astype(np.float32)


@torch.no_grad()
def extract_image_embeddings(model, reader, n):
    """CLS (n,D) + patch tokens (n,P,D) + raw-crop stats (n,2) via the raw-image API."""
    from torchvision.transforms.functional import center_crop

    crop = model.hparams.global_crops_size
    cls_parts, patch_parts, stats_parts = [], [], []
    for i in tqdm(range(0, n, BATCH_SIZE), desc="image embeddings"):
        raw = reader.batch(i, min(i + BATCH_SIZE, n))           # (B, H, W) raw flux
        out = model.compute_embedding_from_raw_image(raw)
        cls_parts.append(out["cls"].float().cpu().numpy())
        patch_parts.append(out["patch"].float().cpu().numpy())
        # Same centre crop the encoder is fed, still in raw flux units.
        cut = center_crop(torch.from_numpy(raw), crop).numpy()   # (B, crop, crop)
        stats_parts.append(raw_stats(cut, np.isfinite(cut)))
    return (np.concatenate(cls_parts, 0),
            np.concatenate(patch_parts, 0),
            np.concatenate(stats_parts, 0))


@torch.no_grad()
def extract_spectrum_embeddings(model, flux_m, valid_m, wave_win):
    """Patch tokens (n,Nspec,D), token mask (n,Nspec), raw stats (n,2) via the raw-spectrum API.

    The returned stats are the raw-flux [mean, std]; the encoder's own `stats`
    output is measured after its per-sample arcsinh normalisation and is
    invariant to an overall rescaling of the spectrum, so it carries no
    absolute scale.
    """
    wave_win = wave_win.astype(np.float32)
    patch_parts, mask_parts, stats_parts = [], [], []
    for i in tqdm(range(0, len(flux_m), BATCH_SIZE), desc="spectrum embeddings"):
        fl = flux_m[i:i + BATCH_SIZE].astype(np.float32)
        vm = valid_m[i:i + BATCH_SIZE]
        if not USE_JANSKY:
            fl = fl / (wave_win[None] ** 2)                      # f_lambda ∝ f_nu / lambda^2
        flux  = torch.from_numpy(fl).to(DEVICE)
        wave  = torch.from_numpy(np.broadcast_to(wave_win, fl.shape).copy()).to(DEVICE)
        vmask = torch.from_numpy(vm).to(DEVICE).bool()
        out = model.compute_embedding_from_raw_spectrum(flux, wave, vmask)
        patch_parts.append(out["patch_token"].cpu().numpy())
        mask_parts.append(out["token_valid_mask"].cpu().numpy())
        stats_parts.append(raw_stats(fl, vm))
    return (np.concatenate(patch_parts, 0),
            np.concatenate(mask_parts, 0),
            np.concatenate(stats_parts, 0))


def main():
    # ── crossmatch table (row order defines the output order) ──
    xt       = Table.read(XMATCH_FITS)
    dja_id   = np.asarray(xt["dja_id"], dtype=np.int64)
    rel      = np.asarray(xt["rel_path"]).astype(str)
    loc      = np.asarray(xt["local_idx"], dtype=np.int64)
    survey   = np.char.strip(np.asarray(xt["survey"]).astype(str))
    image_id = np.asarray(xt["image_id"], dtype=np.int64)
    tile     = np.char.strip(np.asarray(xt["tile"]).astype(str))
    sep      = np.asarray(xt["sep_arcsec"], dtype=np.float32)
    ra       = np.asarray(xt["ra"], dtype=np.float64)
    dec      = np.asarray(xt["dec"], dtype=np.float64)
    N        = len(xt)
    print(f"crossmatch rows: {N}   unique spectra: {len(np.unique(dja_id))}")
    for s in np.unique(survey):
        print(f"  {s:9s} {int((survey == s).sum()):6d}")

    # ── DJA CATALOG: wave grid + per-row spectrum arrays, indexed by dja_id ──
    with fits.open(DJA_FITS, memmap=False) as hdul:
        wave_full = np.asarray(hdul["WAVE"].data, dtype=np.float32)   # (473,)
        cat = hdul["CATALOG"].data
        flux_all  = np.asarray(cat["flux"], dtype=np.float32)         # (Ndja, 473) f_nu
        valid_all = np.asarray(cat["valid_spec"], dtype=bool)
        sn50_all  = np.asarray(cat["sn50"], dtype=np.float32)

    # ── image encoder (jwst_dino) — its `model`/`data` packages load first ──
    print(f"[1/4] image encoder: {IMG_CKPT.name}")
    sys.path.insert(0, str(PROJECT_ROOT / "encoder_image/jwst_dino"))
    from model.jwst_dino import JWST_DINO
    img_model = JWST_DINO.load_from_checkpoint(str(IMG_CKPT), map_location=DEVICE).eval().to(DEVICE)
    print(f"  crop_size={img_model.hparams.global_crops_size}")

    print("[2/4] extracting image embeddings …")
    reader = RawCutoutReader(IMAGE_ROOT, rel, loc)
    img_cls, img_patch, image_stat = extract_image_embeddings(img_model, reader, N)
    print(f"  cls {img_cls.shape}   patch {img_patch.shape}   stats {image_stat.shape}")

    # ── spectrum encoder (LowResPT) — purge jwst_dino's `model`/`data` packages first ──
    print(f"[3/4] spectrum encoder: {SPEC_CKPT.name}")
    for m in list(sys.modules):
        if m in ("model", "data") or m.startswith(("model.", "data.")):
            del sys.modules[m]
    sys.path.insert(0, str(PROJECT_ROOT / "encoder_spectrum/LowResPT"))
    from model.low_res_pt import LowResPT
    spec = LowResPT.load_from_checkpoint(str(SPEC_CKPT), map_location=DEVICE).eval().to(DEVICE)
    # Older checkpoints predate some hparams the current model code reads internally
    # (e.g. data_stretch uses self.hparams.min_std); restore the class default.
    if "min_std" not in spec.hparams:
        spec.hparams["min_std"] = 0.1
    wl_min = spec.hparams.get("wl_ref_min", None)
    wl_max = spec.hparams.get("wl_ref_max", None)
    print(f"  embed_dim={spec.embed_dim}  patch={spec.hparams.patch_size}  "
          f"stride={spec.hparams.stride}  wl=[{wl_min},{wl_max}]  use_jansky={USE_JANSKY}")

    # window the shared grid to the checkpoint's training range
    wl_keep = np.ones(wave_full.shape[0], dtype=bool)
    if wl_min is not None:
        wl_keep &= wave_full > wl_min
    if wl_max is not None:
        wl_keep &= wave_full < wl_max
    wave_win = wave_full[wl_keep]

    # row-aligned spectrum arrays (NaN/Inf → 0, validity anded with finiteness)
    flux_win  = flux_all[dja_id][:, wl_keep]
    finite    = np.isfinite(flux_win)
    flux_win  = np.where(finite, flux_win, 0.0).astype(np.float32)
    valid_win = valid_all[dja_id][:, wl_keep] & finite
    print(f"  spectral window: {wave_win.shape[0]} px "
          f"({wave_win[0]:.3f}–{wave_win[-1]:.3f} µm)")

    print("[4/4] extracting spectrum embeddings …")
    spec_patch, spec_mask, spec_stats = extract_spectrum_embeddings(
        spec, flux_win, valid_win, wave_win)
    print(f"  patch {spec_patch.shape}   mask {spec_mask.shape}   stats {spec_stats.shape}")

    # ── per-row scalars ──
    n_valid    = valid_win.sum(axis=1).astype(np.int32)
    sn50       = sn50_all[dja_id].astype(np.float32)

    for nm, st in [("image_stats", image_stat), ("spectrum_stats", spec_stats)]:
        med = np.nanmedian(np.abs(st), axis=0)
        print(f"  {nm}: NaN={100 * np.isnan(st).mean(axis=0).round(4)}%  "
              f"median|.|={med}  -> suggested stats_scale={med.tolist()}")

    # ── write ──
    print(f"writing → {OUT_H5}")
    OUT_H5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUT_H5, "w") as f:
        for key, data in [
            ("image_cls_embed",      img_cls),
            ("image_patch_embed",    img_patch),
            ("spectrum_patch_embed", spec_patch),
            ("spectrum_token_mask",  spec_mask),
            ("spectrum_stats",       spec_stats),
            ("image_stats",          image_stat),
            ("n_valid",              n_valid),
            ("sn50",                 sn50),
            ("id",                   dja_id),
            ("ra",                   ra),
            ("dec",                  dec),
            ("image_id",             image_id),
            ("sep_arcsec",           sep),
        ]:
            f.create_dataset(key, data=data, compression="gzip")
            print(f"  {key}: {data.shape}  {data.dtype}")
        # fixed-length utf-8 bytes (h5py has no conversion path from numpy '<U' vlen)
        f.create_dataset("survey", data=np.char.encode(survey, "utf-8"), compression="gzip")
        f.create_dataset("tile",   data=np.char.encode(tile,   "utf-8"), compression="gzip")
    print("done.")


if __name__ == "__main__":
    main()
