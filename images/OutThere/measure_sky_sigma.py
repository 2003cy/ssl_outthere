"""Measure per-field background sky sigma for OutThere F150W fields (seg-masked).

Feeds the per-field entries of ``SKY_SIGMA["outthere"]`` in
encoder_image/jwst_dino/data/augmentations.py.

OutThere needs the SAME random-patch + seg-mask + sigma-clip method as the NIRCam
surveys, but measured in the SAME domain the cutouts (and the noise aug) live in:
converted to MJy/sr AND resampled 40 mas -> 30 mas. Measuring on the native 40 mas
mosaic is wrong — bilinear resampling correlates neighbouring pixels and LOWERS the
per-pixel sigma (~x0.82 empirically), and the naive (40/30) area scaling predicts the
opposite. So each candidate blank patch is reprojected to 30 mas (the cutout transform)
before its sigma is taken.

Per field: mosaic ``<field>-f150wn-clear_drc_sci.fits`` (10nJy/pix) + shared IR segmap
``<field>-ir_seg.fits`` (seg==NUMBER). Tile alias = field name.

Run:
    .pixi/envs/default/bin/python images/OutThere/measure_sky_sigma.py
"""

import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from tqdm.auto import tqdm
import warnings
warnings.simplefilter("ignore")

HERE = os.path.dirname(__file__)
NJY10 = 1e-8        # one pixel unit (10 nJy) in Jy
OUT_PIX = 30.0      # measure in the 30 mas cutout domain
PATCH = 128         # native-mosaic patch (px @ 40 mas)
N_PATCHES = 120     # fewer than the NIRCam scripts — each patch costs a reproject
MAX_TRIES = 6000
MIN_COVERAGE = 0.95  # patch must be ~all clean background (finite & nonzero & seg==0)
MIN_BACKGROUND = 0.50
RNG_SEED = 0

# Excluded by hand (must match cutout_export_outthere.py) — bad data / out of scope:
#   alias 'dor' : dense Galactic star field (not extragalactic; crowded, no clean sky).
#   crt-00      : dominated by one huge saturated star (contaminated background).
_EXCLUDE_ALIASES = {"dor"}
_EXCLUDE_FIELDS = {"crt-00"}


def _excluded(field: str) -> bool:
    return field in _EXCLUDE_FIELDS or field.split("-")[0] in _EXCLUDE_ALIASES


def _conv_mjysr(wcs: WCS) -> float:
    pscale_rad = np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix))) * np.pi / 180.0
    return NJY10 / pscale_rad ** 2 / 1e6


def _field_sky_sigma(payload):
    """30mas-domain sky sigma for one OutThere field. Runs in a worker."""
    from reproject import reproject_interp

    field, mosaic_path, seg_path = payload
    hdr = fits.getheader(mosaic_path)
    wcs = WCS(hdr)
    in_pix = np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3.6e6  # mas/pix
    osize = int(round(PATCH * in_pix / OUT_PIX))
    data = np.asarray(fits.getdata(mosaic_path), dtype=np.float32) * _conv_mjysr(wcs)  # MJy/sr
    seg = np.asarray(fits.getdata(seg_path))
    ny, nx = data.shape

    rng = np.random.default_rng(RNG_SEED)
    sigmas = []
    tries = 0
    while len(sigmas) < N_PATCHES and tries < MAX_TRIES:
        tries += 1
        y = rng.integers(0, ny - PATCH)
        x = rng.integers(0, nx - PATCH)
        sub = data[y:y + PATCH, x:x + PATCH]
        sg = seg[y:y + PATCH, x:x + PATCH]
        # Same two gates as the NIRCam scripts: coverage (finite & nonzero) >= 95%,
        # then source-free background (cover & seg==0) >= 50%; measure on the bg pixels.
        cover = np.isfinite(sub) & (sub != 0)
        if cover.mean() < MIN_COVERAGE:
            continue
        bg = cover & (sg == 0)
        if bg.mean() < MIN_BACKGROUND:
            continue
        cut = Cutout2D(data, (x + PATCH / 2 - 0.5, y + PATCH / 2 - 0.5), size=PATCH, wcs=wcs)
        tw = cut.wcs.deepcopy()
        tw.wcs.cd = cut.wcs.wcs.cd * (OUT_PIX / in_pix)
        tw.wcs.crpix = [osize / 2 + 0.5, osize / 2 + 0.5]
        cen = cut.wcs.pixel_to_world((PATCH - 1) / 2, (PATCH - 1) / 2)
        tw.wcs.crval = [cen.ra.deg, cen.dec.deg]
        img30, _ = reproject_interp((np.nan_to_num(cut.data), cut.wcs), tw, shape_out=(osize, osize))
        msk30, _ = reproject_interp((bg.astype(np.float32), cut.wcs), tw, shape_out=(osize, osize))
        valid = (img30 != 0) & (msk30 > 0.5)
        if valid.mean() < MIN_BACKGROUND:
            continue
        _, _, std = sigma_clipped_stats(img30[valid], sigma=3.0, maxiters=5)
        if np.isfinite(std) and std > 0:
            sigmas.append(std)
    sig = float(np.median(sigmas)) if sigmas else float("nan")
    return field, sig, len(sigmas)


def _discover():
    out = []
    for mos in sorted(glob.glob(os.path.join(HERE, "imaging", "*", "*-f150wn-clear_drc_sci.fits"))):
        field = os.path.basename(os.path.dirname(mos))
        if _excluded(field):
            continue
        seg = os.path.join(HERE, "imaging", field, f"{field}-ir_seg.fits")
        if os.path.exists(seg):
            out.append((field, mos, seg))
    return out


def main() -> None:
    fields = _discover()
    print(f"# {len(fields)} OutThere F150W fields (sky sigma measured at {OUT_PIX:.0f} mas)\n")
    result = {}
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, len(fields))) as ex:
        futures = [ex.submit(_field_sky_sigma, pl) for pl in fields]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="fields"):
            field, sig, n = fut.result()
            result[field] = sig
            print(f"  {field:8s} sky_sigma = {sig:.4e} MJy/sr  (n={n})")

    result = dict(sorted(result.items()))
    vals = np.array([v for v in result.values() if np.isfinite(v)])
    print(f"\n# median = {np.median(vals):.4e}  range = [{vals.min():.3e}, {vals.max():.3e}] MJy/sr "
          f"({vals.max() / vals.min():.0f}x scatter)")
    print("\n# ---- paste into SKY_SIGMA['outthere'] in jwst_dino/data/augmentations.py ----")
    print('"outthere": {')
    for field, sig in result.items():
        print(f"    {field!r}: {sig:.4e},")
    print("},")


if __name__ == "__main__":
    main()
