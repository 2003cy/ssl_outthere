#!/usr/bin/env python3
"""
Build jda_spectra_low_image.h5

For each entry in jda_spectra_low.h5:
  1. Find the grizli F150W mosaic via jda_low_res.csv (srcid → file_phot)
  2. Cut 192x192 patch @ 20mas native scale (= 128*1.5 to cover 128px @ 30mas)
  3. Skip if fraction of zero pixels > 30%
  4. Reproject to 128x128 @ 30mas with reproject_exact
     (no unit conversion — grizli mosaics are already in MJy/sr, same as COSMOS-Web)
  6. Write image + ALL spectrum columns from jda_spectra_low.h5 to new h5

Multithreading strategy: one thread per mosaic file (group objects by mosaic,
open the FITS once per thread, cut all objects from it sequentially).

Usage
-----
    python build_jda_image_h5.py                          # defaults
    python build_jda_image_h5.py --num-workers 16 --out jda_spectra_low_image.h5
"""
from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import astropy.units as u
from reproject import reproject_exact
from tqdm.auto import tqdm

# reproject_exact warns about sub-0.05" pixel scales; safe to suppress for
# a simple 1.5x downsampling (20→30mas) on a standard TAN projection.
warnings.filterwarnings("ignore", message=".*reproject_exact.*precision.*", category=UserWarning)


# ─── Constants ────────────────────────────────────────────────────────────────

NATIVE_MAS    = 20      # grizli mosaic native pixel scale (mas/pixel)
TARGET_MAS    = 30      # output pixel scale (mas/pixel)
OUTPUT_SIZE   = 128     # output image side length in pixels at TARGET_MAS
MAX_ZERO_FRAC = 0.20    # reject cutout if fraction of 0 pixels exceeds this

_scale       = TARGET_MAS / NATIVE_MAS                           # 1.5
CUTOUT_NAT   = int(OUTPUT_SIZE * _scale)                         # 192 px @ 20mas

# The grizli JDA mosaics are already in MJy/sr (JWST pipeline standard),
# identical to COSMOS-Web. No unit conversion is needed.

# All columns in jda_spectra_low.h5 (copied verbatim, filtered to valid rows)
SPEC_COLS = ["wave", "flux", "sn50", "z_best", "srcid", "objid",
             "ra", "dec", "phot_f150w_tot_1", "grade"]
_INT_COLS = {"srcid", "objid"}


# ─── Per-mosaic worker ────────────────────────────────────────────────────────

def process_mosaic_group(
    mosaic_path: Path,
    group: List[Dict],
    progress: tqdm,
) -> List[Dict]:
    """
    Open *mosaic_path* once, produce 128x128 images for every object in *group*.

    Parameters
    ----------
    mosaic_path : path to the grizli F150W FITS file
    group       : list of dicts  {h5_idx, ra, dec}
    progress    : shared tqdm bar (updated once per object regardless of outcome)

    Returns
    -------
    List of {h5_idx, image} dicts for objects that pass the zero-fraction QC.
    """
    results: List[Dict] = []
    try:
        with fits.open(mosaic_path, memmap=True) as hdul:
            hdu  = hdul[0]
            wcs  = WCS(hdu.header)
            data = hdu.data  # nJy/pixel

            for obj in group:
                try:
                    h5_idx = obj["h5_idx"]
                    ra     = float(obj["ra"])
                    dec    = float(obj["dec"])
                    coord  = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

                    # ── 20mas cutout (MJy/sr) ────────────────────────────────
                    cutout = Cutout2D(
                        data, coord,
                        size=(CUTOUT_NAT, CUTOUT_NAT),
                        wcs=wcs, mode="partial", fill_value=0.0,
                    )
                    cutout_20mas = cutout.data
                    cutout_wcs   = cutout.wcs

                    # ── QC: fraction of zero pixels ───────────────────────────
                    zero_frac = float(np.sum(cutout_20mas == 0)) / cutout_20mas.size
                    if zero_frac > MAX_ZERO_FRAC:
                        continue  # drop this object

                    # ── Output WCS: 30mas pixel scale, source centred ─────────
                    out_wcs = cutout_wcs.deepcopy()
                    if out_wcs.wcs.has_cd():
                        out_wcs.wcs.cd = out_wcs.wcs.cd * _scale
                    else:
                        out_wcs.wcs.cdelt = out_wcs.wcs.cdelt * _scale
                    out_wcs.wcs.crpix = np.array(
                        [OUTPUT_SIZE / 2 + 0.5, OUTPUT_SIZE / 2 + 0.5]
                    )
                    out_wcs.wcs.crval = np.array([ra, dec])
                    out_wcs.wcs.set()

                    # ── Reproject to 128×128 @ 30mas ─────────────────────────
                    img_30mas, _ = reproject_exact(
                        (cutout_20mas, cutout_wcs),
                        out_wcs,
                        shape_out=(OUTPUT_SIZE, OUTPUT_SIZE),
                    )
                    # NaN at edges (partial footprint) → 0.0
                    img_30mas = np.nan_to_num(img_30mas, nan=0.0).astype(np.float32)

                    results.append({"h5_idx": h5_idx, "image": img_30mas})

                except Exception:
                    pass  # corrupted / out-of-bounds: silently skip
                finally:
                    progress.update(1)

    except Exception:
        # FITS file unreadable: count all objects in group as skipped
        for _ in group:
            progress.update(1)

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Build jda_spectra_low_image.h5 with 128x128 F150W cutouts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",          default="jda_low_res.csv",
                   help="JDA catalog CSV (must contain srcid + file_phot columns)")
    p.add_argument("--h5-in",        default="jda_spectra_low.h5",
                   help="Input spectrum HDF5 (jda_spectra_low.h5)")
    p.add_argument("--mosaic-dir",   default="download_mosaic",
                   help="Directory containing grizli F150W mosaic FITS files")
    p.add_argument("--out",          default="jda_spectra_low_image.h5",
                   help="Output HDF5 file")
    p.add_argument("--num-workers",  type=int, default=8,
                   help="Number of parallel worker threads (one per mosaic)")
    args = p.parse_args(argv)

    csv_path    = Path(args.csv)
    h5_in_path  = Path(args.h5_in)
    mosaic_dir  = Path(args.mosaic_dir)
    output_path = Path(args.out)

    print(f"[info] cutout size  : {CUTOUT_NAT}x{CUTOUT_NAT} px @ {NATIVE_MAS}mas "
          f"→ {OUTPUT_SIZE}x{OUTPUT_SIZE} px @ {TARGET_MAS}mas")
    print(f"[info] zero-frac QC : reject if > {MAX_ZERO_FRAC*100:.0f}% zero pixels")
    print(f"[info] unit         : MJy/sr (grizli mosaics, no conversion applied)")

    # ── Build srcid → mosaic filename mapping from CSV ────────────────────────
    print(f"\n[info] loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    srcid_to_mosaic: Dict[int, str] = {}
    for _, row in df.iterrows():
        fp = row.get("file_phot")
        if pd.isna(fp):
            continue
        try:
            sid = int(row["srcid"])
        except (ValueError, TypeError):
            continue
        if sid not in srcid_to_mosaic:
            mosaic_name = str(fp).replace("-fix_phot", "-f150w-clear_drc_sci")
            srcid_to_mosaic[sid] = mosaic_name
    print(f"[info] {len(srcid_to_mosaic)} srcid → mosaic mappings")

    # ── Load all spectrum data from h5 ────────────────────────────────────────
    print(f"[info] loading spectra: {h5_in_path}")
    with h5py.File(h5_in_path, "r") as h5:
        N = h5["ra"].shape[0]
        spec_data: Dict[str, np.ndarray] = {col: h5[col][:] for col in SPEC_COLS}
    print(f"[info] {N} spectra in input h5")

    # ── Group objects by mosaic file ──────────────────────────────────────────
    mosaic_groups: Dict[str, List[Dict]] = {}
    n_no_mosaic = 0
    for i in range(N):
        try:
            sid = int(spec_data["srcid"][i])
        except (ValueError, TypeError):
            n_no_mosaic += 1
            continue
        mosaic_name = srcid_to_mosaic.get(sid)
        if mosaic_name is None:
            n_no_mosaic += 1
            continue
        mosaic_path = mosaic_dir / mosaic_name
        if not mosaic_path.exists():
            n_no_mosaic += 1
            continue
        key = str(mosaic_path)
        mosaic_groups.setdefault(key, []).append({
            "h5_idx": i,
            "ra":  float(spec_data["ra"][i]),
            "dec": float(spec_data["dec"][i]),
        })

    n_with_mosaic = N - n_no_mosaic
    print(f"[info] {n_with_mosaic} objects matched to a local mosaic "
          f"({len(mosaic_groups)} unique mosaics)")
    print(f"[info] {n_no_mosaic} objects have no local mosaic (will be skipped)")

    if n_with_mosaic == 0:
        print("[warn] no objects can be processed — nothing to write.")
        return

    # ── Process mosaics in parallel ───────────────────────────────────────────
    image_results: List[Dict] = []
    with tqdm(total=n_with_mosaic, desc="cutting & reprojecting") as pbar:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_mosaic_group, Path(mp), grp, pbar): mp
                for mp, grp in mosaic_groups.items()
            }
            for fut in as_completed(futures):
                image_results.extend(fut.result())

    # ── Sort results by original h5 index (preserves catalog order) ──────────
    image_results.sort(key=lambda r: r["h5_idx"])
    valid_idxs = [r["h5_idx"] for r in image_results]
    images     = np.stack([r["image"] for r in image_results], axis=0)

    M = len(valid_idxs)
    n_qc_dropped = n_with_mosaic - M
    print(f"\n[info] {M} objects passed zero-fraction QC")
    print(f"[info] {n_qc_dropped} objects dropped (>{MAX_ZERO_FRAC*100:.0f}% zero pixels)")

    if M == 0:
        print("[warn] no valid images — nothing to write.")
        return

    # ── Write output h5 ───────────────────────────────────────────────────────
    print(f"[info] writing → {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5out:
        # Store metadata as attributes
        h5out.attrs["native_mas"]     = NATIVE_MAS
        h5out.attrs["target_mas"]     = TARGET_MAS
        h5out.attrs["output_size_px"] = OUTPUT_SIZE
        h5out.attrs["max_zero_frac"]  = MAX_ZERO_FRAC
        h5out.attrs["unit_image"]     = "MJy/sr"
        h5out.attrs["unit_wave"]      = "micron"
        h5out.attrs["unit_flux"]      = "uJy"

        # Image dataset: (M, 128, 128)  float32
        h5out.create_dataset(
            "image", data=images,
            compression="gzip", compression_opts=4,
            chunks=(1, OUTPUT_SIZE, OUTPUT_SIZE),
        )
        print(f"       image  : {images.shape}  {images.dtype}")

        # All spectrum columns, filtered to valid indices
        for col in SPEC_COLS:
            arr = spec_data[col][valid_idxs]
            h5out.create_dataset(
                col, data=arr,
                compression="gzip", compression_opts=4,
            )
            print(f"       {col:20s}: {arr.shape}  {arr.dtype}")

    print(f"\n[ok] done — {M} entries written to {output_path}")


if __name__ == "__main__":
    main()
