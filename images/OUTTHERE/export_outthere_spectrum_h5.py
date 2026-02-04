#!/usr/bin/env python3

#python export_outthere_spectrum_h5.py --num-threads 3

"""OUTTHERE spectrum exporter (writes one HDF5 file per filter).

This script aggregates *all* available 1D spectra for each filter into
three files under images/OUTTHERE/spectrum:
- f115w.h5, f150w.h5, f200w.h5

HDF5 layout (per filter file):
- wave/flux/err/line/contam: float32, shape (N, L)
- spec_sn: float32, shape (N,) average S/N per spectrum
- Catalog basics: field (str), id (int), ra/dec/redshift (float)
- Optional extra catalog columns via --catalog-cols

Objects without spectra for a given filter are skipped entirely.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table
from tqdm.auto import tqdm

BASE_DIR = Path(__file__).resolve().parent
MASTER_PATH = BASE_DIR / "spectra-fitting.fits"
DATA_DIR = BASE_DIR / "data"
OUT_DIR_DEFAULT = BASE_DIR / "spectrum"

FILTERS: List[str] = ["F115W", "F150W"]#, "F200W"]
SPEC_COLS: Tuple[str, ...] = ("wave", "flux", "err", "line", "contam")


def _path(field: str, obj_id: int) -> Path:
    """Build OUTTHERE 1D spectrum file path."""
    return DATA_DIR / field / f"{field}_{obj_id:05d}.1D.fits"


def read_spec(one_d_fits: Path, filt: str) -> Optional[Dict[str, np.ndarray]]:
    """Read spectrum for a single filter. Returns None if not available."""
    if not one_d_fits.exists():
        return None
    with fits.open(one_d_fits, memmap=False) as hdul:
        if filt not in hdul or hdul[filt].data is None:
            return None
        tab = Table(hdul[filt].data)
        if not all(c in tab.colnames for c in SPEC_COLS):
            return None
        return {c: np.asarray(tab[c], dtype=np.float32) for c in SPEC_COLS}


def average_snr(flux: np.ndarray, err: np.ndarray) -> np.float32:
    """Compute average spectral S/N."""
    m = np.isfinite(flux) & np.isfinite(err) & (err != 0)
    return np.float32(np.mean(np.abs(flux[m] / err[m]))) if np.any(m) else np.float32(np.nan)


def collect_spectra(master_cat: Table, limit: int) -> Dict[str, List[Tuple[Dict[str, np.ndarray], object]]]:
    """Collect spectra for all filters in one pass.
    
    Args:
        master_cat: Master catalog table
        limit: -1 for all objects, otherwise stop when ALL filters have >= limit spectra
    
    Returns:
        Dict mapping filter -> list of (spectrum_dict, catalog_row)
    """
    result = {f: [] for f in FILTERS}
    
    for obj in tqdm(master_cat, desc="collecting"):
        p = _path(str(obj["field"]), int(obj["id"]))
        
        for filt in FILTERS:
            spec = read_spec(p, filt)
            if spec is not None:
                result[filt].append((spec, obj))
        
        # Check if we've reached limit for ALL filters
        if limit > 0 and all(len(result[f]) >= limit for f in FILTERS):
            break
    
    # Truncate to limit if specified
    if limit > 0:
        for f in FILTERS:
            result[f] = result[f][:limit]
    
    return result


def export_filter(
    spectra_data: List[Tuple[Dict[str, np.ndarray], object]],
    out_dir: Path,
    catalog_cols: List[str],
    filt: str,
    master_cat: Table,
) -> int:
    """Export spectra for a single filter into one HDF5 file."""
    if not spectra_data:
        print(f"[warn] {filt}: no spectra found")
        return 0
    
    spectra = [s for s, _ in spectra_data]
    catalog_rows = [r for _, r in spectra_data]
    
    N = len(spectra)
    L = spectra[0]["wave"].shape[0]
    data = {c: np.stack([s[c] for s in spectra]) for c in SPEC_COLS}
    spec_sn = np.array([average_snr(s["flux"], s["err"]) for s in spectra], dtype=np.float32)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_dir / f"{filt.lower()}.h5", "w") as h5:
        h5.attrs["filter"] = filt
        h5.attrs["spec_len"] = L
        
        for c in SPEC_COLS:
            h5.create_dataset(c, data=data[c], compression="gzip", compression_opts=4)
        h5.create_dataset("spec_sn", data=spec_sn)
        
        for col in catalog_cols:
            if col not in master_cat.colnames:
                continue
            dt = master_cat[col].dtype
            if dt.kind in {"U", "S", "O"}:
                vals = [str(row[col]) if not getattr(row[col], "mask", False) else "" for row in catalog_rows]
                h5.create_dataset(col, data=vals, dtype=h5py.string_dtype("utf-8"))
            elif dt.kind in {"i", "u"}:
                vals = np.array([int(row[col]) if not getattr(row[col], "mask", False) else -1 for row in catalog_rows], dtype=np.int32)
                h5.create_dataset(col, data=vals)
            else:
                vals = np.array([float(row[col]) if not getattr(row[col], "mask", False) else np.nan for row in catalog_rows], dtype=np.float32)
                h5.create_dataset(col, data=vals)
    
    return N


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    p.add_argument("--catalog-cols", nargs="*", default=["field", "id", "ra", "dec", "redshift"])
    p.add_argument("--limit", type=int, default=-1, help="Max spectra per filter (-1 for all)")
    p.add_argument("--num-threads", type=int, default=1)
    args = p.parse_args(argv)

    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing {MASTER_PATH}")

    master_cat = Table.read(str(MASTER_PATH))
    out_dir = Path(args.out_dir)
    
    # Collect all spectra in one pass
    all_spectra = collect_spectra(master_cat, args.limit)

    if args.num_threads > 1:
        with ThreadPoolExecutor(max_workers=min(args.num_threads, len(FILTERS))) as ex:
            futures = {
                ex.submit(export_filter, all_spectra[f], out_dir, args.catalog_cols, f, master_cat): f
                for f in FILTERS
            }
            for fut in as_completed(futures):
                f = futures[fut]
                print(f"[ok] {f}: {fut.result()} spectra -> {out_dir / f'{f.lower()}.h5'}")
    else:
        for f in FILTERS:
            count = export_filter(all_spectra[f], out_dir, args.catalog_cols, f, master_cat)
            print(f"[ok] {f}: {count} spectra -> {out_dir / f'{f.lower()}.h5'}")


if __name__ == "__main__":
    main()