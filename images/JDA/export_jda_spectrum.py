#!/usr/bin/env python3
"""
JDA spectrum exporter.

Reads a CSV (or masked CSV) built from jda_cat.ipynb, loads .spec.fits files
from the download directory, and writes a single HDF5 file containing:

    wave, flux, sn50, z_best, srcid, objid, ra, dec, phot_f150w_tot_1

HDF5 layout
-----------
    wave             float32  (N, L)  wavelength in microns
    flux             float32  (N, L)  flux in uJy
    sn50             float32  (N,)    median S/N from catalog
    z_best           float32  (N,)    best-fit redshift
    srcid            int64    (N,)    source ID
    objid            int64    (N,)    object ID
    ra               float32  (N,)    right ascension (deg)
    dec              float32  (N,)    declination (deg)
    phot_f150w_tot_1 float32  (N,)    F150W total photometry

Spectra with different wavelength array lengths are padded with NaN to the
maximum length so that wave and flux are stored as rectangular (N, L) arrays.
Corrupted or missing FITS files are silently skipped; final statistics are
printed at the end.

Usage
-----
    # single-threaded
    python export_jda_spectrum.py --csv cat_selected.csv --out jda_spectra.h5

    # multi-threaded (4 workers)
    python export_jda_spectrum.py --csv cat_selected.csv --out jda_spectra.h5 --num-workers 4

    # quick test on first 50 rows
    python export_jda_spectrum.py --csv cat_selected.csv --out test.h5 --limit 50
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from tqdm.auto import tqdm


# Catalog columns pulled from each CSV row (order preserved in HDF5)
CATALOG_COLS: List[str] = [
    "sn50",
    "z_best",
    "srcid",
    "objid",
    "ra",
    "dec",
    "phot_f150w_tot_1",
    "grade"
]

# Columns stored as int64; everything else is float32
_INT_COLS = {"srcid", "objid"}


# ---------------------------------------------------------------------------
# Statistics container
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    n_ok:        int = 0
    n_missing:   int = 0  # file not in download dir
    n_corrupted: int = 0  # file exists but unreadable / unexpected format

    def __iadd__(self, other: "Stats") -> "Stats":
        self.n_ok        += other.n_ok
        self.n_missing   += other.n_missing
        self.n_corrupted += other.n_corrupted
        return self

    def print(self, total: int) -> None:
        skipped = self.n_missing + self.n_corrupted
        print(
            f"[stats] total rows : {total}\n"
            f"        ok         : {self.n_ok}\n"
            f"        missing    : {self.n_missing}\n"
            f"        corrupted  : {self.n_corrupted}\n"
            f"        skipped    : {skipped}  "
            f"({100 * skipped / max(total, 1):.1f} %)"
        )


# ---------------------------------------------------------------------------
# Core per-row helpers
# ---------------------------------------------------------------------------

def read_spectrum(fits_path: Path) -> Tuple[Optional[Dict[str, np.ndarray]], str]:
    """Read wave and flux arrays from a .spec.fits file (HDU[1] = SPEC1D).

    Returns
    -------
    (spec_dict, status) where status is one of "ok", "missing", "corrupted".
    spec_dict is None when status != "ok".
    """
    if not fits_path.exists():
        return None, "missing"
    try:
        with fits.open(fits_path, memmap=False) as hdul:
            tab = Table.read(hdul[1])
            wave = np.asarray(tab["wave"], dtype=np.float32)
            flux = np.asarray(tab["flux"], dtype=np.float32)
            flux[~np.isfinite(flux)] = 0.0  # NaN/Inf → 0; existing zeros stay 0
            mask = (wave > 1) & (wave < 2)
            #if more than half of the flux in the wave range is invalid, consider the spectrum corrupted
            valid_count = np.sum(np.isfinite(flux[mask]))
            total_count = len(flux[mask])
            if total_count == 0 or valid_count < total_count / 2:
                return None, "corrupted"
            return {
                "wave": wave[mask],
                "flux": flux[mask],
            }, "ok"
    except Exception:
        return None, "corrupted"


def process_row(row: pd.Series, download_dir: Path) -> Tuple[Optional[Dict], str]:
    """Load the spectrum and catalog fields for a single CSV row.

    Parameters
    ----------
    row:          One row from the catalog DataFrame (pandas Series).
    download_dir: Directory containing downloaded .spec.fits files.

    Returns
    -------
    (result_dict, status).  result_dict is None when status != "ok".
    """
    spec, status = read_spectrum(download_dir / row["file"])
    if spec is None:
        return None, status

    result: Dict = {"wave": spec["wave"], "flux": spec["flux"]}
    for col in CATALOG_COLS:
        result[col] = row.get(col, np.nan)
    return result, "ok"


# ---------------------------------------------------------------------------
# Slice-level processing (used directly and by the multithreaded runner)
# ---------------------------------------------------------------------------

def process_csv_slice(
    df_slice: pd.DataFrame,
    download_dir: Path,
    progress: Optional[tqdm] = None,
) -> Tuple[List[Dict], Stats]:
    """Process a contiguous slice of the catalog DataFrame.

    Parameters
    ----------
    df_slice:     Sub-DataFrame to process (any subset of the full catalog).
    download_dir: Directory containing downloaded .spec.fits files.
    progress:     Optional shared tqdm bar; ``update(1)`` is called per row
                  and is thread-safe.

    Returns
    -------
    (results, stats) — results is a list of successfully loaded spectrum
    dicts; stats holds per-category skip counts for this slice.
    """
    results: List[Dict] = []
    stats = Stats()

    for _, row in df_slice.iterrows():
        res, status = process_row(row, download_dir)
        if status == "ok":
            results.append(res)
            stats.n_ok += 1
        elif status == "missing":
            stats.n_missing += 1
        else:
            stats.n_corrupted += 1

        if progress is not None:
            progress.update(1)

    return results, stats


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def export_to_h5(results: List[Dict], output_path: Path) -> None:
    """Write collected spectra and catalog metadata to a single HDF5 file.

    Parameters
    ----------
    results:     List of dicts as returned by ``process_row`` / ``process_csv_slice``.
    output_path: Destination .h5 file (parent directories are created if needed).
    """
    if not results:
        print("[warn] No valid spectra to write.")
        return

    N = len(results)
    max_len = max(r["wave"].shape[0] for r in results)

    # Build rectangular spectral arrays, padding shorter spectra with NaN
    wave_arr = np.full((N, max_len), np.nan, dtype=np.float32)
    flux_arr = np.full((N, max_len), np.nan, dtype=np.float32)
    for i, r in enumerate(results):
        L = r["wave"].shape[0]
        wave_arr[i, :L] = r["wave"]
        flux_arr[i, :L] = r["flux"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("wave", data=wave_arr, compression="gzip", compression_opts=4)
        h5.create_dataset("flux", data=flux_arr, compression="gzip", compression_opts=4)

        for col in CATALOG_COLS:
            vals = [r[col] for r in results]
            if col in _INT_COLS:
                arr = np.array(
                    [int(v) if not (isinstance(v, float) and np.isnan(v)) else -1
                     for v in vals],
                    dtype=np.int64,
                )
            else:
                arr = np.array(
                    [float(v) if not pd.isna(v) else np.nan for v in vals],
                    dtype=np.float32,
                )
            h5.create_dataset(col, data=arr)

    print(f"[ok] wrote {N} spectra -> {output_path}")


# ---------------------------------------------------------------------------
# Multithreaded runner
# ---------------------------------------------------------------------------

def run_multithreaded(
    df: pd.DataFrame,
    download_dir: Path,
    output_path: Path,
    num_workers: int,
) -> None:
    """Evenly slice *df* across *num_workers* threads, collect results, write h5.

    The DataFrame is split into ``num_workers`` roughly equal chunks with
    ``np.array_split``.  Each chunk is submitted to a ``ThreadPoolExecutor``
    as a separate ``process_csv_slice`` call.  A single shared tqdm bar tracks
    overall row-level progress.  Results and stats are merged in original
    slice order before being written to disk.

    Parameters
    ----------
    df:           Full (or pre-filtered) catalog DataFrame.
    download_dir: Directory containing downloaded .spec.fits files.
    output_path:  Destination .h5 file.
    num_workers:  Number of parallel reader threads.
    """
    slices = np.array_split(df, num_workers)

    # Pre-allocate ordered lists so merge preserves catalog order
    ordered_results: List[Optional[List[Dict]]] = [None] * num_workers
    ordered_stats:   List[Optional[Stats]]       = [None] * num_workers

    with tqdm(total=len(df), desc="processing") as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_csv_slice, sl, download_dir, pbar): i
                for i, sl in enumerate(slices)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                results, stats = fut.result()
                ordered_results[idx] = results
                ordered_stats[idx]   = stats

    merged = [r for chunk in ordered_results if chunk for r in chunk]

    total_stats = Stats()
    for s in ordered_stats:
        if s is not None:
            total_stats += s
    total_stats.print(len(df))

    export_to_h5(merged, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Export JDA spectra to a single HDF5 file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv", required=True,
        help="Input CSV (e.g. cat_selected.csv or test_sample.csv)",
    )
    p.add_argument(
        "--download-dir", default="download",
        help="Directory containing downloaded .spec.fits files",
    )
    p.add_argument(
        "--out", default="jda_spectra.h5",
        help="Output HDF5 file path",
    )
    p.add_argument(
        "--num-workers", type=int, default=1,
        help="Number of parallel reader threads",
    )
    p.add_argument(
        "--limit", type=int, default=-1,
        help="Process only the first N rows (-1 for all)",
    )
    args = p.parse_args(argv)

    csv_path     = Path(args.csv)
    download_dir = Path(args.download_dir)
    output_path  = Path(args.out)

    df = pd.read_csv(csv_path, low_memory=False)
    if args.limit > 0:
        df = df.head(args.limit)

    print(f"[info] {len(df)} rows  from {csv_path}")
    print(f"[info] spectra dir : {download_dir.resolve()}")
    print(f"[info] output      : {output_path.resolve()}")

    if args.num_workers > 1:
        run_multithreaded(df, download_dir, output_path, args.num_workers)
    else:
        with tqdm(total=len(df), desc="processing") as pbar:
            results, stats = process_csv_slice(df, download_dir, pbar)
        stats.print(len(df))
        export_to_h5(results, output_path)


if __name__ == "__main__":
    main()
