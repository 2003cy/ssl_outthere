#!/usr/bin/env python3

#python export_outthere_h5.py --out-dir outputs/h5_all --num-threads 8

"""OUTTHERE per-field exporter (writes one HDF5 file per field).

Conventions (aligned with images/OUTTHERE/check_image_and_spectra.ipynb):
- Catalog: spectra-fitting.fits
- 1D spectra: data/{field}/{field}_{id:05d}.1D.fits (HDUs: F115W/F150W/F200W)
- Images: data/{field}/{field}_{id:05d}.full.fits
    - Science images are selected by EXTNAME='DSCI' and a header FILTER that starts
        with F115W/F150W/F200W (prefix match).
    - Segmentation map is stored in hdul[4].data.

HDF5 layout (one file per field):
- f115w/f150w/f200w: float32, shape (N, 128, 128) centered cutouts
- seg: int32, shape (N, 128, 128) centered cutout from full.fits hdul[4]
- redshift: float32, shape (N,) copied from the catalog (spectra-fitting.fits)
- f115w_wave/flux/err/line/contam etc: float32, shape (N, L_filter)
- f115w_spec_sn / f150w_spec_sn / f200w_spec_sn: float32, shape (N,) average S/N
- *_img_mask / *_spec_mask: uint8 (0/1) availability flags

Missing image/spectrum values are filled with -99. Segmentation padding uses 0.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table
from tqdm.auto import tqdm



BASE_DIR = Path(__file__).resolve().parent
MASTER_PATH = BASE_DIR / "spectra-fitting.fits"
DATA_DIR = BASE_DIR / "data"

FILTERS: List[str] = ["F115W", "F150W", "F200W"]
SPEC_COLS: Tuple[str, ...] = ("wave", "flux", "err", "line", "contam")

IMAGE_EXTNAME = "DSCI"
CUTOUT_SIZE = 128
FILL_VALUE = np.float32(-99.0)
MAX_SCAN = 2000


def _u(x: object) -> str:
    """Uppercase + strip helper for robust FITS header comparisons."""
    return "" if x is None else str(x).strip().upper()


def _paths(field: str, obj_id: int) -> Tuple[Path, Path]:
    """Build OUTTHERE file paths for a given (field, id)."""
    base = f"{field}_{str(int(obj_id)).zfill(5)}"
    p_full = DATA_DIR / field / f"{base}.full.fits"
    p_1d = DATA_DIR / field / f"{base}.1D.fits"
    return p_full, p_1d


def center_cutout(
    img: np.ndarray,
    *,
    size: int = CUTOUT_SIZE,
    dtype: np.dtype = np.float32,
    fill: object = FILL_VALUE,
) -> np.ndarray:
    """Return a centered cutout of shape (size, size).

    If the input image is smaller than (size, size), the output is padded with
    `fill`. The output dtype is controlled by `dtype`.

    Typical usage:
    - DSCI images: dtype=float32, fill=-99
    - Segmentation maps: dtype=int32, fill=0
    """
    H, W = int(img.shape[0]), int(img.shape[1])
    out = np.full((size, size), fill, dtype=dtype)

    y0 = max((H - size) // 2, 0)
    x0 = max((W - size) // 2, 0)
    y1 = min(y0 + size, H)
    x1 = min(x0 + size, W)

    src = np.asarray(img[y0:y1, x0:x1], dtype=dtype)
    dy0 = (size - src.shape[0]) // 2
    dx0 = (size - src.shape[1]) // 2
    out[dy0 : dy0 + src.shape[0], dx0 : dx0 + src.shape[1]] = src
    return out


def read_image(full_fits: Path, filt: str) -> Optional[np.ndarray]:
    """Read a DSCI science image for the given filter from a .full.fits file.

    The correct image extension is identified by:
    - EXTNAME == IMAGE_EXTNAME (typically 'DSCI')
    - FILTER header starts with the filter prefix (e.g., 'F150W')
    """
    if not full_fits.exists():
        return None
    want_prefix = _u(filt)
    with fits.open(full_fits, memmap=False) as hdul:
        # Iterate over all HDUs because the DSCI/filter image is not guaranteed
        # to live at a fixed index.
        for hdu in hdul:
            if getattr(hdu, "data", None) is None:
                continue
            hdr = getattr(hdu, "header", None)
            if hdr is None:
                continue
            if _u(hdr.get("EXTNAME")) != IMAGE_EXTNAME:
                continue
            if _u(hdr.get("FILTER")).startswith(want_prefix):
                return np.asarray(hdu.data, dtype=np.float32)
    return None


def read_segmentation(full_fits: Path) -> Optional[np.ndarray]:
    """Read the segmentation map from a .full.fits file.

    Convention in this dataset: hdul[4].data stores the segmentation map.
    Returns None if the file does not exist or the expected HDU is missing.
    """
    if not full_fits.exists():
        return None
    with fits.open(full_fits, memmap=False) as hdul:
        if len(hdul) <= 4:
            return None
        seg = hdul[4].data
        if seg is None:
            return None
        return np.asarray(seg)


def read_spec(one_d_fits: Path, filt: str) -> Optional[Dict[str, np.ndarray]]:
    """Read 1D spectral arrays from a .1D.fits file for a given filter.

    Expects an HDU named exactly like the filter (e.g., 'F150W') and columns:
    wave/flux/err/line/contam.
    """
    if not one_d_fits.exists():
        return None
    with fits.open(one_d_fits, memmap=False) as hdul:
        if filt not in hdul:
            return None
        hdu = hdul[filt]
        if hdu.data is None:
            return None
        tab = Table(hdu.data)
        out: Dict[str, np.ndarray] = {}
        for c in SPEC_COLS:
            out[c] = np.asarray(tab[c], dtype=np.float32)
        return out


def average_snr(flux: np.ndarray, err: np.ndarray, *, fill: float = float(FILL_VALUE)) -> np.float32:
    """Compute average spectral S/N as mean(|flux/err|) over valid points.

    The user-requested rule is to include points where flux != 0. In practice we
    also exclude fill values, non-finite values, and err==0 to avoid inf/NaN.
    Returns `fill` if no valid points exist.
    """
    flux = np.asarray(flux, dtype=np.float32)
    err = np.asarray(err, dtype=np.float32)

    m = (
        np.isfinite(flux)
        & np.isfinite(err)
        & (flux != 0)
        & (flux != fill)
        & (err != 0)
        & (err != fill)
    )
    if not np.any(m):
        return np.float32(fill)
    sn = np.abs(flux[m] / err[m])
    if sn.size == 0:
        return np.float32(fill)
    return np.float32(np.mean(sn))


def infer_spec_len(rows: Table) -> Dict[str, int]:
    """Infer per-filter spectrum length (L) by scanning up to MAX_SCAN rows.

    Assumption (from the dataset convention): within a given field, each filter
    uses a fixed-length wavelength grid, so L is constant per filter.
    """
    spec_len: Dict[str, int] = {}
    n = min(len(rows), MAX_SCAN)
    for obj in rows[:n]:
        field = str(obj["field"])
        obj_id = int(obj["id"])
        _, p_1d = _paths(field, obj_id)
        for f in FILTERS:
            if f in spec_len:
                continue
            spec = read_spec(p_1d, f)
            if spec is None:
                continue
            spec_len[f] = int(spec["wave"].shape[0])
        if len(spec_len) == len(FILTERS):
            break
    missing = [f for f in FILTERS if f not in spec_len]
    if missing:
        raise RuntimeError(f"Could not infer spectrum length for filters: {missing}")
    return spec_len


def export_field(master_cat: Table, field: str, out_path: Path, limit: Optional[int]) -> None:
    """Export one field into a single HDF5 file."""
    rows = master_cat[master_cat["field"] == field]
    if len(rows) == 0:
        print(f"[skip] {field}: no rows")
        return
    if limit is not None:
        rows = rows[:limit]

    n = len(rows)
    spec_len = infer_spec_len(master_cat[master_cat["field"] == field])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        dt_str = h5py.string_dtype("utf-8")

        # File-level metadata for downstream consumers.
        h5.attrs["field"] = str(field)
        h5.attrs["filters"] = np.array(FILTERS, dtype=dt_str)
        h5.attrs["fill_value"] = float(FILL_VALUE)
        h5.attrs["cutout_size"] = int(CUTOUT_SIZE)
        h5.attrs["image_extname"] = str(IMAGE_EXTNAME)

        h5.create_dataset("field", shape=(n,), dtype=dt_str)
        h5.create_dataset("id", shape=(n,), dtype=np.int32)
        h5.create_dataset(
            "redshift",
            shape=(n,),
            dtype=np.float32,
            chunks=True,
            compression="gzip",
            compression_opts=4,
            fillvalue=FILL_VALUE,
        )

        h5.create_dataset(
            "seg",
            shape=(n, CUTOUT_SIZE, CUTOUT_SIZE),
            dtype=np.int32,
            chunks=(1, CUTOUT_SIZE, CUTOUT_SIZE),
            compression="gzip",
            compression_opts=4,
            fillvalue=0,
        )

        for f in FILTERS:
            # Image dataset per filter.
            h5.create_dataset(
                f.lower(),
                shape=(n, CUTOUT_SIZE, CUTOUT_SIZE),
                dtype=np.float32,
                chunks=(1, CUTOUT_SIZE, CUTOUT_SIZE),
                compression="gzip",
                compression_opts=4,
                fillvalue=FILL_VALUE,
            )
            h5.create_dataset(f"{f.lower()}_img_mask", shape=(n,), dtype=np.uint8, chunks=True, fillvalue=0)

            L = spec_len[f]
            for c in SPEC_COLS:
                # Spectral arrays per filter/column.
                h5.create_dataset(
                    f"{f.lower()}_{c}",
                    shape=(n, L),
                    dtype=np.float32,
                    chunks=(min(n, 256), L),
                    compression="gzip",
                    compression_opts=4,
                    fillvalue=FILL_VALUE,
                )
            h5.create_dataset(f"{f.lower()}_spec_mask", shape=(n,), dtype=np.uint8, chunks=True, fillvalue=0)

            # Per-object scalar feature: average S/N for the spectrum.
            h5.create_dataset(
                f"{f.lower()}_spec_sn",
                shape=(n,),
                dtype=np.float32,
                chunks=True,
                compression="gzip",
                compression_opts=4,
                fillvalue=FILL_VALUE,
            )

        for i, obj in enumerate(tqdm(rows, desc=f"export {field}", total=n)):
            obj_id = int(obj["id"])
            p_full, p_1d = _paths(field, obj_id)
            h5["field"][i] = str(field)
            h5["id"][i] = obj_id

            # Catalog scalar (optional if the column is missing or value is masked).
            z = FILL_VALUE
            if "redshift" in obj.colnames:
                try:
                    zv = obj["redshift"]
                    # Handle masked values from astropy Tables.
                    if getattr(zv, "mask", False):
                        z = FILL_VALUE
                    else:
                        zf = float(zv)
                        z = np.float32(zf) if np.isfinite(zf) else FILL_VALUE
                except Exception:
                    z = FILL_VALUE
            h5["redshift"][i] = z

            # Segmentation map (optional).
            seg = read_segmentation(p_full)
            if seg is not None:
                h5["seg"][i] = center_cutout(seg, dtype=np.int32, fill=0)

            for f in FILTERS:
                # Science image (optional).
                im = read_image(p_full, f)
                if im is not None:
                    h5[f.lower()][i] = center_cutout(im)
                    h5[f"{f.lower()}_img_mask"][i] = 1

                # 1D spectra (optional).
                spec = read_spec(p_1d, f)
                if spec is not None:
                    L = int(h5[f"{f.lower()}_wave"].shape[1])
                    if int(spec["wave"].shape[0]) != L:
                        raise RuntimeError(f"{p_1d}: spectrum length mismatch for {f}: {spec['wave'].shape[0]} != {L}")
                    for c in SPEC_COLS:
                        h5[f"{f.lower()}_{c}"][i] = spec[c]
                    h5[f"{f.lower()}_spec_sn"][i] = average_snr(spec["flux"], spec["err"], fill=float(FILL_VALUE))
                    h5[f"{f.lower()}_spec_mask"][i] = 1

    print(f"[ok] wrote {out_path} (N={n})")


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint."""
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="outputs/h5", help="Output directory (one .h5 per field)")
    p.add_argument("--fields", nargs="*", default=None, help="Optional subset of fields")
    p.add_argument("--limit", type=int, default=None, help="Limit objects per field (for testing)")
    p.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for exporting multiple fields (only used when --fields is not provided)",
    )
    args = p.parse_args(argv)

    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing {MASTER_PATH} (run from images/OUTTHERE or keep file next to this script)")

    master_cat = Table.read(str(MASTER_PATH))
    all_fields = np.unique(master_cat["field"]).tolist()
    fields = args.fields if args.fields else all_fields

    out_dir = Path(args.out_dir)

    # Parallelize only in the "export everything" mode to avoid surprising behavior
    # when the user explicitly provided a subset via --fields.
    if args.fields is None and int(args.num_threads) != 1:
        num_threads = max(int(args.num_threads), 1)
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futures = {}
            for field in fields:
                out_path = out_dir / f"{field}.h5"
                fut = ex.submit(export_field, master_cat, str(field), out_path, args.limit)
                futures[fut] = str(field)

            for fut in as_completed(futures):
                field = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    raise RuntimeError(f"Export failed for field={field}") from e
    else:
        for field in fields:
            out_path = out_dir / f"{field}.h5"
            export_field(master_cat, str(field), out_path, args.limit)



if __name__ == "__main__":
    main()
