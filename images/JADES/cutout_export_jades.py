"""Export JADES (GOODS-S + GOODS-N) f150w cutouts -> memmap .npy + slim image-index FITS.

Same on-disk contract as COSMOS/CEERS/OutThere (images/cosmos_2025/cutout_export_npy.py)
so encoder_image/jwst_dino loads every survey identically. Per filter::

    <output-dir>/<filter>/nircam_jades_<filter>_<field>.npy       (N,128,128) float16, MJy/sr
    <output-dir>/<filter>/nircam_jades_<filter>_<field>_seg.npy    (N,128,128) uint8
    <output-dir>/image_index_jades_<filter>.fits                   slim index (BOTH fields)

JADES = NIRCam, MJy/sr @ 30 mas — IDENTICAL units & pixel scale to COSMOS/CEERS, so
cutouts need NO flux conversion and NO resampling (the simplest survey to add). It is
essentially "CEERS x 2 fields":
  * two independent products (GOODS-S, GOODS-N), each one mosaic (SCI ext 1) + one segmap
    (SEGMENTATION ext 1) + one photometry catalog -> two pseudo-tiles 'gds' / 'gdn',
    concatenated into a SINGLE image_index_jades_f150w.fits (field alias = tile);
  * pixel centers from RA/DEC via each mosaic's WCS (0-based, frame-safe);
  * segmap pixel value == catalog ID (NOT a row index), so the central object's binary
    mask is ``seg_cut == ID``;
  * quality cut F150W_FLAG==0 (the per-source flagged-pixel COUNT is 0, i.e. a clean
    source — mirrors CEERS BAD_REGION_FLAG==0) AND a KRON_S total-flux SNR floor
    (F150W_KRON_S/F150W_KRON_S_e > snr_min, default 0 = keep the full faint population
    incl. SNR<3). The catalog columns live in two row-aligned extensions: ID/RA/DEC/
    F150W_FLAG in 'FLAG', F150W_KRON_S(+_e) in 'KRON'.

`id` stored = the JADES catalog ID (== segmap value), unique only WITHIN a field. Object
identity is the COMPOSITE (tile=field, id=ID), like OutThere; downstream labels join
(field, ID) into the JADES catalog. There is no single global master.

Parallelism: each field's mosaic + segmap are placed in shared memory once; rows fan out
to a process pool (true parallelism). Fields are processed ONE AT A TIME so peak RAM is a
single field's mosaic+segmap (~17 GB for GOODS-S), not both.
"""

from __future__ import annotations

'''
cd /nexus/posix0/MIA-astro-env/ivemo/yacheng/ssl_outthere/images/JADES

python cutout_export_jades.py \
  --jades-dir . \
  --output-dir ../../data/image \
  --filter f150w --image-size 128 --seg-size 128 \
  --max-workers 64 --max-empty-frac 0.40 \
'''

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from tqdm.auto import tqdm

warnings.simplefilter('ignore')  # silence astropy WCS FITSFixedWarning spam

# field alias (tile, globally unique) -> file-name slug. Both fields go into ONE index.
FIELDS = {'gds': 'goods-s', 'gdn': 'goods-n'}

# Worker-process globals: the shared mosaic/segmap, attached once per pool.
_MOSAIC: np.ndarray | None = None
_SEG: np.ndarray | None = None
_HANDLES: list = []


def _safe_cutout(image: np.ndarray, x_center: float, y_center: float, size: int) -> np.ndarray | None:
    half = size // 2
    x = int(round(float(x_center)))
    y = int(round(float(y_center)))
    y0, y1 = y - half, y + half
    x0, x1 = x - half, x + half
    if y0 < 0 or x0 < 0:
        return None
    if y1 > image.shape[-2] or x1 > image.shape[-1]:
        return None
    return image[..., y0:y1, x0:x1]


def _to_shm(arr: np.ndarray) -> tuple[shared_memory.SharedMemory, tuple]:
    """Copy ``arr`` into a fresh shared-memory block; return the block + its meta."""
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[:] = arr
    return shm, (shm.name, arr.shape, arr.dtype)


def _attach(meta: tuple) -> np.ndarray:
    name, shape, dtype = meta
    shm = shared_memory.SharedMemory(name=name)
    _HANDLES.append(shm)  # keep alive for the worker's lifetime
    return np.ndarray(shape, dtype=dtype, buffer=shm.buf)


def _init_worker(mosaic_meta: tuple, seg_meta: tuple) -> None:
    global _MOSAIC, _SEG, _HANDLES
    _HANDLES = []
    _MOSAIC = _attach(mosaic_meta)
    _SEG = _attach(seg_meta)


def _process_chunk(payload):
    """Cut one chunk of objects from the shared mosaic/segmap. Runs in a worker.

    payload = (xs, ys, ids, ras, decs, image_size, seg_size, max_empty_frac).
    `ids` are JADES catalog IDs == segmap pixel values (mask = seg_cut == id).
    Returns stacked (image, seg, id, ra, dec) for the kept cutouts, or None.
    """
    xs, ys, ids, ras, decs, image_size, seg_size, max_empty_frac = payload
    imgs, segs, oid, ora, odec = [], [], [], [], []
    for k in range(len(ids)):
        img_cut = _safe_cutout(_MOSAIC, xs[k], ys[k], image_size)
        if img_cut is None:
            continue
        empty = ~np.isfinite(img_cut) | (img_cut == 0)
        if empty.mean() > max_empty_frac:
            continue
        seg_cut = _safe_cutout(_SEG, xs[k], ys[k], seg_size)
        if seg_cut is None:
            continue
        imgs.append(img_cut.astype(np.float16))
        segs.append((seg_cut == ids[k]).astype(np.uint8))  # central object mask (seg==ID)
        oid.append(int(ids[k])); ora.append(float(ras[k])); odec.append(float(decs[k]))
    if not imgs:
        return None
    return (
        np.stack(imgs), np.stack(segs),
        np.asarray(oid, dtype=np.int64),
        np.asarray(ora, dtype=np.float64),
        np.asarray(odec, dtype=np.float64),
    )


def _ext(hdul, name: str) -> int:
    """Index of the HDU whose EXTNAME == name (case-insensitive)."""
    for i, hd in enumerate(hdul):
        if str(hd.header.get('EXTNAME', '')).upper() == name.upper():
            return i
    raise KeyError(f'EXTNAME {name!r} not found')


def _select(cat_path: str, snr_min: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read ID/RA/DEC + quality (F150W_FLAG==0) & KRON_S SNR floor from the catalog.

    The columns live in two row-aligned extensions: ID/RA/DEC/F150W_FLAG in 'FLAG',
    F150W_KRON_S(+_e) in 'KRON'. ``snr_min=None`` keeps every clean source (incl. SNR<3);
    pass a number to additionally require KRON_S SNR > snr_min.
    Returns (ids, ras, decs) of selected rows.
    """
    with fits.open(cat_path, memmap=True) as h:
        flag = h[_ext(h, 'FLAG')]
        ids = np.asarray(flag.data['ID'], dtype=np.int64)
        ras = np.asarray(flag.data['RA'], dtype=np.float64)
        decs = np.asarray(flag.data['DEC'], dtype=np.float64)
        f150flag = np.asarray(flag.data['F150W_FLAG'])
        mask = f150flag == 0  # clean source: zero flagged pixels in the F150W footprint
        if snr_min is not None:
            kron = h[_ext(h, 'KRON')]
            with np.errstate(divide='ignore', invalid='ignore'):
                snr = np.asarray(kron.data['F150W_KRON_S'], float) / \
                      np.asarray(kron.data['F150W_KRON_S_e'], float)
            snr = np.nan_to_num(snr, nan=-1.0, posinf=-1.0, neginf=-1.0)
            mask &= snr > snr_min
    return ids[mask], ras[mask], decs[mask]


def _export_field(field: str, slug: str, args) -> tuple | None:
    """Cut every selected object in one field; write its npy shards; return index rows."""
    cat = os.path.join(args.jades_dir, f'hlsp_jades_jwst_nircam_{slug}_photometry_v5.0_catalog.fits')
    mos = os.path.join(args.jades_dir, f'hlsp_jades_jwst_nircam_{slug}_f150w_v5.0_drz.fits')
    seg = os.path.join(args.jades_dir, f'hlsp_jades_jwst_nircam_{slug}_segmentation_v5.0_drz.fits')
    image_size = args.image_size - (args.image_size % 2)
    seg_size = args.seg_size - (args.seg_size % 2)

    print(f'\n=== {field} ({slug}) ===')
    print(f'reading catalog {os.path.basename(cat)}')
    ids, ras, decs = _select(cat, args.snr_min)
    cut = 'F150W_FLAG==0' + (f' & KRON_S snr>{args.snr_min}' if args.snr_min is not None else ' (no SNR cut)')
    print(f'selected {len(ids)} objects ({cut})')
    if args.limit is not None:
        ids, ras, decs = ids[:args.limit], ras[:args.limit], decs[:args.limit]
        print(f'--limit: restricted to first {len(ids)} objects')

    print('computing pixel coordinates from WCS')
    with fits.open(mos, memmap=True) as h:
        sci = _ext(h, 'SCI')
        wcs = WCS(h[sci].header)
    xs, ys = wcs.world_to_pixel(SkyCoord(ras, decs, unit='deg'))

    print(f'loading mosaic (SCI ext {sci}) into shared memory')
    mosaic = np.asarray(fits.getdata(mos, ext=sci), dtype=np.float32)
    m_shm, m_meta = _to_shm(mosaic)
    del mosaic
    print('loading segmap (SEGMENTATION ext 1) into shared memory')
    with fits.open(seg, memmap=True) as h:
        segdata = np.asarray(h[_ext(h, 'SEGMENTATION')].data, dtype=np.int32)
    s_shm, s_meta = _to_shm(segdata)
    del segdata

    n = len(ids)
    n_chunks = max(1, -(-n // args.chunk_size))
    idx_chunks = np.array_split(np.arange(n), n_chunks)
    payloads = [
        (xs[c], ys[c], ids[c], ras[c], decs[c], image_size, seg_size, args.max_empty_frac)
        for c in idx_chunks if len(c) > 0
    ]
    print(f'mosaic {m_meta[1]}; {n} objects -> {len(payloads)} chunks across {args.max_workers} workers')

    try:
        with ProcessPoolExecutor(
            max_workers=min(args.max_workers, len(payloads)),
            initializer=_init_worker, initargs=(m_meta, s_meta),
        ) as pool:
            futures = [pool.submit(_process_chunk, pl) for pl in payloads]
            results = [f.result() for f in tqdm(futures, desc=f'{field} chunks')]
    finally:
        for shm in (m_shm, s_shm):
            shm.close()
            shm.unlink()

    results = [r for r in results if r is not None]
    if not results:
        print(f'{field}: no cutouts produced (all edge/empty?) — skipped.')
        return None

    images = np.concatenate([r[0] for r in results])
    segs = np.concatenate([r[1] for r in results])
    out_ids = np.concatenate([r[2] for r in results])
    out_ras = np.concatenate([r[3] for r in results])
    out_decs = np.concatenate([r[4] for r in results])
    print(f'{field}: kept {len(images)} cutouts ({n - len(images)} dropped as edge/empty)')

    filter_dir = os.path.join(args.output_dir, args.filter)
    os.makedirs(filter_dir, exist_ok=True)
    rel_path = os.path.join(args.filter, f'nircam_{args.survey}_{args.filter}_{field}.npy')
    np.save(os.path.join(args.output_dir, rel_path), images)
    np.save(os.path.join(args.output_dir, rel_path[:-4] + '_seg.npy'), segs)
    print(f'{field}: wrote shards -> {os.path.join(args.output_dir, rel_path)}')
    return rel_path, out_ids, out_ras, out_decs, field


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='JADES (GOODS-S+N) cutout exporter (npy + slim image-index FITS).')
    p.add_argument('--jades-dir', default='.', help='Dir holding the JADES mosaics/segmaps/catalogs.')
    p.add_argument('--output-dir', required=True, help='Root for <filter>/*.npy and the index FITS.')
    p.add_argument('--survey', default='jades')
    p.add_argument('--filter', default='f150w')
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--seg-size', type=int, default=128)
    p.add_argument('--max-empty-frac', type=float, default=0.40,
                   help='Drop a cutout whose empty (NaN/inf or 0) pixel fraction exceeds this.')
    p.add_argument('--snr-min', type=float, default=0.0,
                   help='KRON_S SNR floor (default 0 = keep all clean sources incl. SNR<3). '
                        'Pass None-like negative to disable; 0 keeps F150W_KRON_S/_e > 0.')
    p.add_argument('--chunk-size', type=int, default=400, help='Rows per process-pool task.')
    p.add_argument('--max-workers', type=int, default=os.cpu_count() or 4)
    p.add_argument('--limit', type=int, default=None, help='Process only the first N selected objects per field (debug).')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    image_size = args.image_size - (args.image_size % 2)

    results = [r for field, slug in FIELDS.items()
               if (r := _export_field(field, slug, args)) is not None]
    if not results:
        print('no cutouts produced for any field — nothing written.')
        return

    index = Table({
        'id': np.concatenate([r[1] for r in results]),
        'ra': np.concatenate([r[2] for r in results]),
        'dec': np.concatenate([r[3] for r in results]),
        'tile': np.concatenate([np.full(len(r[1]), r[4]) for r in results]),
        'filter': np.full(sum(len(r[1]) for r in results), args.filter),
        'local_idx': np.concatenate([np.arange(len(r[1]), dtype=np.int32) for r in results]),
        'rel_path': np.concatenate([np.full(len(r[1]), r[0]) for r in results]),
    })
    # NOTE: `id` = per-field JADES catalog ID (== seg value); identity is (tile, id).
    index.meta.update({'PIXSCALE': 30.0, 'BUNIT': 'MJy/sr', 'IMGSIZE': image_size,
                       'DTYPE': 'float16', 'SURVEY': args.survey, 'INSTRUME': 'NIRCAM',
                       'IDKEY': 'field+ID'})
    index_path = os.path.join(args.output_dir, f'image_index_{args.survey}_{args.filter}.fits')
    index.write(index_path, overwrite=True)
    print(f'\nwrote {len(index)} cutouts across {len(results)} fields '
          f'({", ".join(r[4] for r in results)}) -> {index_path}')


if __name__ == '__main__':
    main()
