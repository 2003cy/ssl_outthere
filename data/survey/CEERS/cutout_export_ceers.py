"""Export CEERS f150w cutouts -> memmap-friendly .npy + a slim image-index FITS.

Mirrors data/survey/cosmos_2025/cutout_export_npy.py (same on-disk contract), so the
JWST dataset (encoder_image/jwst_dino/data/dataset.py) loads COSMOS and CEERS
identically. Per filter the layout is::

    <output-dir>/<filter>/nircam_ceers_<filter>_<tile>.npy       (N,128,128) float16
    <output-dir>/<filter>/nircam_ceers_<filter>_<tile>_seg.npy   (N,128,128) uint8
    <output-dir>/image_index_ceers_<filter>.fits                 slim index

``rel_path`` in the index is relative to the index file's directory; labels are
joined downstream by ``id`` (row index into ceers_cat_v1.0.fits) and ``ra``/``dec``
match third-party catalogs positionally.

CEERS specifics vs COSMOS:
  * one 'fullceers' mosaic (SCI in ext 1) instead of 20 tiles -> a single pseudo-tile
    ('EGS', the field name);
  * pixel centers from RA/DEC via the mosaic WCS (0-based, frame-safe) -- the catalog
    X/Y_IMAGE are SExtractor 1-based and unused;
  * segmap value == catalog NUMBER (a detection id, NOT the row index), so the central
    object's binary mask is ``seg_cut == NUMBER``;
  * quality cut BAD_REGION_FLAG==0 only (no SNR cut by default) — mirrors COSMOS's
    quality-flag-only export so the full faint population (incl. SNR<3) is kept;
    an optional --snr-min floor is available.

Parallelism: the mosaic + segmap are placed in shared memory once; rows fan out to a
process pool (true parallelism -- the per-object loop would be GIL-bound under threads),
every worker attaching the same single physical copy.
"""

from __future__ import annotations

'''
cd /nexus/posix0/MIA-astro-env/ivemo/yacheng/ssl_outthere/data/survey/CEERS

pixi shell  # (default env)

python cutout_export_ceers.py \
  --catalog ceers_cat_v1.0.fits \
  --mosaic hlsp_ceers_jwst_nircam_fullceers_f150w_v1_sci-bkgsub.fits \
  --segmap ceers_segmap_v1.0.fits \
  --output-dir ../../data/image \
  --filter f150w --image-size 128 --seg-size 128 \
  --max-workers 64 --max-empty-frac 0.40 \
  2>&1 | tee export_ceers_npy_f150w_$(date +%Y%m%d_%H%M%S).log
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
    global _MOSAIC, _SEG
    _MOSAIC = _attach(mosaic_meta)
    _SEG = _attach(seg_meta)


def _process_chunk(payload):
    """Cut one chunk of objects from the shared mosaic/segmap. Runs in a worker.

    payload = (xs, ys, ids, segids, ras, decs, image_size, seg_size, max_empty_frac).
    Returns stacked (image, seg, id, ra, dec) for the kept cutouts, or None.
    """
    xs, ys, ids, segids, ras, decs, image_size, seg_size, max_empty_frac = payload
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
        segs.append((seg_cut == segids[k]).astype(np.uint8))  # central object mask
        oid.append(int(ids[k])); ora.append(float(ras[k])); odec.append(float(decs[k]))
    if not imgs:
        return None
    return (
        np.stack(imgs), np.stack(segs),
        np.asarray(oid, dtype=np.int64),
        np.asarray(ora, dtype=np.float64),
        np.asarray(odec, dtype=np.float64),
    )


def _select(cat: Table, snr_min: float | None) -> np.ndarray:
    """Quality cut (BAD_REGION_FLAG==0), optionally with a single-band SNR floor.

    ``snr_min=None`` keeps ALL objects in good regions (incl. SNR<3 / faint / no
    f150w detection) — mirrors COSMOS's quality-flag-only export so the model sees
    the full population. Pass a number to additionally require SNR_f150w>snr_min.
    """
    mask = np.asarray(cat['BAD_REGION_FLAG']) == 0
    if snr_min is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            snr150 = np.asarray(cat['F150W_FLUX']) / np.asarray(cat['F150W_FLUXERR_EMP'])
        snr150 = np.nan_to_num(snr150, nan=-1.0, posinf=-1.0, neginf=-1.0)
        mask &= snr150 > snr_min
    return mask


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='CEERS cutout exporter (npy + slim image-index FITS).')
    p.add_argument('--catalog', default='ceers_cat_v1.0.fits')
    p.add_argument('--mosaic', default='hlsp_ceers_jwst_nircam_fullceers_f150w_v1_sci-bkgsub.fits')
    p.add_argument('--segmap', default='ceers_segmap_v1.0.fits')
    p.add_argument('--ext', type=int, default=1, help='Mosaic HDU holding SCI data (CEERS: 1).')
    p.add_argument('--seg-ext', type=int, default=0, help='Segmap HDU (CEERS: 0).')
    p.add_argument('--output-dir', required=True, help='Root for <filter>/*.npy and the index FITS.')
    p.add_argument('--survey', default='ceers')
    p.add_argument('--tile', default='EGS', help='Pseudo-tile alias for the single mosaic.')
    p.add_argument('--filter', default='f150w')
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--seg-size', type=int, default=128)
    p.add_argument('--max-empty-frac', type=float, default=0.40,
                   help='Drop a cutout whose empty (NaN/inf or 0) pixel fraction exceeds this.')
    p.add_argument('--snr-min', type=float, default=None,
                   help='Optional SNR_f150w floor. Default None = keep all objects in '
                        'good regions (incl. SNR<3); pass e.g. 3 to additionally cut on SNR.')
    p.add_argument('--chunk-size', type=int, default=400, help='Rows per process-pool task.')
    p.add_argument('--max-workers', type=int, default=os.cpu_count() or 4)
    p.add_argument('--limit', type=int, default=None, help='Process only the first N selected objects (debug).')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    image_size = args.image_size - (args.image_size % 2)
    seg_size = args.seg_size - (args.seg_size % 2)

    print(f'reading catalog {args.catalog}')
    cat = Table.read(args.catalog)
    # `id` = row index into the FULL catalog (so ceers_cat[col][id] indexes directly,
    # matching COSMOS). `segid` = SExtractor NUMBER == the segmap pixel value.
    row_index = np.arange(len(cat), dtype=np.int64)
    mask = _select(cat, args.snr_min)
    cat, row_index = cat[mask], row_index[mask]
    cut = 'BAD_REGION_FLAG==0' + (f' & snr_f150w>{args.snr_min}' if args.snr_min is not None else ' (no SNR cut)')
    print(f'selected {len(cat)} objects ({cut})')
    if args.limit is not None:
        cat, row_index = cat[:args.limit], row_index[:args.limit]
        print(f'--limit: restricted to first {len(cat)} objects')

    print('computing pixel coordinates from WCS')
    with fits.open(args.mosaic, memmap=True) as h:
        wcs = WCS(h[args.ext].header)
    coords = SkyCoord(np.asarray(cat['RA']), np.asarray(cat['DEC']), unit='deg')
    xs, ys = wcs.world_to_pixel(coords)
    ids = row_index
    segids = np.asarray(cat['NUMBER'], dtype=np.int64)
    ras = np.asarray(cat['RA'], dtype=np.float64)
    decs = np.asarray(cat['DEC'], dtype=np.float64)

    print(f'loading mosaic {args.mosaic} (ext {args.ext}) into shared memory')
    mosaic = np.asarray(fits.getdata(args.mosaic, ext=args.ext), dtype=np.float32)
    m_shm, m_meta = _to_shm(mosaic)
    del mosaic
    print(f'loading segmap {args.segmap} (ext {args.seg_ext}) into shared memory')
    segmap = np.asarray(fits.getdata(args.segmap, ext=args.seg_ext), dtype=np.int32)
    s_shm, s_meta = _to_shm(segmap)
    del segmap

    n = len(ids)
    n_chunks = max(1, -(-n // args.chunk_size))
    idx_chunks = np.array_split(np.arange(n), n_chunks)
    payloads = [
        (xs[c], ys[c], ids[c], segids[c], ras[c], decs[c], image_size, seg_size, args.max_empty_frac)
        for c in idx_chunks if len(c) > 0
    ]
    print(f'mosaic {m_meta[1]}; {n} objects -> {len(payloads)} chunks across {args.max_workers} workers')

    try:
        with ProcessPoolExecutor(
            max_workers=min(args.max_workers, len(payloads)),
            initializer=_init_worker, initargs=(m_meta, s_meta),
        ) as pool:
            futures = [pool.submit(_process_chunk, pl) for pl in payloads]
            # Iterate in submission order so npy row order (-> local_idx) is deterministic.
            results = [f.result() for f in tqdm(futures, desc='chunks')]
    finally:
        for shm in (m_shm, s_shm):
            shm.close()
            shm.unlink()

    results = [r for r in results if r is not None]
    if not results:
        print('no cutouts produced (all edge/empty?) — nothing written.')
        return

    images = np.concatenate([r[0] for r in results])
    segs = np.concatenate([r[1] for r in results])
    out_ids = np.concatenate([r[2] for r in results])
    out_ras = np.concatenate([r[3] for r in results])
    out_decs = np.concatenate([r[4] for r in results])
    print(f'kept {len(images)} cutouts ({n - len(images)} dropped as edge/empty objects)')

    filter_dir = os.path.join(args.output_dir, args.filter)
    os.makedirs(filter_dir, exist_ok=True)
    rel_path = os.path.join(args.filter, f'nircam_{args.survey}_{args.filter}_{args.tile}.npy')
    np.save(os.path.join(args.output_dir, rel_path), images)
    np.save(os.path.join(args.output_dir, rel_path[:-4] + '_seg.npy'), segs)

    index = Table({
        'id': out_ids,
        'ra': out_ras,
        'dec': out_decs,
        'tile': np.full(len(out_ids), args.tile),
        'filter': np.full(len(out_ids), args.filter),
        'local_idx': np.arange(len(out_ids), dtype=np.int32),
        'rel_path': np.full(len(out_ids), rel_path),
    })
    index.meta.update({'PIXSCALE': 30.0, 'BUNIT': 'MJy/sr', 'IMGSIZE': image_size,
                       'DTYPE': 'float16', 'SURVEY': args.survey})
    index_path = os.path.join(args.output_dir, f'image_index_{args.survey}_{args.filter}.fits')
    index.write(index_path, overwrite=True)
    print(f'wrote {len(images)} cutouts -> {os.path.join(args.output_dir, rel_path)}')
    print(f'{len(index)} rows -> {index_path}')


if __name__ == '__main__':
    main()
