"""Export CEERS f150w cutouts to a SINGLE minimal-image-store HDF5.

CEERS differs from COSMOS:
  * one large mosaic (data in ext 1, SCI_BKSUB), not tiled  -> single output file
  * no segmentation map                                     -> no `seg` dataset
  * pixel centers computed from RA/DEC via the mosaic WCS (0-based, frame-safe)

Stored datasets: image, id (catalog NUMBER), ra, dec. File attrs carry survey,
filter, pixscale, bunit and the measured sky_sigma. Metadata is joined downstream
by `id` against ceers_cat_v1.0.fits; third-party catalogs match by ra/dec.

Parallelism: the cutout loop is embarrassingly parallel. We use *processes*
(true parallelism — threads would be GIL-bound on the per-object Python loop). The
mosaic is opened with memmap=True in every worker, so all workers share one physical
copy via the OS page cache instead of duplicating 5.5 GB each.
"""

from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from tqdm.auto import tqdm

warnings.simplefilter('ignore')  # silence astropy WCS FITSFixedWarning spam

# Per-process globals populated by the pool initializer.
_MOSAIC = None
_HDUL = None


def _safe_cutout(image, x_center, y_center, size):
    """Centered square cutout; None if the box leaves the image (edge object)."""
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


def _init_worker(mosaic_path: str, ext: int):
    global _MOSAIC, _HDUL
    _HDUL = fits.open(mosaic_path, memmap=True)
    _MOSAIC = _HDUL[ext].data


def _process_chunk(payload):
    """Extract cutouts for one chunk of objects. Runs in a worker process."""
    ids, ras, decs, xs, ys, size = payload
    imgs, oid, ora, odec = [], [], [], []
    for k in range(len(ids)):
        cut = _safe_cutout(_MOSAIC, xs[k], ys[k], size)
        if cut is None:
            continue
        imgs.append(np.asarray(cut, dtype=np.float32))  # forces read+copy off the memmap
        oid.append(ids[k]); ora.append(ras[k]); odec.append(decs[k])
    if not imgs:
        return None
    return (
        np.stack(imgs),
        np.asarray(oid, dtype=np.int64),
        np.asarray(ora, dtype=np.float64),
        np.asarray(odec, dtype=np.float64),
    )


def _select(cat: Table, snr_min: float = 3.0) -> np.ndarray:
    """Quality + single-band SNR cut for CEERS.

    CEERS is ~1 dex deeper than COSMOS, so a single f150w SNR>3 cut is used
    (lower threshold, one band) rather than COSMOS's f115w|f150w>5. The bad-region
    flag is kept as an orthogonal data-quality guard. Uses empirical flux errors.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        snr150 = np.asarray(cat['F150W_FLUX']) / np.asarray(cat['F150W_FLUXERR_EMP'])
    snr150 = np.nan_to_num(snr150, nan=-1.0, posinf=-1.0, neginf=-1.0)
    mask = (np.asarray(cat['BAD_REGION_FLAG']) == 0) & (snr150 > snr_min)
    return mask


def main():
    p = argparse.ArgumentParser(description='Export CEERS f150w cutouts to a single HDF5.')
    p.add_argument('--catalog', default='ceers_cat_v1.0.fits')
    p.add_argument('--mosaic', default='hlsp_ceers_jwst_nircam_fullceers_f150w_v1_sci-bkgsub.fits')
    p.add_argument('--ext', type=int, default=1, help='Mosaic HDU index holding SCI data (CEERS: 1).')
    p.add_argument('--output', required=True, help='Output HDF5 path (single file).')
    p.add_argument('--filter', default='f150w')
    p.add_argument('--survey', default='ceers')
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--sky-sigma', type=float, default=0.0070, help='Measured background RMS (MJy/sr) stored as attr.')
    p.add_argument('--max-workers', type=int, default=os.cpu_count() or 4)
    p.add_argument('--chunks-per-worker', type=int, default=4, help='Chunk granularity for load balancing.')
    p.add_argument('--limit', type=int, default=None, help='Process only the first N selected objects (for testing).')
    args = p.parse_args()

    size = args.image_size - (args.image_size % 2)

    print(f'reading catalog {args.catalog}')
    cat = Table.read(args.catalog)
    # Row index into the FULL, unfiltered catalog — stored as `id` so that
    # ceers_cat[col][id] indexes directly, matching COSMOS (id == row index).
    row_index = np.arange(len(cat), dtype=np.int64)
    mask = _select(cat)
    cat = cat[mask]
    row_index = row_index[mask]
    print(f'selected {len(cat)} objects (BAD_REGION_FLAG==0 & snr_f150w>3)')
    if args.limit is not None:
        cat = cat[:args.limit]
        row_index = row_index[:args.limit]
        print(f'--limit: restricted to first {len(cat)} objects')

    # 0-based pixel centers from RA/DEC via the mosaic WCS (frame-safe).
    print('computing pixel coordinates from WCS')
    with fits.open(args.mosaic, memmap=True) as h:
        wcs = WCS(h[args.ext].header)
        ny, nx = h[args.ext].data.shape
    coords = SkyCoord(np.asarray(cat['RA']), np.asarray(cat['DEC']), unit='deg')
    xs, ys = wcs.world_to_pixel(coords)
    ids = row_index  # == row position in the full ceers_cat_v1.0.fits
    ras = np.asarray(cat['RA'], dtype=np.float64)
    decs = np.asarray(cat['DEC'], dtype=np.float64)

    # Sort by y to localize the memmap reads (more page-cache friendly).
    order = np.argsort(ys)
    ids, ras, decs, xs, ys = ids[order], ras[order], decs[order], xs[order], ys[order]

    n = len(ids)
    n_chunks = max(1, args.max_workers * args.chunks_per_worker)
    splits = np.array_split(np.arange(n), n_chunks)
    payloads = [
        (ids[s], ras[s], decs[s], xs[s], ys[s], size)
        for s in splits if len(s) > 0
    ]
    print(f'mosaic {ny}x{nx}; {n} objects -> {len(payloads)} chunks across {args.max_workers} workers')

    results = []
    with ProcessPoolExecutor(
        max_workers=args.max_workers,
        initializer=_init_worker,
        initargs=(args.mosaic, args.ext),
    ) as ex:
        futures = [ex.submit(_process_chunk, pl) for pl in payloads]
        for fut in tqdm(as_completed(futures), total=len(futures), desc='chunks'):
            r = fut.result()
            if r is not None:
                results.append(r)

    if not results:
        print('no cutouts produced (all edge/empty?) — nothing written.')
        return

    images = np.concatenate([r[0] for r in results])
    out_ids = np.concatenate([r[1] for r in results])
    out_ras = np.concatenate([r[2] for r in results])
    out_decs = np.concatenate([r[3] for r in results])
    print(f'kept {len(images)} cutouts ({n - len(images)} dropped as edge objects)')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('image', data=images,
                         chunks=(min(256, len(images)), size, size), dtype=np.float32)
        f.create_dataset('id', data=out_ids, dtype=np.int64)
        f.create_dataset('ra', data=out_ras, dtype=np.float64)
        f.create_dataset('dec', data=out_decs, dtype=np.float64)
        f.attrs['survey'] = args.survey
        f.attrs['filter'] = args.filter
        f.attrs['pixscale_mas'] = 30.0
        f.attrs['bunit'] = 'MJy/sr'
        f.attrs['image_size'] = int(size)
        f.attrs['sky_sigma'] = float(args.sky_sigma)
    print(f'wrote {len(images)} samples to {args.output}')


if __name__ == '__main__':
    main()
