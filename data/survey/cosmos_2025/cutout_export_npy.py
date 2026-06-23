"""Per-tile cutout exporter -> memmap-friendly .npy + a slim image-index FITS.

Storage/index separation (see also cutout_export_parallel.py, the h5 variant):
  * pixels live in fixed-shape per-(filter,tile) ``.npy`` arrays (memmap-friendly,
    file count grows with shards not objects);
  * everything else lives in a slim ``image_index_cosmos_<filter>.fits`` catalog.
    Labels are joined on demand by ``id`` (row index into the COSMOS-Web master
    catalogs) downstream; ``ra``/``dec`` match third-party catalogs positionally.

Per filter the layout is::

    <output-dir>/<filter>/nircam_cosmos_<filter>_<tile>.npy       (N,128,128) float16
    <output-dir>/<filter>/nircam_cosmos_<filter>_<tile>_seg.npy   (N,128,128) uint8
    <output-dir>/image_index_cosmos_<filter>.fits                 slim index

``rel_path`` in the index is relative to the index file's directory, so the
catalog and the arrays move together with no absolute paths.

Parallelism: tiles run sequentially (only one tile's mosaic+segmap resident at a
time), while each tile fans its rows out to a process pool. The mosaic and
segmap are placed in shared memory once per tile so every worker attaches the
same single copy. This lets the worker count exceed the tile count.
"""

from __future__ import annotations

'''
cd /nexus/posix0/MIA-astro-env/ivemo/yacheng/ssl_outthere/data/survey/cosmos_2025

pixi shell

python cutout_export_npy.py \
    --catalog COSMOSWeb_mastercatalog_v1_photom_primary.fits \
    --filters f150w \
    --output-dir ../../data/image \
    --base-dir . \
    --snr-min 0 \
    --image-size 128 --seg-size 128 \
    --max-empty-frac 0.40 \
    --max-workers 64 \
'''


import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np
from astropy.io import fits
from astropy.table import Table
from tqdm.auto import tqdm

# id/ra/dec are stored in the index; segment-id/x_image/y_image drive the cutout.
REQUIRED_COLUMNS: tuple[str, ...] = (
    'id', 'ra', 'dec', 'segment-id', 'x_image', 'y_image',
)

# Worker-process globals: the tile's shared mosaic/segmap, attached once per pool.
_MOSAIC: np.ndarray | None = None
_SEG: np.ndarray | None = None
_HANDLES: list = []


@dataclass
class ExportConfig:
    base_dir: str
    segmentation_dir: str
    filter_dir: str
    image_size: int
    seg_size: int
    chunk_size: int
    max_workers: int
    max_empty_frac: float
    segmentation_template: str
    filter_template: str
    survey: str
    show_progress: bool


@dataclass
class TileResult:
    tile: str
    n: int
    error: str | None = None
    rel_path: str | None = None
    id: np.ndarray | None = None
    ra: np.ndarray | None = None
    dec: np.ndarray | None = None


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


def _process_chunk(rows: np.ndarray, image_size: int, seg_size: int, max_empty_frac: float):
    """Cut one row-chunk from the shared mosaic/segmap. Returns stacked arrays or None."""
    imgs, segs, ids, ras, decs = [], [], [], [], []
    for row in rows:
        img_cut = _safe_cutout(_MOSAIC, row['x_image'], row['y_image'], image_size)
        if img_cut is None:
            continue
        empty = ~np.isfinite(img_cut) | (img_cut == 0)
        if empty.mean() > max_empty_frac:
            continue
        seg_cut = _safe_cutout(_SEG, row['x_image'], row['y_image'], seg_size)
        if seg_cut is None:
            continue
        imgs.append(img_cut.astype(np.float16))
        segs.append((seg_cut == row['segment-id']).astype(np.uint8))
        ids.append(int(row['id']))
        ras.append(float(row['ra']))
        decs.append(float(row['dec']))
    if not imgs:
        return None
    return (
        np.stack(imgs), np.stack(segs),
        np.asarray(ids, dtype=np.int64),
        np.asarray(ras, dtype=np.float64),
        np.asarray(decs, dtype=np.float64),
    )


def _process_tile(tile_rows: np.ndarray, tile: str, filter_name: str, config: ExportConfig) -> TileResult:
    seg_path = os.path.join(config.base_dir, config.segmentation_dir,
                            config.segmentation_template.format(tile=tile))
    filt_path = os.path.join(config.base_dir, config.filter_dir,
                             config.filter_template.format(filter=filter_name, tile=tile))
    if not (os.path.exists(seg_path) and os.path.exists(filt_path)):
        return TileResult(tile=tile, n=0, error='missing FITS files'), None, None

    mosaic = np.asarray(fits.getdata(filt_path), dtype=np.float32)
    segmap = np.asarray(fits.getdata(seg_path))
    m_shm, m_meta = _to_shm(mosaic)
    s_shm, s_meta = _to_shm(segmap)
    del mosaic, segmap  # the shared copies are authoritative

    n_chunks = max(1, -(-len(tile_rows) // config.chunk_size))
    chunks = np.array_split(tile_rows, n_chunks)
    try:
        with ProcessPoolExecutor(
            max_workers=min(config.max_workers, n_chunks),
            initializer=_init_worker, initargs=(m_meta, s_meta),
        ) as pool:
            futures = [
                pool.submit(_process_chunk, c, config.image_size, config.seg_size, config.max_empty_frac)
                for c in chunks
            ]
            # Iterate in submission order so npy row order is deterministic.
            results = [f.result() for f in tqdm(
                futures, desc=f'{filter_name} {tile}', leave=False, disable=not config.show_progress)]
    finally:
        for shm in (m_shm, s_shm):
            shm.close()
            shm.unlink()

    results = [r for r in results if r is not None]
    if not results:
        return TileResult(tile=tile, n=0, error='no valid cutouts'), None, None

    image_arr = np.concatenate([r[0] for r in results])
    seg_arr = np.concatenate([r[1] for r in results])
    id_arr = np.concatenate([r[2] for r in results])
    ra_arr = np.concatenate([r[3] for r in results])
    dec_arr = np.concatenate([r[4] for r in results])

    rel_path = os.path.join(filter_name, f'nircam_cosmos_{filter_name}_{tile}.npy')
    return TileResult(tile=tile, n=len(id_arr), rel_path=rel_path,
                      id=id_arr, ra=ra_arr, dec=dec_arr), image_arr, seg_arr


def _export_filter(catalog: Table, filter_name: str, output_dir: str, config: ExportConfig) -> None:
    filter_dir = os.path.join(output_dir, filter_name)
    os.makedirs(filter_dir, exist_ok=True)

    tiles = [t.decode() if isinstance(t, (bytes, np.bytes_)) else str(t)
             for t in np.unique(np.asarray(catalog['tile']))]
    print(f'{filter_name}: {len(tiles)} tiles')

    index_parts: list[TileResult] = []
    tile_col = np.asarray(catalog['tile'])
    for tile in tqdm(tiles, desc=f'{filter_name} tiles', disable=not config.show_progress):
        mask = tile_col == (tile.encode() if tile_col.dtype.kind == 'S' else tile)
        tile_rows = catalog[mask][list(REQUIRED_COLUMNS)].as_array()
        if len(tile_rows) == 0:
            continue

        result, image_arr, seg_arr = _process_tile(tile_rows, tile, filter_name, config)
        if result.error:
            print(f'{filter_name} {tile}: skipped ({result.error})')
            continue

        np.save(os.path.join(output_dir, result.rel_path), image_arr)
        np.save(os.path.join(output_dir, result.rel_path[:-4] + '_seg.npy'), seg_arr)
        print(f'{filter_name} {tile}: saved {result.n} cutouts')
        index_parts.append(result)

    if not index_parts:
        print(f'{filter_name}: nothing written')
        return

    index = Table({
        'id': np.concatenate([r.id for r in index_parts]),
        'ra': np.concatenate([r.ra for r in index_parts]),
        'dec': np.concatenate([r.dec for r in index_parts]),
        'tile': np.concatenate([np.full(r.n, r.tile) for r in index_parts]),
        'filter': np.full(sum(r.n for r in index_parts), filter_name),
        'local_idx': np.concatenate([np.arange(r.n, dtype=np.int32) for r in index_parts]),
        'rel_path': np.concatenate([np.full(r.n, r.rel_path) for r in index_parts]),
    })
    index.meta.update({'PIXSCALE': 30.0, 'BUNIT': 'MJy/sr', 'IMGSIZE': config.image_size,
                       'DTYPE': 'float16', 'SURVEY': config.survey})
    index_path = os.path.join(output_dir, f'image_index_cosmos_{filter_name}.fits')
    index.write(index_path, overwrite=True)
    print(f'{filter_name}: {len(index)} rows -> {index_path}')


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Per-tile cutout exporter (npy + slim image-index FITS).')
    p.add_argument('--catalog', required=True, help='COSMOS master catalog FITS.')
    p.add_argument('--filters', nargs='+', required=True, help='Filters, e.g. f115w f150w.')
    p.add_argument('--output-dir', required=True, help='Root for <filter>/*.npy and the index FITS.')
    p.add_argument('--survey', default='cosmos')
    p.add_argument('--base-dir', default='.', help='Base dir for resolving FITS mosaics.')
    p.add_argument('--segmentation-dir', default='segmentation_maps')
    p.add_argument('--filter-dir', default=None, help='Mosaic folder (defaults to the filter name).')
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--seg-size', type=int, default=128)
    p.add_argument('--chunk-size', type=int, default=2000, help='Rows per process-pool task.')
    p.add_argument('--max-workers', type=int, default=os.cpu_count() or 4)
    p.add_argument('--max-empty-frac', type=float, default=0.40,
                   help='Drop a cutout whose empty (NaN/inf or 0) pixel fraction exceeds this.')
    p.add_argument('--max-sample', type=int, default=None, help='Cap rows per tile (debug).')
    p.add_argument('--snr-min', type=float, default=None,
                   help='Optional SNR floor on --snr-col (e.g. 0 drops negative-flux / no-detection '
                        'sources, matching the CEERS export). Default None = quality-flag-only.')
    p.add_argument('--snr-col', default='snr_f150w',
                   help='Catalog SNR column for --snr-min (single detection band, default snr_f150w).')
    p.add_argument('--segmentation-template', default='detection_chi2pos_SWLW_{tile}_segmap_v1.3.fits.gz')
    p.add_argument('--filter-template', default='mosaic_nircam_{filter}_COSMOS-Web_30mas_{tile}_v1.0_sci.fits')
    p.add_argument('--no-progress', action='store_true')
    return p.parse_args()


'''
python cutout_export_npy.py \
  --catalog COSMOSWeb_mastercatalog_v1_photom_primary.fits \
  --filters f150w --output-dir ../../data/image \
  --base-dir . --image-size 128 --seg-size 128 \
  --snr-min 0 \
  --max-workers 64 --max-empty-frac 0.40 \
  2>&1 | tee export_npy_f150w_$(date +%Y%m%d_%H%M%S).log
'''


def main() -> None:
    args = _parse_args()
    catalog = Table.read(args.catalog)
    catalog = catalog[catalog['warn_flag'] <= 3]
    print(f'warn_flag<=3: {len(catalog)} objects')
    if args.snr_min is not None:
        # Drop negative-flux / no-detection sources (snr>0), matching the CEERS export.
        snr = np.nan_to_num(np.asarray(catalog[args.snr_col], dtype=float),
                            nan=-1.0, posinf=-1.0, neginf=-1.0)
        catalog = catalog[snr > args.snr_min]
        print(f'{args.snr_col}>{args.snr_min}: {len(catalog)} objects')
    if args.max_sample is not None:
        catalog = catalog[:args.max_sample]
    os.makedirs(args.output_dir, exist_ok=True)

    for filter_name in args.filters:
        config = ExportConfig(
            base_dir=args.base_dir,
            segmentation_dir=args.segmentation_dir,
            filter_dir=args.filter_dir or filter_name,
            image_size=args.image_size - (args.image_size % 2),
            seg_size=args.seg_size - (args.seg_size % 2),
            chunk_size=args.chunk_size,
            max_workers=max(1, args.max_workers),
            max_empty_frac=args.max_empty_frac,
            segmentation_template=args.segmentation_template,
            filter_template=args.filter_template,
            survey=args.survey,
            show_progress=not args.no_progress,
        )
        _export_filter(catalog, filter_name, args.output_dir, config)


if __name__ == '__main__':
    main()
