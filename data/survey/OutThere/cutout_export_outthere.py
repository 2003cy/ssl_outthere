"""Export OutThere <filter> cutouts -> memmap .npy + slim image-index FITS.

Same on-disk contract as COSMOS/CEERS (data/survey/cosmos_2025/cutout_export_npy.py,
data/survey/CEERS/cutout_export_ceers.py) so encoder_image/jwst_dino loads every survey
identically. Per filter::

    <output-dir>/<filter>/niriss_outthere_<filter>_<field>.npy       (N,128,128) float16, MJy/sr @30mas
    <output-dir>/<filter>/niriss_outthere_<filter>_<field>_seg.npy   (N,128,128) uint8
    <output-dir>/image_index_outthere_<filter>.fits

OutThere = grizli/NIRISS direct imaging, ONE drizzle per field. Differs from the
NIRCam surveys, handled here:
  * units `10*nanoJansky/pix` -> MJy/sr via TRUE pixel area from the WCS (NOT the
    stale header PIXAR_SR, which encodes the 65mas native pixel);
  * 40mas pixels -> RESAMPLED to 30mas with reproject_interp (match COSMOS/CEERS
    sampling + FoV: native 96px@40mas = 3.84" = 128px@30mas);
  * per-field SExtractor catalog `<field>-ir.cat.fits` + shared IR segmap
    `<field>-ir_seg.fits` (seg pixel == catalog NUMBER), both filter-independent;
  * a given filter exists in only SOME fields (43 f150w / 118 f200w); fields without
    the filter mosaic are simply not globbed -> skipped.

`id` stored = the field's SExtractor NUMBER, unique only WITHIN a field. Object identity
is the COMPOSITE (tile=field, id=NUMBER); downstream labels join (field, NUMBER) into
phomoetry.fits / spectra-fitting.fits. There is no single global master catalog (unlike
COSMOS/CEERS where `id` alone indexes the master).

Quality cut: BAD-flag analog `FLAG < 4` (SExtractor: drop saturated/truncated/edge;
keep clean/neighbor/deblend) AND `snr = FLUX_AUTO/FLUXERR_AUTO > 0` (IR-detection
significance — the catalog has no per-filter SNR). Matches COSMOS/CEERS snr>0 + quality.

Parallelism: fields are independent and each mosaic is small (~80 MB), so we fan the
43/118 fields out to a process pool (one field fully handled per task: load -> cut ->
convert -> resample -> save shard). reproject is the per-cutout cost.
"""

from __future__ import annotations

'''
cd /nexus/posix0/MIA-astro-env/ivemo/yacheng/ssl_outthere/data/survey/OutThere

python cutout_export_outthere.py \
  --imaging-dir imaging \
  --output-dir ../../data/image \
  --filter f150w --image-size 128 --out-pixscale 30 \
  --max-workers 32 \
  2>&1 | tee export_outthere_f150w_$(date +%Y%m%d_%H%M%S).log
'''

import argparse
import glob
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS
from tqdm.auto import tqdm

warnings.simplefilter('ignore')  # silence astropy WCS FITSFixedWarning spam

NJY10 = 1e-8  # one pixel unit (10 nanoJansky) in Jy; == header PHOTFNU for these mosaics

# Fields excluded by hand — bad data / out of scope, must never appear as a tile:
#   alias 'dor' : dense Galactic star field (not extragalactic; crowded, no clean sky).
#   crt-00      : dominated by one huge saturated star (contaminated background).
_EXCLUDE_ALIASES = {"dor"}
_EXCLUDE_FIELDS = {"crt-00"}


def _excluded(field: str) -> bool:
    return field in _EXCLUDE_FIELDS or field.split("-")[0] in _EXCLUDE_ALIASES


def _conv_mjysr(wcs: WCS) -> float:
    """10nJy/pix -> MJy/sr using the TRUE pixel solid angle from the WCS (not PIXAR_SR)."""
    pscale_rad = np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix))) * np.pi / 180.0
    omega = pscale_rad ** 2  # sr/pixel
    return NJY10 / omega / 1e6  # (Jy/pixval) / (sr) / (Jy per MJy)


def _select(cat: Table, snr_min: float, flag_max: int) -> np.ndarray:
    """FLAG < flag_max  AND  FLUX_AUTO/FLUXERR_AUTO > snr_min (IR-detection SNR)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.asarray(cat['FLUX_AUTO'], float) / np.asarray(cat['FLUXERR_AUTO'], float)
    snr = np.nan_to_num(snr, nan=-1.0, posinf=-1.0, neginf=-1.0)
    flag = np.asarray(cat['FLAG'], float)
    return (flag < flag_max) & (snr > snr_min)


def _process_field(payload):
    """Cut+convert+resample all selected objects in one field. Runs in a worker.

    Writes the field's .npy + _seg.npy shards; returns the index rows (or None).
    """
    from reproject import reproject_interp  # worker-local import

    (field, mosaic_path, seg_path, cat_path, out_dir, filt, survey,
     out_size, out_pix, max_empty_frac, snr_min, flag_max) = payload

    hdr = fits.getheader(mosaic_path)
    wcs = WCS(hdr)
    in_pix = np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3.6e6  # mas/pix
    native = int(round(out_size * out_pix / in_pix))  # native box matching the 30mas FoV
    data = np.asarray(fits.getdata(mosaic_path), dtype=np.float32) * _conv_mjysr(wcs)  # -> MJy/sr
    seg = np.asarray(fits.getdata(seg_path))
    cat = Table.read(cat_path)

    keep = _select(cat, snr_min, flag_max)
    nums = np.asarray(cat['NUMBER'])[keep]
    xs = np.asarray(cat['X_IMAGE'], float)[keep] - 1.0  # 1-based -> 0-based
    ys = np.asarray(cat['Y_IMAGE'], float)[keep] - 1.0
    ras = np.asarray(cat['RA'], float)[keep]
    decs = np.asarray(cat['DEC'], float)[keep]

    imgs, segs, oid, ora, odec = [], [], [], [], []
    for k in range(len(nums)):
        try:
            cut = Cutout2D(data, (xs[k], ys[k]), size=native, wcs=wcs, mode='partial', fill_value=0.0)
        except (ValueError, Exception):
            continue
        if cut.data.shape != (native, native):
            continue
        empty = ~np.isfinite(cut.data) | (cut.data == 0)
        if empty.mean() > max_empty_frac:
            continue
        # target WCS: out_size px at out_pix mas, centered on the cutout centre.
        tw = cut.wcs.deepcopy()
        tw.wcs.cd = cut.wcs.wcs.cd * (out_pix / in_pix)
        tw.wcs.crpix = [out_size / 2 + 0.5, out_size / 2 + 0.5]
        cen = cut.wcs.pixel_to_world((native - 1) / 2, (native - 1) / 2)
        tw.wcs.crval = [cen.ra.deg, cen.dec.deg]
        img30, _ = reproject_interp((np.nan_to_num(cut.data), cut.wcs), tw,
                                    shape_out=(out_size, out_size))

        seg_cut = Cutout2D(seg, (xs[k], ys[k]), size=native, mode='partial', fill_value=0).data
        mask40 = (seg_cut == nums[k]).astype(np.float32)
        mask30, _ = reproject_interp((mask40, cut.wcs), tw, shape_out=(out_size, out_size))

        imgs.append(np.nan_to_num(img30).astype(np.float16))
        segs.append((mask30 > 0.5).astype(np.uint8))
        oid.append(int(nums[k])); ora.append(float(ras[k])); odec.append(float(decs[k]))

    if not imgs:
        return None
    rel_path = os.path.join(filt, f'niriss_{survey}_{filt}_{field}.npy')
    os.makedirs(os.path.join(out_dir, filt), exist_ok=True)
    np.save(os.path.join(out_dir, rel_path), np.stack(imgs))
    np.save(os.path.join(out_dir, rel_path[:-4] + '_seg.npy'), np.stack(segs))
    return (rel_path, np.asarray(oid, np.int64), np.asarray(ora, np.float64),
            np.asarray(odec, np.float64), field, len(oid))


def _discover(imaging_dir: str, filt: str):
    """(field, mosaic, seg, cat) for every field that HAS this filter's mosaic."""
    out = []
    for mos in sorted(glob.glob(os.path.join(imaging_dir, '*', f'*-{filt}n-clear_drc_sci.fits'))):
        field = os.path.basename(os.path.dirname(mos))
        if _excluded(field):
            continue
        seg = os.path.join(imaging_dir, field, f'{field}-ir_seg.fits')
        cat = os.path.join(imaging_dir, field, f'{field}-ir.cat.fits')
        if os.path.exists(seg) and os.path.exists(cat):
            out.append((field, mos, seg, cat))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description='OutThere per-field cutout exporter (npy + image-index FITS).')
    p.add_argument('--imaging-dir', default='imaging')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--survey', default='outthere')
    p.add_argument('--filter', default='f150w', help='f150w or f200w (the NIRISS pupil; file uses <filter>n).')
    p.add_argument('--image-size', type=int, default=128, help='Output cutout size (px) at --out-pixscale.')
    p.add_argument('--out-pixscale', type=float, default=30.0, help='Output mas/pix (match COSMOS/CEERS).')
    p.add_argument('--max-empty-frac', type=float, default=0.40)
    p.add_argument('--snr-min', type=float, default=0.0)
    p.add_argument('--flag-max', type=int, default=4, help='Keep SExtractor FLAG < this.')
    p.add_argument('--max-workers', type=int, default=os.cpu_count() or 4)
    args = p.parse_args()

    out_size = args.image_size - (args.image_size % 2)
    fields = _discover(args.imaging_dir, args.filter)
    print(f'{args.filter}: {len(fields)} fields have a mosaic (others skipped)')
    if not fields:
        print('no fields with this filter — nothing to do.')
        return

    payloads = [
        (fd, mos, seg, cat, args.output_dir, args.filter, args.survey,
         out_size, args.out_pixscale, args.max_empty_frac, args.snr_min, args.flag_max)
        for fd, mos, seg, cat in fields
    ]
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    with ProcessPoolExecutor(max_workers=min(args.max_workers, len(payloads))) as ex:
        futures = [ex.submit(_process_field, pl) for pl in payloads]
        for fut in tqdm(as_completed(futures), total=len(futures), desc='fields'):
            r = fut.result()
            if r is not None:
                results.append(r)

    if not results:
        print('no cutouts produced — nothing written.')
        return

    index = Table({
        'id': np.concatenate([r[1] for r in results]),
        'ra': np.concatenate([r[2] for r in results]),
        'dec': np.concatenate([r[3] for r in results]),
        'tile': np.concatenate([np.full(r[5], r[4]) for r in results]),
        'filter': np.full(sum(r[5] for r in results), args.filter),
        'local_idx': np.concatenate([np.arange(r[5], dtype=np.int32) for r in results]),
        'rel_path': np.concatenate([np.full(r[5], r[0]) for r in results]),
    })
    # NOTE: `id` = per-field SExtractor NUMBER (unique within a field, == seg value).
    # Object identity is the COMPOSITE (tile, id); join (field, NUMBER) -> phomoetry/spectra.
    index.meta.update({'PIXSCALE': args.out_pixscale, 'BUNIT': 'MJy/sr', 'IMGSIZE': out_size,
                       'DTYPE': 'float16', 'SURVEY': args.survey, 'INSTRUME': 'NIRISS',
                       'IDKEY': 'field+NUMBER'})
    index_path = os.path.join(args.output_dir, f'image_index_{args.survey}_{args.filter}.fits')
    index.write(index_path, overwrite=True)
    print(f'wrote {len(index)} cutouts across {len(results)} fields -> {index_path}')


if __name__ == '__main__':
    main()
