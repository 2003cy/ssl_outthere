"""Frozen-encoder redshift benchmark for LowResPT.

The encoder is never updated: embeddings are extracted under `no_grad` and the
probes are fitted on the resulting numpy arrays. `eval_redshift` is what both
`neurips_spectrum_bench.ipynb` and the Optuna callback call.

CLI, from the LowResPT directory:

    python eval/run_eval.py --run low_res_pt_1_2_micron_noz_cut_tokenweight --version 0
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from astropy.table import Table
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from data.dataset import LowResDataset          # noqa: E402
from data.datamodule import LowResDataModule    # noqa: E402
from eval.probes import knn_zeroshot, lasso_probe, metrics, ridge_probe  # noqa: E402

DEFAULT_FITS = "/home/yacheng/ssl_outthere/data/spectrum/DJA_spectra_v4.5.fits"

# Cuts that define the redshift-probe sample. `frac_valid_pix=0.9` is a probe-only
# requirement: near-empty spectra collapse to an identical all-zero embedding and
# would pin the probe's predictions to a constant.
PROBE_CUTS = dict(min_sn50=0, min_redshift=1, max_redshift=3, frac_valid_pix=0.9)

SPLIT_FRAC = 0.5
POOL = "concat"


def best_ckpt(run: str, version: int = 0, outputs: Optional[Path] = None) -> Path:
    """Lowest-val_hid_loss checkpoint of a run, read out of the filename."""
    outputs = Path(outputs) if outputs is not None else REPO / "outputs"
    ckpts = list(outputs.glob(f"{run}/version_{version}/checkpoints/*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints under {outputs}/{run}/version_{version}")
    return min(ckpts, key=lambda p: float(p.stem.split("val_hid_loss=")[-1]))


def build_probe_dataset(fits_path: str = DEFAULT_FITS, **cuts) -> LowResDataset:
    """The redshift-probe sample: `PROBE_CUTS` unless overridden."""
    return LowResDataset(fits_path, **{**PROBE_CUTS, **cuts})


def label_from_catalog(ds: LowResDataset, col: str, fits_path: str = DEFAULT_FITS):
    """Any catalog column, re-indexed into dataset row order via `ds.valid_indices`."""
    return np.asarray(Table.read(fits_path)[col])[ds.valid_indices]


def _pool(raw: torch.Tensor, tvm: torch.Tensor, pool: str) -> torch.Tensor:
    """Reduce (B, N, D) patch tokens to a per-spectrum vector.

    `concat` flattens the whole token sequence with invalid tokens zeroed, which
    preserves position on the fixed wavelength grid; the other modes collapse it.
    """
    vm3 = tvm.float().unsqueeze(-1)
    tok = raw * vm3
    mean_ = tok.sum(1) / vm3.sum(1).clamp(min=1)
    max_ = torch.nan_to_num(
        raw.masked_fill(tvm.unsqueeze(-1) == 0, float("-inf")).max(1).values, neginf=0.0)
    if pool == "concat":
        return tok.reshape(tok.shape[0], -1)
    if pool == "mean":
        return mean_
    if pool == "max":
        return max_
    if pool == "meanmax":
        return torch.cat([mean_, max_], dim=-1)
    raise ValueError(pool)


@torch.no_grad()
def compute_embeddings(enc, dataset, pool: str = POOL, batch_size: int = 256,
                       device=None, num_workers: int = 0):
    """Frozen embeddings and redshift labels, in dataset row order."""
    device = device if device is not None else next(enc.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=LowResDataModule._pad_collate)
    Xs, ys = [], []
    for b in loader:
        out = enc.compute_embedding_from_raw_spectrum(
            b["flux"].to(device), b["wavelength"].to(device), b["valid_mask"].to(device))
        Xs.append(_pool(out["patch_token"].float(), out["token_valid_mask"], pool).cpu().numpy())
        ys.append(b["redshift"].numpy())
    return np.concatenate(Xs), np.concatenate(ys)


def group_split(objid, n_rows: int, split_frac: float = SPLIT_FRAC, seed: int = 42):
    """50/50 split grouped by object, so no object appears on both sides.

    A plain random split lets repeat observations of the same object straddle the
    split, which the k-NN probe can exploit by retrieving its own twin.
    """
    gss = GroupShuffleSplit(n_splits=1, train_size=split_frac, random_state=seed)
    tr, va = next(gss.split(np.arange(n_rows), groups=objid))
    assert not (set(objid[tr]) & set(objid[va])), "objid leaked across split"
    return tr, va


def eval_redshift(model, ds, device=None, *, pool: str = POOL,
                  split_frac: float = SPLIT_FRAC,
                  split_seeds: Sequence[int] = (42,),
                  probes: Sequence[str] = ("knn", "ridge"),
                  lasso_alpha: Optional[float] = None, lasso_kwargs: Optional[dict] = None,
                  X=None, y=None, batch_size: int = 256, n_jobs: int = 1) -> dict:
    """Run the requested probes on `model`'s frozen embeddings of `ds`.

    Metrics are averaged over one group-aware split per entry of `split_seeds`,
    the way the stellar-mass and Sersic probes average over 40 splits; each
    metric is returned alongside a `*_std` giving the split-to-split spread.
    Splits are independent, so they are fitted in a thread pool (`n_jobs`) —
    scikit-learn's coordinate descent releases the GIL, and threads keep the
    9172 x 3456 feature matrix shared instead of pickling it per worker.

    Pass `X`/`y` to reuse embeddings already extracted from the same model.
    """
    if X is None or y is None:
        X, y = compute_embeddings(model, ds, pool=pool, batch_size=batch_size, device=device)

    def _one(seed):
        tr, va = group_split(ds.objid, len(X), split_frac=split_frac, seed=seed)
        X_tr, X_va, y_tr, y_va = X[tr], X[va], y[tr], y[va]
        out = {}
        for name in probes:
            if name == "knn":
                pred, info = knn_zeroshot(X_tr, y_tr, X_va), {}
            elif name == "ridge":
                pred, info = ridge_probe(X_tr, y_tr, X_va)
            elif name == "lasso":
                pred, info = lasso_probe(X_tr, y_tr, X_va, alpha=lasso_alpha,
                                         **(lasso_kwargs or {}))
            else:
                raise ValueError(f"unknown probe {name!r}")
            out[name] = {**metrics(y_va, pred), **info}
        return out, len(tr), len(va)

    runs = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_one)(s) for s in split_seeds)

    agg = {}
    for name in probes:
        agg[name] = {}
        for k in runs[0][0][name]:
            vals = [r[0][name][k] for r in runs]
            agg[name][k] = float(np.mean(vals))
            agg[name][f"{k}_std"] = float(np.std(vals))

    return {"n": int(len(X)), "n_train": runs[0][1], "n_val": runs[0][2],
            "n_splits": len(list(split_seeds)), "dim": int(X.shape[1]), "probes": agg}


def _main():
    import argparse

    from model.low_res_pt import LowResPT

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--run", type=str, default="low_res_pt_1_2_micron_noz_cut_tokenweight")
    ap.add_argument("--version", type=int, default=0)
    ap.add_argument("--fits", type=str, default=DEFAULT_FITS)
    ap.add_argument("--pool", type=str, default=POOL)
    ap.add_argument("--split-seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--n-splits", type=int, default=None,
                    help="shorthand for --split-seeds 0 1 ... n-1")
    ap.add_argument("--probes", nargs="+", default=["knn", "ridge"])
    ap.add_argument("--lasso-alpha", type=float, default=None)
    ap.add_argument("--n-jobs", type=int, default=1)
    args = ap.parse_args()
    seeds = list(range(args.n_splits)) if args.n_splits else args.split_seeds

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.ckpt) if args.ckpt else best_ckpt(args.run, args.version)
    print(f"device={device}\nckpt={ckpt}")

    model = LowResPT.load_from_checkpoint(ckpt, map_location=device).eval().to(device)
    ds = build_probe_dataset(args.fits)
    res = eval_redshift(model, ds, device, pool=args.pool, split_seeds=seeds,
                        probes=args.probes, lasso_alpha=args.lasso_alpha,
                        n_jobs=args.n_jobs)

    print(f"\nX=({res['n']}, {res['dim']})  train={res['n_train']}  val={res['n_val']}  "
          f"pool={args.pool}  splits={res['n_splits']}")
    for name, m in res["probes"].items():
        extra = "".join(f"  {k}={m[k]:.4g}" for k in ("alpha", "n_active") if k in m)
        print(f"{name:8s} R2={m['r2']:.3f}+/-{m['r2_std']:.3f}  "
              f"sNMAD={m['snmad']:.4f}+/-{m['snmad_std']:.4f}  "
              f"MAE={m['mae']:.3f}  out={m['out']:.1%}{extra}")


if __name__ == "__main__":
    _main()
