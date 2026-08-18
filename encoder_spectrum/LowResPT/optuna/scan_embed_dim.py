"""Fixed-configuration scan over embed_dim, holding every other hyperparameter
at the winning trial of study r1.

The sweep's optimum sat on the upper bound of embed_dim, so this checks whether
the objective keeps improving past it. Everything except embed_dim (and the head
count it implies) is frozen, and the d=192 point is a rerun of the winner: the
gap between it and the recorded 0.0264 is a first estimate of run-to-run noise,
without which a difference at d=384 cannot be called real.
"""
import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))

import lightning.pytorch as pl
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

from data.dataset import LowResDataset
from data.gpu_datamodule import GPULowResDataModule
from model.low_res_pt import LowResPT
from optuna_train import OptunaRedshiftProbe, build_data, load_base_cfg

# study r1, trial #63
WINNER = dict(num_layers=12, head_dim=8, continuous_patch_length=3,
              mask_ratio=0.6974068226979588, dropout=0.2651642582041546,
              lr=0.0009533236470842136, weight_decay=1.181695080107626e-05,
              warmup_steps=272, err_weight_sigma_min=0.10496807023170782)
DIMS = [192, 256, 384]


class _NoTrial:
    """Stands in for an optuna Trial: report and prune are no-ops."""
    number = -1
    def report(self, value, step): pass
    def should_prune(self): return False


def main():
    cfg = load_base_cfg(HERE / "configs" / "base.yaml")
    ds_pt, ds_z = build_data(cfg)
    out = []
    for d in DIMS:
        hp = dict(WINNER); hd = hp.pop("head_dim")
        mcfg = {**cfg["model"], **hp, "embed_dim": d, "num_heads": d // hd}
        dm = GPULowResDataModule(**cfg["data"]); dm.dataset = ds_pt
        model = LowResPT(wl_ref_min=cfg["data"]["wl_ref_min"],
                         wl_ref_max=cfg["data"]["wl_ref_max"], **mcfg)
        n_par = sum(p.numel() for p in model.parameters())
        cb = OptunaRedshiftProbe(_NoTrial(), ds_z,
                                 every_n_epochs=cfg["probe"]["every_n_epochs"],
                                 split_seeds=range(cfg["probe"]["n_splits"]),
                                 n_jobs=cfg["probe"]["n_jobs"],
                                 probe=cfg["probe"]["probe"],
                                 lasso_alpha=cfg["probe"]["lasso_alpha"],
                                 lasso_kwargs=cfg["probe"]["lasso_kwargs"],
                                 track_knn=True)
        ck = ModelCheckpoint(dirpath=HERE / "studies" / "scan" / f"d{d}",
                             filename="best-{epoch:03d}-{val_hid_loss:.4f}",
                             monitor="val_hid_loss", mode="min", save_top_k=1)
        tr = pl.Trainer(max_epochs=cfg["trainer"]["max_epochs"], accelerator="gpu",
                        devices=1, precision=cfg["trainer"]["precision"],
                        logger=False, callbacks=[cb, ck],
                        enable_progress_bar=False, enable_model_summary=False)
        t0 = time.time(); tr.fit(model, datamodule=dm); dt = (time.time() - t0) / 60
        last = cb.history[-1]
        rec = dict(embed_dim=d, num_heads=d // hd, params_M=n_par / 1e6,
                   readout=27 * d, objective=cb.objective_value(),
                   n_active=last["lasso"]["n_active"], r2=last["lasso"]["r2"],
                   knn=last["knn"]["snmad"], minutes=dt,
                   best_ckpt=str(ck.best_model_path))
        out.append(rec)
        print(f"\n>>> d={d} heads={d//hd} {n_par/1e6:.1f}M  obj={rec['objective']:.4f}  "
              f"n_active={rec['n_active']:.0f}/{27*d} ({100*rec['n_active']/(27*d):.1f}%)  "
              f"knn={rec['knn']:.4f}  {dt:.1f} min\n", flush=True)
        (HERE / "studies" / "scan").mkdir(parents=True, exist_ok=True)
        (HERE / "studies" / "scan" / "results.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 78)
    print(f"{'d':>5} {'heads':>6} {'params':>8} {'readout':>8} {'objective':>10} "
          f"{'n_active':>9} {'%sel':>6} {'kNN':>7} {'min':>6}")
    for r in out:
        print(f"{r['embed_dim']:>5} {r['num_heads']:>6} {r['params_M']:7.1f}M {r['readout']:>8} "
              f"{r['objective']:10.4f} {r['n_active']:9.0f} {100*r['n_active']/r['readout']:5.1f}% "
              f"{r['knn']:7.4f} {r['minutes']:6.1f}")
    print("\nreference: study r1 trial #63 (d=192) objective 0.0264, n_active 1120 (21.6%), knn 0.0050")


if __name__ == "__main__":
    main()
