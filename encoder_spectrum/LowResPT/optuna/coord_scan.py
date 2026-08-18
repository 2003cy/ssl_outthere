"""Stage 2 of the coordinate scan: depth, at the embed_dim chosen by stage 1.

Stage 1 (scan_embed_dim.py) sweeps embed_dim at fixed depth. This waits for it,
picks d* under an explicit rule, then sweeps num_layers at d*, and finally
repeats the best configuration to estimate run-to-run noise.

d* rule: the smallest embed_dim whose objective is within 2x the measured
run-to-run noise of the best one. Width beyond that point buys nothing and only
inflates the probe readout, so the tie is broken towards the smaller model.
"""
import json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from data.gpu_datamodule import GPULowResDataModule
from model.low_res_pt import LowResPT
from optuna_train import OptunaRedshiftProbe, build_data, load_base_cfg
from scan_embed_dim import WINNER, _NoTrial

SCAN = HERE / "studies" / "scan"
OUT = HERE / "studies" / "coord"
NOISE = 0.0003          # measured: d=192 rerun 0.0267 vs recorded 0.0264
BUDGET_H = 3.0
DEPTHS = [16, 20, 24]
N_REPEAT = 2


def run(cfg, ds_pt, ds_z, d, L, tag):
    hp = dict(WINNER); hd = hp.pop("head_dim"); hp.pop("num_layers")
    mcfg = {**cfg["model"], **hp, "embed_dim": d, "num_layers": L, "num_heads": d // hd}
    dm = GPULowResDataModule(**cfg["data"]); dm.dataset = ds_pt
    model = LowResPT(wl_ref_min=cfg["data"]["wl_ref_min"],
                     wl_ref_max=cfg["data"]["wl_ref_max"], **mcfg)
    cb = OptunaRedshiftProbe(_NoTrial(), ds_z, every_n_epochs=cfg["probe"]["every_n_epochs"],
                             split_seeds=range(cfg["probe"]["n_splits"]),
                             n_jobs=cfg["probe"]["n_jobs"], probe=cfg["probe"]["probe"],
                             lasso_alpha=cfg["probe"]["lasso_alpha"],
                             lasso_kwargs=cfg["probe"]["lasso_kwargs"], track_knn=True)
    ck = ModelCheckpoint(dirpath=OUT / tag, filename="best-{epoch:03d}-{val_hid_loss:.4f}",
                         monitor="val_hid_loss", mode="min", save_top_k=1)
    tr = pl.Trainer(max_epochs=cfg["trainer"]["max_epochs"], accelerator="gpu", devices=1,
                    precision=cfg["trainer"]["precision"], logger=False, callbacks=[cb, ck],
                    enable_progress_bar=False, enable_model_summary=False)
    t0 = time.time(); tr.fit(model, datamodule=dm); dt = (time.time() - t0) / 60
    last = cb.history[-1]
    r = dict(tag=tag, embed_dim=d, num_layers=L, num_heads=d // hd,
             params_M=sum(p.numel() for p in model.parameters()) / 1e6,
             readout=27 * d, objective=cb.objective_value(),
             n_active=last["lasso"]["n_active"], r2=last["lasso"]["r2"],
             knn=last["knn"]["snmad"], minutes=dt, best_ckpt=str(ck.best_model_path))
    print(f"\n>>> {tag}: d={d} L={L} {r['params_M']:.1f}M  obj={r['objective']:.4f}  "
          f"n_active={r['n_active']:.0f}/{27*d} ({100*r['n_active']/(27*d):.1f}%)  "
          f"knn={r['knn']:.4f}  {dt:.1f} min\n", flush=True)
    return r


def main():
    t_start = time.time()
    while subprocess.run(["pgrep", "-f", "scan_embed_dim.py"],
                         capture_output=True).returncode == 0:
        print("waiting for stage 1 ...", flush=True); time.sleep(60)

    s1 = json.loads((SCAN / "results.json").read_text())
    print("\n=== stage 1: embed_dim at L=12 ===")
    for r in s1:
        print(f"  d={r['embed_dim']:>4} obj={r['objective']:.4f} "
              f"n_active={r['n_active']:.0f}/{r['readout']} "
              f"({100*r['n_active']/r['readout']:.1f}%) knn={r['knn']:.4f}")
    best = min(r["objective"] for r in s1)
    d_star = min(r["embed_dim"] for r in s1 if r["objective"] <= best + 2 * NOISE)
    print(f"\n  best {best:.4f}; within 2x noise ({2*NOISE:.4f}) -> d* = {d_star}")

    cfg = load_base_cfg(HERE / "configs" / "base.yaml")
    ds_pt, ds_z = build_data(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    res = []
    base = [r for r in s1 if r["embed_dim"] == d_star][0]
    res.append(dict(base, tag=f"d{d_star}_L12", num_layers=12))

    print(f"\n=== stage 2: num_layers at d={d_star} ===")
    for L in DEPTHS:
        if (time.time() - t_start) / 3600 > BUDGET_H:
            print("budget reached, stopping"); break
        r = run(cfg, ds_pt, ds_z, d_star, L, f"d{d_star}_L{L}")
        res.append(r)
        (OUT / "results.json").write_text(json.dumps(res, indent=2))
        if r["objective"] > res[0]["objective"] + 2 * NOISE:
            print(f"L={L} is worse than L=12 by more than 2x noise; deeper will not help")
            break

    bestr = min(res, key=lambda r: r["objective"])
    print(f"\n=== stage 3: repeat best (d={bestr['embed_dim']} L={bestr['num_layers']}) ===")
    reps = [bestr["objective"]]
    for i in range(N_REPEAT):
        if (time.time() - t_start) / 3600 > BUDGET_H:
            break
        r = run(cfg, ds_pt, ds_z, bestr["embed_dim"], bestr["num_layers"], f"repeat{i}")
        reps.append(r["objective"]); res.append(r)
        (OUT / "results.json").write_text(json.dumps(res, indent=2))

    print("\n" + "=" * 76)
    print(f"{'tag':>14} {'d':>5} {'L':>4} {'params':>8} {'objective':>10} {'%sel':>6} {'kNN':>7} {'min':>6}")
    for r in res:
        print(f"{r['tag']:>14} {r['embed_dim']:>5} {r['num_layers']:>4} {r['params_M']:7.1f}M "
              f"{r['objective']:10.4f} {100*r['n_active']/r['readout']:5.1f}% {r['knn']:7.4f} "
              f"{r.get('minutes',0):6.1f}")
    if len(reps) > 1:
        import statistics
        print(f"\nrepeats of best: {['%.4f' % x for x in reps]}  "
              f"mean {statistics.mean(reps):.4f} std {statistics.pstdev(reps):.4f}")
    print(f"\nreference: study r1 trial #63 = 0.0264 | baseline ckpt = 0.0471 +/- 0.0009")
    print(f"total {(time.time()-t_start)/3600:.2f} h")


if __name__ == "__main__":
    main()
