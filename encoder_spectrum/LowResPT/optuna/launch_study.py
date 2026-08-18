"""Launch / resume the LowResPT Optuna study.

    python optuna/launch_study.py --study-name r1 --n-trials 40

The study lives in optuna/studies/<name>/study.db (sqlite). Re-running the same
command resumes it; running it again in another shell (optionally with a
different --gpu) adds a worker to the same study -- sqlite handles the
concurrency. When running more than one worker, pin threads
(OMP_NUM_THREADS=2) so the workers' Lasso paths don't fight over every core.

Objective = best redshift-probe sigma_NMAD per trial (minimize); MedianPruner
cuts weak trials early using the per-N-epoch reports.
"""

import argparse
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-name", required=True)
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--timeout", type=float, default=None,
                    help="seconds; stop when reached even if n-trials is not met")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="override trainer.max_epochs from base.yaml (smoke tests)")
    ap.add_argument("--probe-every", type=int, default=None,
                    help="override probe.every_n_epochs from base.yaml (smoke tests)")
    ap.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES for this worker")
    ap.add_argument("--base-config", type=str, default=str(HERE / "configs" / "base.yaml"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--startup-trials", type=int, default=10)
    # 180, so the first pruning decision lands at epoch 199. Two reasons: a
    # full-length dry run showed the objective is not monotonic (it degrades
    # between epochs ~39 and ~79 while the learning rate ramps), and
    # warmup_steps is now searched up to 2000 steps = epoch 95, after which a
    # run still needs ~200 epochs of cosine decay to approach its best. Judging
    # earlier would systematically cut the long-warmup configurations.
    ap.add_argument("--warmup-epochs", type=int, default=180,
                    help="no trial is pruned before this epoch")
    args = ap.parse_args()

    # Pin the GPU before torch is imported (via optuna_train)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    import sys
    sys.path.insert(0, str(HERE))
    from optuna_train import load_base_cfg, make_objective

    study_dir = HERE / "studies" / args.study_name
    (study_dir / "trials").mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{study_dir / 'study.db'}"

    cfg = load_base_cfg(Path(args.base_config))
    if args.probe_every is not None:
        cfg.setdefault("probe", {})["every_n_epochs"] = args.probe_every
    interval = cfg.get("probe", {}).get("every_n_epochs", 20)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",                       # sigma_NMAD, lower is better
        sampler=TPESampler(seed=args.seed, multivariate=True, group=True,
                           n_startup_trials=args.startup_trials),
        # steps are epochs: no pruning before --warmup-epochs, then a decision at
        # every probe report.
        pruner=MedianPruner(n_startup_trials=args.startup_trials,
                            n_warmup_steps=args.warmup_epochs,
                            interval_steps=interval),
        load_if_exists=True,
    )

    objective = make_objective(Path(args.base_config), study_dir,
                               max_epochs=args.max_epochs,
                               probe_every=args.probe_every)
    print(f"[{time.strftime('%H:%M:%S')}] study '{args.study_name}' gpu={args.gpu} "
          f"target {args.n_trials} trials -> {storage}", flush=True)

    # A worker dying mid-trial is recoverable: the trial is recorded FAIL and the
    # sweep continues rather than taking the process down.
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   gc_after_trial=True, catch=(RuntimeError, ValueError, TypeError))

    try:
        print(f"\n=== best trial #{study.best_trial.number}  "
              f"sigma_NMAD={study.best_value:.4f} ===")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
    except ValueError:
        print("\nno completed trials")


if __name__ == "__main__":
    main()
