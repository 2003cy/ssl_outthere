"""Multi-process Optuna launcher.

Spawns N worker processes that share one SQLite study (concurrent writes are
safe). Each worker runs `n_trials_per_worker` trials, picking unseen param
combinations from the TPE sampler.

Examples:

    # First time — create and run with 4 parallel workers, 100 trials total
    python launch_study.py \
        --study-name sweep_v1 \
        --n-workers 4 \
        --n-trials  100 \
        --max-epochs 200

    # Resume / add more trials to the same study
    python launch_study.py --study-name sweep_v1 --n-workers 4 --n-trials 50

    # Inspect live progress in another shell:
    optuna-dashboard sqlite:///optuna/studies/sweep_v1/study.db
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import yaml
import multiprocessing as mp
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from optuna_train import make_objective, load_base_cfg


def _worker(worker_id: int, args, study_dir: Path):
    # Pin each worker to one GPU (set CUDA_VISIBLE_DEVICES *before* importing torch in train code)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Optional: less contention on the small dataloader pool
    os.environ.setdefault("OMP_NUM_THREADS", "2")

    storage = f"sqlite:///{study_dir / 'study.db'}"
    study = optuna.load_study(
        study_name=args.study_name,
        storage=storage,
    )
    objective = make_objective(
        base_cfg_path=Path(args.base_config),
        study_dir=study_dir,
        max_epochs=args.max_epochs,
        early_stop_patience=args.patience,
    )
    n = math.ceil(args.n_trials / args.n_workers)
    print(f"[worker {worker_id}] starting, target {n} trials, gpu={args.gpu}", flush=True)
    study.optimize(objective, n_trials=n, gc_after_trial=True,
                   show_progress_bar=False, catch=())
    print(f"[worker {worker_id}] done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-name", required=True)
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--n-trials",  type=int, default=100,
                    help="Total trials across all workers")
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience",   type=int, default=30)
    ap.add_argument("--gpu",        type=str, default="0",
                    help="CUDA_VISIBLE_DEVICES value for every worker")
    ap.add_argument("--base-config", type=str,
                    default=str(HERE / "configs" / "base.yaml"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    study_dir = HERE / "studies" / args.study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "trials").mkdir(exist_ok=True)
    storage = f"sqlite:///{study_dir / 'study.db'}"

    # Create-or-load the study (idempotent)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        sampler=TPESampler(seed=args.seed, multivariate=True, group=True,
                           n_startup_trials=10),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=20, interval_steps=5),
        load_if_exists=True,
    )
    # Snapshot launch metadata for reproducibility
    meta = {
        "study_name": args.study_name,
        "n_workers":  args.n_workers,
        "n_trials":   args.n_trials,
        "max_epochs": args.max_epochs,
        "patience":   args.patience,
        "gpu":        args.gpu,
        "base_config": args.base_config,
        "seed":       args.seed,
        "started":    time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (study_dir / "launch_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    # Sanity-check base config exists
    load_base_cfg(Path(args.base_config))

    print(f"Study      : {args.study_name}")
    print(f"Storage    : {storage}")
    print(f"Workers    : {args.n_workers}  × ~{math.ceil(args.n_trials/args.n_workers)} trials each")
    print(f"Max epochs : {args.max_epochs}    Patience : {args.patience}")
    print(f"GPU        : {args.gpu}")
    print(f"Dashboard  : optuna-dashboard {storage}")

    # Spawn workers (use spawn — CUDA-safe)
    ctx = mp.get_context("spawn")
    procs = []
    for wid in range(args.n_workers):
        p = ctx.Process(target=_worker, args=(wid, args, study_dir))
        p.start()
        procs.append(p)
        time.sleep(2)   # stagger CUDA init

    exit_code = 0
    for p in procs:
        p.join()
        if p.exitcode != 0:
            exit_code = p.exitcode

    # Final summary
    try:
        best = study.best_trial
        print(f"\n=== BEST trial #{best.number}  val_loss={best.value:.4f} ===")
        for k, v in best.params.items():
            print(f"  {k:>18} = {v}")
    except ValueError:
        print("No completed trials.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
