#!/usr/bin/env python
"""Optuna hyperparameter search for ma_specformer on JDA PRISM/F100LP spectra.

Run from the ma_specformer directory:
    python optuna_train.py
    python optuna_train.py --n-trials 50 --epochs 60
    python optuna_train.py --study-name my_study   # resumes if DB exists

Config loading order per trial:
    ma_specformer_jda_low_res.yaml  (base config)
    _optuna.yaml  (or a derived copy)  (HPO overrides: epochs, callbacks, warmup)
    --key=value CLI args            (per-trial: model arch, lr, masking, regularization)

LR scaling:
    Per-trial lr → --optimizer.init_args.lr.
    WarmupCosineLR reads it as base_lr when base_lr=null.  min_lr = base_lr * 0.1.
    warmup_steps is scaled proportionally to epochs (REF: 400 steps / 200 epochs).
"""

import argparse
import copy
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import optuna
import yaml

BASE_DIR    = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
BASE_CFG    = BASE_DIR / "ma_specformer_jda_low_res.yaml"
HPO_CFG     = BASE_DIR / "_optuna.yaml"

REF_EPOCHS       = 200
REF_WARMUP_STEPS = 400


def _scale_warmup(trial_epochs: int) -> int:
    return max(20, round(REF_WARMUP_STEPS * trial_epochs / REF_EPOCHS))


def _build_hpo_config(trial_epochs: int) -> Path:
    """Return path to an HPO config with warmup_steps scaled for trial_epochs.

    Returns HPO_CFG directly when epochs match its default (40).
    Otherwise writes a temporary YAML and returns its path; the caller
    must delete it after the study finishes.
    """
    with open(HPO_CFG) as f:
        hpo = yaml.safe_load(f)

    warmup = _scale_warmup(trial_epochs)
    patience = max(5, trial_epochs // 4)
    val_freq = max(1, trial_epochs // 10)

    hpo["trainer"]["max_epochs"] = trial_epochs
    hpo["trainer"]["check_val_every_n_epoch"] = val_freq
    for cb in hpo["trainer"]["callbacks"]:
        cp = cb.get("class_path", "")
        ia = cb.setdefault("init_args", {})
        if "WarmupCosineLR" in cp:
            ia["warmup_steps"] = warmup
        elif "EarlyStopping" in cp:
            ia["patience"] = patience

    # Check if any values actually differ from _optuna.yaml defaults
    with open(HPO_CFG) as f:
        original = yaml.safe_load(f)
    if hpo == original:
        return HPO_CFG

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", dir=BASE_DIR, delete=False, prefix="_optuna_run_"
    )
    yaml.dump(hpo, tmp, default_flow_style=False, allow_unicode=True)
    tmp.close()
    return Path(tmp.name)


def _build_cli_overrides(trial: optuna.Trial) -> list[str]:
    """Return per-trial --key=value CLI args for LightningCLI."""

    # ── Architecture ──────────────────────────────────────────────────────────
    embed_dim  = trial.suggest_categorical("embed_dim",  [32, 64])
    num_heads  = trial.suggest_categorical("num_heads",  [4, 8])
    num_layers = trial.suggest_int("num_layers", 4, 10, step=2)

    # ── Learning rate (log-uniform; WarmupCosineLR derives min_lr = lr * 0.1) ─
    lr = trial.suggest_float("lr", 5e-6, 5e-4, log=True)

    # ── Masking ───────────────────────────────────────────────────────────────
    mask_ratio      = trial.suggest_float("mask_ratio",      0.20, 0.60)
    line_prominence = trial.suggest_float("line_prominence", 0.50, 2.50)

    # ── Regularization ────────────────────────────────────────────────────────
    dropout      = trial.suggest_float("dropout",      0.00, 0.20)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.10, log=True)

    # ── CLS auxiliary loss ────────────────────────────────────────────────────
    cls_aux_weight = trial.suggest_float("cls_aux_weight", 0.0, 0.30)

    return [
        f"--model.embed_dim={embed_dim}",
        f"--model.num_heads={num_heads}",
        f"--model.num_layers={num_layers}",
        f"--model.mask_ratio={mask_ratio:.4f}",
        f"--model.line_prominence={line_prominence:.4f}",
        f"--model.dropout={dropout:.4f}",
        f"--model.cls_aux_weight={cls_aux_weight:.4f}",
        f"--optimizer.init_args.lr={lr:.3e}",
        f"--optimizer.init_args.weight_decay={weight_decay:.3e}",
    ]


def _best_val_loss(since: float) -> float:
    """Return best val_loss from the metrics file written after `since` (epoch time)."""
    candidates = [
        p for p in OUTPUTS_DIR.glob("metrics_optuna_*.json")
        if p.stat().st_mtime >= since
    ]
    if not candidates:
        print("  [warn] no metrics file found for this trial")
        return float("inf")

    path = max(candidates, key=lambda p: p.stat().st_mtime)
    with open(path) as f:
        history = json.load(f)

    val_losses = [
        entry["metrics"]["val_loss"]
        for entry in history
        if "val_loss" in entry.get("metrics", {})
    ]
    if not val_losses:
        print(f"  [warn] no val_loss entries in {path.name}")
        return float("inf")

    return float(min(val_losses))


def objective(
    trial: optuna.Trial,
    hpo_cfg: Path,
    show_output: bool,
) -> float:
    overrides = _build_cli_overrides(trial)

    def _get(key: str) -> str:
        return next(v.split("=", 1)[1] for v in overrides if v.startswith(f"--{key}="))

    print(
        f"\n[Trial {trial.number:3d}] "
        f"embed={_get('model.embed_dim')} "
        f"heads={_get('model.num_heads')} "
        f"layers={_get('model.num_layers')} | "
        f"lr={_get('optimizer.init_args.lr')} "
        f"wd={_get('optimizer.init_args.weight_decay')} | "
        f"mask={_get('model.mask_ratio')} "
        f"prom={_get('model.line_prominence')} "
        f"drop={_get('model.dropout')} "
        f"cls_w={_get('model.cls_aux_weight')}"
    )

    cmd = [
        sys.executable,
        str(BASE_DIR / "data" / "trainer_jda.py"),
        "fit",
        "--config", str(BASE_CFG),
        "--config", str(hpo_cfg),
        *overrides,
    ]

    t0 = time.time()
    kwargs = {} if show_output else {"stdout": subprocess.DEVNULL, "stderr": subprocess.STDOUT}
    result = subprocess.run(cmd, cwd=str(BASE_DIR), **kwargs)

    if result.returncode != 0:
        print(f"  [error] trainer exited with code {result.returncode}")
        return float("inf")

    val_loss = _best_val_loss(since=t0)
    print(f"  → val_loss = {val_loss:.6f}")
    return val_loss


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for ma_specformer JDA")
    parser.add_argument("--n-trials",    type=int,  default=30)
    parser.add_argument("--epochs",      type=int,  default=40,
                        help="Epochs per trial (default 40 ≈ 20%% of full 200). "
                             "warmup_steps scales as round(400 * epochs / 200).")
    parser.add_argument("--study-name",  default="ma_specformer_jda")
    parser.add_argument("--storage",     default=None,
                        help="Optuna storage URL (default: sqlite in BASE_DIR)")
    parser.add_argument("--n-jobs",      type=int,  default=1,
                        help="Parallel trials (each needs its own GPU)")
    parser.add_argument("--show-output", action="store_true",
                        help="Print trainer stdout/stderr for each trial")
    args = parser.parse_args()

    storage = args.storage or f"sqlite:///{BASE_DIR}/optuna_{args.study_name}.db"
    hpo_cfg = _build_hpo_config(args.epochs)
    tmp_hpo = hpo_cfg if hpo_cfg == HPO_CFG else hpo_cfg  # track for cleanup

    print(f"Study       : {args.study_name}")
    print(f"Storage     : {storage}")
    print(f"Base config : {BASE_CFG.name}")
    print(f"HPO config  : {hpo_cfg.name}")
    print(f"Epochs/trial: {args.epochs}  warmup_steps={_scale_warmup(args.epochs)}")
    print(f"Trials      : {args.n_trials}")

    try:
        study = optuna.create_study(
            direction="minimize",
            storage=storage,
            study_name=args.study_name,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        )

        study.optimize(
            lambda t: objective(t, hpo_cfg, args.show_output),
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
        )
    finally:
        if hpo_cfg != HPO_CFG:
            hpo_cfg.unlink(missing_ok=True)

    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"Best val_loss : {best.value:.6f}")
    print("Best params   :")
    for k, v in best.params.items():
        print(f"  {k:24s} = {v}")

    best_path = BASE_DIR / f"optuna_best_{args.study_name}.json"
    with open(best_path, "w") as f:
        json.dump({"val_loss": best.value, "params": best.params}, f, indent=2)
    print(f"\nSaved → {best_path}")
    print(
        "\nTo launch a full run with best params:\n"
        "  Copy best params into ma_specformer_jda_low_res.yaml,\n"
        "  set max_epochs=200, patience=30, warmup_steps=400."
    )


if __name__ == "__main__":
    main()
