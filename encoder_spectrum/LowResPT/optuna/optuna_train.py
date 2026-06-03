"""Optuna objective for LowResPT hyperparameter search.

One trial = one Lightning training run. Per-trial artifacts (ckpts, metrics,
hparams) are saved under  studies/{study}/trials/trial_{number:04d}/.

Run via launch_study.py (multi-process) or import objective() in a notebook.
"""

from __future__ import annotations

import os
import sys
import json
import yaml
import logging
import traceback
from pathlib import Path
from typing import Any, Dict

# Ensure local model/ and data/ are importable
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import torch
import optuna
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from model.low_res_pt import LowResPT
from data.datamodule import LowResDataModule


# ─────────────────────────────────────────────────────────────────────────────
# Search space — edit freely. Comment lines mark "fixed" knobs we don't tune.
# ─────────────────────────────────────────────────────────────────────────────
def suggest_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    # Architecture. We sweep `head_dim` instead of `num_heads` so that the
    # value space stays fixed across trials (Optuna forbids dynamic
    # categoricals). num_heads is derived: must divide embed_dim cleanly.
    embed_dim  = trial.suggest_categorical("embed_dim",  [128, 192, 256, 384])
    num_layers = trial.suggest_categorical("num_layers", [4, 6, 8, 12])
    head_dim   = trial.suggest_categorical("head_dim",   [16, 32, 64])
    if embed_dim % head_dim != 0:
        # Incompatible combo — prune cheaply rather than error
        raise optuna.TrialPruned(
            f"embed_dim={embed_dim} not divisible by head_dim={head_dim}"
        )
    num_heads = embed_dim // head_dim

    return dict(
        # — model architecture
        embed_dim   = embed_dim,
        num_layers  = num_layers,
        num_heads   = num_heads,
        dropout     = trial.suggest_float("dropout",  0.0, 0.3),
        # — masking / loss
        mask_ratio       = trial.suggest_float("mask_ratio", 0.3, 0.7),
        line_loss_weight = trial.suggest_float("line_loss_weight", 1.0, 10.0, log=True),
        # — optimizer
        lr            = trial.suggest_float("lr",           1e-4, 1e-3, log=True),
        weight_decay  = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
        # — fixed (override here to broaden later)
        # patch_size / stride / mlp_ratio / warmup_steps / min_std left at YAML defaults
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optuna ⇄ Lightning bridge: report val_loss each epoch, allow pruning
# ─────────────────────────────────────────────────────────────────────────────
class OptunaPruningCallback(pl.Callback):
    """Self-contained pruning callback (avoids the version-fragile
    optuna.integration import). Reports val_loss to optuna on epoch end."""

    def __init__(self, trial: optuna.Trial, monitor: str = "val_loss"):
        self.trial   = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        value = float(metric.detach().cpu().item() if hasattr(metric, "detach") else metric)
        epoch = trainer.current_epoch
        self.trial.report(value, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch {epoch} (val_loss={value:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Single trial
# ─────────────────────────────────────────────────────────────────────────────
def run_trial(
    trial: optuna.Trial,
    base_cfg: Dict[str, Any],
    study_dir: Path,
    max_epochs: int = 200,
    early_stop_patience: int = 30,
) -> float:
    hp = suggest_hparams(trial)

    # Build per-trial config (deep merge)
    cfg = {
        "model": {**base_cfg["model"]},
        "data":  {**base_cfg["data"]},
    }
    for k in ("embed_dim", "num_layers", "num_heads", "dropout",
              "mask_ratio", "line_loss_weight", "lr", "weight_decay"):
        cfg["model"][k] = hp[k]

    # Per-trial artifact dir
    trial_dir = study_dir / "trials" / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "hparams.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Per-trial file logger
    log_path = trial_dir / "trial.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    logging.getLogger().addHandler(fh)

    try:
        # Build model / data
        dm = LowResDataModule(**cfg["data"])
        model = LowResPT(
            wl_ref_min=cfg["data"]["wl_ref_min"],
            wl_ref_max=cfg["data"]["wl_ref_max"],
            **cfg["model"],
        )

        ckpt_cb = ModelCheckpoint(
            dirpath=trial_dir / "checkpoints",
            filename="best-{epoch:03d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=False,
        )
        es_cb = EarlyStopping(monitor="val_loss", patience=early_stop_patience, mode="min")
        prune_cb = OptunaPruningCallback(trial, monitor="val_loss")

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu",
            devices=1,                      # one trial = one GPU context
            precision="16-mixed",
            logger=CSVLogger(save_dir=trial_dir, name="", version=""),
            callbacks=[ckpt_cb, es_cb, prune_cb],
            enable_progress_bar=False,
            enable_model_summary=False,
            log_every_n_steps=5,
        )
        trainer.fit(model, datamodule=dm)

        best = float(ckpt_cb.best_model_score.detach().cpu().item())
        # Persist final result for easy grepping
        (trial_dir / "result.json").write_text(json.dumps({
            "trial_number": trial.number,
            "best_val_loss": best,
            "best_ckpt": str(ckpt_cb.best_model_path),
            "hparams": hp,
            "epochs_run": trainer.current_epoch,
        }, indent=2))
        return best

    except optuna.TrialPruned:
        raise
    except (RuntimeError, ValueError) as e:
        # OOM / NaN / config-incompatible → mark trial failed, don't kill worker
        logging.error(f"Trial {trial.number} failed: {e}\n{traceback.format_exc()}")
        (trial_dir / "FAILED.txt").write_text(f"{e}\n{traceback.format_exc()}")
        return float("inf")
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()
        # Release GPU mem between trials in the same worker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: load base config once
# ─────────────────────────────────────────────────────────────────────────────
def load_base_cfg(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def make_objective(base_cfg_path: Path, study_dir: Path,
                   max_epochs: int = 200, early_stop_patience: int = 30):
    base_cfg = load_base_cfg(base_cfg_path)
    def _objective(trial: optuna.Trial) -> float:
        return run_trial(trial, base_cfg, study_dir,
                         max_epochs=max_epochs,
                         early_stop_patience=early_stop_patience)
    return _objective
