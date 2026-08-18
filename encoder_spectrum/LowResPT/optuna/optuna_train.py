"""Optuna objective for LowResPT hyperparameter search.

One trial = one Lightning run at the production length. Every
`probe.every_n_epochs` a frozen-encoder redshift probe is fitted on the live
model, averaged over 40 group-aware splits, and its sigma_NMAD is reported to
Optuna for pruning; the trial's objective is the MEDIAN of the last three
reports (minimize).

`val_hid_loss` is deliberately NOT the objective. It still drives checkpointing
and early stopping *inside* a run, it just doesn't rank trials -- reconstruction
loss is dominated by the noise floor and says little about whether redshift is
linearly decodable from the representation.

The pretraining dataset is built ONCE and shared by every trial, so the FITS is
not re-read and re-filtered 40 times.

Run via launch_study.py.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                      # the LowResPT/ package root
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import lightning.pytorch as pl          # noqa: E402
import numpy as np                     # noqa: E402
import optuna                           # noqa: E402
import torch                            # noqa: E402
import yaml                             # noqa: E402
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402
from lightning.pytorch.loggers import CSVLogger                          # noqa: E402

from data.dataset import LowResDataset            # noqa: E402
from data.gpu_datamodule import GPULowResDataModule  # noqa: E402
from eval.run_eval import build_probe_dataset, compute_embeddings, eval_redshift  # noqa: E402
from model.low_res_pt import LowResPT             # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Search space -- nine dimensions. Everything else is fixed in configs/base.yaml;
# README.md carries the justification for each range and each fixed knob.
# ─────────────────────────────────────────────────────────────────────────────
# Width. Shifted DOWN relative to the obvious {128..384} so the grid straddles
# one parameter per training scalar and reaches the under-parameterised side,
# and so the concat(27 x d) readout stays near the 4582 available rows.
EMBED_DIM_CHOICES = [32, 64, 128, 192]
# Head width. 4 is excluded on purpose: scaled dot-product attention over a
# 4-dimensional head is close to noise, and d=192 would give 48 of them.
HEAD_DIM_CHOICES = [8, 16, 32]
NUM_LAYERS_CHOICES = [4, 6, 8, 12]
DROPOUT_RANGE = (0.0, 0.3)
MASK_RATIO_RANGE = (0.30, 0.75)
# log. Measured token sigma^2 (164k tokens, production cuts): p1 0.0037,
# p25 0.075, p50 0.29, p90 1.17, p99 2.97. The current 0.1 clamps 29% of tokens.
# 0.01 clamps 5% (near-pure inverse-variance weighting); 2.0 clamps ~95%, i.e.
# effectively uniform MSE -- so the top of this range contains the "no
# weighting" ablation without spending a separate categorical on it.
ERR_SIGMA_MIN_RANGE = (0.01, 2.0)
LR_RANGE = (1e-4, 1e-3)                # log
WEIGHT_DECAY_RANGE = (1e-5, 1e-1)      # log
# log. A run is ~6300 steps at 21 steps/epoch, so this spans 3% to 32% of it;
# the shipped 1000 is 16%. Strongly coupled to lr, which is why it is searched
# jointly rather than pinned. The upper bound is 2000 rather than 3000 so that
# warmup always ends (epoch 95) well before the pruner's first decision at
# epoch 199 -- otherwise MedianPruner would cut long-warmup trials while their
# learning rate is still ramping.
WARMUP_STEPS_RANGE = (200, 2000)
CONT_PATCH_RANGE = (1, 4)


def suggest_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    embed_dim = trial.suggest_categorical("embed_dim", EMBED_DIM_CHOICES)
    head_dim = trial.suggest_categorical("head_dim", HEAD_DIM_CHOICES)
    # Every (embed_dim, head_dim) pair here divides exactly, so no trial is ever
    # spent on an incompatible combination.
    num_heads = embed_dim // head_dim
    return dict(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=trial.suggest_categorical("num_layers", NUM_LAYERS_CHOICES),
        dropout=trial.suggest_float("dropout", *DROPOUT_RANGE),
        # masking: at patch_size=4/stride=2 adjacent tokens share half their
        # pixels, so a single masked token is largely recoverable from its
        # neighbours -- the block length controls how much is genuinely hidden.
        continuous_patch_length=trial.suggest_int("continuous_patch_length", *CONT_PATCH_RANGE),
        mask_ratio=trial.suggest_float("mask_ratio", *MASK_RATIO_RANGE),
        # inverse-variance floor: how far the loss may concentrate on the
        # best-measured tokens.
        err_weight_sigma_min=trial.suggest_float("err_weight_sigma_min",
                                                 *ERR_SIGMA_MIN_RANGE, log=True),
        lr=trial.suggest_float("lr", *LR_RANGE, log=True),
        weight_decay=trial.suggest_float("weight_decay", *WEIGHT_DECAY_RANGE, log=True),
        warmup_steps=trial.suggest_int("warmup_steps", *WARMUP_STEPS_RANGE, log=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optuna ⇄ Lightning bridge: probe redshift every N epochs, allow pruning
# ─────────────────────────────────────────────────────────────────────────────
class OptunaRedshiftProbe(pl.Callback):
    """Fit a frozen-encoder redshift probe on the live model and report it.

    The objective probe is averaged over `n_splits` group-aware splits, the way
    the stellar-mass and Sersic probes are, so that split-to-split scatter does
    not drive the ranking. The k-NN guard uses a single split -- it is only
    watched for degradation, never optimised, and 40 neighbour searches over a
    3456-dimensional space would cost more than the objective itself.

    The probe is a forward pass under `no_grad` followed by scikit-learn, so
    unlike SpecML's learnable attention-pool probe it needs no escape from
    Lightning's `inference_mode`. The global RNG is still saved and restored so
    that anything added here later cannot shift the training mask stream.
    """

    # The trial score is the median of the last OBJECTIVE_WINDOW reports rather
    # than the best over the run: the best-scoring epoch is a live model that no
    # checkpoint ever saved, and taking a minimum over many reports rewards a
    # trial for being noisy. The median of the final reports describes the state
    # the run actually ends in, which is the checkpoint that gets used.
    OBJECTIVE_WINDOW = 3

    def __init__(self, trial, ds, every_n_epochs=20, split_seeds=(7,), n_jobs=1,
                 probe="lasso", lasso_alpha=None, lasso_kwargs=None, track_knn=True):
        self.trial = trial
        self.ds = ds
        self.every = every_n_epochs
        self.split_seeds = list(split_seeds)
        self.n_jobs = n_jobs
        self.probe = probe
        self.lasso_alpha = lasso_alpha
        self.lasso_kwargs = lasso_kwargs or {}
        self.track_knn = track_knn
        self.history = []

    def objective_value(self) -> float:
        vals = [h[self.probe]["snmad"] for h in self.history]
        if not vals:
            return float("inf")
        return float(np.median(vals[-self.OBJECTIVE_WINDOW:]))

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        if (trainer.current_epoch + 1) % self.every != 0:
            return

        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        was_training = pl_module.training
        pl_module.eval()
        try:
            X, y = compute_embeddings(pl_module, self.ds, device=pl_module.device)
            res = eval_redshift(pl_module, self.ds, pl_module.device, X=X, y=y,
                                split_seeds=self.split_seeds, n_jobs=self.n_jobs,
                                probes=(self.probe,), lasso_alpha=self.lasso_alpha,
                                lasso_kwargs=self.lasso_kwargs)
            entry = dict(res["probes"])
            if self.track_knn:
                kres = eval_redshift(pl_module, self.ds, pl_module.device, X=X, y=y,
                                     split_seeds=self.split_seeds[:1], probes=("knn",))
                entry["knn"] = kres["probes"]["knn"]

            p = entry[self.probe]
            sn = p["snmad"]
            self.history.append({"epoch": trainer.current_epoch, **entry})

            log = {f"ds_z_{self.probe}_snmad": sn,
                   f"ds_z_{self.probe}_snmad_std": p["snmad_std"],
                   f"ds_z_{self.probe}_r2": p["r2"],
                   f"ds_z_{self.probe}_out": p["out"],
                   f"ds_z_{self.probe}_nactive": float(p.get("n_active", 0))}
            if self.track_knn:
                log.update({"ds_z_knn_snmad": entry["knn"]["snmad"],
                            "ds_z_knn_r2": entry["knn"]["r2"]})
            if trainer.logger is not None:
                trainer.logger.log_metrics(log, step=trainer.global_step)

            knn_txt = f"  knn={entry['knn']['snmad']:.4f}" if self.track_knn else ""
            print(f"  [trial {self.trial.number}] epoch {trainer.current_epoch:3d}  "
                  f"{self.probe} sNMAD={sn:.4f}+/-{p['snmad_std']:.4f} "
                  f"R2={p['r2']:.3f}{knn_txt}  (obj={self.objective_value():.4f})",
                  flush=True)

            # Pruning compares the current report against other trials at the
            # same epoch, so every report is still needed even though only the
            # last few enter the trial score.
            self.trial.report(sn, step=trainer.current_epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned(
                    f"pruned at epoch {trainer.current_epoch} (sNMAD={sn:.4f})")
        finally:
            if was_training:
                pl_module.train()
            torch.set_rng_state(rng_cpu)
            if rng_cuda is not None:
                torch.cuda.set_rng_state_all(rng_cuda)


# ─────────────────────────────────────────────────────────────────────────────
# Single trial
# ─────────────────────────────────────────────────────────────────────────────
def run_trial(trial, base_cfg, ds_pt, ds_z, study_dir: Path, max_epochs=None) -> float:
    hp = suggest_hparams(trial)
    model_cfg = {**base_cfg["model"], **hp}
    data_cfg = dict(base_cfg["data"])
    tr_cfg = dict(base_cfg.get("trainer", {}))
    probe_cfg = dict(base_cfg.get("probe", {}))
    max_epochs = max_epochs or tr_cfg.get("max_epochs", 300)

    trial_dir = study_dir / "trials" / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    # Not "hparams.yaml": Lightning's CSVLogger writes its own into this same
    # directory and would clobber it. This one is written before training starts,
    # so it survives a trial that dies in the first epoch.
    (trial_dir / "trial_config.yaml").write_text(
        yaml.safe_dump({"searched": hp, "model": model_cfg, "data": data_cfg},
                       sort_keys=False))

    probe_cb = OptunaRedshiftProbe(
        trial, ds_z,
        every_n_epochs=probe_cfg.get("every_n_epochs", 20),
        split_seeds=range(probe_cfg.get("n_splits", 40)),
        n_jobs=probe_cfg.get("n_jobs", 8),
        probe=probe_cfg.get("probe", "lasso"),
        lasso_alpha=probe_cfg.get("lasso_alpha"),
        lasso_kwargs=probe_cfg.get("lasso_kwargs"),
        track_knn=probe_cfg.get("track_knn", True),
    )

    try:
        dm = GPULowResDataModule(**data_cfg)
        dm.dataset = ds_pt                      # share the prebuilt pretraining dataset
        model = LowResPT(wl_ref_min=data_cfg["wl_ref_min"],
                         wl_ref_max=data_cfg["wl_ref_max"], **model_cfg)

        ckpt_cb = ModelCheckpoint(dirpath=trial_dir / "checkpoints",
                                  filename="best-{epoch:03d}-{val_hid_loss:.4f}",
                                  monitor="val_hid_loss", mode="min",
                                  save_top_k=1, save_last=False)
        es_cb = EarlyStopping(monitor="val_hid_loss", mode="min",
                              patience=tr_cfg.get("patience", 100))

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=tr_cfg.get("accelerator", "gpu"),
            devices=1,
            precision=tr_cfg.get("precision", "bf16-mixed"),
            logger=CSVLogger(save_dir=trial_dir, name="", version=""),
            callbacks=[ckpt_cb, es_cb, probe_cb],
            enable_progress_bar=False,
            enable_model_summary=False,
            log_every_n_steps=tr_cfg.get("log_every_n_steps", 10),
        )
        trainer.fit(model, datamodule=dm)

        if not probe_cb.history:
            raise RuntimeError(
                f"probe never ran: max_epochs={max_epochs} < every_n_epochs="
                f"{probe_cb.every}")

        (trial_dir / "result.json").write_text(json.dumps({
            "trial_number": trial.number,
            "objective": probe_cb.objective_value(),
            "objective_probe": probe_cb.probe,
            "objective_window": probe_cb.OBJECTIVE_WINDOW,
            "n_splits": len(probe_cb.split_seeds),
            "hparams": {k: v for k, v in hp.items()},
            "epochs_run": trainer.current_epoch,
            "best_val_hid_loss": float(ckpt_cb.best_model_score),
            "best_ckpt": str(ckpt_cb.best_model_path),
            "history": probe_cb.history,
        }, indent=2))
        return probe_cb.objective_value()

    except optuna.TrialPruned:
        (trial_dir / "result.json").write_text(json.dumps({
            "trial_number": trial.number, "pruned": True,
            "objective": probe_cb.objective_value(), "hparams": hp,
            "history": probe_cb.history}, indent=2))
        raise
    except Exception as e:
        # TypeError included on purpose: a stale kwarg against the live
        # LowResPT signature is exactly what killed the previous sweep's workers.
        (trial_dir / "FAILED.txt").write_text(f"{e}\n{traceback.format_exc()}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
def load_base_cfg(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_data(base_cfg):
    """Pretraining and probe datasets, built once and reused by every trial."""
    d = base_cfg["data"]
    ds_pt = LowResDataset(
        d["fits_path"], grades=tuple(d.get("grades", (1, 2, 3))),
        min_sn50=d.get("min_sn50"), min_redshift=d.get("min_redshift"),
        max_redshift=d.get("max_redshift"), frac_valid_pix=d.get("frac_valid_pix"),
        wl_ref_min=d["wl_ref_min"], wl_ref_max=d["wl_ref_max"],
        use_jansky=d.get("use_jansky", False), err_column=d.get("err_column", "full_err"))
    ds_z = build_probe_dataset(d["fits_path"], wl_ref_min=d["wl_ref_min"],
                               wl_ref_max=d["wl_ref_max"],
                               use_jansky=d.get("use_jansky", False),
                               err_column=d.get("err_column", "full_err"))
    print(f"pretraining sample {len(ds_pt)} | probe sample {len(ds_z)}", flush=True)
    return ds_pt, ds_z


def make_objective(base_cfg_path: Path, study_dir: Path, max_epochs=None, probe_every=None):
    base_cfg = load_base_cfg(base_cfg_path)
    if probe_every is not None:
        base_cfg.setdefault("probe", {})["every_n_epochs"] = probe_every
    ds_pt, ds_z = build_data(base_cfg)     # FITS read once, shared by all trials

    def _objective(trial: optuna.Trial) -> float:
        return run_trial(trial, base_cfg, ds_pt, ds_z, study_dir, max_epochs=max_epochs)

    return _objective
