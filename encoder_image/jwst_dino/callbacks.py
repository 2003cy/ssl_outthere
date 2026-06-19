"""Lightweight Lightning callbacks for JWST_DINO training."""

import os

import torch
from lightning.pytorch.callbacks import Callback


class EpochPrinter(Callback):
    """Print one plain line per validation instead of a live progress bar.

    Same rationale as LowResPT/callbacks.py: renders cleanly when training is
    launched as a subprocess (notebook / srun) where ANSI cursor codes pile up.
    """

    @staticmethod
    def _fmt(metrics, key, nd=4):
        v = metrics.get(key)
        if v is None:
            return "  n/a "
        try:
            return f"{v.item():.{nd}f}"
        except AttributeError:
            return f"{float(v):.{nd}f}"

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        lr = m.get("lr")
        lr_s = f"{lr.item():.2e}" if lr is not None else "n/a"
        print(
            f"epoch {trainer.current_epoch:3d} | step {trainer.global_step:6d} | "
            f"train_loss={self._fmt(m, 'train_loss')} | "
            f"val_loss={self._fmt(m, 'val_loss')} | "
            f"val_dino_global={self._fmt(m, 'val_dino_global')} | "
            f"val_ibot={self._fmt(m, 'val_ibot')} | "
            f"val_koleo={self._fmt(m, 'val_koleo')} | "
            f"lr={lr_s}",
            flush=True,
        )


class ExportTeacherBackbone(Callback):
    """Dump the teacher backbone in the dinov2 'teacher checkpoint' layout.

    Fires at the validation cadence (= the checkpoint cadence). The .pth has the
    {"teacher": state_dict} structure that the existing benchmark /
    compute_embeddings loaders expect.
    """

    def on_validation_epoch_end(self, trainer, pl_module):
        # log_dir is None when logging is suppressed (e.g. fast_dev_run).
        if trainer.sanity_checking or not trainer.is_global_zero or trainer.log_dir is None:
            return
        eval_dir = os.path.join(trainer.log_dir, "eval", str(trainer.global_step))
        os.makedirs(eval_dir, exist_ok=True)
        path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save(pl_module.export_teacher_backbone(), path)
        print(f"[ExportTeacherBackbone] saved {path}", flush=True)
