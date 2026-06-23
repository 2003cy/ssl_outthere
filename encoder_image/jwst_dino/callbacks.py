"""Lightweight Lightning callbacks for JWST_DINO training."""

import os
import re

import torch
from lightning.pytorch.callbacks import Callback

from visualize_metrics import plot_metrics


class EpochPrinter(Callback):
    """Print one plain line per validation instead of a live progress bar.

    Same rationale as LowResPT/callbacks.py: renders cleanly when training is
    launched as a subprocess (notebook / srun) where ANSI cursor codes pile up.

    Validation logs only per-survey ``val_<survey>_<k>`` (see model.validation_step);
    the combined ``val_<k>`` shown here is derived as the mean over surveys.
    """

    @staticmethod
    def _fmt(v, nd=4):
        if v is None:
            return "  n/a "
        try:
            return f"{v.item():.{nd}f}"
        except AttributeError:
            return f"{float(v):.{nd}f}"

    @staticmethod
    def _combined(metrics, base):
        """Mean over per-survey val_<survey>_<base> entries, or plain val_<base>."""
        if (direct := metrics.get(f"val_{base}")) is not None:
            return direct
        pat = re.compile(rf"val_(.+)_{re.escape(base)}$")
        vals = [float(v) for k, v in metrics.items() if pat.fullmatch(k)]
        return sum(vals) / len(vals) if vals else None

    def on_validation_epoch_end(self, trainer, pl_module):
        # Runs on every DDP rank; print only on rank 0 (else the line repeats N_gpu times).
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        m = trainer.callback_metrics
        lr = m.get("lr")
        lr_s = f"{lr.item():.2e}" if lr is not None else "n/a"
        print(
            f"epoch {trainer.current_epoch:3d} | step {trainer.global_step:6d} | "
            f"train_loss={self._fmt(m.get('train_loss'))} | "
            f"val_loss={self._fmt(self._combined(m, 'loss'))} | "
            f"val_dino_global={self._fmt(self._combined(m, 'dino_global'))} | "
            f"val_ibot={self._fmt(self._combined(m, 'ibot'))} | "
            f"val_koleo={self._fmt(self._combined(m, 'koleo'))} | "
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


class PlotMetrics(Callback):
    """Refresh ``<log_dir>/metrics.png`` from the CSVLogger CSV at every validation.

    Flushes the logger first so the just-finished epoch is on disk, then delegates
    the drawing to visualize_metrics.plot_metrics. Wrapped so a plotting hiccup can
    never interrupt training.
    """

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero or trainer.log_dir is None:
            return
        try:
            if trainer.logger is not None:
                trainer.logger.save()  # flush pending rows to metrics.csv
            csv_path = os.path.join(trainer.log_dir, "metrics.csv")
            run_name = os.path.basename(trainer.log_dir.rstrip("/"))
            plot_metrics(csv_path, os.path.join(trainer.log_dir, "metrics.png"), run_name)
        except Exception as e:  # never let plotting kill a run
            print(f"[PlotMetrics] skipped ({type(e).__name__}: {e})", flush=True)
