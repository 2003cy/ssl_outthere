"""Lightweight Lightning callbacks for LowResPT training."""

import os

from lightning.pytorch.callbacks import Callback

from visualize_metrics import plot_metrics


class EpochPrinter(Callback):
    """Print one plain line per epoch instead of a live progress bar.

    tqdm / RichProgressBar redraw in place via ANSI cursor codes. When training
    is launched as a subprocess (`!python trainer.py ...` in a notebook), stdout
    is a pipe and the notebook's output renderer does NOT honor those codes, so
    every refresh piles up instead of overwriting. A plain `print` with a newline
    needs no cursor control, so it renders cleanly in that exact setting.
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
            f"epoch {trainer.current_epoch:3d} | "
            f"train_loss={self._fmt(m, 'train_loss')} | "
            f"val_loss={self._fmt(m, 'val_loss')} | "
            f"val_hid_loss={self._fmt(m, 'val_hid_loss')} | "
            f"lr={lr_s}",
            flush=True,
        )


class PlotMetrics(Callback):
    """Refresh ``<log_dir>/metrics.png`` from the CSVLogger CSV every PLOT_EVERY_N_EPOCHS epochs.

    Fires on train-epoch end so train curves refresh during training. Flushes the
    logger first, then delegates drawing to visualize_metrics.plot_metrics. Wrapped
    so a plotting hiccup can never interrupt training.
    """

    PLOT_EVERY_N_EPOCHS = 2

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero or trainer.log_dir is None:
            return
        if trainer.current_epoch % self.PLOT_EVERY_N_EPOCHS != 0:
            return
        try:
            if trainer.logger is not None:
                trainer.logger.save()  # flush pending rows to metrics.csv
            csv_path = os.path.join(trainer.log_dir, "metrics.csv")
            run_name = os.path.basename(trainer.log_dir.rstrip("/"))
            plot_metrics(csv_path, os.path.join(trainer.log_dir, "metrics.png"), run_name)
        except Exception as e:  # never let plotting kill a run
            print(f"[PlotMetrics] skipped ({type(e).__name__}: {e})", flush=True)
