"""Lightweight Lightning callbacks for LowResPT training."""

from lightning.pytorch.callbacks import Callback


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
