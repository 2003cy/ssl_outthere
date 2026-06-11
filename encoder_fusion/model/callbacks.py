"""Training callbacks for encoder_fusion (EpochPrinter).

LR scheduling (warmup + cosine) lives in MultimodalFusionModule.configure_optimizers
(single definition, mirroring encoder_spectrum/LowResPT) — there is no LR callback.

Run artifacts (metrics.csv, hparams.yaml, checkpoints/) follow the standard
Lightning CSVLogger convention used by encoder_spectrum/LowResPT: configure a
CSVLogger with ``save_dir`` + ``name`` and Lightning writes everything under
``<save_dir>/<name>/version_N/``. ModelCheckpoint with no explicit ``dirpath``
nests its ``checkpoints/`` in that same version directory.
"""

from lightning.pytorch.callbacks import Callback


class EpochPrinter(Callback):
    """Print one plain line per epoch instead of a live progress bar.

    tqdm / RichProgressBar redraw in place via ANSI cursor codes. When training
    is launched as a subprocess (`!python trainer.py ...` in a notebook), stdout
    is a pipe and the notebook's output renderer does NOT honor those codes, so
    every refresh piles up instead of overwriting. A plain `print` with a newline
    needs no cursor control, so it renders cleanly in that exact setting.

    Pair with ``trainer.enable_progress_bar: false`` in the config.
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
        line = (
            f"epoch {trainer.current_epoch:3d} | "
            f"train_loss={self._fmt(m, 'train_loss')} | "
            f"val_loss={self._fmt(m, 'val_loss')} | "
            f"R@1={self._fmt(m, 'val_R@1', nd=3)} | "
            f"R@10={self._fmt(m, 'val_R@10', nd=3)} | "
            f"lr={lr_s}"
        )
        # Append per-pair validation losses (e.g. val_image_spectrum), if any.
        pair_keys = sorted(
            k for k in m if k.startswith("val_") and k != "val_loss" and "_" in k[len("val_"):]
        )
        if pair_keys:
            line += " | " + " | ".join(f"{k}={self._fmt(m, k)}" for k in pair_keys)
        print(line, flush=True)
