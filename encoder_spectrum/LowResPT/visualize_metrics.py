"""Render a training-metrics figure (metrics.png) from a Lightning CSVLogger CSV.

Used two ways:
  * imported by the ``PlotMetrics`` callback (callbacks.py), which refreshes
    ``<log_dir>/metrics.png`` every few epochs during training;
  * standalone:  ``python visualize_metrics.py <metrics.csv> [out.png]``.

Same information as the live CSV (the reconstruction MSE terms + the lr schedule +
token counts); just laid out as a clean panel grid with smoothed train curves and
val markers.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (train column, val column or None, panel title). Title format:
# "<description> (<exact logged CSV column name(s)>) [mse]". The [mse] tag marks an
# (unweighted) mean-squared-error term. Caveat: with loss_weighting=invvar both
# train_loss/val_loss and the hid/vis losses are INVERSE-VARIANCE weighted MSE,
# so they are not directly comparable to the unweighted train_full_mse /
# val_hid_mse_unw / line / continuum reference terms.
_LOSS_PANELS = [
    ("train_loss",     "val_loss",     "total loss (train_loss / val_loss) [mse]"),
    ("train_hid_mse",  "val_hid_loss", "hidden tokens (masked recon) (train_hid_mse / val_hid_loss) [mse]"),
    ("train_vis_mse",  "val_vis_loss", "visible tokens (train_vis_mse / val_vis_loss) [mse]"),
    ("train_full_mse", "val_hid_mse_unw", "unweighted recon (train_full_mse / val_hid_mse_unw) [mse]"),
    ("train_line_mse", "val_hid_line_loss", "line region (train_line_mse / val_hid_line_loss) [mse]"),
    ("train_cont_mse", "val_hid_cont_loss", "continuum region (train_cont_mse / val_hid_cont_loss) [mse]"),
]
_SMOOTH = 80      # rolling window for the (noisy) per-step train curve
_SKIP_FIRST = 100  # drop the first logging points (unstable warmup régime);
                   # clamped to <=20% of the run below.

_C_TRAIN = "#1f77b4"
_C_VAL = "#d62728"


def _style() -> None:
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.edgecolor": "#888888",
        "axes.linewidth": 0.8,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "font.size": 9,
        "legend.frameon": False,
    })


def plot_metrics(csv_path: str, out_path: str | None = None, run_name: str | None = None,
                 skip_first: int = _SKIP_FIRST) -> str | None:
    """Write a metrics figure next to ``csv_path`` (or to ``out_path``). Returns the path."""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, OSError):
        return None
    if "step" not in df.columns or "train_loss" not in df.columns:
        return None

    # Lightning logs train and val in separate rows; forward-fill epoch so val rows
    # inherit theirs, and derive a step->epoch scale for the top axis.
    if "epoch" in df.columns:
        df["epoch"] = df["epoch"].ffill()
        ref = df.dropna(subset=["epoch", "step"])
        x_to_epoch = (float(ref["epoch"].iloc[-1]) / float(ref["step"].iloc[-1])
                      if len(ref) and ref["step"].iloc[-1] else 0.0)
    else:
        x_to_epoch = 0.0

    out_path = out_path or os.path.join(os.path.dirname(csv_path), "metrics.png")
    tr_all = df.dropna(subset=["train_loss"]).sort_values("step")
    skip = min(skip_first, len(tr_all) // 5)  # never drop more than ~20% (short runs)
    tr = tr_all.iloc[skip:]
    va = df.dropna(subset=["val_loss"]).sort_values("step")
    if len(tr) == 0:
        return None
    x_tr = tr["step"].to_numpy()

    _style()
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.4), constrained_layout=True)
    axes = axes.ravel()

    for ax, (tcol, vcol, title) in zip(axes, _LOSS_PANELS):
        has_pos = False  # only log-scale panels with positive data (a term may be
                         # identically 0 early in training)
        if tcol in tr.columns and tr[tcol].notna().any():
            ax.plot(x_tr, tr[tcol], color=_C_TRAIN, alpha=0.35, lw=0.9, label="train")
            ax.plot(x_tr, tr[tcol].rolling(_SMOOTH, min_periods=5).mean(),
                    color=_C_TRAIN, lw=1.2, alpha=0.9, label="_nolegend_")
            has_pos |= bool((tr[tcol] > 0).any())
        if vcol and vcol in va.columns and va[vcol].notna().any():
            ax.plot(va["step"], va[vcol], "o-", color=_C_VAL, ms=4, lw=1.2, label="val")
            has_pos |= bool((va[vcol] > 0).any())
        ax.set_title(title, pad=22)
        if has_pos:
            ax.set_yscale("log")  # recon MSE terms span orders of magnitude
        ax.legend(loc="best", fontsize=8)
        ax.tick_params(labelsize=8)

    # lr schedule
    ax = axes[6]
    if "lr" in tr.columns and tr["lr"].notna().any():
        ax.plot(x_tr, tr["lr"].to_numpy(dtype=float), color="#2ca02c", lw=1.5, label="lr")
    ax.set_title("lr schedule (lr)", pad=22)
    ax.legend(loc="best", fontsize=8)
    ax.tick_params(labelsize=8)

    # token counts / masking ratio
    ax = axes[7]
    for col, lab in [("selected_tokens", "selected"), ("hidden_tokens", "hidden"),
                     ("valid_tokens", "valid")]:
        if col in tr.columns and tr[col].notna().any():
            ax.plot(x_tr, tr[col].rolling(_SMOOTH, min_periods=5).mean(), lw=1.4, label=lab)
    ax.set_title("token counts (selected_tokens / hidden_tokens / valid_tokens)", pad=22)
    ax.legend(loc="best", fontsize=8)
    ax.tick_params(labelsize=8)

    for ax in axes:
        ax.set_xlabel("global step", fontsize=9)
        if x_to_epoch:  # top axis: same range in epochs
            sec = ax.secondary_xaxis("top", functions=(lambda x: x * x_to_epoch,
                                                        lambda e: e / x_to_epoch))
            sec.set_xlabel("epoch", fontsize=8)
            sec.tick_params(labelsize=7)

    epoch = int(df["epoch"].max()) if "epoch" in df.columns and df["epoch"].notna().any() else -1
    step = int(tr["step"].iloc[-1])
    vlast = float(va["val_loss"].iloc[-1]) if len(va) else float("nan")
    name = run_name or os.path.basename(os.path.dirname(os.path.abspath(csv_path)))
    fig.suptitle(f"{name}    |    epoch {epoch}  ·  step {step}  ·  val_loss {vlast:.4f}",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python visualize_metrics.py <metrics.csv> [out.png]")
    written = plot_metrics(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"wrote {written}")
