"""Render a training-metrics figure (metrics.png) from a Lightning CSVLogger CSV.

Used two ways:
  * imported by the ``PlotMetrics`` callback (callbacks.py), which refreshes
    ``<log_dir>/metrics.png`` at every validation;
  * standalone:  ``python visualize_metrics.py <metrics.csv> [out.png]``.

Same information as the live CSV (the five loss terms + the schedules); just laid
out as a clean 2x3 panel with a smoothed train curve and val markers.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (metric base name, panel title). train_<name> / val_<name> are paired automatically.
_LOSS_PANELS = [
    ("loss", "total loss"),
    ("dino_global", "DINO global (CLS)"),
    ("dino_local", "DINO local (CLS)"),
    ("ibot", "iBOT (patch)"),
    ("koleo", "KoLeo"),
]
_SMOOTH = 80      # rolling window for the (noisy) per-step train curve
_SKIP_FIRST = 10  # drop the first few logging points (very-early warmup spikes);
                   # keep most of the early warmup. Clamped to <=20% of the run below.

_C_TRAIN = "#1f77b4"
_C_VAL = "#d62728"
# per-survey val overlay colors (cycled by detected survey)
_C_SURVEY = ["#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def _detect_surveys(columns) -> list[str]:
    """Survey names X for which a ``val_X_loss`` column exists (per-survey val split)."""
    return sorted({m.group(1) for c in columns
                   if (m := re.fullmatch(r"val_(.+)_loss", c))})


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

    # Validation logs only per-survey val_<survey>_<key>; synthesize the combined
    # val_<key> as the row-mean over the per-survey columns when it isn't logged directly.
    surveys = _detect_surveys(df.columns)
    for key, _ in _LOSS_PANELS:
        vcol = f"val_{key}"
        if vcol not in df.columns and surveys:
            cols = [f"val_{s}_{key}" for s in surveys if f"val_{s}_{key}" in df.columns]
            if cols:
                df[vcol] = df[cols].mean(axis=1, skipna=True)
    if "val_loss" not in df.columns:
        df["val_loss"] = np.nan  # keep downstream dropna()/title robust on train-only CSVs

    # Bottom x-axis: cumulative tokens_seen (config-invariant). It is linear in step, so
    # derive the per-step scale to also place val rows (which don't log tokens_seen). The
    # top x-axis (see _epoch_axis) shows the same range in epochs.
    ref = df.dropna(subset=["tokens_seen", "step"])
    tok_per_step = float((ref["tokens_seen"] / ref["step"].clip(lower=1)).iloc[-1])
    df["_x"] = df["step"] * tok_per_step / 1e9
    x_to_epoch = df["epoch"].iloc[-1] / df["_x"].iloc[-1] if df["_x"].iloc[-1] else 0.0

    out_path = out_path or os.path.join(os.path.dirname(csv_path), "metrics.png")
    tr_all = df.dropna(subset=["train_loss"])
    skip = min(skip_first, len(tr_all) // 5)  # never drop more than ~20% (short runs)
    tr = tr_all.iloc[skip:]
    va = df.dropna(subset=["val_loss"])
    if len(tr) == 0:
        return None
    x_tr = tr["_x"].to_numpy()

    _style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), constrained_layout=True)
    axes = axes.ravel()

    surveys = _detect_surveys(df.columns)
    for ax, (key, title) in zip(axes, _LOSS_PANELS):
        tcol, vcol = f"train_{key}", f"val_{key}"
        if tcol in tr.columns:
            ax.plot(x_tr, tr[tcol], color=_C_TRAIN, alpha=0.12, lw=0.8)
            ax.plot(x_tr, tr[tcol].rolling(_SMOOTH, min_periods=5).mean(),
                    color=_C_TRAIN, lw=1.8, label="train")
        if vcol in va.columns and va[vcol].notna().any():
            ax.plot(va["_x"], va[vcol], "o-", color=_C_VAL, ms=4, lw=1.2, label="val")
        for i, s in enumerate(surveys):  # per-survey val overlay (if present)
            sc = f"val_{s}_{key}"
            if sc in va.columns and va[sc].notna().any():
                ax.plot(va["_x"], va[sc], "--", color=_C_SURVEY[i % len(_C_SURVEY)],
                        lw=1.0, marker=".", ms=4, alpha=0.85, label=f"val·{s}")
        ax.set_title(title, pad=22)
        ax.legend(loc="best", fontsize=8)
        ax.tick_params(labelsize=8)

    # schedules in the last panel
    ax = axes[5]
    for col, lab, norm in [("lr", "lr", True),
                           ("teacher_temp", "teacher_temp", False),
                           ("mom", "momentum", False)]:
        if col in tr.columns:
            y = tr[col].to_numpy(dtype=float)
            if norm and np.nanmax(y) > 0:
                y, lab = y / np.nanmax(y), "lr (norm)"
            ax.plot(x_tr, y, lw=1.5, label=lab)
    ax.set_title("schedules", pad=22)
    ax.legend(loc="best", fontsize=8)
    ax.tick_params(labelsize=8)

    for ax in axes:
        ax.set_xlabel("tokens seen (x1e9)", fontsize=9)
        if x_to_epoch:  # top axis: same range in epochs
            sec = ax.secondary_xaxis("top", functions=(lambda x: x * x_to_epoch,
                                                        lambda e: e / x_to_epoch))
            sec.set_xlabel("epoch", fontsize=8)
            sec.tick_params(labelsize=7)

    epoch = int(df["epoch"].max()) if "epoch" in df.columns else -1
    step = int(tr["step"].iloc[-1])
    vlast = float(va["val_loss"].iloc[-1]) if len(va) else float("nan")
    name = run_name or os.path.basename(os.path.dirname(os.path.abspath(csv_path)))
    fig.suptitle(f"{name}    |    epoch {epoch}  ·  step {step}  ·  val_loss {vlast:.3f}",
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
