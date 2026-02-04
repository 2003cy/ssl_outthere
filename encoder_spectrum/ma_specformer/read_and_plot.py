# Section 1: Import libraries

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Image
import glob
import os

sns.set_theme(style='whitegrid')

def stat_and_plot(metrics_history):
    """Process and plot training metrics from a list of metric dicts.

    Args:
        metrics_history: List of dicts, each containing metrics logged at a training step.
    """
# Section 4: Build DataFrame with all metrics

    rows = []
    for e in metrics_history:
        row = {
            'epoch': e.get('epoch'),
            'global_step': e.get('global_step'),
            'batch_idx': e.get('batch_idx')
        }
        metrics = e.get('metrics', {}) or {}
        # Flatten all metric keys dynamically
        for k, v in metrics.items():
            row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    # ensure numeric and sorted by step
    df = df.sort_values('global_step').reset_index(drop=True)
    print('DataFrame shape:', df.shape)
    print('Available metrics:', [c for c in df.columns if c not in ['epoch', 'global_step', 'batch_idx']])

    # Section 5: Epoch-level aggregation (mean, median, std)

    # Get numeric columns that are actual metrics
    metric_cols = [c for c in df.columns if c not in ['epoch', 'global_step', 'batch_idx']]

    # Build aggregation dict dynamically
    agg_dict = {}
    for col in metric_cols:
        agg_dict[f'{col}_mean'] = (col, 'mean')
        agg_dict[f'{col}_std'] = (col, 'std')

    if agg_dict:
        epoch_summary = df.groupby('epoch').agg(**agg_dict).fillna(np.nan)
        print('Epoch summary:')
        display(epoch_summary)

    # Section 6: Smoothing (moving average and exponential moving average)

    window = max(5, int(len(df) * 0.02))  # adapt window
    alpha = 0.1

    # Apply smoothing to all numeric metric columns
    for col in metric_cols:
        if col in df.columns:
            df[f'{col}_sma'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{col}_ewm'] = df[col].ewm(alpha=alpha).mean()



    # Section 8: Plot epoch-level aggregated statistics

    if not epoch_summary.empty:
        plt.figure(figsize=(8,5))
        epochs = epoch_summary.index.values
        mean_train = epoch_summary['train_loss_mean'].values
        std_train = epoch_summary['train_loss_std'].values
        plt.plot(epochs, mean_train, marker='o', label='train_loss_mean')
        plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, alpha=0.2)
        if 'val_loss_mean' in epoch_summary.columns and not np.all(np.isnan(epoch_summary['val_loss_mean'].values)):
            mean_val = epoch_summary['val_loss_mean'].values
            std_val = epoch_summary['val_loss_std'].values
            plt.plot(epochs, mean_val, marker='o', label='val_loss_mean')
            plt.fill_between(epochs, mean_val - std_val, mean_val + std_val, alpha=0.2)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Epoch-level aggregated loss (mean ± std)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Section 9: Detect anomalies (sudden jumps)
    if 'train_loss' in df.columns:
        df['abs_diff'] = df['train_loss'].diff().abs()
        thr = df['abs_diff'].mean() + 3 * df['abs_diff'].std()
        anomalies = df[df['abs_diff'] > thr][['global_step','train_loss','abs_diff']]
        print(f'Anomaly threshold: {thr:.6f}, found {len(anomalies)} anomalies')


    # New plotting: multiple subplots — each subplot shows all variants for a metric vs global_step

    # Determine metric base names (exclude index/helper cols)
    exclude = {'epoch','global_step','batch_idx','abs_diff'}
    metric_cols = [c for c in df.columns if c not in exclude]
    # normalize col naming to find bases (remove _sma/_ewm/_diff suffixes)
    bases = []
    for c in metric_cols:
        base = c
        for suf in ['_sma','_ewm','_diff']:
            if base.endswith(suf):
                base = base[: -len(suf)]
        if base not in bases:
            bases.append(base)

    if not bases:
        print('No metric columns found for subplots.')
    else:
        n = len(bases)
        fig, axs = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=True)
        if n == 1:
            axs = [axs]

        for ax, base in zip(axs, bases):
            # raw
            if base in df.columns:
                ax.plot(df['global_step'], df[base], color='C0', alpha=0.3, label=f'{base}')
            # sma
            sma_col = f'{base}_sma'
            if sma_col in df.columns:
                ax.plot(df['global_step'], df[sma_col], color='C0', linewidth=2, label=f'{base} SMA')
            # ewm
            ewm_col = f'{base}_ewm'
            if ewm_col in df.columns:
                ax.plot(df['global_step'], df[ewm_col], color='C1', linestyle='--', linewidth=2, label=f'{base} EWM')
            # diff/abs diff
            diff_col = f'{base}_diff'
            if diff_col in df.columns:
                ax_twin = ax.twinx()
                ax_twin.plot(df['global_step'], df[diff_col], color='C3', alpha=0.4, label=f'{base} diff')
                ax_twin.set_ylabel(f'{base} diff')
            # anomalies if computed
            if 'abs_diff' in df.columns:
                anoms = df[df['abs_diff'] > (df['abs_diff'].mean() + 3 * df['abs_diff'].std())]
                if not anoms.empty:
                    ax.scatter(anoms['global_step'], anoms[base].values, color='red', s=30, label='anomaly')

            ax.set_ylabel(base)
            ax.grid(True)
            ax.legend(loc='upper right')

        axs[-1].set_xlabel('global_step')
        plt.tight_layout()
        #out_file = 'training_metrics_subplots.png'
        #plt.savefig(out_file)
        plt.show()
        #display(Image(out_file))
