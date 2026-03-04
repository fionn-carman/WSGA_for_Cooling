"""
FOM1 Parity Plot — Direct Model vs Calculated from Thermophysical Predictions

Creates publication-quality parity plots comparing:
  - Direct FOM1 model predictions (XGBoost trained on FOM1 directly)
  - Calculated FOM1 from 4 separate thermophysical property model predictions

Reads the fom1_comparison.csv produced by train_shared_test.py (--compare_only).

Usage:
    python FOM1_ParityPlot.py --input_dir ./output_shared
    python FOM1_ParityPlot.py --input_dir ./output_shared --output_dir ./FOM1_Plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import argparse
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# Plotting
# ============================================================

def compute_metrics(y_true, y_pred):
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
    }


def create_fom1_parity_plot(df, output_path, figsize=(12, 5.5)):
    """
    Two-panel parity plot (40C and 100C) comparing direct and calculated FOM1.

    Style matches ParityPlot.py: square subplots, k-- parity line, monospace
    metrics box, colored scatter with white edgecolors, clean grid-free look.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    color_direct = '#1f77b4'    # blue
    color_computed = '#d62728'   # red

    for ax_idx, temp in enumerate(['40', '100']):
        true_col = f'FOM1_true_{temp}'
        direct_col = f'FOM1_direct_{temp}'
        computed_col = f'FOM1_computed_{temp}'

        # Drop NaN rows for this temperature
        cols = [true_col, direct_col, computed_col]
        sub = df[cols].dropna()
        y_true = sub[true_col].values
        y_direct = sub[direct_col].values
        y_computed = sub[computed_col].values

        ax = axes[ax_idx]

        # Axis limits
        all_vals = np.concatenate([y_true, y_direct, y_computed])
        min_val = all_vals.min()
        max_val = all_vals.max()
        margin = (max_val - min_val) * 0.08
        plot_min = min_val - margin
        plot_max = max_val + margin

        # Parity line
        ax.plot(
            [plot_min, plot_max], [plot_min, plot_max],
            'k--', linewidth=1.5, zorder=1,
        )

        # Scatter — direct model (filled circles)
        ax.scatter(
            y_true, y_direct,
            c=color_direct, s=40, alpha=0.7,
            edgecolors='white', linewidth=0.5,
            label='Direct', zorder=3,
        )

        # Scatter — calculated from thermo (filled triangles)
        ax.scatter(
            y_true, y_computed,
            c=color_computed, s=40, alpha=0.7,
            edgecolors='white', linewidth=0.5,
            marker='^', label='Calculated', zorder=3,
        )

        # Metrics
        m_direct = compute_metrics(y_true, y_direct)
        m_computed = compute_metrics(y_true, y_computed)

        metrics_text = (
            f"Direct\n"
            f"  R\u00b2   = {m_direct['r2']:.4f}\n"
            f"  RMSE = {m_direct['rmse']:.4f}\n"
            f"  MAE  = {m_direct['mae']:.4f}\n"
            f"\n"
            f"Calculated\n"
            f"  R\u00b2   = {m_computed['r2']:.4f}\n"
            f"  RMSE = {m_computed['rmse']:.4f}\n"
            f"  MAE  = {m_computed['mae']:.4f}"
        )

        ax.text(
            0.05, 0.95, metrics_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                alpha=0.85,
                edgecolor='gray',
            ),
        )

        # Labels
        ax.set_xlabel('FOM1 — Experimental', fontsize=12)
        ax.set_ylabel('FOM1 — Predicted', fontsize=12)
        ax.set_title(f'{temp}\u00b0C', fontsize=13, fontweight='bold')

        ax.legend(loc='lower right', fontsize=9, frameon=False)

        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        ax.set_aspect('equal')
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def create_fom1_single_panel(df, temp, output_path, figsize=(6, 6)):
    """
    Single-panel parity plot for one temperature.
    """
    true_col = f'FOM1_true_{temp}'
    direct_col = f'FOM1_direct_{temp}'
    computed_col = f'FOM1_computed_{temp}'

    cols = [true_col, direct_col, computed_col]
    sub = df[cols].dropna()
    y_true = sub[true_col].values
    y_direct = sub[direct_col].values
    y_computed = sub[computed_col].values

    color_direct = '#1f77b4'
    color_computed = '#d62728'

    fig, ax = plt.subplots(figsize=figsize)

    all_vals = np.concatenate([y_true, y_direct, y_computed])
    min_val = all_vals.min()
    max_val = all_vals.max()
    margin = (max_val - min_val) * 0.08
    plot_min = min_val - margin
    plot_max = max_val + margin

    ax.plot(
        [plot_min, plot_max], [plot_min, plot_max],
        'k--', linewidth=1.5, zorder=1,
    )

    ax.scatter(
        y_true, y_direct,
        c=color_direct, s=40, alpha=0.7,
        edgecolors='white', linewidth=0.5,
        label='Direct', zorder=3,
    )

    ax.scatter(
        y_true, y_computed,
        c=color_computed, s=40, alpha=0.7,
        edgecolors='white', linewidth=0.5,
        marker='^', label='Calculated', zorder=3,
    )

    m_direct = compute_metrics(y_true, y_direct)
    m_computed = compute_metrics(y_true, y_computed)

    metrics_text = (
        f"Direct\n"
        f"  R\u00b2   = {m_direct['r2']:.4f}\n"
        f"  RMSE = {m_direct['rmse']:.4f}\n"
        f"  MAE  = {m_direct['mae']:.4f}\n"
        f"\n"
        f"Calculated\n"
        f"  R\u00b2   = {m_computed['r2']:.4f}\n"
        f"  RMSE = {m_computed['rmse']:.4f}\n"
        f"  MAE  = {m_computed['mae']:.4f}"
    )

    ax.text(
        0.05, 0.95, metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            alpha=0.85,
            edgecolor='gray',
        ),
    )

    ax.set_xlabel('FOM1 — Experimental', fontsize=12)
    ax.set_ylabel('FOM1 — Predicted', fontsize=12)
    ax.set_title(f'FOM1 at {temp}\u00b0C', fontsize=13, fontweight='bold')

    ax.legend(loc='lower right', fontsize=9, frameon=False)

    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_aspect('equal')
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Publication-quality FOM1 parity plots: direct vs calculated'
    )
    parser.add_argument(
        '--input_dir', type=str, default='.',
        help='Directory containing fom1_comparison.csv (output of train_shared_test.py)',
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for plots (defaults to input_dir/FOM1_Plots)',
    )
    parser.add_argument(
        '--single', action='store_true',
        help='Also generate individual single-panel plots for 40C and 100C',
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    csv_path = input_dir / 'fom1_comparison.csv'
    if not csv_path.exists():
        raise FileNotFoundError(
            f'fom1_comparison.csv not found in {input_dir}. '
            f'Run train_shared_test.py --compare_only first.'
        )

    output_dir = Path(args.output_dir) if args.output_dir else input_dir / 'FOM1_Plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f'Loaded {len(df)} molecules from {csv_path}')

    # Two-panel figure (40C + 100C side by side)
    create_fom1_parity_plot(df, output_dir / 'FOM1_parity_direct_vs_calculated.png')

    # Optional single-panel plots
    if args.single:
        create_fom1_single_panel(df, '40', output_dir / 'FOM1_parity_40C.png')
        create_fom1_single_panel(df, '100', output_dir / 'FOM1_parity_100C.png')

    print('Done.')
