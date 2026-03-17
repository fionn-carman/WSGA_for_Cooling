#!/usr/bin/env python3
"""
Analyse NSGA-II multi-objective sweep results and compare against
the parametric MolPrice threshold sweep.

Produces:
  1. 2x2 Pareto front overlay (NSGA-II vs sweep, per category)
  2. Hypervolume convergence curves
  3. Comparison metrics table (hypervolume, front size, spread, cheap-end coverage)
  4. Molecule grid of NSGA-II Pareto front per category

Usage:
    python analyse_nsga2.py \
        --nsga2_dir ../outputs/nsga2_sweep \
        --sweep_dir ../outputs/stability_molprice_sweep \
        --output_dir ../outputs/nsga2_analysis
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw


# ============================================================
# Hypervolume computation (2D, both objectives maximised)
# ============================================================

def compute_hypervolume_2d(front_points, ref_point):
    """2D hypervolume by sorting + rectangle summation."""
    pts = np.asarray(front_points)
    ref = np.asarray(ref_point)
    mask = np.all(pts > ref, axis=1)
    pts = pts[mask]
    if len(pts) == 0:
        return 0.0
    order = np.argsort(-pts[:, 0])
    pts = pts[order]
    hv = 0.0
    prev_y = ref[1]
    for x, y in pts:
        if y > prev_y:
            hv += (x - ref[0]) * (y - prev_y)
            prev_y = y
    return hv


def extract_pareto_front(fom1, molprice):
    """Extract non-dominated points from FOM1 and -MolPrice."""
    pts = np.column_stack([fom1, -molprice])
    n = len(pts)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i]):
                is_pareto[i] = False
                break
    return is_pareto


# ============================================================
# Data Loading
# ============================================================

CATEGORIES = ['bio_stable', 'bio_unstable', 'nonbio_stable', 'nonbio_unstable']
CATEGORY_LABELS = {
    'bio_stable': 'Biodegradable + Stable',
    'bio_unstable': 'Biodegradable',
    'nonbio_stable': 'Stable',
    'nonbio_unstable': 'Unconstrained',
}
HV_REF_POINT = np.array([0.0, -10.0])


def load_nsga2_results(nsga2_dir):
    """Load NSGA-II results across seeds per category."""
    results = {}
    for cat in CATEGORIES:
        cat_runs = []
        for seed_dir in sorted(glob.glob(os.path.join(nsga2_dir, f"{cat}_seed*"))):
            pareto_path = os.path.join(seed_dir, "final_pareto_front.csv")
            stats_path = os.path.join(seed_dir, "generation_stats.csv")
            all_mols_path = os.path.join(seed_dir, "all_evaluated_molecules.csv")

            run_data = {'seed_dir': seed_dir}

            if os.path.exists(pareto_path):
                run_data['pareto'] = pd.read_csv(pareto_path)
            if os.path.exists(stats_path):
                run_data['stats'] = pd.read_csv(stats_path)
            if os.path.exists(all_mols_path):
                run_data['all_mols'] = pd.read_csv(all_mols_path)

            if 'pareto' in run_data:
                cat_runs.append(run_data)

        results[cat] = cat_runs
    return results


def load_sweep_results(sweep_dir):
    """Load threshold sweep results, merging all runs per category."""
    results = {}
    for cat in CATEGORIES:
        all_mols = []
        for run_dir in sorted(glob.glob(os.path.join(sweep_dir, f"*_{cat}_seed*"))):
            csv_path = os.path.join(run_dir, "all_evaluated_molecules.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                all_mols.append(df)
        if all_mols:
            results[cat] = pd.concat(all_mols, ignore_index=True)
    return results


def merge_pareto_fronts(nsga2_runs, target_col='FOM1_direct_avg'):
    """Merge Pareto fronts from multiple NSGA-II seeds."""
    all_pareto = []
    for run in nsga2_runs:
        all_pareto.append(run['pareto'])
    if not all_pareto:
        return pd.DataFrame()
    merged = pd.concat(all_pareto, ignore_index=True).drop_duplicates(subset='SMILES')

    # Re-extract non-dominated from merged set
    if target_col in merged.columns and 'MolPrice' in merged.columns:
        valid = merged[target_col].notna() & merged['MolPrice'].notna()
        merged_valid = merged[valid].copy()
        is_nd = extract_pareto_front(
            merged_valid[target_col].values,
            merged_valid['MolPrice'].values
        )
        return merged_valid[is_nd].copy()
    return merged


def get_sweep_pareto(sweep_df, target_col='FOM1_direct_avg'):
    """Extract non-dominated front from sweep results."""
    if target_col not in sweep_df.columns or 'MolPrice' not in sweep_df.columns:
        return pd.DataFrame()

    # Filter valid molecules
    valid = sweep_df.copy()
    if 'is_valid' in valid.columns:
        valid = valid[valid['is_valid'] == 1]

    valid = valid[valid[target_col].notna() & valid['MolPrice'].notna()]
    valid = valid.drop_duplicates(subset='SMILES')

    if len(valid) == 0:
        return pd.DataFrame()

    is_nd = extract_pareto_front(
        valid[target_col].values,
        valid['MolPrice'].values
    )
    return valid[is_nd].copy()


# ============================================================
# Plotting
# ============================================================

def plot_pareto_overlay(nsga2_results, sweep_results, output_dir, target_col='FOM1_direct_avg'):
    """2x2 grid: NSGA-II vs sweep Pareto fronts per category."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
    axes = axes.flatten()

    for i, cat in enumerate(CATEGORIES):
        ax = axes[i]

        # Sweep Pareto front
        if cat in sweep_results:
            sweep_pareto = get_sweep_pareto(sweep_results[cat], target_col)
            if len(sweep_pareto) > 0:
                sweep_sorted = sweep_pareto.sort_values('MolPrice')
                ax.scatter(sweep_sorted['MolPrice'], sweep_sorted[target_col],
                          s=15, alpha=0.5, color='C1', label=f'Sweep (n={len(sweep_pareto)})', zorder=2)
                ax.plot(sweep_sorted['MolPrice'], sweep_sorted[target_col],
                       alpha=0.3, color='C1', linewidth=0.8, zorder=1)

        # NSGA-II merged Pareto front
        if cat in nsga2_results and nsga2_results[cat]:
            nsga2_pareto = merge_pareto_fronts(nsga2_results[cat], target_col)
            if len(nsga2_pareto) > 0:
                nsga2_sorted = nsga2_pareto.sort_values('MolPrice')
                ax.scatter(nsga2_sorted['MolPrice'], nsga2_sorted[target_col],
                          s=15, alpha=0.7, color='C0', label=f'NSGA-II (n={len(nsga2_pareto)})', zorder=4)
                ax.plot(nsga2_sorted['MolPrice'], nsga2_sorted[target_col],
                       alpha=0.4, color='C0', linewidth=0.8, zorder=3)

        ax.set_xlabel('MolPrice / log(USD/mmol)')
        ax.set_ylabel(f'{target_col}')
        ax.legend(frameon=False, fontsize=8)
        ax.text(0.02, 0.98, CATEGORY_LABELS[cat], transform=ax.transAxes,
               fontsize=9, fontweight='bold', va='top')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'pareto_overlay.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_hypervolume_convergence(nsga2_results, output_dir):
    """Hypervolume convergence per category (all seeds)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
    axes = axes.flatten()

    for i, cat in enumerate(CATEGORIES):
        ax = axes[i]
        if cat in nsga2_results:
            for run in nsga2_results[cat]:
                if 'stats' in run:
                    stats = run['stats']
                    if 'hypervolume' in stats.columns:
                        seed_label = os.path.basename(run['seed_dir'])
                        ax.plot(stats['generation'], stats['hypervolume'],
                               alpha=0.6, linewidth=1.0, label=seed_label)

        ax.set_xlabel('Generation')
        ax.set_ylabel('Hypervolume')
        ax.legend(frameon=False, fontsize=7)
        ax.text(0.02, 0.98, CATEGORY_LABELS[cat], transform=ax.transAxes,
               fontsize=9, fontweight='bold', va='top')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'hypervolume_convergence.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def compute_comparison_metrics(nsga2_results, sweep_results, target_col='FOM1_direct_avg'):
    """Compute comparison metrics between NSGA-II and sweep."""
    rows = []
    for cat in CATEGORIES:
        row = {'category': cat}

        # NSGA-II metrics
        if cat in nsga2_results and nsga2_results[cat]:
            nsga2_pareto = merge_pareto_fronts(nsga2_results[cat], target_col)
            if len(nsga2_pareto) > 0 and target_col in nsga2_pareto.columns:
                fom_vals = nsga2_pareto[target_col].values
                mp_vals = nsga2_pareto['MolPrice'].values
                pts = np.column_stack([fom_vals, -mp_vals])
                hv = compute_hypervolume_2d(pts, HV_REF_POINT)
                row['nsga2_hv'] = hv
                row['nsga2_front_size'] = len(nsga2_pareto)
                row['nsga2_fom1_max'] = fom_vals.max()
                row['nsga2_fom1_spread'] = fom_vals.max() - fom_vals.min()
                row['nsga2_mp_spread'] = mp_vals.max() - mp_vals.min()
                row['nsga2_cheap_count'] = int((mp_vals < 3).sum())

        # Sweep metrics
        if cat in sweep_results:
            sweep_pareto = get_sweep_pareto(sweep_results[cat], target_col)
            if len(sweep_pareto) > 0 and target_col in sweep_pareto.columns:
                fom_vals = sweep_pareto[target_col].values
                mp_vals = sweep_pareto['MolPrice'].values
                pts = np.column_stack([fom_vals, -mp_vals])
                hv = compute_hypervolume_2d(pts, HV_REF_POINT)
                row['sweep_hv'] = hv
                row['sweep_front_size'] = len(sweep_pareto)
                row['sweep_fom1_max'] = fom_vals.max()
                row['sweep_fom1_spread'] = fom_vals.max() - fom_vals.min()
                row['sweep_mp_spread'] = mp_vals.max() - mp_vals.min()
                row['sweep_cheap_count'] = int((mp_vals < 3).sum())

        rows.append(row)

    return pd.DataFrame(rows)


def save_nsga2_molecule_grids(nsga2_results, output_dir, target_col='FOM1_direct_avg'):
    """Save molecule grid images for NSGA-II Pareto fronts."""
    for cat in CATEGORIES:
        if cat not in nsga2_results or not nsga2_results[cat]:
            continue

        pareto = merge_pareto_fronts(nsga2_results[cat], target_col)
        if len(pareto) == 0:
            continue

        # Sort by FOM1 descending, take top 25
        pareto = pareto.sort_values(target_col, ascending=False).head(25)

        mols, legends = [], []
        for _, row in pareto.iterrows():
            smi = row['SMILES']
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            if mol:
                mols.append(mol)
                short_smi = smi if len(smi) <= 30 else smi[:27] + "..."
                fom1 = row.get(target_col, float('nan'))
                mp = row.get('MolPrice', float('nan'))
                legends.append(f"{short_smi}\nFOM1:{fom1:.2f} MP:{mp:.2f}")

        if mols:
            img = Draw.MolsToGridImage(
                mols, molsPerRow=5, subImgSize=(300, 300),
                legends=legends, useSVG=False, returnPNG=False
            )
            out_path = os.path.join(output_dir, f'nsga2_pareto_mols_{cat}.png')
            img.save(out_path)
            print(f"Saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Analyse NSGA-II vs threshold sweep")
    parser.add_argument("--nsga2_dir", type=str, required=True,
                       help="NSGA-II sweep output directory")
    parser.add_argument("--sweep_dir", type=str, default=None,
                       help="Threshold sweep output directory (optional)")
    parser.add_argument("--output_dir", type=str, default="../outputs/nsga2_analysis",
                       help="Analysis output directory")
    parser.add_argument("--target_col", type=str, default="FOM1_direct_avg",
                       help="FOM1 column to use")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading NSGA-II results...")
    nsga2_results = load_nsga2_results(args.nsga2_dir)
    for cat, runs in nsga2_results.items():
        print(f"  {cat}: {len(runs)} seeds")

    sweep_results = {}
    if args.sweep_dir and os.path.isdir(args.sweep_dir):
        print("\nLoading sweep results...")
        sweep_results = load_sweep_results(args.sweep_dir)
        for cat, df in sweep_results.items():
            print(f"  {cat}: {len(df)} total molecules")

    # Comparison metrics
    print("\nComputing comparison metrics...")
    metrics_df = compute_comparison_metrics(nsga2_results, sweep_results, args.target_col)
    metrics_path = os.path.join(args.output_dir, 'comparison_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")
    print(metrics_df.to_string(index=False))

    # Plots
    print("\nGenerating plots...")
    plot_pareto_overlay(nsga2_results, sweep_results, args.output_dir, args.target_col)
    plot_hypervolume_convergence(nsga2_results, args.output_dir)
    save_nsga2_molecule_grids(nsga2_results, args.output_dir, args.target_col)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
