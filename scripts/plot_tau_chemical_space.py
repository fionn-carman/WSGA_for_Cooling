"""
Chemical Space Analysis — Tau Sweep (Shared UMAP).

Loads ALL molecules from the tau sweep (all tau values x all seeds),
fits a single shared UMAP embedding with the n-gram reference population,
and generates a 2x4 grid figure showing chemical space coverage per tau.

The shared embedding ensures panels are directly comparable across tau values.

Figures produced:
  (grid) 2x4 coverage grid — one panel per tau value
  (fg)   Side-by-side FG prevalence: reference vs GA (all tau combined)

Requirements:
    pip install umap-learn rdkit-pypi tqdm

Usage:
    python plot_tau_chemical_space.py --sweep_dir ../outputs/tau_sweep
    python plot_tau_chemical_space.py --sweep_dir ../outputs/tau_sweep --no_cache
    python plot_tau_chemical_space.py --npz path/to/tau_umap_cache.npz --out_dir ./figs
"""

import os
import re
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
except ImportError:
    print("ERROR: RDKit required.  Install: conda install -c conda-forge rdkit")
    sys.exit(1)

RDLogger.DisableLog("rdApp.*")

# Block TensorFlow before importing UMAP
import types as _types
_tf_stub = _types.ModuleType("tensorflow")
_tf_stub.__version__ = "0.0.0"
sys.modules.setdefault("tensorflow", _tf_stub)

try:
    import umap.umap_ as umap_module
    UMAP = umap_module.UMAP
except (ImportError, RuntimeError):
    try:
        import umap
        UMAP = umap.UMAP
    except (ImportError, RuntimeError):
        print("ERROR: umap-learn required.  Install: pip install umap-learn")
        sys.exit(1)

# Import shared utilities from plot_chemical_space
from plot_chemical_space import (
    detect_functional_groups, smiles_to_fingerprint, fps_to_numpy,
    load_reference_set, _save_fig, FUNCTIONAL_GROUPS, FG_SHORT_NAMES,
    plot_fg_comparison,
)


# ======================================================================
# Data loading
# ======================================================================

def parse_tau_dir(dirname):
    """Extract tau and seed from directory name."""
    m = re.match(r"tau([\d.]+)_seed(\d+)", dirname)
    if not m:
        return None
    return {"tau": float(m.group(1)), "seed": int(m.group(2))}


def load_all_tau_molecules(sweep_dir, sample_ga=50000):
    """Load all_evaluated_molecules.csv from every tau/seed dir.

    Tags each molecule with its tau value and seed.
    Caps total molecules at sample_ga with stratified sampling by (tau, generation).

    Returns:
        ga_df: DataFrame with SMILES, generation, fitness, tau, seed columns
        tau_values: sorted list of unique tau values found
    """
    frames = []
    for dirname in sorted(os.listdir(sweep_dir)):
        run_path = os.path.join(sweep_dir, dirname)
        if not os.path.isdir(run_path):
            continue
        params = parse_tau_dir(dirname)
        if params is None:
            continue

        eval_path = os.path.join(run_path, "all_evaluated_molecules.csv")
        if not os.path.exists(eval_path):
            continue

        try:
            df = pd.read_csv(eval_path)
            if df.empty:
                continue
            df["tau"] = params["tau"]
            df["seed"] = params["seed"]
            frames.append(df)
        except Exception as e:
            print(f"  WARNING: Could not read {eval_path}: {e}")

    if not frames:
        print("ERROR: No evaluated molecules found in any tau sweep directory.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    # Normalise column names
    if "CanonicalSMILES" in combined.columns and "SMILES" not in combined.columns:
        combined.rename(columns={"CanonicalSMILES": "SMILES"}, inplace=True)

    tau_values = sorted(combined["tau"].unique())
    print(f"  Loaded {len(combined)} molecules across {len(tau_values)} tau values, "
          f"{combined['seed'].nunique()} seeds")

    # Stratified sampling by (tau, generation) to keep balanced representation
    if sample_ga > 0 and len(combined) > sample_ga:
        print(f"  Sampling {sample_ga} from {len(combined)} (stratified by tau + generation)...")
        if "generation" in combined.columns:
            combined["_strat"] = combined["tau"].astype(str) + "_" + combined["generation"].astype(str)
            combined = combined.groupby("_strat", group_keys=False).apply(
                lambda g: g.sample(
                    n=min(len(g), max(1, int(sample_ga * len(g) / len(combined)))),
                    random_state=42
                )
            ).reset_index(drop=True)
            combined.drop(columns=["_strat"], inplace=True)
            if len(combined) > sample_ga:
                combined = combined.sample(n=sample_ga, random_state=42)
        else:
            combined = combined.sample(n=sample_ga, random_state=42)

    print(f"  Final dataset: {len(combined)} molecules")
    for tv in tau_values:
        n = (combined["tau"] == tv).sum()
        print(f"    tau={tv:.2f}: {n} molecules")

    return combined, tau_values


# ======================================================================
# UMAP computation
# ======================================================================

def compute_or_load_umap(ref_smiles, ga_df, cache_path, no_cache=False, n_top=50):
    """Compute shared UMAP across reference + all tau GA molecules.

    Returns:
        coords_2d, labels, generations, fitness, tau_arr, valid_smiles
    """
    # Check cache
    if not no_cache and os.path.exists(cache_path):
        print(f"  Loading cached UMAP from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["coords_2d"],
            data["labels"],
            data["generations"],
            data["fitness"],
            data["tau_values"],
            data["smiles"].tolist(),
        )

    smiles_col = "SMILES"
    fom_col = "FOM1_avg" if "FOM1_avg" in ga_df.columns else "FitnessScore"

    # Identify top molecules per tau
    top_smiles_set = set()
    for tau_val in ga_df["tau"].unique():
        tau_subset = ga_df[ga_df["tau"] == tau_val]
        top_df = tau_subset.nlargest(n_top, fom_col).drop_duplicates(subset=[smiles_col])
        top_smiles_set.update(top_df[smiles_col].dropna().tolist())

    # Build combined SMILES list
    all_smiles = []
    all_labels = []
    all_generations = []
    all_fitness = []
    all_tau = []

    for smi in ref_smiles:
        all_smiles.append(smi)
        all_labels.append("reference")
        all_generations.append(-1)
        all_fitness.append(0)
        all_tau.append(-1.0)

    for _, row in ga_df.iterrows():
        smi = row.get(smiles_col)
        if not isinstance(smi, str):
            continue
        all_smiles.append(smi)
        all_labels.append("explored")
        all_generations.append(row.get("generation", 0))
        all_fitness.append(row.get(fom_col, 0))
        all_tau.append(row.get("tau", -1.0))

    # Top molecules as separate layer
    for smi in top_smiles_set:
        all_smiles.append(smi)
        all_labels.append("top")
        all_generations.append(-1)
        all_fitness.append(0)
        # Find the tau for this top molecule
        match = ga_df[ga_df[smiles_col] == smi]
        if len(match) > 0:
            all_tau.append(match.iloc[0]["tau"])
        else:
            all_tau.append(-1.0)

    # Compute fingerprints
    print(f"  Computing fingerprints for {len(all_smiles)} molecules...")
    fps = []
    valid_indices = []
    for i, smi in enumerate(tqdm(all_smiles, desc="Fingerprinting")):
        fp = smiles_to_fingerprint(smi)
        if fp is not None:
            fps.append(fp)
            valid_indices.append(i)

    labels = np.array([all_labels[i] for i in valid_indices])
    generations = np.array([all_generations[i] for i in valid_indices], dtype=float)
    fitness = np.array([all_fitness[i] for i in valid_indices], dtype=float)
    tau_arr = np.array([all_tau[i] for i in valid_indices], dtype=float)
    valid_smiles = [all_smiles[i] for i in valid_indices]

    print(f"  Valid fingerprints: {len(fps)} / {len(all_smiles)}")
    print(f"    Reference: {(labels == 'reference').sum()}")
    print(f"    Explored:  {(labels == 'explored').sum()}")
    print(f"    Top:       {(labels == 'top').sum()}")

    fp_matrix = fps_to_numpy(fps)

    # Fit UMAP
    print(f"  Fitting UMAP on {fp_matrix.shape[0]} molecules (this may take a few minutes)...")
    reducer = UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric="jaccard",
        random_state=42,
        n_jobs=1,
    )
    coords_2d = reducer.fit_transform(fp_matrix)
    print(f"  UMAP complete: {coords_2d.shape}")

    # Save cache
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez_compressed(
        cache_path,
        coords_2d=coords_2d,
        labels=labels,
        generations=generations,
        fitness=fitness,
        tau_values=tau_arr,
        smiles=np.array(valid_smiles, dtype=object),
    )
    print(f"  Cached UMAP to {cache_path}")

    return coords_2d, labels, generations, fitness, tau_arr, valid_smiles


# ======================================================================
# Figures
# ======================================================================

def plot_tau_coverage_grid(coords_2d, labels, tau_arr, tau_values,
                           out_dir, dpi=300):
    """2x4 grid: one coverage panel per tau value, shared UMAP coordinates."""
    ref_mask = labels == "reference"
    ref_coords = coords_2d[ref_mask]

    # Shared axis limits from full embedding
    x_pad = (coords_2d[:, 0].max() - coords_2d[:, 0].min()) * 0.05
    y_pad = (coords_2d[:, 1].max() - coords_2d[:, 1].min()) * 0.05
    xlim = (coords_2d[:, 0].min() - x_pad, coords_2d[:, 0].max() + x_pad)
    ylim = (coords_2d[:, 1].min() - y_pad, coords_2d[:, 1].max() + y_pad)

    n_tau = len(tau_values)
    ncols = 4
    nrows = (n_tau + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    panel_labels = [chr(ord("a") + i) for i in range(n_tau)]

    for idx, tau_val in enumerate(tau_values):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        # Reference background
        ax.scatter(ref_coords[:, 0], ref_coords[:, 1],
                   c="#DDDDDD", s=3, alpha=0.3, rasterized=True)

        # GA molecules for this tau
        tau_exp_mask = (labels == "explored") & (np.abs(tau_arr - tau_val) < 1e-6)
        if tau_exp_mask.sum() > 0:
            ax.scatter(coords_2d[tau_exp_mask, 0], coords_2d[tau_exp_mask, 1],
                       c="#4C72B0", s=3, alpha=0.15, rasterized=True)

        # Top molecules for this tau
        tau_top_mask = (labels == "top") & (np.abs(tau_arr - tau_val) < 1e-6)
        if tau_top_mask.sum() > 0:
            ax.scatter(coords_2d[tau_top_mask, 0], coords_2d[tau_top_mask, 1],
                       c="red", s=30, alpha=0.9, marker="*", zorder=5)

        n_explored = tau_exp_mask.sum()
        ax.set_title(f"\u03c4 = {tau_val:.2f}  (n={n_explored:,})", fontsize=11)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.text(0.02, 0.98, f"({panel_labels[idx]})", transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left")

    # Hide unused axes
    for idx in range(n_tau, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Chemical Space Coverage by Niching Threshold (\u03c4)",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, out_dir, "tau_coverage_grid", dpi)


def print_coverage_stats_per_tau(coords_2d, labels, tau_arr, tau_values):
    """Print grid-based coverage statistics per tau value."""
    ref_mask = labels == "reference"
    ref_coords = coords_2d[ref_mask]

    x_range = (coords_2d[:, 0].min(), coords_2d[:, 0].max())
    y_range = (coords_2d[:, 1].min(), coords_2d[:, 1].max())
    n_bins = 50

    def grid_cells(coords):
        if len(coords) == 0:
            return set()
        x_bins = np.linspace(x_range[0], x_range[1], n_bins + 1)
        y_bins = np.linspace(y_range[0], y_range[1], n_bins + 1)
        xi = np.clip(np.digitize(coords[:, 0], x_bins) - 1, 0, n_bins - 1)
        yi = np.clip(np.digitize(coords[:, 1], y_bins) - 1, 0, n_bins - 1)
        return set(zip(xi, yi))

    ref_set = grid_cells(ref_coords)

    print(f"\n{'='*70}")
    print(f"  COVERAGE STATISTICS PER TAU ({n_bins}x{n_bins} grid)")
    print(f"{'='*70}")
    print(f"\n  Reference occupies: {len(ref_set)} cells\n")
    print(f"  {'Tau':<8} {'Cells':>8} {'% Grid':>8} {'Overlap':>8} {'Novel':>8}")
    print(f"  {'-'*44}")

    for tau_val in tau_values:
        tau_mask = (labels == "explored") & (np.abs(tau_arr - tau_val) < 1e-6)
        tau_coords = coords_2d[tau_mask]
        tau_set = grid_cells(tau_coords)
        overlap = ref_set & tau_set
        novel = len(tau_set) - len(overlap)
        print(f"  {tau_val:<8.2f} {len(tau_set):>8} {100*len(tau_set)/(n_bins*n_bins):>7.1f}% "
              f"{len(overlap):>8} {novel:>8}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chemical space analysis for the tau sweep (shared UMAP)"
    )
    parser.add_argument("--sweep_dir", type=str, default=None,
                        help="Tau sweep directory (e.g. ../outputs/tau_sweep)")
    parser.add_argument("--npz", type=str, default=None,
                        help="Path to existing .npz UMAP cache — replot only")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory (auto-detected)")
    parser.add_argument("--ref_csv", type=str, default=None,
                        help="Path to reference set CSV")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--n_top", type=int, default=50,
                        help="Top molecules to highlight per tau (default: 50)")
    parser.add_argument("--sample_ga", type=int, default=50000,
                        help="Max GA molecules to load (0=all, default: 50000)")
    parser.add_argument("--cache", type=str, default=None,
                        help="Path to .npz UMAP cache")
    parser.add_argument("--no_cache", action="store_true",
                        help="Force recomputation of UMAP")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for PNG output (default: 300)")
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    if args.data_dir is None:
        args.data_dir = os.path.join(repo_root, "data")

    if args.ref_csv is None:
        args.ref_csv = os.path.join(repo_root, "data", "ngram_reference_10k.csv")

    if args.out_dir is None:
        if args.sweep_dir is not None:
            args.out_dir = os.path.join(args.sweep_dir, "analysis", "chemical_space")
        elif args.npz is not None:
            args.out_dir = os.path.dirname(os.path.abspath(args.npz))
        else:
            args.out_dir = "."

    if args.cache is None:
        args.cache = os.path.join(args.out_dir, "tau_umap_cache.npz")

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Mode A: --npz supplied -> load cache and replot
    # ------------------------------------------------------------------
    if args.npz is not None:
        if not os.path.exists(args.npz):
            print(f"ERROR: {args.npz} not found.")
            sys.exit(1)

        print("=" * 60)
        print("  NPZ MODE: Loading cached UMAP and replotting")
        print("=" * 60)

        data = np.load(args.npz, allow_pickle=True)
        coords_2d = data["coords_2d"]
        labels = data["labels"]
        generations = data["generations"]
        fitness = data["fitness"]
        tau_arr = data["tau_values"]
        valid_smiles = data["smiles"].tolist()

        tau_values = sorted(set(tau_arr[tau_arr >= 0]))
        print(f"  Loaded {len(labels)} points, {len(tau_values)} tau values")

        # Generate figures
        print("\n  --- Tau coverage grid ---")
        plot_tau_coverage_grid(coords_2d, labels, tau_arr, tau_values,
                               args.out_dir, dpi=args.dpi)

        # FG comparison: reference vs all GA
        ref_mask = labels == "reference"
        exp_mask = labels == "explored"
        ref_smi = [s for s, m in zip(valid_smiles, ref_mask) if m]
        ga_smi = [s for s, m in zip(valid_smiles, exp_mask) if m]
        if ref_smi and ga_smi:
            print("\n  --- FG comparison ---")
            plot_fg_comparison(ref_smi, ga_smi, args.out_dir, dpi=args.dpi)

        print_coverage_stats_per_tau(coords_2d, labels, tau_arr, tau_values)
        return

    # ------------------------------------------------------------------
    # Mode B: Full pipeline (--sweep_dir required)
    # ------------------------------------------------------------------
    if args.sweep_dir is None:
        print("ERROR: Either --sweep_dir or --npz is required.")
        sys.exit(1)

    # ==================================================================
    # 1. Load reference set
    # ==================================================================
    print("=" * 60)
    print("  STEP 1: Load reference set")
    print("=" * 60)

    ref_smiles = load_reference_set(args.ref_csv, args.data_dir)

    # ==================================================================
    # 2. Load all tau sweep molecules
    # ==================================================================
    print(f"\n{'='*60}")
    print("  STEP 2: Load tau sweep molecules")
    print("=" * 60)

    ga_df, tau_values = load_all_tau_molecules(args.sweep_dir, sample_ga=args.sample_ga)

    # ==================================================================
    # 3. Compute shared UMAP
    # ==================================================================
    print(f"\n{'='*60}")
    print("  STEP 3: Compute shared UMAP embedding")
    print("=" * 60)

    coords_2d, labels, generations, fitness, tau_arr, valid_smiles = \
        compute_or_load_umap(
            ref_smiles, ga_df, args.cache,
            no_cache=args.no_cache, n_top=args.n_top,
        )

    # ==================================================================
    # 4. Generate figures
    # ==================================================================
    print(f"\n{'='*60}")
    print("  STEP 4: Generate figures")
    print("=" * 60)

    print("\n  --- Tau coverage grid ---")
    plot_tau_coverage_grid(coords_2d, labels, tau_arr, tau_values,
                           args.out_dir, dpi=args.dpi)

    # FG comparison: reference vs all GA combined
    ref_mask = labels == "reference"
    exp_mask = labels == "explored"
    ref_smi = [s for s, m in zip(valid_smiles, ref_mask) if m]
    ga_smi = [s for s, m in zip(valid_smiles, exp_mask) if m]
    if ref_smi and ga_smi:
        print("\n  --- FG comparison: reference vs GA ---")
        plot_fg_comparison(ref_smi, ga_smi, args.out_dir, dpi=args.dpi)

    # ==================================================================
    # 5. Coverage statistics
    # ==================================================================
    print_coverage_stats_per_tau(coords_2d, labels, tau_arr, tau_values)

    print(f"\nAll figures saved to: {args.out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
