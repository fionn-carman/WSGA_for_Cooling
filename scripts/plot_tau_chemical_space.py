"""
Chemical Space Analysis — Tau Comparison (Shared UMAP).

Loads molecules from multiple tau configurations (tau/seed directories),
fits a single shared UMAP embedding with the n-gram reference population,
and generates a grid figure showing chemical space coverage per tau.

The shared embedding ensures panels are directly comparable across tau values.

Figures produced:
  (grid)       NxM coverage grid — one panel per tau value
  (standalone) Individual coverage panels per tau (with invisible colorbars)
  (per-model)  Per-property 2x1 grids: Mahalanobis distance + binary OOD
               for each of the 12 regression models, per tau value

Requirements:
    pip install umap-learn rdkit-pypi tqdm

Usage:
    python plot_tau_chemical_space.py --sweep_dir ../outputs/tau_comparison
    python plot_tau_chemical_space.py --sweep_dir ../outputs/tau_comparison --no_cache
    python plot_tau_chemical_space.py --npz path/to/tau_umap_cache.npz
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
    _robust_axis_limits, _add_invisible_colorbar, _add_side_colorbar,
    PANEL_SIZE,
)


# ======================================================================
# Per-model constants
# ======================================================================

REGRESSION_TARGETS = [
    "BP-Measured", "DC_exp", "Density_100C_g_cm^3", "Density_40C_g_cm^3",
    "flashpoint", "Heat_Capacity_Constant_Pressure_100C_J_K_Mol",
    "Heat_Capacity_Constant_Pressure_40C_J_K_Mol",
    "Kinematic_Viscosity_40C", "Kinematic_Viscosity_100C",
    "MP-Measured", "Thermal_Conductivity_100C", "Thermal_Conductivity_40C",
]

# Human-readable short names for plot titles
MODEL_DISPLAY_NAMES = {
    "BP-Measured": "Boiling Point",
    "DC_exp": "Dielectric Constant",
    "Density_100C_g_cm^3": "Density (100\u00b0C)",
    "Density_40C_g_cm^3": "Density (40\u00b0C)",
    "flashpoint": "Flash Point",
    "Heat_Capacity_Constant_Pressure_100C_J_K_Mol": "Heat Capacity (100\u00b0C)",
    "Heat_Capacity_Constant_Pressure_40C_J_K_Mol": "Heat Capacity (40\u00b0C)",
    "Kinematic_Viscosity_40C": "Kinematic Viscosity (40\u00b0C)",
    "Kinematic_Viscosity_100C": "Kinematic Viscosity (100\u00b0C)",
    "MP-Measured": "Melting Point",
    "Thermal_Conductivity_100C": "Thermal Conductivity (100\u00b0C)",
    "Thermal_Conductivity_40C": "Thermal Conductivity (40\u00b0C)",
}


def _sanitise_target(target):
    """Sanitise target name the same way as compute_mahalanobis.py."""
    return target.replace("^", "").replace("/", "_")


# ======================================================================
# Data loading
# ======================================================================

def parse_tau_dir(dirname):
    """Extract tau and seed from directory name like 'tau0.2_seed42'."""
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
        print(f"  Loading cached UMAP from {os.path.abspath(cache_path)}...")
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
    print(f"  Cached UMAP to {os.path.abspath(cache_path)}")

    return coords_2d, labels, generations, fitness, tau_arr, valid_smiles


# ======================================================================
# Figures
# ======================================================================

def _draw_tau_panel(ax, coords_2d, labels, tau_arr, tau_val):
    """Draw a single tau coverage panel (reference + GA + top)."""
    ref_mask = labels == "reference"
    tau_exp_mask = (labels == "explored") & (np.abs(tau_arr - tau_val) < 1e-6)
    tau_top_mask = (labels == "top") & (np.abs(tau_arr - tau_val) < 1e-6)

    # Reference background
    ax.scatter(coords_2d[ref_mask, 0], coords_2d[ref_mask, 1],
               c="#DDDDDD", s=3, alpha=0.3, label="Reference", rasterized=True)

    # GA molecules for this tau
    n_explored = tau_exp_mask.sum()
    if n_explored > 0:
        ax.scatter(coords_2d[tau_exp_mask, 0], coords_2d[tau_exp_mask, 1],
                   c="#4C72B0", s=3, alpha=0.15,
                   label=f"GA explored ({n_explored:,})", rasterized=True)

    # Top molecules for this tau
    n_top = tau_top_mask.sum()
    if n_top > 0:
        ax.scatter(coords_2d[tau_top_mask, 0], coords_2d[tau_top_mask, 1],
                   c="red", s=30, alpha=0.9, marker="*",
                   label=f"Top {n_top}", zorder=5)

    # Shared axis limits from full embedding (percentile-clipped)
    xlim, ylim = _robust_axis_limits(coords_2d)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.legend(fontsize=12, markerscale=2, frameon=False, loc="upper right")


def plot_tau_coverage_grid(coords_2d, labels, tau_arr, tau_values,
                           out_dir, dpi=300):
    """Grid figure: one coverage panel per tau value, shared UMAP coordinates.

    Adapts layout to number of tau values (1x2, 1x3, 2x4, etc.).
    """
    n_tau = len(tau_values)

    # Choose grid layout
    if n_tau <= 2:
        nrows, ncols = 1, n_tau
    elif n_tau <= 4:
        nrows, ncols = 1, n_tau
    elif n_tau <= 8:
        ncols = 4
        nrows = (n_tau + ncols - 1) // ncols
    else:
        ncols = 4
        nrows = (n_tau + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 7 * nrows))

    # Normalise axes to 2D array
    if n_tau == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    panel_labels = [chr(ord("a") + i) for i in range(n_tau)]

    for idx, tau_val in enumerate(tau_values):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        _draw_tau_panel(ax, coords_2d, labels, tau_arr, tau_val)
        _add_invisible_colorbar(fig, ax)

        ax.text(0.02, 0.98, f"({panel_labels[idx]})", transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top", ha="left")

    # Hide unused axes
    for idx in range(n_tau, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, out_dir, "tau_coverage_grid", dpi)


def plot_tau_standalone_panels(coords_2d, labels, tau_arr, tau_values,
                                out_dir, dpi=300):
    """Individual standalone panel per tau value (with invisible colorbars)."""
    for tau_val in tau_values:
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        _draw_tau_panel(ax, coords_2d, labels, tau_arr, tau_val)
        _add_invisible_colorbar(fig, ax)
        fig.tight_layout()

        tau_str = f"{tau_val:.2f}".replace(".", "")
        _save_fig(fig, out_dir, f"tau_{tau_str}_coverage", dpi)


def _draw_tau_generations_panel(ax, coords_2d, labels, tau_arr, generations,
                                tau_val, fig=None):
    """Draw a generation-coloured panel for a single tau value."""
    ref_mask = labels == "reference"
    tau_exp_mask = (labels == "explored") & (np.abs(tau_arr - tau_val) < 1e-6)
    tau_top_mask = (labels == "top") & (np.abs(tau_arr - tau_val) < 1e-6)

    ax.scatter(coords_2d[ref_mask, 0], coords_2d[ref_mask, 1],
               c="#EEEEEE", s=2, alpha=0.2, rasterized=True)

    gen_vals = generations[tau_exp_mask]
    if len(gen_vals) > 0:
        sc = ax.scatter(coords_2d[tau_exp_mask, 0], coords_2d[tau_exp_mask, 1],
                        c=gen_vals, cmap="plasma", s=3, alpha=0.3, rasterized=True)
        if fig is not None:
            _add_side_colorbar(fig, ax, sc, "Generation")

    if tau_top_mask.sum() > 0:
        ax.scatter(coords_2d[tau_top_mask, 0], coords_2d[tau_top_mask, 1],
                   c="red", s=30, alpha=0.9, marker="*", zorder=5)

    xlim, ylim = _robust_axis_limits(coords_2d)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_tau_generations_grid(coords_2d, labels, tau_arr, generations,
                              tau_values, out_dir, dpi=300):
    """Side-by-side generation-coloured panels, one per tau value."""
    n_tau = len(tau_values)

    if n_tau <= 2:
        nrows, ncols = 1, n_tau
    elif n_tau <= 4:
        nrows, ncols = 1, n_tau
    else:
        ncols = 4
        nrows = (n_tau + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 7 * nrows))

    if n_tau == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    panel_labels = [chr(ord("a") + i) for i in range(n_tau)]

    for idx, tau_val in enumerate(tau_values):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        _draw_tau_generations_panel(ax, coords_2d, labels, tau_arr,
                                    generations, tau_val, fig=fig)

        ax.text(0.02, 0.98, f"({panel_labels[idx]})", transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top", ha="left")

    for idx in range(n_tau, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, out_dir, "tau_generations_grid", dpi)


def _draw_tau_fitness_panel(ax, coords_2d, labels, tau_arr, fitness,
                            tau_val, fig=None):
    """Draw a fitness-landscape panel for a single tau value."""
    ref_mask = labels == "reference"
    tau_exp_mask = (labels == "explored") & (np.abs(tau_arr - tau_val) < 1e-6)
    tau_top_mask = (labels == "top") & (np.abs(tau_arr - tau_val) < 1e-6)

    ax.scatter(coords_2d[ref_mask, 0], coords_2d[ref_mask, 1],
               c="#EEEEEE", s=2, alpha=0.2, rasterized=True)

    fit_vals = fitness[tau_exp_mask]
    valid_fit = fit_vals[fit_vals > 0]
    if len(valid_fit) > 0:
        vmin, vmax = np.percentile(valid_fit, [5, 95])
        sc = ax.scatter(coords_2d[tau_exp_mask, 0], coords_2d[tau_exp_mask, 1],
                        c=fit_vals, cmap="viridis", s=3, alpha=0.3,
                        vmin=vmin, vmax=vmax, rasterized=True)
        if fig is not None:
            _add_side_colorbar(fig, ax, sc, "FOM1 (avg)")

    if tau_top_mask.sum() > 0:
        ax.scatter(coords_2d[tau_top_mask, 0], coords_2d[tau_top_mask, 1],
                   c="red", s=30, alpha=0.9, marker="*", zorder=5)

    xlim, ylim = _robust_axis_limits(coords_2d)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_tau_fitness_grid(coords_2d, labels, tau_arr, fitness,
                          tau_values, out_dir, dpi=300):
    """Side-by-side fitness-landscape panels, one per tau value."""
    n_tau = len(tau_values)

    if n_tau <= 2:
        nrows, ncols = 1, n_tau
    elif n_tau <= 4:
        nrows, ncols = 1, n_tau
    else:
        ncols = 4
        nrows = (n_tau + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 7 * nrows))

    if n_tau == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    panel_labels = [chr(ord("a") + i) for i in range(n_tau)]

    for idx, tau_val in enumerate(tau_values):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        _draw_tau_fitness_panel(ax, coords_2d, labels, tau_arr,
                                fitness, tau_val, fig=fig)

        ax.text(0.02, 0.98, f"({panel_labels[idx]})", transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top", ha="left")

    for idx in range(n_tau, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, out_dir, "tau_fitness_grid", dpi)


def plot_tau_standalone_fitness(coords_2d, labels, tau_arr, fitness,
                                tau_values, out_dir, dpi=300):
    """Individual standalone fitness panels per tau value."""
    for tau_val in tau_values:
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        _draw_tau_fitness_panel(ax, coords_2d, labels, tau_arr,
                                fitness, tau_val, fig=fig)
        fig.tight_layout()

        tau_str = f"{tau_val:.2f}".replace(".", "")
        _save_fig(fig, out_dir, f"tau_{tau_str}_fitness", dpi)


def plot_tau_standalone_generations(coords_2d, labels, tau_arr, generations,
                                    tau_values, out_dir, dpi=300):
    """Individual standalone generation panels per tau value."""
    for tau_val in tau_values:
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        _draw_tau_generations_panel(ax, coords_2d, labels, tau_arr,
                                    generations, tau_val, fig=fig)
        fig.tight_layout()

        tau_str = f"{tau_val:.2f}".replace(".", "")
        _save_fig(fig, out_dir, f"tau_{tau_str}_generations", dpi)


# ======================================================================
# Mahalanobis / OOD panels
# ======================================================================

def load_mahal_data(mahal_csv, valid_smiles, labels, tau_arr):
    """Load Mahalanobis CSV and match OOD_any + mean Mahal to UMAP molecules.

    Returns:
        ood_any : ndarray (n_points,)  — 0/1 for GA molecules, -1 for reference/top
        mahal_mean : ndarray (n_points,) — mean Mahalanobis distance across models,
                     NaN for reference/top
    """
    import re as _re
    df = pd.read_csv(mahal_csv)

    # Normalise SMILES column
    if "CanonicalSMILES" in df.columns and "SMILES" not in df.columns:
        df.rename(columns={"CanonicalSMILES": "SMILES"}, inplace=True)

    # Build lookup: SMILES -> (OOD_any, mean_mahal)
    mahal_cols = [c for c in df.columns if c.startswith("Mahal_")]
    if not mahal_cols:
        print("  WARNING: No Mahal_* columns found in mahal CSV")
        n = len(valid_smiles)
        return np.full(n, -1, dtype=int), np.full(n, np.nan)

    df["_mean_mahal"] = df[mahal_cols].mean(axis=1)
    ood_col = "OOD_any" if "OOD_any" in df.columns else None

    lookup = {}
    for _, row in df.iterrows():
        smi = row.get("SMILES")
        if isinstance(smi, str):
            ood_val = int(row[ood_col]) if ood_col else 0
            mean_m = row["_mean_mahal"]
            lookup[smi] = (ood_val, mean_m)

    n = len(valid_smiles)
    ood_any = np.full(n, -1, dtype=int)
    mahal_mean = np.full(n, np.nan)

    matched = 0
    for i, smi in enumerate(valid_smiles):
        if labels[i] in ("reference", "top"):
            continue
        if smi in lookup:
            ood_any[i], mahal_mean[i] = lookup[smi]
            matched += 1

    print(f"  Matched {matched} / {(labels == 'explored').sum()} explored molecules to Mahalanobis data")
    return ood_any, mahal_mean


def load_mahal_per_model(mahal_csv, valid_smiles, labels, tau_arr):
    """Load per-model Mahalanobis distances and OOD flags from mahal CSV.

    Returns:
        per_model : dict  — keys are target names, values are dicts with:
            'mahal': ndarray (n_points,) — Mahalanobis distance, NaN for ref/top
            'ood':   ndarray (n_points,) — 0/1 for GA molecules, -1 for ref/top
            'display_name': str — human-readable property name
    """
    df = pd.read_csv(mahal_csv)

    # Normalise SMILES column
    if "CanonicalSMILES" in df.columns and "SMILES" not in df.columns:
        df.rename(columns={"CanonicalSMILES": "SMILES"}, inplace=True)

    # Build SMILES -> row index lookup
    smi_to_idx = {}
    for i, row in df.iterrows():
        smi = row.get("SMILES")
        if isinstance(smi, str):
            smi_to_idx[smi] = i

    n = len(valid_smiles)
    per_model = {}
    found_targets = []

    for target in REGRESSION_TARGETS:
        col_safe = _sanitise_target(target)
        mahal_col = f"Mahal_{col_safe}"
        ood_col = f"OOD_{col_safe}"

        if mahal_col not in df.columns:
            continue

        found_targets.append(target)
        mahal_arr = np.full(n, np.nan)
        ood_arr = np.full(n, -1, dtype=int)

        for i, smi in enumerate(valid_smiles):
            if labels[i] in ("reference", "top"):
                continue
            if smi in smi_to_idx:
                row_idx = smi_to_idx[smi]
                mahal_arr[i] = df.at[row_idx, mahal_col]
                if ood_col in df.columns:
                    ood_arr[i] = int(df.at[row_idx, ood_col])

        per_model[target] = {
            "mahal": mahal_arr,
            "ood": ood_arr,
            "display_name": MODEL_DISPLAY_NAMES.get(target, target),
        }

    print(f"  Loaded per-model Mahalanobis data for {len(found_targets)} / {len(REGRESSION_TARGETS)} models")
    return per_model


def _draw_tau_mahal_panel(ax, coords_2d, labels, tau_arr, mahal_mean,
                           tau_val, fig=None):
    """Draw a Mahalanobis distance colorbar panel for a single tau value."""
    ref_mask = labels == "reference"
    tau_exp_mask = (labels == "explored") & (np.abs(tau_arr - tau_val) < 1e-6)
    tau_top_mask = (labels == "top") & (np.abs(tau_arr - tau_val) < 1e-6)

    ax.scatter(coords_2d[ref_mask, 0], coords_2d[ref_mask, 1],
               c="#EEEEEE", s=2, alpha=0.2, rasterized=True)

    mahal_vals = mahal_mean[tau_exp_mask]
    valid_mahal = mahal_vals[np.isfinite(mahal_vals)]
    if len(valid_mahal) > 0:
        vmin, vmax = np.percentile(valid_mahal, [2, 98])
        sc = ax.scatter(coords_2d[tau_exp_mask, 0], coords_2d[tau_exp_mask, 1],
                        c=mahal_vals, cmap="inferno", s=3, alpha=0.3,
                        vmin=vmin, vmax=vmax, rasterized=True)
        if fig is not None:
            _add_side_colorbar(fig, ax, sc, "Mahalanobis Distance")

    if tau_top_mask.sum() > 0:
        ax.scatter(coords_2d[tau_top_mask, 0], coords_2d[tau_top_mask, 1],
                   c="red", s=30, alpha=0.9, marker="*", zorder=5)

    xlim, ylim = _robust_axis_limits(coords_2d)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_tau_ood_panel(ax, coords_2d, labels, tau_arr, ood_any, tau_val):
    """Draw a binary OOD panel for a single tau value."""
    ref_mask = labels == "reference"
    tau_exp_mask = (labels == "explored") & (np.abs(tau_arr - tau_val) < 1e-6)
    tau_top_mask = (labels == "top") & (np.abs(tau_arr - tau_val) < 1e-6)

    ax.scatter(coords_2d[ref_mask, 0], coords_2d[ref_mask, 1],
               c="#EEEEEE", s=2, alpha=0.2, rasterized=True)

    ood_vals = ood_any[tau_exp_mask]
    exp_coords = coords_2d[tau_exp_mask]

    in_mask = ood_vals == 0
    out_mask = ood_vals == 1

    if in_mask.sum() > 0:
        ax.scatter(exp_coords[in_mask, 0], exp_coords[in_mask, 1],
                   c="#2CA02C", s=3, alpha=0.2,
                   label=f"In-domain ({in_mask.sum():,})", rasterized=True)
    if out_mask.sum() > 0:
        ax.scatter(exp_coords[out_mask, 0], exp_coords[out_mask, 1],
                   c="#D64545", s=3, alpha=0.3,
                   label=f"OOD ({out_mask.sum():,})", rasterized=True)

    if tau_top_mask.sum() > 0:
        ax.scatter(coords_2d[tau_top_mask, 0], coords_2d[tau_top_mask, 1],
                   c="red", s=30, alpha=0.9, marker="*",
                   label=f"Top {tau_top_mask.sum()}", zorder=5)

    xlim, ylim = _robust_axis_limits(coords_2d)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=12, markerscale=2, frameon=False, loc="upper right")


def plot_tau_mahal_grid(coords_2d, labels, tau_arr, mahal_mean,
                         tau_values, out_dir, dpi=300):
    """Side-by-side Mahalanobis distance colorbar panels, one per tau value."""
    n_tau = len(tau_values)

    if n_tau <= 4:
        nrows, ncols = 1, n_tau
    else:
        ncols = 4
        nrows = (n_tau + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows))

    if n_tau == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    panel_labels = [chr(ord("a") + i) for i in range(n_tau)]

    for idx, tau_val in enumerate(tau_values):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        _draw_tau_mahal_panel(ax, coords_2d, labels, tau_arr,
                               mahal_mean, tau_val, fig=fig)
        ax.text(0.02, 0.98, f"({panel_labels[idx]})", transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top", ha="left")

    for idx in range(n_tau, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, out_dir, "tau_mahalanobis_grid", dpi)


def plot_tau_ood_grid(coords_2d, labels, tau_arr, ood_any,
                       tau_values, out_dir, dpi=300):
    """Side-by-side binary OOD panels, one per tau value."""
    n_tau = len(tau_values)

    if n_tau <= 4:
        nrows, ncols = 1, n_tau
    else:
        ncols = 4
        nrows = (n_tau + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows))

    if n_tau == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    panel_labels = [chr(ord("a") + i) for i in range(n_tau)]

    for idx, tau_val in enumerate(tau_values):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        _draw_tau_ood_panel(ax, coords_2d, labels, tau_arr, ood_any, tau_val)
        _add_invisible_colorbar(fig, ax)
        ax.text(0.02, 0.98, f"({panel_labels[idx]})", transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top", ha="left")

    for idx in range(n_tau, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, out_dir, "tau_ood_grid", dpi)


def plot_tau_standalone_mahal(coords_2d, labels, tau_arr, mahal_mean,
                               tau_values, out_dir, dpi=300):
    """Individual standalone Mahalanobis distance panels per tau value."""
    for tau_val in tau_values:
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        _draw_tau_mahal_panel(ax, coords_2d, labels, tau_arr,
                               mahal_mean, tau_val, fig=fig)
        fig.tight_layout()

        tau_str = f"{tau_val:.2f}".replace(".", "")
        _save_fig(fig, out_dir, f"tau_{tau_str}_mahalanobis", dpi)


def plot_tau_standalone_ood(coords_2d, labels, tau_arr, ood_any,
                             tau_values, out_dir, dpi=300):
    """Individual standalone OOD panels per tau value."""
    for tau_val in tau_values:
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        _draw_tau_ood_panel(ax, coords_2d, labels, tau_arr, ood_any, tau_val)
        _add_invisible_colorbar(fig, ax)
        fig.tight_layout()

        tau_str = f"{tau_val:.2f}".replace(".", "")
        _save_fig(fig, out_dir, f"tau_{tau_str}_ood", dpi)


# ======================================================================
# Per-model OOD figures (2x1 grids: Mahalanobis + binary OOD)
# ======================================================================

def plot_per_model_ood_figures(coords_2d, labels, tau_arr, per_model,
                               tau_values, out_dir, dpi=300):
    """Generate per-model 2x1 grids (Mahalanobis distance | binary OOD).

    For each tau value and each of the 12 regression models, produces
    a 2x1 figure:
      Left:  Mahalanobis distance coloured by magnitude
      Right: Binary in-domain / OOD classification

    Output files: tau_{tau}_model_{model}_ood.png
    """
    os.makedirs(out_dir, exist_ok=True)

    for tau_val in tau_values:
        tau_str = f"{tau_val:.2f}".replace(".", "")

        for target, mdata in per_model.items():
            col_safe = _sanitise_target(target)
            display_name = mdata["display_name"]
            mahal_arr = mdata["mahal"]
            ood_arr = mdata["ood"]

            fig, (ax_mahal, ax_ood) = plt.subplots(
                1, 2, figsize=(PANEL_SIZE[0] * 2 + 1, PANEL_SIZE[1])
            )

            # --- Left panel: Mahalanobis distance ---
            _draw_per_model_mahal_panel(
                ax_mahal, coords_2d, labels, tau_arr, mahal_arr,
                tau_val, display_name, fig=fig,
            )

            # --- Right panel: binary OOD ---
            _draw_per_model_ood_panel(
                ax_ood, coords_2d, labels, tau_arr, ood_arr,
                tau_val, display_name,
            )
            _add_invisible_colorbar(fig, ax_ood)

            fig.suptitle(f"{display_name} — \u03c4 = {tau_val}",
                         fontsize=16, fontweight="bold", y=1.02)
            fig.tight_layout()

            fname = f"tau_{tau_str}_model_{col_safe}_ood"
            _save_fig(fig, out_dir, fname, dpi)

    n_figs = len(tau_values) * len(per_model)
    print(f"  Saved {n_figs} per-model OOD figures to {out_dir}/")


def _draw_per_model_mahal_panel(ax, coords_2d, labels, tau_arr, mahal_arr,
                                 tau_val, display_name, fig=None):
    """Left panel of per-model 2x1: Mahalanobis distance colormap."""
    ref_mask = labels == "reference"
    tau_exp_mask = (labels == "explored") & (np.abs(tau_arr - tau_val) < 1e-6)
    tau_top_mask = (labels == "top") & (np.abs(tau_arr - tau_val) < 1e-6)

    ax.scatter(coords_2d[ref_mask, 0], coords_2d[ref_mask, 1],
               c="#EEEEEE", s=2, alpha=0.2, rasterized=True)

    mahal_vals = mahal_arr[tau_exp_mask]
    valid_mahal = mahal_vals[np.isfinite(mahal_vals)]
    if len(valid_mahal) > 0:
        vmin, vmax = np.percentile(valid_mahal, [2, 98])
        sc = ax.scatter(coords_2d[tau_exp_mask, 0], coords_2d[tau_exp_mask, 1],
                        c=mahal_vals, cmap="inferno", s=3, alpha=0.3,
                        vmin=vmin, vmax=vmax, rasterized=True)
        if fig is not None:
            _add_side_colorbar(fig, ax, sc, "Mahalanobis Distance")

    if tau_top_mask.sum() > 0:
        ax.scatter(coords_2d[tau_top_mask, 0], coords_2d[tau_top_mask, 1],
                   c="red", s=30, alpha=0.9, marker="*", zorder=5)

    xlim, ylim = _robust_axis_limits(coords_2d)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Mahalanobis Distance", fontsize=13)


def _draw_per_model_ood_panel(ax, coords_2d, labels, tau_arr, ood_arr,
                                tau_val, display_name):
    """Right panel of per-model 2x1: binary in-domain / OOD."""
    ref_mask = labels == "reference"
    tau_exp_mask = (labels == "explored") & (np.abs(tau_arr - tau_val) < 1e-6)
    tau_top_mask = (labels == "top") & (np.abs(tau_arr - tau_val) < 1e-6)

    ax.scatter(coords_2d[ref_mask, 0], coords_2d[ref_mask, 1],
               c="#EEEEEE", s=2, alpha=0.2, rasterized=True)

    ood_vals = ood_arr[tau_exp_mask]
    exp_coords = coords_2d[tau_exp_mask]

    in_mask = ood_vals == 0
    out_mask = ood_vals == 1

    if in_mask.sum() > 0:
        ax.scatter(exp_coords[in_mask, 0], exp_coords[in_mask, 1],
                   c="#2CA02C", s=3, alpha=0.2,
                   label=f"In-domain ({in_mask.sum():,})", rasterized=True)
    if out_mask.sum() > 0:
        ax.scatter(exp_coords[out_mask, 0], exp_coords[out_mask, 1],
                   c="#D64545", s=3, alpha=0.3,
                   label=f"OOD ({out_mask.sum():,})", rasterized=True)

    if tau_top_mask.sum() > 0:
        ax.scatter(coords_2d[tau_top_mask, 0], coords_2d[tau_top_mask, 1],
                   c="red", s=30, alpha=0.9, marker="*",
                   label=f"Top {tau_top_mask.sum()}", zorder=5)

    xlim, ylim = _robust_axis_limits(coords_2d)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"OOD Classification", fontsize=13)
    ax.legend(fontsize=11, markerscale=2, frameon=False, loc="upper right")


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
        description="Chemical space analysis for tau comparison (shared UMAP)"
    )
    parser.add_argument("--sweep_dir", type=str, default=None,
                        help="Directory containing tau{X}_seed{Y} subdirectories")
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
    parser.add_argument("--mahal_csv", type=str, default=None,
                        help="Path to Mahalanobis-augmented CSV (from compute_mahalanobis.py). "
                             "Enables OOD panels.")
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

    # Output dir priority: npz location > sweep_dir/analysis
    if args.out_dir is None:
        if args.npz is not None:
            args.out_dir = os.path.dirname(os.path.abspath(args.npz))
        elif args.sweep_dir is not None:
            args.out_dir = os.path.join(args.sweep_dir, "analysis", "chemical_space")
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
        tau_arr = data["tau_values"]
        valid_smiles = data["smiles"].tolist()

        tau_values = sorted(set(tau_arr[tau_arr >= 0]))
        print(f"  Loaded {len(labels)} points, {len(tau_values)} tau values")
        for tv in tau_values:
            n = ((labels == "explored") & (np.abs(tau_arr - tv) < 1e-6)).sum()
            print(f"    tau={tv:.2f}: {n} explored molecules")

        print("\n  --- Tau coverage grid ---")
        plot_tau_coverage_grid(coords_2d, labels, tau_arr, tau_values,
                               args.out_dir, dpi=args.dpi)

        print("\n  --- Tau generations grid ---")
        plot_tau_generations_grid(coords_2d, labels, tau_arr, generations,
                                  tau_values, args.out_dir, dpi=args.dpi)

        fitness = data["fitness"] if "fitness" in data else np.zeros(len(labels))
        print("\n  --- Tau fitness grid ---")
        plot_tau_fitness_grid(coords_2d, labels, tau_arr, fitness,
                              tau_values, args.out_dir, dpi=args.dpi)

        print("\n  --- Standalone coverage panels ---")
        plot_tau_standalone_panels(coords_2d, labels, tau_arr, tau_values,
                                   args.out_dir, dpi=args.dpi)

        print("\n  --- Standalone generation panels ---")
        plot_tau_standalone_generations(coords_2d, labels, tau_arr, generations,
                                        tau_values, args.out_dir, dpi=args.dpi)

        print("\n  --- Standalone fitness panels ---")
        plot_tau_standalone_fitness(coords_2d, labels, tau_arr, fitness,
                                    tau_values, args.out_dir, dpi=args.dpi)

        # OOD panels (if Mahalanobis CSV provided)
        if args.mahal_csv is not None:
            if not os.path.exists(args.mahal_csv):
                print(f"\n  WARNING: {args.mahal_csv} not found, skipping OOD panels")
            else:
                print(f"\n  --- Loading Mahalanobis data from {args.mahal_csv} ---")
                ood_any, mahal_mean = load_mahal_data(
                    args.mahal_csv, valid_smiles, labels, tau_arr)

                print("\n  --- Tau Mahalanobis distance grid ---")
                plot_tau_mahal_grid(coords_2d, labels, tau_arr, mahal_mean,
                                     tau_values, args.out_dir, dpi=args.dpi)

                print("\n  --- Tau OOD grid ---")
                plot_tau_ood_grid(coords_2d, labels, tau_arr, ood_any,
                                   tau_values, args.out_dir, dpi=args.dpi)

                print("\n  --- Standalone Mahalanobis panels ---")
                plot_tau_standalone_mahal(coords_2d, labels, tau_arr, mahal_mean,
                                           tau_values, args.out_dir, dpi=args.dpi)

                print("\n  --- Standalone OOD panels ---")
                plot_tau_standalone_ood(coords_2d, labels, tau_arr, ood_any,
                                         tau_values, args.out_dir, dpi=args.dpi)

                # Per-model OOD figures (12 x 2x1 grids per tau)
                print(f"\n  --- Loading per-model Mahalanobis data ---")
                per_model = load_mahal_per_model(
                    args.mahal_csv, valid_smiles, labels, tau_arr)
                if per_model:
                    per_model_dir = os.path.join(args.out_dir, "per_model_ood")
                    print(f"\n  --- Per-model OOD figures ({len(per_model)} models x {len(tau_values)} tau) ---")
                    plot_per_model_ood_figures(
                        coords_2d, labels, tau_arr, per_model,
                        tau_values, per_model_dir, dpi=args.dpi)

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

    print("\n  --- Tau generations grid ---")
    plot_tau_generations_grid(coords_2d, labels, tau_arr, generations,
                              tau_values, args.out_dir, dpi=args.dpi)

    print("\n  --- Tau fitness grid ---")
    plot_tau_fitness_grid(coords_2d, labels, tau_arr, fitness,
                          tau_values, args.out_dir, dpi=args.dpi)

    print("\n  --- Standalone coverage panels ---")
    plot_tau_standalone_panels(coords_2d, labels, tau_arr, tau_values,
                               args.out_dir, dpi=args.dpi)

    print("\n  --- Standalone generation panels ---")
    plot_tau_standalone_generations(coords_2d, labels, tau_arr, generations,
                                    tau_values, args.out_dir, dpi=args.dpi)

    print("\n  --- Standalone fitness panels ---")
    plot_tau_standalone_fitness(coords_2d, labels, tau_arr, fitness,
                                tau_values, args.out_dir, dpi=args.dpi)

    # OOD panels (if Mahalanobis CSV provided)
    if args.mahal_csv is not None:
        if not os.path.exists(args.mahal_csv):
            print(f"\n  WARNING: {args.mahal_csv} not found, skipping OOD panels")
        else:
            print(f"\n  --- Loading Mahalanobis data from {args.mahal_csv} ---")
            ood_any, mahal_mean = load_mahal_data(
                args.mahal_csv, valid_smiles, labels, tau_arr)

            print("\n  --- Tau Mahalanobis distance grid ---")
            plot_tau_mahal_grid(coords_2d, labels, tau_arr, mahal_mean,
                                 tau_values, args.out_dir, dpi=args.dpi)

            print("\n  --- Tau OOD grid ---")
            plot_tau_ood_grid(coords_2d, labels, tau_arr, ood_any,
                               tau_values, args.out_dir, dpi=args.dpi)

            print("\n  --- Standalone Mahalanobis panels ---")
            plot_tau_standalone_mahal(coords_2d, labels, tau_arr, mahal_mean,
                                       tau_values, args.out_dir, dpi=args.dpi)

            print("\n  --- Standalone OOD panels ---")
            plot_tau_standalone_ood(coords_2d, labels, tau_arr, ood_any,
                                     tau_values, args.out_dir, dpi=args.dpi)

            # Per-model OOD figures (12 x 2x1 grids per tau)
            print(f"\n  --- Loading per-model Mahalanobis data ---")
            per_model = load_mahal_per_model(
                args.mahal_csv, valid_smiles, labels, tau_arr)
            if per_model:
                per_model_dir = os.path.join(args.out_dir, "per_model_ood")
                print(f"\n  --- Per-model OOD figures ({len(per_model)} models x {len(tau_values)} tau) ---")
                plot_per_model_ood_figures(
                    coords_2d, labels, tau_arr, per_model,
                    tau_values, per_model_dir, dpi=args.dpi)

    # ==================================================================
    # 5. Coverage statistics
    # ==================================================================
    print_coverage_stats_per_tau(coords_2d, labels, tau_arr, tau_values)

    print(f"\nAll figures saved to: {os.path.abspath(args.out_dir)}/")
    print("Done.")


if __name__ == "__main__":
    main()
