"""
Chemical Space — Step 2: Compare configs on shared UMAP embedding.

Loads the .npz file produced by compute_chemical_space.py and generates:

  1. Per-config UMAP panels (reference + that config's explored molecules)
  2. Grid comparison figure (all configs side by side)
  3. Coverage comparison bar chart
  4. Density contour overlay (all configs on one plot)
  5. Functional group comparison heatmap across configs

Usage:
    python compare_chemical_space.py --input chemical_space_data.npz
    python compare_chemical_space.py --input chemical_space_data.npz --out_dir ../outputs/analysis/chem_space_comparison
"""

import os
import sys
import ast
import argparse
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from collections import Counter

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    print("ERROR: RDKit required. Install: conda install -c conda-forge rdkit")
    sys.exit(1)

RDLogger.DisableLog("rdApp.*")


# ======================================================================
# Functional groups (same as analyse_chemical_space.py)
# ======================================================================

FUNCTIONAL_GROUPS = {
    "Ether (R-O-R)":       Chem.MolFromSmarts("[CX4][OX2][CX4]"),
    "Vinyl ether":         Chem.MolFromSmarts("[CX3]=[CX3][OX2]"),
    "Alcohol (-OH)":       Chem.MolFromSmarts("[OX2H]"),
    "Aldehyde (-CHO)":     Chem.MolFromSmarts("[CX3H1](=O)"),
    "Ketone (C=O)":        Chem.MolFromSmarts("[#6][CX3](=O)[#6]"),
    "Ester (-COO-)":       Chem.MolFromSmarts("[CX3](=O)[OX2][#6]"),
    "Carboxylic acid":     Chem.MolFromSmarts("[CX3](=O)[OX2H]"),
    "Epoxide":             Chem.MolFromSmarts("[OX2r3]"),
    "Alkene (C=C)":        Chem.MolFromSmarts("[CX3]=[CX3]"),
    "Terminal alkene":     Chem.MolFromSmarts("[CX3H2]=[CX3]"),
    "Acetal (R-O-CH-O-R)": Chem.MolFromSmarts("[OX2][CX4][OX2]"),
}


def detect_functional_groups(smi):
    if not isinstance(smi, str):
        return {}
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {}
    counts = {}
    for name, pattern in FUNCTIONAL_GROUPS.items():
        if pattern is not None:
            counts[name] = len(mol.GetSubstructMatches(pattern))
    return counts


# ======================================================================
# Grid coverage utility
# ======================================================================

def grid_coverage(coords, x_range, y_range, n_bins=50):
    """Return set of occupied grid cells."""
    if len(coords) == 0:
        return set()
    x_bins = np.linspace(x_range[0], x_range[1], n_bins + 1)
    y_bins = np.linspace(y_range[0], y_range[1], n_bins + 1)
    x_idx = np.clip(np.digitize(coords[:, 0], x_bins) - 1, 0, n_bins - 1)
    y_idx = np.clip(np.digitize(coords[:, 1], y_bins) - 1, 0, n_bins - 1)
    return set(zip(x_idx.tolist(), y_idx.tolist()))


# ======================================================================
# Figures
# ======================================================================

def plot_grid_comparison(coords_2d, source, generation, fitness, config_labels,
                         out_dir, n_bins=50):
    """Side-by-side UMAP panels for each config, all on the same coordinate system."""
    n_configs = len(config_labels)
    ref_mask = source == -1

    # Layout: 2 rows if > 4 configs
    if n_configs <= 4:
        ncols = n_configs
        nrows = 1
    else:
        ncols = 4
        nrows = (n_configs + 3) // 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                             squeeze=False)

    for ci in range(n_configs):
        row, col = divmod(ci, ncols)
        ax = axes[row][col]
        config_mask = source == ci

        # Reference background
        ax.scatter(coords_2d[ref_mask, 0], coords_2d[ref_mask, 1],
                   c="#DDDDDD", s=1, alpha=0.2, rasterized=True)

        # Explored molecules coloured by generation
        gen_vals = generation[config_mask]
        if len(gen_vals) > 0:
            sc = ax.scatter(coords_2d[config_mask, 0], coords_2d[config_mask, 1],
                            c=gen_vals, cmap="plasma", s=2, alpha=0.25, rasterized=True)

        # Top 50 molecules for this config (by fitness)
        config_fitness = fitness[config_mask]
        if len(config_fitness) > 50:
            top_idx = np.where(config_mask)[0]
            top_50 = top_idx[np.argsort(config_fitness)[-50:]]
            ax.scatter(coords_2d[top_50, 0], coords_2d[top_50, 1],
                       c="red", s=20, alpha=0.9, marker="*", zorder=5)

        ax.set_title(config_labels[ci], fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for ci in range(n_configs, nrows * ncols):
        row, col = divmod(ci, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Chemical Space Exploration by Hyperparameter Configuration",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_grid.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "comparison_grid.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: comparison_grid.png/pdf")


def plot_density_overlay(coords_2d, source, config_labels, out_dir):
    """Density contour overlay of all configs on one plot."""
    ref_mask = source == -1
    n_configs = len(config_labels)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Reference background as scatter
    ax.scatter(coords_2d[ref_mask, 0], coords_2d[ref_mask, 1],
               c="#EEEEEE", s=1, alpha=0.15, rasterized=True, label="Reference")

    # Colour cycle for configs
    cmap = plt.cm.tab10
    colours = [cmap(i / max(n_configs - 1, 1)) for i in range(n_configs)]

    for ci in range(n_configs):
        config_mask = source == ci
        config_coords = coords_2d[config_mask]

        if len(config_coords) < 10:
            continue

        # Scatter with low alpha
        ax.scatter(config_coords[:, 0], config_coords[:, 1],
                   c=[colours[ci]], s=2, alpha=0.1, rasterized=True)

        # Centroid marker
        cx, cy = config_coords[:, 0].mean(), config_coords[:, 1].mean()
        ax.scatter(cx, cy, c=[colours[ci]], s=150, marker="D",
                   edgecolors="black", linewidth=1.5, zorder=10,
                   label=config_labels[ci])

        # Density contour
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(config_coords.T, bw_method=0.3)
            xmin, xmax = coords_2d[:, 0].min(), coords_2d[:, 0].max()
            ymin, ymax = coords_2d[:, 1].min(), coords_2d[:, 1].max()
            xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100),
                                 np.linspace(ymin, ymax, 100))
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contour(xx, yy, zz, levels=3, colors=[colours[ci]],
                       linewidths=1.5, alpha=0.7)
        except Exception:
            pass  # Skip contour if scipy unavailable or KDE fails

    ax.set_title("Density Overlay — All Configurations", fontsize=14)
    ax.legend(fontsize=8, markerscale=1.5, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_density_overlay.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "comparison_density_overlay.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: comparison_density_overlay.png/pdf")


def plot_coverage_comparison(coords_2d, source, config_labels, config_mean_foms,
                             out_dir, n_bins=50):
    """Bar chart comparing grid coverage and overlap for each config."""
    ref_mask = source == -1
    x_range = (coords_2d[:, 0].min(), coords_2d[:, 0].max())
    y_range = (coords_2d[:, 1].min(), coords_2d[:, 1].max())
    total_cells = n_bins * n_bins

    ref_cells = grid_coverage(coords_2d[ref_mask], x_range, y_range, n_bins)

    n_configs = len(config_labels)
    coverages = []
    overlaps = []
    novel_cells = []
    unique_smiles_counts = []

    for ci in range(n_configs):
        config_mask = source == ci
        config_coords = coords_2d[config_mask]
        exp_cells = grid_coverage(config_coords, x_range, y_range, n_bins)

        cov_pct = 100 * len(exp_cells) / total_cells
        overlap_pct = 100 * len(exp_cells & ref_cells) / len(ref_cells) if ref_cells else 0
        novel = len(exp_cells - ref_cells)

        coverages.append(cov_pct)
        overlaps.append(overlap_pct)
        novel_cells.append(novel)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(n_configs)
    short_labels = [f"Config {i}" for i in range(n_configs)]

    # Panel 1: Coverage and overlap
    bar_width = 0.35
    bars1 = ax1.bar(x - bar_width/2, coverages, bar_width, label="Total coverage (%)",
                    color="#4C72B0", alpha=0.8)
    bars2 = ax1.bar(x + bar_width/2, overlaps, bar_width, label="Overlap with ref (%)",
                    color="#55A868", alpha=0.8)

    # Add reference line
    ref_cov = 100 * len(ref_cells) / total_cells
    ax1.axhline(y=ref_cov, color="gray", linestyle="--", alpha=0.5,
                label=f"Reference coverage ({ref_cov:.1f}%)")

    ax1.set_xlabel("Configuration", fontsize=11)
    ax1.set_ylabel("Percentage", fontsize=11)
    ax1.set_title("Grid Coverage Comparison", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax1.legend(fontsize=9)

    # Panel 2: Coverage vs mean FOM (scatter)
    ax2.scatter(coverages, config_mean_foms, s=100, c="#4C72B0", edgecolors="black",
                zorder=5)
    for i, (cov, fom) in enumerate(zip(coverages, config_mean_foms)):
        ax2.annotate(f"C{i}", (cov, fom), fontsize=8, ha="center",
                     xytext=(0, 8), textcoords="offset points")

    ax2.set_xlabel("Grid Coverage (%)", fontsize=11)
    ax2.set_ylabel("Mean Best FOM1", fontsize=11)
    ax2.set_title("Coverage vs Performance", fontsize=13)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_coverage.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "comparison_coverage.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: comparison_coverage.png/pdf")

    # Print table
    print(f"\n  {'Config':<8} {'Label':<40} {'Coverage%':>10} {'RefOverlap%':>12} {'Novel cells':>12} {'MeanFOM1':>10}")
    print(f"  {'-'*92}")
    for ci in range(n_configs):
        print(f"  C{ci:<6} {config_labels[ci]:<40} {coverages[ci]:>9.1f}% {overlaps[ci]:>11.1f}% {novel_cells[ci]:>12} {config_mean_foms[ci]:>10.4f}")


def plot_functional_group_comparison(smiles, source, config_labels, out_dir):
    """Heatmap of functional group prevalence for each config (top 50 molecules)."""
    n_configs = len(config_labels)

    # For each config, get functional group frequencies of top-fitness molecules
    fg_names = list(FUNCTIONAL_GROUPS.keys())
    fg_matrix = np.zeros((n_configs, len(fg_names)))

    for ci in range(n_configs):
        config_mask = source == ci
        config_smiles = smiles[config_mask]

        # Take a sample (up to 500) for functional group analysis
        if len(config_smiles) > 500:
            idx = np.random.RandomState(42).choice(len(config_smiles), 500, replace=False)
            config_smiles = config_smiles[idx]

        n_mols = 0
        total_counts = Counter()
        for smi in config_smiles:
            counts = detect_functional_groups(smi)
            if counts:
                n_mols += 1
                for name, c in counts.items():
                    if c > 0:
                        total_counts[name] += 1

        if n_mols > 0:
            for fi, name in enumerate(fg_names):
                fg_matrix[ci, fi] = total_counts[name] / n_mols

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, max(4, n_configs * 0.6 + 2)))

    im = ax.imshow(fg_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label("Fraction of molecules", fontsize=10)

    ax.set_xticks(range(len(fg_names)))
    ax.set_xticklabels(fg_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_configs))
    ax.set_yticklabels([f"C{i}: {config_labels[i]}" for i in range(n_configs)],
                       fontsize=8)

    # Annotate cells
    for ci in range(n_configs):
        for fi in range(len(fg_names)):
            val = fg_matrix[ci, fi]
            if val > 0.01:
                color = "white" if val > 0.5 else "black"
                ax.text(fi, ci, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    ax.set_title("Functional Group Prevalence by Configuration", fontsize=13)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_functional_groups.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "comparison_functional_groups.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: comparison_functional_groups.png/pdf")


def plot_trajectory_comparison(coords_2d, source, generation, config_labels,
                               out_dir, n_gen_bins=8):
    """Overlay search trajectories (generation centroids) for all configs."""
    ref_mask = source == -1
    n_configs = len(config_labels)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Reference background
    ax.scatter(coords_2d[ref_mask, 0], coords_2d[ref_mask, 1],
               c="#EEEEEE", s=1, alpha=0.15, rasterized=True)

    cmap_configs = plt.cm.tab10
    colours = [cmap_configs(i / max(n_configs - 1, 1)) for i in range(n_configs)]

    for ci in range(n_configs):
        config_mask = source == ci
        config_coords = coords_2d[config_mask]
        config_gen = generation[config_mask]

        if len(config_coords) == 0:
            continue

        max_gen = config_gen.max()
        if max_gen <= 0:
            continue

        bin_edges = np.linspace(0, max_gen, n_gen_bins + 1)

        centroids_x = []
        centroids_y = []

        for b in range(n_gen_bins):
            mask = (config_gen >= bin_edges[b]) & (config_gen < bin_edges[b + 1])
            if mask.sum() == 0:
                continue
            centroids_x.append(config_coords[mask, 0].mean())
            centroids_y.append(config_coords[mask, 1].mean())

        if len(centroids_x) > 1:
            ax.plot(centroids_x, centroids_y, "-", color=colours[ci],
                    linewidth=2, alpha=0.7)

        # Start marker (circle) and end marker (star)
        if centroids_x:
            ax.scatter(centroids_x[0], centroids_y[0], c=[colours[ci]],
                       s=80, marker="o", edgecolors="black", linewidth=1, zorder=10)
            ax.scatter(centroids_x[-1], centroids_y[-1], c=[colours[ci]],
                       s=120, marker="*", edgecolors="black", linewidth=1, zorder=10,
                       label=config_labels[ci])

    ax.set_title("Search Trajectories Through Chemical Space", fontsize=14)
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left",
              title="Config (circle=start, star=end)", title_fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_trajectories.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "comparison_trajectories.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: comparison_trajectories.png/pdf")


def plot_reference_functional_groups(coords_2d, source, smiles, out_dir):
    """UMAP of reference population coloured by functional group.

    Creates two figures:
      1. Multi-panel: one panel per functional group, coloured by presence/count.
      2. Single panel: each molecule coloured by its *dominant* functional group,
         showing how UMAP clusters correspond to chemical features.
    """
    ref_mask = source == -1
    ref_coords = coords_2d[ref_mask]
    ref_smiles = smiles[ref_mask]
    n_ref = len(ref_smiles)

    if n_ref == 0:
        print("  WARNING: No reference molecules, skipping FG reference map")
        return

    print(f"  Detecting functional groups for {n_ref} reference molecules...")

    fg_names = list(FUNCTIONAL_GROUPS.keys())
    # Matrix: (n_ref, n_fg) — count of each FG per molecule
    fg_counts = np.zeros((n_ref, len(fg_names)), dtype=np.int32)
    # Boolean presence
    fg_present = np.zeros((n_ref, len(fg_names)), dtype=bool)

    for i, smi in enumerate(ref_smiles):
        counts = detect_functional_groups(smi)
        for fi, name in enumerate(fg_names):
            c = counts.get(name, 0)
            fg_counts[i, fi] = c
            fg_present[i, fi] = c > 0

    # ------------------------------------------------------------------
    # Figure 1: Multi-panel — one panel per functional group
    # ------------------------------------------------------------------
    # Only show FGs that are present in at least 1% of molecules
    active_fgs = [(fi, name) for fi, name in enumerate(fg_names)
                  if fg_present[:, fi].mean() > 0.01]

    if not active_fgs:
        print("  WARNING: No functional groups detected in reference set")
        return

    n_panels = len(active_fgs)
    ncols = min(4, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                             squeeze=False)

    for idx, (fi, name) in enumerate(active_fgs):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        has_fg = fg_present[:, fi]
        pct = 100 * has_fg.mean()

        # Plot molecules without this FG as grey background
        ax.scatter(ref_coords[~has_fg, 0], ref_coords[~has_fg, 1],
                   c="#DDDDDD", s=3, alpha=0.3, rasterized=True)

        # Plot molecules with this FG, coloured by count
        if has_fg.sum() > 0:
            counts_fg = fg_counts[has_fg, fi]
            sc = ax.scatter(ref_coords[has_fg, 0], ref_coords[has_fg, 1],
                            c=counts_fg, cmap="YlOrRd", s=5, alpha=0.6,
                            vmin=0, vmax=max(counts_fg.max(), 1), rasterized=True)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Count")

        ax.set_title(f"{name} ({pct:.0f}%)", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused panels
    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Reference Population — Functional Group Distribution in Chemical Space",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reference_fg_panels.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "reference_fg_panels.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: reference_fg_panels.png/pdf")

    # ------------------------------------------------------------------
    # Figure 2: Single panel — dominant functional group per molecule
    # ------------------------------------------------------------------
    # Assign each molecule its dominant (most frequent) FG.
    # Ties broken by order in FUNCTIONAL_GROUPS dict. Molecules with
    # no FGs labelled "Hydrocarbon only".
    fg_short_names = {
        "Ether (R-O-R)": "Ether",
        "Vinyl ether": "Vinyl ether",
        "Alcohol (-OH)": "Alcohol",
        "Aldehyde (-CHO)": "Aldehyde",
        "Ketone (C=O)": "Ketone",
        "Ester (-COO-)": "Ester",
        "Carboxylic acid": "Carboxylic acid",
        "Epoxide": "Epoxide",
        "Alkene (C=C)": "Alkene",
        "Terminal alkene": "Terminal alkene",
        "Acetal (R-O-CH-O-R)": "Acetal",
    }

    # Determine dominant FG per molecule
    dominant_labels = []
    for i in range(n_ref):
        max_count = 0
        dominant = "Hydrocarbon only"
        for fi, name in enumerate(fg_names):
            if fg_counts[i, fi] > max_count:
                max_count = fg_counts[i, fi]
                dominant = fg_short_names.get(name, name)
        dominant_labels.append(dominant)
    dominant_labels = np.array(dominant_labels)

    # Get unique labels sorted by frequency
    label_counts = Counter(dominant_labels)
    unique_labels = [lab for lab, _ in label_counts.most_common()]

    # Colour map — use tab20 for up to 20 categories
    cmap = plt.cm.tab20
    label_to_colour = {}
    for i, lab in enumerate(unique_labels):
        if lab == "Hydrocarbon only":
            label_to_colour[lab] = "#BBBBBB"
        else:
            label_to_colour[lab] = cmap(i / max(len(unique_labels) - 1, 1))

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot "Hydrocarbon only" first (background)
    if "Hydrocarbon only" in label_to_colour:
        mask = dominant_labels == "Hydrocarbon only"
        if mask.sum() > 0:
            ax.scatter(ref_coords[mask, 0], ref_coords[mask, 1],
                       c=label_to_colour["Hydrocarbon only"], s=4, alpha=0.3,
                       label=f"Hydrocarbon only ({mask.sum()})", rasterized=True)

    # Plot each FG category
    for lab in unique_labels:
        if lab == "Hydrocarbon only":
            continue
        mask = dominant_labels == lab
        if mask.sum() < 5:
            continue
        ax.scatter(ref_coords[mask, 0], ref_coords[mask, 1],
                   c=[label_to_colour[lab]], s=6, alpha=0.5,
                   label=f"{lab} ({mask.sum()})", rasterized=True)

    ax.set_title("Reference Population — Dominant Functional Group",
                 fontsize=14)
    ax.legend(fontsize=8, markerscale=3, bbox_to_anchor=(1.02, 1),
              loc="upper left", title="Dominant FG (count)")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reference_fg_dominant.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "reference_fg_dominant.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: reference_fg_dominant.png/pdf")


def plot_scaffold_comparison(smiles, source, fitness, config_labels, out_dir,
                             top_n=50):
    """Compare Murcko scaffolds of top molecules across configs."""
    n_configs = len(config_labels)

    # Collect top N scaffolds per config
    config_scaffolds = []
    for ci in range(n_configs):
        config_mask = source == ci
        config_smiles = smiles[config_mask]
        config_fitness = fitness[config_mask]

        if len(config_smiles) == 0:
            config_scaffolds.append(Counter())
            continue

        # Take top_n by fitness
        top_idx = np.argsort(config_fitness)[-top_n:]
        top_smi = config_smiles[top_idx]

        scaffolds = Counter()
        for smi in top_smi:
            if not isinstance(smi, str):
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                scaffold = MurckoScaffold.MakeScaffoldGeneric(
                    MurckoScaffold.GetScaffoldForMol(mol))
                scaffolds[Chem.MolToSmiles(scaffold)] += 1
            except Exception:
                pass
        config_scaffolds.append(scaffolds)

    # Get all unique scaffolds across configs
    all_scaffolds = Counter()
    for sc in config_scaffolds:
        all_scaffolds.update(sc)

    # Top 15 most common scaffolds overall
    top_scaffolds = [s for s, _ in all_scaffolds.most_common(15)]

    if not top_scaffolds:
        print("  WARNING: No scaffolds found, skipping scaffold comparison")
        return

    # Build matrix
    scaffold_matrix = np.zeros((n_configs, len(top_scaffolds)))
    for ci in range(n_configs):
        for si, scaffold in enumerate(top_scaffolds):
            scaffold_matrix[ci, si] = config_scaffolds[ci].get(scaffold, 0)

    fig, ax = plt.subplots(figsize=(max(12, len(top_scaffolds) * 0.8), max(4, n_configs * 0.6 + 2)))

    im = ax.imshow(scaffold_matrix, cmap="Blues", aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label("Count in top molecules", fontsize=10)

    # Shorten scaffold labels
    short_scaffolds = []
    for s in top_scaffolds:
        if len(s) > 20:
            short_scaffolds.append(s[:18] + "..")
        else:
            short_scaffolds.append(s)

    ax.set_xticks(range(len(top_scaffolds)))
    ax.set_xticklabels(short_scaffolds, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n_configs))
    ax.set_yticklabels([f"C{i}: {config_labels[i]}" for i in range(n_configs)],
                       fontsize=8)

    # Annotate non-zero cells
    for ci in range(n_configs):
        for si in range(len(top_scaffolds)):
            val = int(scaffold_matrix[ci, si])
            if val > 0:
                ax.text(si, ci, str(val), ha="center", va="center",
                        fontsize=8, color="white" if val > 2 else "black")

    ax.set_title(f"Murcko Scaffold Distribution (Top {top_n} Molecules per Config)", fontsize=13)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_scaffolds.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "comparison_scaffolds.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: comparison_scaffolds.png/pdf")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare chemical space exploration across hyperparameter configs"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to .npz file from compute_chemical_space.py")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for figures (default: alongside .npz)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading {args.input}...")
    data = np.load(args.input, allow_pickle=True)

    coords_2d = data["coords_2d"]
    source = data["source"]
    generation = data["generation"]
    fitness = data["fitness"]
    smiles = data["smiles"]
    config_labels = data["config_labels"].tolist()
    config_mean_foms = data["config_mean_foms"]

    n_configs = len(config_labels)
    n_total = len(coords_2d)
    n_ref = (source == -1).sum()

    print(f"  {n_total} molecules ({n_ref} reference, {n_total - n_ref} from {n_configs} configs)")
    for ci, label in enumerate(config_labels):
        n = (source == ci).sum()
        print(f"    C{ci}: {label} ({n} molecules, mean FOM1={config_mean_foms[ci]:.4f})")

    # ------------------------------------------------------------------
    # 2. Output directory
    # ------------------------------------------------------------------
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(args.input), "chemical_space_comparison")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nOutput directory: {args.out_dir}")

    # ------------------------------------------------------------------
    # 3. Generate figures
    # ------------------------------------------------------------------
    print(f"\nGenerating comparison figures...")

    plot_grid_comparison(coords_2d, source, generation, fitness,
                         config_labels, args.out_dir)

    plot_density_overlay(coords_2d, source, config_labels, args.out_dir)

    plot_coverage_comparison(coords_2d, source, config_labels,
                             config_mean_foms, args.out_dir)

    plot_trajectory_comparison(coords_2d, source, generation,
                               config_labels, args.out_dir)

    plot_functional_group_comparison(smiles, source, config_labels, args.out_dir)

    plot_reference_functional_groups(coords_2d, source, smiles, args.out_dir)

    plot_scaffold_comparison(smiles, source, fitness, config_labels, args.out_dir)

    print(f"\nAll figures saved to: {args.out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
