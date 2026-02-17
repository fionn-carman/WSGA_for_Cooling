"""
Chemical Space Coverage Analysis.

Visualises how thoroughly the GA explores the accessible chemical space
using UMAP projections of Morgan fingerprints.

Three layers are plotted:
  1. Background: random molecules from the n-gram generator (unoptimised)
  2. Explored: all evaluated molecules from GA runs (coloured by generation)
  3. Top molecules: the highest-scoring individuals (highlighted)

This helps diagnose whether the GA converges to a local optimum by
showing whether it explored broadly before converging, or was always
confined to a narrow region.

Additional figures:
  - Functional group frequency across generations
  - UMAP coloured by fitness score
  - Exploration trajectory (generation centroids over time)

Requirements:
    pip install umap-learn rdkit-pypi

Usage:
    python analyse_chemical_space.py --sweep_dir ../outputs/hyperparam_sweep
    python analyse_chemical_space.py --run_dir ../outputs/hyperparam_sweep/pop2000_mr0.8_er0.3_k3_seed42
    python analyse_chemical_space.py --sweep_dir ../outputs/hyperparam_sweep --n_runs 5
"""

import os
import re
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from collections import Counter

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, Descriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    print("ERROR: RDKit is required. Install with: conda install -c conda-forge rdkit")
    sys.exit(1)

try:
    import umap
except ImportError:
    print("ERROR: umap-learn is required. Install with: pip install umap-learn")
    sys.exit(1)


# ======================================================================
# Fingerprint utilities
# ======================================================================

def smiles_to_fingerprint(smi, radius=2, n_bits=2048):
    """Convert SMILES to Morgan fingerprint bit vector. Returns None on failure."""
    if not isinstance(smi, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None


def fps_to_numpy(fps):
    """Convert list of RDKit fingerprints to numpy array."""
    arr = np.zeros((len(fps), fps[0].GetNumBits()), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


# ======================================================================
# Functional group detection
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
    """Return dict of functional group counts for a SMILES string."""
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
# Data loading
# ======================================================================

def load_evaluated_molecules(run_path, sample_n=None):
    """Load all_evaluated_molecules.csv from a run, optionally sampling."""
    path = os.path.join(run_path, "all_evaluated_molecules.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        if sample_n and len(df) > sample_n:
            df = df.sample(n=sample_n, random_state=42)
        return df
    except Exception:
        return None


def generate_reference_molecules(data_dir, n_molecules=10000):
    """Generate random molecules using the n-gram generator as a reference set.

    Falls back to loading the training SMILES directly if the generator
    module can't be imported.
    """
    # Try importing the generator
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    sys.path.insert(0, src_dir)

    try:
        from generate_molecules import generate_initial_population, load_combined_training_data
        print(f"  Generating {n_molecules} random reference molecules via n-gram model...")
        training_smiles = load_combined_training_data(data_dir)
        ref_df = generate_initial_population(
            n=n_molecules,
            max_heavy_atoms=30,
            min_heavy_atoms=5,
            max_carbons=30,
            max_oxygens=6,
            training_smiles=training_smiles,
        )
        print(f"  Generated {len(ref_df)} reference molecules")
        return ref_df["SMILES"].tolist()
    except Exception as e:
        print(f"  Could not run n-gram generator ({e})")
        print(f"  Falling back to training SMILES as reference...")

        # Fallback: load training data SMILES directly
        all_smiles = []
        data_files = [
            "processed_full_hydrocarbon_dataset.csv",
            "biodegradability_cleaned.csv",
            "BP-Measured_cleaned.csv",
            "DC_exp_cleaned.csv",
            "flashpoint_cleaned.csv",
            "MP-Measured_cleaned.csv",
        ]
        for fname in data_files:
            fpath = os.path.join(data_dir, fname)
            if os.path.exists(fpath):
                try:
                    df = pd.read_csv(fpath)
                    for col in df.columns:
                        if col.lower() in ("smiles", "canonical_smiles", "canonicalsmiles"):
                            all_smiles.extend(df[col].dropna().tolist())
                            break
                except Exception:
                    pass

        unique = list(set(all_smiles))
        if len(unique) > n_molecules:
            np.random.seed(42)
            unique = list(np.random.choice(unique, size=n_molecules, replace=False))
        print(f"  Loaded {len(unique)} training SMILES as reference")
        return unique


def find_best_runs(sweep_dir, n_runs=5):
    """Find the top N runs by best FOM1_avg from a sweep directory."""
    results = []
    for dirname in os.listdir(sweep_dir):
        run_path = os.path.join(sweep_dir, dirname)
        if not os.path.isdir(run_path):
            continue
        # Check for evaluated molecules
        eval_path = os.path.join(run_path, "all_evaluated_molecules.csv")
        if not os.path.exists(eval_path):
            continue

        # Quick read of top molecules for ranking
        top_files = [f for f in os.listdir(run_path)
                     if f.startswith("top_") and f.endswith(".csv")]
        if not top_files:
            continue
        try:
            top_df = pd.read_csv(os.path.join(run_path, top_files[0]))
            if top_df.empty:
                continue
            best_fom = top_df.iloc[0].get("FOM1_avg", 0)
            results.append((dirname, best_fom))
        except Exception:
            continue

    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:n_runs]]


# ======================================================================
# Figures
# ======================================================================

def plot_umap_exploration(umap_coords, labels, generation, fitness, out_dir):
    """Main UMAP figure with three layers: reference, explored, top molecules.

    labels: 'reference', 'explored', or 'top'
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ref_mask = labels == "reference"
    exp_mask = labels == "explored"
    top_mask = labels == "top"

    # --- Panel 1: Reference vs Explored ---
    ax = axes[0]
    ax.scatter(umap_coords[ref_mask, 0], umap_coords[ref_mask, 1],
               c="#DDDDDD", s=3, alpha=0.3, label="Reference", rasterized=True)
    ax.scatter(umap_coords[exp_mask, 0], umap_coords[exp_mask, 1],
               c="#4C72B0", s=3, alpha=0.15, label="Explored", rasterized=True)
    ax.scatter(umap_coords[top_mask, 0], umap_coords[top_mask, 1],
               c="red", s=30, alpha=0.9, marker="*", label="Top 50", zorder=5)
    ax.set_title("Chemical Space Coverage", fontsize=13)
    ax.legend(fontsize=9, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Panel 2: Coloured by generation ---
    ax = axes[1]
    ax.scatter(umap_coords[ref_mask, 0], umap_coords[ref_mask, 1],
               c="#EEEEEE", s=2, alpha=0.2, rasterized=True)

    gen_vals = generation[exp_mask]
    if len(gen_vals) > 0:
        sc = ax.scatter(umap_coords[exp_mask, 0], umap_coords[exp_mask, 1],
                        c=gen_vals, cmap="plasma", s=3, alpha=0.3, rasterized=True)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Generation", fontsize=10)

    ax.scatter(umap_coords[top_mask, 0], umap_coords[top_mask, 1],
               c="red", s=30, alpha=0.9, marker="*", zorder=5)
    ax.set_title("Exploration Over Time", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Panel 3: Coloured by fitness ---
    ax = axes[2]
    ax.scatter(umap_coords[ref_mask, 0], umap_coords[ref_mask, 1],
               c="#EEEEEE", s=2, alpha=0.2, rasterized=True)

    fit_vals = fitness[exp_mask]
    valid_fit = fit_vals[fit_vals > 0]
    if len(valid_fit) > 0:
        vmin, vmax = np.percentile(valid_fit, [5, 95])
        sc = ax.scatter(umap_coords[exp_mask, 0], umap_coords[exp_mask, 1],
                        c=fit_vals, cmap="viridis", s=3, alpha=0.3,
                        vmin=vmin, vmax=vmax, rasterized=True)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("FOM1 (avg)", fontsize=10)

    ax.scatter(umap_coords[top_mask, 0], umap_coords[top_mask, 1],
               c="red", s=30, alpha=0.9, marker="*", zorder=5)
    ax.set_title("Fitness Landscape", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle("Chemical Space Exploration — UMAP Projection of Morgan Fingerprints",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "umap_chemical_space.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "umap_chemical_space.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: umap_chemical_space.png/pdf")


def plot_exploration_trajectory(umap_coords, labels, generation, out_dir,
                                n_gen_bins=10):
    """Plot centroids of explored molecules over generation bins,
    showing the trajectory of the search through chemical space."""
    exp_mask = labels == "explored"
    ref_mask = labels == "reference"

    exp_coords = umap_coords[exp_mask]
    exp_gen = generation[exp_mask]

    if len(exp_coords) == 0:
        return

    max_gen = exp_gen.max()
    bin_edges = np.linspace(0, max_gen, n_gen_bins + 1)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Background
    ax.scatter(umap_coords[ref_mask, 0], umap_coords[ref_mask, 1],
               c="#EEEEEE", s=2, alpha=0.2, rasterized=True)
    ax.scatter(exp_coords[:, 0], exp_coords[:, 1],
               c="#CCCCCC", s=1, alpha=0.1, rasterized=True)

    # Compute centroids per bin
    cmap = plt.cm.plasma
    centroids_x = []
    centroids_y = []
    colors = []

    for i in range(n_gen_bins):
        mask = (exp_gen >= bin_edges[i]) & (exp_gen < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        cx = exp_coords[mask, 0].mean()
        cy = exp_coords[mask, 1].mean()
        centroids_x.append(cx)
        centroids_y.append(cy)
        gen_mid = (bin_edges[i] + bin_edges[i + 1]) / 2
        colors.append(cmap(gen_mid / max_gen))

        # Label with generation range
        label = f"Gen {int(bin_edges[i])}-{int(bin_edges[i+1])}"
        ax.annotate(label, (cx, cy), fontsize=7, ha="center",
                    xytext=(5, 5), textcoords="offset points")

    # Draw trajectory line
    if len(centroids_x) > 1:
        ax.plot(centroids_x, centroids_y, "k-", linewidth=1.5, alpha=0.6, zorder=3)

    # Draw centroid markers
    for i, (cx, cy, c) in enumerate(zip(centroids_x, centroids_y, colors)):
        size = 80 + i * 15  # Bigger markers for later generations
        ax.scatter(cx, cy, c=[c], s=size, edgecolors="black",
                   linewidth=1, zorder=4)

    ax.set_title("Search Trajectory Through Chemical Space", fontsize=14)
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "umap_trajectory.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "umap_trajectory.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: umap_trajectory.png/pdf")


def plot_functional_groups_over_generations(eval_df, out_dir, n_gen_bins=8):
    """Stacked area chart of functional group frequencies across generations."""
    smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in eval_df.columns else "SMILES"

    max_gen = eval_df["generation"].max()
    bin_edges = np.linspace(0, max_gen, n_gen_bins + 1)

    fg_data = {name: [] for name in FUNCTIONAL_GROUPS}
    bin_labels = []

    for i in range(n_gen_bins):
        mask = (eval_df["generation"] >= bin_edges[i]) & (eval_df["generation"] < bin_edges[i + 1])
        subset = eval_df.loc[mask, smiles_col].dropna()

        if len(subset) == 0:
            for name in FUNCTIONAL_GROUPS:
                fg_data[name].append(0)
            bin_labels.append(f"{int(bin_edges[i])}-{int(bin_edges[i+1])}")
            continue

        # Sample if too many
        if len(subset) > 2000:
            subset = subset.sample(2000, random_state=42)

        total_counts = Counter()
        n_mols = 0
        for smi in subset:
            counts = detect_functional_groups(smi)
            if counts:
                n_mols += 1
                for name, c in counts.items():
                    if c > 0:
                        total_counts[name] += 1

        for name in FUNCTIONAL_GROUPS:
            freq = total_counts[name] / n_mols if n_mols > 0 else 0
            fg_data[name].append(freq)

        bin_labels.append(f"{int(bin_edges[i])}-{int(bin_edges[i+1])}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(bin_labels))

    # Sort by final frequency for better readability
    sorted_names = sorted(FUNCTIONAL_GROUPS.keys(),
                          key=lambda n: fg_data[n][-1] if fg_data[n] else 0,
                          reverse=True)

    cmap = plt.cm.tab20
    for i, name in enumerate(sorted_names):
        vals = fg_data[name]
        if max(vals) > 0.01:  # Only plot groups that appear
            ax.plot(x, vals, "o-", label=name, color=cmap(i / len(sorted_names)),
                    linewidth=2, markersize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Generation Range", fontsize=12)
    ax.set_ylabel("Fraction of Molecules Containing Group", fontsize=12)
    ax.set_title("Functional Group Prevalence Across Generations", fontsize=14)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "functional_groups_over_generations.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "functional_groups_over_generations.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: functional_groups_over_generations.png/pdf")


# ======================================================================
# Main
# ======================================================================

def analyse_chemical_space(run_dir=None, sweep_dir=None, data_dir=None,
                           n_runs=3, n_ref=10000, sample_per_run=20000):
    """Run the full chemical space analysis."""

    # ------------------------------------------------------------------
    # 1. Determine which runs to analyse
    # ------------------------------------------------------------------
    run_dirs = []

    if run_dir:
        run_dirs = [run_dir]
        print(f"Analysing single run: {run_dir}")
    elif sweep_dir:
        print(f"Finding top {n_runs} runs from {sweep_dir}...")
        best_names = find_best_runs(sweep_dir, n_runs=n_runs)
        run_dirs = [os.path.join(sweep_dir, name) for name in best_names]
        print(f"  Selected: {', '.join(best_names)}")
    else:
        print("ERROR: Provide either --run_dir or --sweep_dir")
        return

    # ------------------------------------------------------------------
    # 2. Load evaluated molecules
    # ------------------------------------------------------------------
    print(f"\nLoading evaluated molecules...")
    all_eval = []
    for rd in run_dirs:
        df = load_evaluated_molecules(rd, sample_n=sample_per_run)
        if df is not None:
            all_eval.append(df)
            print(f"  {os.path.basename(rd)}: {len(df)} molecules")

    if not all_eval:
        print("ERROR: No evaluated molecules found")
        return

    eval_df = pd.concat(all_eval, ignore_index=True)
    smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in eval_df.columns else "SMILES"
    print(f"  Total: {len(eval_df)} evaluated molecules")

    # ------------------------------------------------------------------
    # 3. Generate reference set
    # ------------------------------------------------------------------
    if data_dir is None:
        # Try to infer data_dir
        for candidate in ["../data", "../../data", os.path.join(os.path.dirname(run_dirs[0]), "../../data")]:
            if os.path.isdir(candidate):
                data_dir = os.path.abspath(candidate)
                break

    if data_dir:
        print(f"\nGenerating reference set from {data_dir}...")
        ref_smiles = generate_reference_molecules(data_dir, n_molecules=n_ref)
    else:
        print("\nWARNING: Could not find data directory, skipping reference set")
        ref_smiles = []

    # ------------------------------------------------------------------
    # 4. Get top molecules
    # ------------------------------------------------------------------
    fom_col = "FOM1_avg" if "FOM1_avg" in eval_df.columns else "FitnessScore"
    top_df = eval_df.nlargest(50, fom_col).drop_duplicates(subset=[smiles_col])

    # ------------------------------------------------------------------
    # 5. Compute fingerprints
    # ------------------------------------------------------------------
    print(f"\nComputing fingerprints...")

    all_smiles = []
    all_labels = []
    all_generations = []
    all_fitness = []

    # Reference
    for smi in ref_smiles:
        all_smiles.append(smi)
        all_labels.append("reference")
        all_generations.append(-1)
        all_fitness.append(0)

    # Explored
    for _, row in eval_df.iterrows():
        all_smiles.append(row[smiles_col])
        all_labels.append("explored")
        all_generations.append(row.get("generation", 0))
        all_fitness.append(row.get(fom_col, 0))

    # Top (add separately so they're on top layer)
    top_smiles_set = set(top_df[smiles_col].tolist())
    for smi in top_smiles_set:
        all_smiles.append(smi)
        all_labels.append("top")
        all_generations.append(-1)
        all_fitness.append(0)

    # Compute fingerprints, filtering failures
    fps = []
    valid_indices = []
    for i, smi in enumerate(all_smiles):
        fp = smiles_to_fingerprint(smi)
        if fp is not None:
            fps.append(fp)
            valid_indices.append(i)

    labels = np.array([all_labels[i] for i in valid_indices])
    generations = np.array([all_generations[i] for i in valid_indices], dtype=float)
    fitness = np.array([all_fitness[i] for i in valid_indices], dtype=float)

    print(f"  Valid fingerprints: {len(fps)} / {len(all_smiles)}")
    print(f"    Reference: {(labels == 'reference').sum()}")
    print(f"    Explored:  {(labels == 'explored').sum()}")
    print(f"    Top:       {(labels == 'top').sum()}")

    # Convert to numpy
    fp_matrix = fps_to_numpy(fps)

    # ------------------------------------------------------------------
    # 6. UMAP projection
    # ------------------------------------------------------------------
    print(f"\nRunning UMAP (this may take a few minutes)...")
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric="jaccard",
        random_state=42,
        n_jobs=1,
    )
    umap_coords = reducer.fit_transform(fp_matrix)
    print(f"  UMAP complete: {umap_coords.shape}")

    # ------------------------------------------------------------------
    # 7. Output directory
    # ------------------------------------------------------------------
    if sweep_dir:
        out_dir = os.path.join(sweep_dir, "analysis", "chemical_space")
    else:
        out_dir = os.path.join(run_dir, "chemical_space")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 8. Generate figures
    # ------------------------------------------------------------------
    print(f"\nGenerating figures...")

    plot_umap_exploration(umap_coords, labels, generations, fitness, out_dir)
    plot_exploration_trajectory(umap_coords, labels, generations, out_dir)
    plot_functional_groups_over_generations(eval_df, out_dir)

    # ------------------------------------------------------------------
    # 9. Coverage statistics
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  COVERAGE STATISTICS")
    print(f"{'='*60}")

    ref_coords = umap_coords[labels == "reference"]
    exp_coords = umap_coords[labels == "explored"]
    top_coords = umap_coords[labels == "top"]

    # Grid-based coverage: divide UMAP space into bins
    all_coords = umap_coords
    x_range = (all_coords[:, 0].min(), all_coords[:, 0].max())
    y_range = (all_coords[:, 1].min(), all_coords[:, 1].max())
    n_bins = 50

    def grid_coverage(coords, x_range, y_range, n_bins):
        if len(coords) == 0:
            return 0
        x_bins = np.linspace(x_range[0], x_range[1], n_bins + 1)
        y_bins = np.linspace(y_range[0], y_range[1], n_bins + 1)
        x_idx = np.clip(np.digitize(coords[:, 0], x_bins) - 1, 0, n_bins - 1)
        y_idx = np.clip(np.digitize(coords[:, 1], y_bins) - 1, 0, n_bins - 1)
        occupied = set(zip(x_idx, y_idx))
        return len(occupied)

    total_cells = n_bins * n_bins
    ref_cells = grid_coverage(ref_coords, x_range, y_range, n_bins)
    exp_cells = grid_coverage(exp_coords, x_range, y_range, n_bins)

    # How much of the reference space was explored?
    ref_set = set()
    if len(ref_coords) > 0:
        x_bins = np.linspace(x_range[0], x_range[1], n_bins + 1)
        y_bins = np.linspace(y_range[0], y_range[1], n_bins + 1)
        x_idx = np.clip(np.digitize(ref_coords[:, 0], x_bins) - 1, 0, n_bins - 1)
        y_idx = np.clip(np.digitize(ref_coords[:, 1], y_bins) - 1, 0, n_bins - 1)
        ref_set = set(zip(x_idx, y_idx))

    exp_set = set()
    if len(exp_coords) > 0:
        x_idx = np.clip(np.digitize(exp_coords[:, 0], x_bins) - 1, 0, n_bins - 1)
        y_idx = np.clip(np.digitize(exp_coords[:, 1], y_bins) - 1, 0, n_bins - 1)
        exp_set = set(zip(x_idx, y_idx))

    overlap = ref_set & exp_set

    print(f"\n  Grid-based coverage ({n_bins}x{n_bins} = {total_cells} cells):")
    print(f"    Reference occupies:    {ref_cells:>5} cells ({100*ref_cells/total_cells:.1f}%)")
    print(f"    GA explored:           {exp_cells:>5} cells ({100*exp_cells/total_cells:.1f}%)")
    if ref_cells > 0:
        print(f"    Overlap with ref:      {len(overlap):>5} cells ({100*len(overlap)/ref_cells:.1f}% of reference)")
    print(f"    GA-only (novel) cells: {exp_cells - len(overlap):>5}")

    # Unique SMILES stats
    eval_unique = eval_df[smiles_col].nunique()
    print(f"\n  Unique molecules evaluated: {eval_unique}")
    print(f"  Total evaluations:         {len(eval_df)}")

    print(f"\nFigures saved to: {out_dir}/")
    print(f"Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chemical space coverage analysis with UMAP"
    )
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Single run directory to analyse")
    parser.add_argument("--sweep_dir", type=str, default=None,
                        help="Sweep directory (analyses top N runs)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory for reference molecule generation")
    parser.add_argument("--n_runs", type=int, default=3,
                        help="Number of top runs to analyse from sweep")
    parser.add_argument("--n_ref", type=int, default=10000,
                        help="Number of reference molecules to generate")
    parser.add_argument("--sample_per_run", type=int, default=20000,
                        help="Max molecules to sample per run (for speed)")
    args = parser.parse_args()

    analyse_chemical_space(
        run_dir=args.run_dir,
        sweep_dir=args.sweep_dir,
        data_dir=args.data_dir,
        n_runs=args.n_runs,
        n_ref=args.n_ref,
        sample_per_run=args.sample_per_run,
    )
