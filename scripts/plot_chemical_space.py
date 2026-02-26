"""
Chemical Space Analysis — Single Best Configuration.

Identifies the top hyperparameter configuration from a sweep, loads all
evaluated molecules across seeds, computes a shared UMAP embedding with
the n-gram reference population, and generates publication-quality figures.

Figures produced:
  (a)      Reference population coloured by dominant functional group
  (fg_ga)  GA explored population coloured by dominant functional group
  (b)      GA coverage overlay on reference
  (c)      Exploration over time (generation colourmap)
  (d)      FOM1 fitness landscape
  (grid)   1x3 combined figure of (b)-(d)
  (fg_gen) Functional group prevalence vs generation

Requirements:
    pip install umap-learn rdkit-pypi tqdm

Usage:
    python plot_chemical_space.py --sweep_dir ../outputs/hyperparam_sweep
    python plot_chemical_space.py --sweep_dir ../outputs/hyperparam_sweep --no_cache
    python plot_chemical_space.py --npz path/to/umap_cache.npz --out_dir ./figs
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
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from tqdm import tqdm
except ImportError:
    # Minimal fallback if tqdm not installed
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


# ======================================================================
# Functional groups (consistent with other scripts)
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

FG_SHORT_NAMES = {
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

# Fixed colour assignments for functional groups — data-independent so every
# panel/grid uses identical colours regardless of subset or ordering.
FIXED_FG_COLOURS = {
    "Ether":           "#1f77b4",  # blue
    "Vinyl ether":     "#aec7e8",  # light blue
    "Alcohol":         "#ff7f0e",  # orange
    "Aldehyde":        "#ffbb78",  # light orange
    "Ketone":          "#2ca02c",  # green
    "Ester":           "#98df8a",  # light green
    "Carboxylic acid": "#d62728",  # red
    "Epoxide":         "#ff9896",  # light red
    "Alkene":          "#9467bd",  # purple
    "Terminal alkene": "#c5b0d5",  # light purple
    "Acetal":          "#8c564b",  # brown
    "Hydrocarbon only": "#BBBBBB", # grey
}


# ======================================================================
# Utility functions
# ======================================================================

def detect_functional_groups(smi):
    """Return dict of {fg_name: count} for a SMILES string."""
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


def smiles_to_fingerprint(smi, radius=2, n_bits=2048):
    """Convert SMILES to Morgan fingerprint bit vector."""
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
    """Convert list of RDKit fingerprints to numpy uint8 array."""
    arr = np.zeros((len(fps), fps[0].GetNumBits()), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


def assign_dominant_fg(smiles_list):
    """Detect FGs and assign the dominant one per molecule.

    Returns:
        dominant_labels: np.ndarray of str
        fg_counts: np.ndarray of int32 (n_mol, n_fg)
    """
    fg_names = list(FUNCTIONAL_GROUPS.keys())
    n_mol = len(smiles_list)
    fg_counts = np.zeros((n_mol, len(fg_names)), dtype=np.int32)

    for i, smi in enumerate(tqdm(smiles_list, desc="Detecting FGs")):
        counts = detect_functional_groups(smi)
        for fi, name in enumerate(fg_names):
            fg_counts[i, fi] = counts.get(name, 0)

    dominant_labels = []
    for i in range(n_mol):
        max_count = 0
        dominant = "Hydrocarbon only"
        for fi, name in enumerate(fg_names):
            if fg_counts[i, fi] > max_count:
                max_count = fg_counts[i, fi]
                dominant = FG_SHORT_NAMES.get(name, name)
        dominant_labels.append(dominant)

    return np.array(dominant_labels), fg_counts


# ======================================================================
# Data loading
# ======================================================================

def parse_run_dir(dirname):
    """Extract hyperparameters from directory name.

    Supports Stage 1 (with k) and legacy (with tau) formats.
    """
    m = re.match(r"pop(\d+)_mr([\d.]+)_er([\d.]+)_k(\d+)_seed(\d+)", dirname)
    if m:
        return {
            "pop": int(m.group(1)), "mr": float(m.group(2)),
            "er": float(m.group(3)), "k": int(m.group(4)),
            "seed": int(m.group(5)), "format": "stage1",
        }
    m = re.match(r"pop(\d+)_mr([\d.]+)_er([\d.]+)_tau([\d.]+)_seed(\d+)", dirname)
    if m:
        return {
            "pop": int(m.group(1)), "mr": float(m.group(2)),
            "er": float(m.group(3)), "tau": float(m.group(4)),
            "seed": int(m.group(5)), "format": "legacy",
        }
    return None


def identify_top_config(sweep_dir):
    """Read configs_aggregated.csv and return the top configuration.

    Returns:
        params: dict (e.g. {"pop": 2000, "mr": 0.8, "er": 0.3, "k": 3})
        label: human-readable string
        mean_fom: mean best FOM1_avg across seeds
    """
    csv_path = os.path.join(sweep_dir, "analysis", "configs_aggregated.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        print("       Run analyse_hyperparam_sweep.py first to generate it.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    possible_groups = ["pop", "mr", "er", "k", "tau"]
    group_cols = [c for c in possible_groups if c in df.columns]

    if df.empty or not group_cols:
        print(f"ERROR: {csv_path} is empty or missing expected columns.")
        sys.exit(1)

    # File is pre-sorted by best_FOM1_avg_mean descending
    top_row = df.iloc[0]
    params = {}
    for col in group_cols:
        val = top_row[col]
        params[col] = int(val) if col in ("pop", "k") else float(val)

    label = ", ".join(f"{k}={v}" for k, v in params.items())
    mean_fom = top_row.get("best_FOM1_avg_mean", np.nan)

    return params, label, mean_fom


def find_seed_dirs(sweep_dir, params):
    """Find all run directories matching the given config params."""
    matching = []
    for dirname in sorted(os.listdir(sweep_dir)):
        run_path = os.path.join(sweep_dir, dirname)
        if not os.path.isdir(run_path):
            continue
        parsed = parse_run_dir(dirname)
        if parsed is None:
            continue

        match = True
        for key, val in params.items():
            parsed_val = parsed.get(key)
            if parsed_val is None:
                match = False
                break
            if isinstance(val, float):
                if abs(parsed_val - val) > 1e-6:
                    match = False
                    break
            elif parsed_val != val:
                match = False
                break
        if match:
            matching.append(run_path)

    return matching


def load_ga_molecules(seed_dirs, sample_ga=50000):
    """Load and concatenate all_evaluated_molecules.csv from seed dirs.

    Caps total molecules at sample_ga via stratified sampling by generation.
    """
    frames = []
    for rd in tqdm(seed_dirs, desc="Loading GA data"):
        eval_path = os.path.join(rd, "all_evaluated_molecules.csv")
        if not os.path.exists(eval_path):
            print(f"  WARNING: {eval_path} not found, skipping")
            continue
        try:
            df = pd.read_csv(eval_path)
            if df.empty:
                continue
            # Tag with seed directory for provenance
            df["_source_dir"] = os.path.basename(rd)
            frames.append(df)
        except Exception as e:
            print(f"  WARNING: Could not read {eval_path}: {e}")

    if not frames:
        print("ERROR: No evaluated molecules found in any seed directory.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    # Normalise column names
    if "CanonicalSMILES" in combined.columns and "SMILES" not in combined.columns:
        combined.rename(columns={"CanonicalSMILES": "SMILES"}, inplace=True)

    # Stratified sampling by generation if over limit
    if sample_ga > 0 and len(combined) > sample_ga:
        print(f"  Sampling {sample_ga} from {len(combined)} molecules (stratified by generation)...")
        if "generation" in combined.columns:
            combined = combined.groupby("generation", group_keys=False).apply(
                lambda g: g.sample(
                    n=min(len(g), max(1, int(sample_ga * len(g) / len(combined)))),
                    random_state=42
                )
            ).reset_index(drop=True)
            # Trim to exact count if slight overshoot
            if len(combined) > sample_ga:
                combined = combined.sample(n=sample_ga, random_state=42)
        else:
            combined = combined.sample(n=sample_ga, random_state=42)

    return combined


def load_reference_set(ref_csv, data_dir):
    """Load reference SMILES from CSV, generating if missing."""
    if os.path.exists(ref_csv):
        df = pd.read_csv(ref_csv)
        smiles = df["SMILES"].dropna().tolist()
        print(f"  Loaded {len(smiles)} reference molecules from {ref_csv}")
        return smiles

    print(f"  Reference CSV not found at {ref_csv}, generating inline...")
    import random as _random
    _random.seed(42)
    np.random.seed(42)

    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    sys.path.insert(0, os.path.abspath(src_dir))
    from generate_molecules import generate_initial_population, load_combined_training_data

    training_smiles = sorted(load_combined_training_data(data_dir))
    ref_df = generate_initial_population(
        n=10000, max_heavy_atoms=30, min_heavy_atoms=5,
        max_carbons=30, max_oxygens=6, training_smiles=training_smiles,
        similarity_threshold=1.0, max_attempts_factor=50,
    )
    os.makedirs(os.path.dirname(os.path.abspath(ref_csv)), exist_ok=True)
    ref_df[["SMILES"]].to_csv(ref_csv, index=False)
    smiles = ref_df["SMILES"].tolist()
    print(f"  Generated and saved {len(smiles)} reference molecules to {ref_csv}")
    return smiles


# ======================================================================
# UMAP computation
# ======================================================================

def compute_or_load_umap(ref_smiles, ga_df, cache_path, no_cache=False, n_top=50):
    """Compute shared UMAP or load from cache.

    Returns:
        coords_2d, labels, generations, fitness, is_valid, valid_smiles, dominant_fgs
    """
    # Check cache
    if not no_cache and os.path.exists(cache_path):
        print(f"  Loading cached UMAP from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        dominant_fgs = data["dominant_fgs"] if "dominant_fgs" in data else None
        is_valid = data["is_valid"] if "is_valid" in data else None
        return (
            data["coords_2d"],
            data["labels"],
            data["generations"],
            data["fitness"],
            is_valid,
            data["smiles"].tolist(),
            dominant_fgs,
        )

    # Determine column names
    smiles_col = "SMILES"
    fom_col = "FOM1_avg" if "FOM1_avg" in ga_df.columns else "FitnessScore"

    # Identify top molecules
    top_df = ga_df.nlargest(n_top, fom_col).drop_duplicates(subset=[smiles_col])
    top_smiles_set = set(top_df[smiles_col].dropna().tolist())

    # Build combined SMILES list
    all_smiles = []
    all_labels = []
    all_generations = []
    all_fitness = []
    all_is_valid = []

    for smi in ref_smiles:
        all_smiles.append(smi)
        all_labels.append("reference")
        all_generations.append(-1)
        all_fitness.append(0)
        all_is_valid.append(-1)  # not applicable for reference

    has_valid_col = "is_valid" in ga_df.columns
    for _, row in ga_df.iterrows():
        smi = row.get(smiles_col)
        if not isinstance(smi, str):
            continue
        all_smiles.append(smi)
        all_labels.append("explored")
        all_generations.append(row.get("generation", 0))
        all_fitness.append(row.get(fom_col, 0))
        all_is_valid.append(int(row["is_valid"]) if has_valid_col else -1)

    # Top molecules as separate layer for z-order
    for smi in top_smiles_set:
        all_smiles.append(smi)
        all_labels.append("top")
        all_generations.append(-1)
        all_fitness.append(0)
        all_is_valid.append(-1)

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
    is_valid = np.array([all_is_valid[i] for i in valid_indices], dtype=np.int8)
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

    # Pre-compute dominant FG labels for caching
    print("  Pre-computing dominant functional groups...")
    dominant_fgs, _ = assign_dominant_fg(valid_smiles)

    # Save cache
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez_compressed(
        cache_path,
        coords_2d=coords_2d,
        labels=labels,
        generations=generations,
        fitness=fitness,
        is_valid=is_valid,
        smiles=np.array(valid_smiles, dtype=object),
        dominant_fgs=dominant_fgs,
    )
    print(f"  Cached UMAP to {cache_path}")

    return coords_2d, labels, generations, fitness, is_valid, valid_smiles, dominant_fgs


# ======================================================================
# Drawing helpers (shared by standalone + combined grid)
# ======================================================================

def _draw_panel_a(ax, ref_coords, dominant_labels, label_counts, label_to_colour,
                  show_legend=True, all_coords=None):
    """Panel (a): FG-coloured population scatter.

    Parameters
    ----------
    all_coords : array, optional
        Full UMAP coordinates (all populations) for consistent axis limits.
        Falls back to ref_coords if not provided.
    """
    unique_labels = [lab for lab, _ in label_counts.most_common()]

    # Hydrocarbons first (background)
    if "Hydrocarbon only" in label_to_colour:
        mask = dominant_labels == "Hydrocarbon only"
        if mask.sum() > 0:
            ax.scatter(ref_coords[mask, 0], ref_coords[mask, 1],
                       c=label_to_colour["Hydrocarbon only"], s=4, alpha=0.3,
                       label=f"Hydrocarbon ({mask.sum()})", rasterized=True)

    for lab in unique_labels:
        if lab == "Hydrocarbon only":
            continue
        mask = dominant_labels == lab
        if mask.sum() < 3:
            continue
        ax.scatter(ref_coords[mask, 0], ref_coords[mask, 1],
                   c=[label_to_colour[lab]], s=8, alpha=0.5,
                   label=f"{lab} ({mask.sum()})", rasterized=True)

    _apply_axis_limits(ax, all_coords if all_coords is not None else ref_coords)
    ax.set_xticks([])
    ax.set_yticks([])

    if show_legend:
        ax.legend(fontsize=11, markerscale=2, frameon=False, loc="upper right",
                  ncol=1, handletextpad=0.3, labelspacing=0.3)


def _draw_panel_b(ax, coords_2d, labels):
    """Panel (b): GA overlay on reference — Chemical Space Coverage."""
    ref_mask = labels == "reference"
    exp_mask = labels == "explored"
    top_mask = labels == "top"

    ax.scatter(coords_2d[ref_mask, 0], coords_2d[ref_mask, 1],
               c="#DDDDDD", s=3, alpha=0.3, label="Reference", rasterized=True)
    ax.scatter(coords_2d[exp_mask, 0], coords_2d[exp_mask, 1],
               c="#4C72B0", s=3, alpha=0.15, label="GA explored", rasterized=True)
    ax.scatter(coords_2d[top_mask, 0], coords_2d[top_mask, 1],
               c="red", s=30, alpha=0.9, marker="*", label="Top 50", zorder=5)

    ax.legend(fontsize=12, markerscale=2, frameon=False, loc="upper right")
    _apply_axis_limits(ax, coords_2d)
    ax.set_xticks([])
    ax.set_yticks([])


def _robust_axis_limits(coords_2d, margin=0.08, percentile=0.5):
    """Compute axis limits that exclude extreme outliers.

    Uses the given percentile to trim outliers from each side, then adds
    a margin (fraction of the trimmed range) for padding.
    """
    lo, hi = percentile, 100 - percentile
    xmin, xmax = np.percentile(coords_2d[:, 0], [lo, hi])
    ymin, ymax = np.percentile(coords_2d[:, 1], [lo, hi])
    x_pad = (xmax - xmin) * margin
    y_pad = (ymax - ymin) * margin
    return (xmin - x_pad, xmax + x_pad), (ymin - y_pad, ymax + y_pad)


def _apply_axis_limits(ax, coords_2d):
    """Set axis limits that trim outlier whitespace."""
    xlim, ylim = _robust_axis_limits(coords_2d)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


_CBAR_FRACTION = 0.046
_CBAR_PAD = 0.02
_CBAR_SHRINK = 1.0


def _add_side_colorbar(fig, ax, mappable, label):
    """Add a vertical colorbar to the right of the axes."""
    cbar = fig.colorbar(mappable, ax=ax, fraction=_CBAR_FRACTION,
                        pad=_CBAR_PAD, shrink=_CBAR_SHRINK)
    cbar.set_label(label, fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    return cbar


def _add_invisible_colorbar(fig, ax):
    """Add an invisible colorbar spacer so the axes width matches panels with colorbars."""
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=_CBAR_FRACTION,
                        pad=_CBAR_PAD, shrink=_CBAR_SHRINK)
    cbar.ax.set_visible(False)


def _draw_panel_c(ax, coords_2d, labels, generations, fig=None):
    """Panel (c): Exploration over time (generation colourmap)."""
    ref_mask = labels == "reference"
    exp_mask = labels == "explored"
    top_mask = labels == "top"

    ax.scatter(coords_2d[ref_mask, 0], coords_2d[ref_mask, 1],
               c="#EEEEEE", s=2, alpha=0.2, rasterized=True)

    gen_vals = generations[exp_mask]
    if len(gen_vals) > 0:
        sc = ax.scatter(coords_2d[exp_mask, 0], coords_2d[exp_mask, 1],
                        c=gen_vals, cmap="plasma", s=3, alpha=0.3, rasterized=True)
        if fig is not None:
            _add_side_colorbar(fig, ax, sc, "Generation")

    ax.scatter(coords_2d[top_mask, 0], coords_2d[top_mask, 1],
               c="red", s=30, alpha=0.9, marker="*", zorder=5)

    _apply_axis_limits(ax, coords_2d)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_panel_d(ax, coords_2d, labels, fitness, fig=None):
    """Panel (d): FOM1 fitness landscape."""
    ref_mask = labels == "reference"
    exp_mask = labels == "explored"
    top_mask = labels == "top"

    ax.scatter(coords_2d[ref_mask, 0], coords_2d[ref_mask, 1],
               c="#EEEEEE", s=2, alpha=0.2, rasterized=True)

    fit_vals = fitness[exp_mask]
    valid_fit = fit_vals[fit_vals > 0]
    if len(valid_fit) > 0:
        vmin, vmax = np.percentile(valid_fit, [5, 95])
        sc = ax.scatter(coords_2d[exp_mask, 0], coords_2d[exp_mask, 1],
                        c=fit_vals, cmap="viridis", s=3, alpha=0.3,
                        vmin=vmin, vmax=vmax, rasterized=True)
        if fig is not None:
            _add_side_colorbar(fig, ax, sc, "FOM1 (avg)")

    ax.scatter(coords_2d[top_mask, 0], coords_2d[top_mask, 1],
               c="red", s=30, alpha=0.9, marker="*", zorder=5)

    _apply_axis_limits(ax, coords_2d)
    ax.set_xticks([])
    ax.set_yticks([])


# ======================================================================
# Standalone figure functions
# ======================================================================

def _build_fg_colour_map(label_counts):
    """Build FG label -> colour mapping using fixed colour assignments.

    Uses FIXED_FG_COLOURS for deterministic colours across all panels.
    Falls back to tab20 only for unexpected labels not in the fixed map.
    """
    label_to_colour = {}
    cmap = plt.cm.tab20
    fallback_idx = 0
    for lab, _ in label_counts.most_common():
        if lab in FIXED_FG_COLOURS:
            label_to_colour[lab] = FIXED_FG_COLOURS[lab]
        else:
            label_to_colour[lab] = cmap(fallback_idx / 20)
            fallback_idx += 1
    return label_to_colour


def _prepare_panel_a_data(coords_2d, labels, valid_smiles, dominant_fgs=None):
    """Compute FG labels and colour map for panel (a). Returns reusable data."""
    ref_mask = labels == "reference"
    ref_coords = coords_2d[ref_mask]

    if dominant_fgs is not None:
        dominant_labels = dominant_fgs[ref_mask]
    else:
        ref_smiles_list = [s for s, m in zip(valid_smiles, ref_mask) if m]
        dominant_labels, _ = assign_dominant_fg(ref_smiles_list)

    label_counts = Counter(dominant_labels)
    label_to_colour = _build_fg_colour_map(label_counts)

    return ref_coords, dominant_labels, label_counts, label_to_colour


def _save_fig(fig, out_dir, basename, dpi=300):
    """Save figure as PNG with consistent settings."""
    path = os.path.join(out_dir, f"{basename}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


PANEL_SIZE = (7, 7)  # consistent size for all standalone panels


def plot_panel_a(coords_2d, labels, valid_smiles, out_dir, dominant_fgs=None,
                 dpi=300):
    """Standalone panel (a): FG-coloured reference population."""
    ref_coords, dominant_labels, label_counts, label_to_colour = \
        _prepare_panel_a_data(coords_2d, labels, valid_smiles, dominant_fgs)

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    _draw_panel_a(ax, ref_coords, dominant_labels, label_counts, label_to_colour,
                  all_coords=coords_2d)
    _add_invisible_colorbar(fig, ax)
    fig.tight_layout()
    _save_fig(fig, out_dir, "panel_a_fg_reference", dpi)
    return ref_coords, dominant_labels, label_counts, label_to_colour


def plot_fg_ga(coords_2d, labels, valid_smiles, label_to_colour, out_dir,
               dominant_fgs=None, dpi=300):
    """FG-coloured UMAP of GA-explored molecules (same colour map as reference)."""
    exp_mask = labels == "explored"
    ga_coords = coords_2d[exp_mask]

    if dominant_fgs is not None:
        dominant_labels = dominant_fgs[exp_mask]
    else:
        ga_smiles_list = [s for s, m in zip(valid_smiles, exp_mask) if m]
        dominant_labels, _ = assign_dominant_fg(ga_smiles_list)

    label_counts = Counter(dominant_labels)

    # Add any new FG labels not in the reference colour map
    for lab in label_counts:
        if lab not in label_to_colour:
            label_to_colour[lab] = "#999999"

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    _draw_panel_a(ax, ga_coords, dominant_labels, label_counts, label_to_colour,
                  all_coords=coords_2d)
    _add_invisible_colorbar(fig, ax)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fg_ga_explored", dpi)


def plot_fg_combined(coords_2d, labels, valid_smiles, out_dir,
                     dominant_fgs=None, dpi=300):
    """1x2 combined grid: reference FG + GA FG."""
    ref_coords, dominant_labels_ref, label_counts_ref, label_to_colour = \
        _prepare_panel_a_data(coords_2d, labels, valid_smiles, dominant_fgs)

    exp_mask = labels == "explored"
    ga_coords = coords_2d[exp_mask]
    if dominant_fgs is not None:
        dominant_labels_ga = dominant_fgs[exp_mask]
    else:
        ga_smiles_list = [s for s, m in zip(valid_smiles, exp_mask) if m]
        dominant_labels_ga, _ = assign_dominant_fg(ga_smiles_list)
    label_counts_ga = Counter(dominant_labels_ga)
    for lab in label_counts_ga:
        if lab not in label_to_colour:
            label_to_colour[lab] = "#999999"

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    _draw_panel_a(axes[0], ref_coords, dominant_labels_ref, label_counts_ref,
                  label_to_colour, all_coords=coords_2d)
    _add_invisible_colorbar(fig, axes[0])

    _draw_panel_a(axes[1], ga_coords, dominant_labels_ga, label_counts_ga,
                  label_to_colour, all_coords=coords_2d)
    _add_invisible_colorbar(fig, axes[1])

    for ax, lab in zip(axes, ["(a)", "(b)"]):
        ax.text(0.02, 0.98, lab, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top", ha="left")

    fig.tight_layout()
    _save_fig(fig, out_dir, "fg_combined", dpi)
    return label_to_colour


def _draw_panel_validity(ax, coords_2d, labels, is_valid):
    """Panel: GA molecules coloured by validity (valid=green, invalid=red)."""
    ref_mask = labels == "reference"
    exp_mask = labels == "explored"
    top_mask = labels == "top"

    ax.scatter(coords_2d[ref_mask, 0], coords_2d[ref_mask, 1],
               c="#EEEEEE", s=2, alpha=0.2, rasterized=True)

    if is_valid is not None:
        exp_valid = is_valid[exp_mask]
        valid_mask_exp = exp_valid == 1
        invalid_mask_exp = exp_valid == 0

        exp_coords = coords_2d[exp_mask]
        # Invalid first (background)
        if invalid_mask_exp.sum() > 0:
            ax.scatter(exp_coords[invalid_mask_exp, 0], exp_coords[invalid_mask_exp, 1],
                       c="#D64545", s=3, alpha=0.2,
                       label=f"Invalid ({invalid_mask_exp.sum()})", rasterized=True)
        # Valid on top
        if valid_mask_exp.sum() > 0:
            ax.scatter(exp_coords[valid_mask_exp, 0], exp_coords[valid_mask_exp, 1],
                       c="#2CA02C", s=3, alpha=0.2,
                       label=f"Valid ({valid_mask_exp.sum()})", rasterized=True)
    else:
        # No validity data — plot all explored as grey
        ax.scatter(coords_2d[exp_mask, 0], coords_2d[exp_mask, 1],
                   c="#999999", s=3, alpha=0.15,
                   label="Explored (no validity data)", rasterized=True)

    ax.scatter(coords_2d[top_mask, 0], coords_2d[top_mask, 1],
               c="red", s=30, alpha=0.9, marker="*", label="Top 50", zorder=5)

    ax.legend(fontsize=12, markerscale=2, frameon=False, loc="upper right")
    _apply_axis_limits(ax, coords_2d)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_panel_validity(coords_2d, labels, is_valid, out_dir, dpi=300):
    """Standalone validity UMAP."""
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    _draw_panel_validity(ax, coords_2d, labels, is_valid)
    _add_invisible_colorbar(fig, ax)
    fig.tight_layout()
    _save_fig(fig, out_dir, "panel_validity", dpi)


# ======================================================================
# Mahalanobis / OOD panels
# ======================================================================

def load_mahal_data_single(mahal_csv, valid_smiles, labels):
    """Load Mahalanobis CSV and match OOD_any + mean Mahal to UMAP molecules.

    Returns:
        ood_any : ndarray (n_points,) — 0/1 for GA molecules, -1 for reference/top
        mahal_mean : ndarray (n_points,) — mean Mahalanobis distance across models,
                     NaN for reference/top
    """
    df = pd.read_csv(mahal_csv)

    if "CanonicalSMILES" in df.columns and "SMILES" not in df.columns:
        df.rename(columns={"CanonicalSMILES": "SMILES"}, inplace=True)

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


def _draw_panel_mahal(ax, coords_2d, labels, mahal_mean, fig=None):
    """Panel: Mahalanobis distance continuous colorbar."""
    ref_mask = labels == "reference"
    exp_mask = labels == "explored"
    top_mask = labels == "top"

    ax.scatter(coords_2d[ref_mask, 0], coords_2d[ref_mask, 1],
               c="#EEEEEE", s=2, alpha=0.2, rasterized=True)

    mahal_vals = mahal_mean[exp_mask]
    valid_mahal = mahal_vals[np.isfinite(mahal_vals)]
    if len(valid_mahal) > 0:
        vmin, vmax = np.percentile(valid_mahal, [2, 98])
        sc = ax.scatter(coords_2d[exp_mask, 0], coords_2d[exp_mask, 1],
                        c=mahal_vals, cmap="inferno", s=3, alpha=0.3,
                        vmin=vmin, vmax=vmax, rasterized=True)
        if fig is not None:
            _add_side_colorbar(fig, ax, sc, "Mahalanobis Distance")

    ax.scatter(coords_2d[top_mask, 0], coords_2d[top_mask, 1],
               c="red", s=30, alpha=0.9, marker="*", zorder=5)

    _apply_axis_limits(ax, coords_2d)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_panel_ood(ax, coords_2d, labels, ood_any):
    """Panel: binary OOD (in-domain=green, OOD=red)."""
    ref_mask = labels == "reference"
    exp_mask = labels == "explored"
    top_mask = labels == "top"

    ax.scatter(coords_2d[ref_mask, 0], coords_2d[ref_mask, 1],
               c="#EEEEEE", s=2, alpha=0.2, rasterized=True)

    ood_vals = ood_any[exp_mask]
    exp_coords = coords_2d[exp_mask]

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

    ax.scatter(coords_2d[top_mask, 0], coords_2d[top_mask, 1],
               c="red", s=30, alpha=0.9, marker="*", label="Top 50", zorder=5)

    ax.legend(fontsize=12, markerscale=2, frameon=False, loc="upper right")
    _apply_axis_limits(ax, coords_2d)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_panel_mahal(coords_2d, labels, mahal_mean, out_dir, dpi=300):
    """Standalone Mahalanobis distance panel."""
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    _draw_panel_mahal(ax, coords_2d, labels, mahal_mean, fig=fig)
    fig.tight_layout()
    _save_fig(fig, out_dir, "panel_mahalanobis", dpi)


def plot_panel_ood(coords_2d, labels, ood_any, out_dir, dpi=300):
    """Standalone binary OOD panel."""
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    _draw_panel_ood(ax, coords_2d, labels, ood_any)
    _add_invisible_colorbar(fig, ax)
    fig.tight_layout()
    _save_fig(fig, out_dir, "panel_ood", dpi)


def plot_panel_b(coords_2d, labels, out_dir, dpi=300):
    """Standalone panel (b): Coverage overlay."""
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    _draw_panel_b(ax, coords_2d, labels)
    _add_invisible_colorbar(fig, ax)
    fig.tight_layout()
    _save_fig(fig, out_dir, "panel_b_coverage", dpi)


def plot_panel_c(coords_2d, labels, generations, out_dir, dpi=300):
    """Standalone panel (c): Exploration over time."""
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    _draw_panel_c(ax, coords_2d, labels, generations, fig=fig)
    fig.tight_layout()
    _save_fig(fig, out_dir, "panel_c_generations", dpi)


def plot_panel_d(coords_2d, labels, fitness, out_dir, dpi=300):
    """Standalone panel (d): Fitness landscape."""
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    _draw_panel_d(ax, coords_2d, labels, fitness, fig=fig)
    fig.tight_layout()
    _save_fig(fig, out_dir, "panel_d_fitness", dpi)


def plot_combined_grid(coords_2d, labels, generations, fitness, is_valid,
                       out_dir, dpi=300):
    """2x2 combined grid: coverage, validity, generations, fitness."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    _draw_panel_b(axes[0, 0], coords_2d, labels)
    _add_invisible_colorbar(fig, axes[0, 0])   # spacer so width matches
    _draw_panel_validity(axes[0, 1], coords_2d, labels, is_valid)
    _add_invisible_colorbar(fig, axes[0, 1])   # spacer so width matches
    _draw_panel_c(axes[1, 0], coords_2d, labels, generations, fig=fig)
    _draw_panel_d(axes[1, 1], coords_2d, labels, fitness, fig=fig)

    # Panel labels
    for ax, label in zip(axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top", ha="left")

    fig.tight_layout()
    _save_fig(fig, out_dir, "chemical_space_combined", dpi)


def plot_fg_over_generations(ga_df, out_dir, n_gen_bins=8, dpi=300):
    """Functional group prevalence vs generation bins — line chart."""
    smiles_col = "SMILES"

    if "generation" not in ga_df.columns:
        print("  WARNING: No 'generation' column, skipping FG-over-generations plot")
        return

    max_gen = ga_df["generation"].max()
    bin_edges = np.linspace(0, max_gen, n_gen_bins + 1)

    fg_data = {name: [] for name in FUNCTIONAL_GROUPS}
    bin_labels = []

    for i in range(n_gen_bins):
        mask = (ga_df["generation"] >= bin_edges[i]) & (ga_df["generation"] < bin_edges[i + 1])
        subset = ga_df.loc[mask, smiles_col].dropna()
        bin_label = f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
        bin_labels.append(bin_label)

        if len(subset) == 0:
            for name in FUNCTIONAL_GROUPS:
                fg_data[name].append(0)
            continue

        # Sample if too many
        if len(subset) > 2000:
            subset = subset.sample(2000, random_state=42)

        total_counts = Counter()
        n_mols = 0
        for smi in tqdm(subset, desc=f"FGs gen {bin_label}", leave=False):
            counts = detect_functional_groups(smi)
            if counts:
                n_mols += 1
                for name, c in counts.items():
                    if c > 0:
                        total_counts[name] += 1

        for name in FUNCTIONAL_GROUPS:
            freq = total_counts[name] / n_mols if n_mols > 0 else 0
            fg_data[name].append(freq)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(bin_labels))

    sorted_names = sorted(FUNCTIONAL_GROUPS.keys(),
                          key=lambda n: fg_data[n][-1] if fg_data[n] else 0,
                          reverse=True)

    cmap = plt.cm.tab20
    for i, name in enumerate(sorted_names):
        vals = fg_data[name]
        if max(vals) > 0.01:
            short = FG_SHORT_NAMES.get(name, name)
            ax.plot(x, vals, "o-", label=short, color=cmap(i / len(sorted_names)),
                    linewidth=2, markersize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Generation Range", fontsize=12)
    ax.set_ylabel("Fraction of Molecules Containing Group", fontsize=12)
    ax.set_title("Functional Group Prevalence Across Generations", fontsize=14)
    ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=1,
              handletextpad=0.3, labelspacing=0.3)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    _save_fig(fig, out_dir, "fg_over_generations", dpi)


def plot_fg_comparison(ref_smiles_list, ga_smiles_list, out_dir, dpi=300):
    """Side-by-side bar chart comparing FG prevalence in reference vs GA molecules."""
    print("  Computing FG distributions...")

    def _fg_fractions(smiles_list, desc):
        """Return {fg_short_name: fraction} for a list of SMILES."""
        total_counts = Counter()
        n_valid = 0
        sample = smiles_list
        if len(sample) > 5000:
            import random as _rng
            _rng_inst = _rng.Random(42)
            sample = _rng_inst.sample(smiles_list, 5000)
        for smi in tqdm(sample, desc=desc):
            counts = detect_functional_groups(smi)
            if counts:
                n_valid += 1
                for name, c in counts.items():
                    if c > 0:
                        total_counts[name] += 1
        fracs = {}
        for name in FUNCTIONAL_GROUPS:
            short = FG_SHORT_NAMES.get(name, name)
            fracs[short] = total_counts[name] / n_valid if n_valid > 0 else 0
        return fracs

    ref_fracs = _fg_fractions(ref_smiles_list, "FGs (reference)")
    ga_fracs = _fg_fractions(ga_smiles_list, "FGs (GA)")

    # Sort by reference prevalence descending
    fg_names = sorted(ref_fracs.keys(), key=lambda n: ref_fracs[n], reverse=True)
    # Filter out groups with < 1% in both
    fg_names = [n for n in fg_names if ref_fracs[n] > 0.01 or ga_fracs[n] > 0.01]

    x = np.arange(len(fg_names))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ref_vals = [ref_fracs[n] for n in fg_names]
    ga_vals = [ga_fracs[n] for n in fg_names]

    ax.bar(x - bar_width / 2, ref_vals, bar_width, label="N-gram reference",
           color="#7FB3D8", edgecolor="white", linewidth=0.5)
    ax.bar(x + bar_width / 2, ga_vals, bar_width, label="GA explored",
           color="#E07B54", edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(fg_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of Molecules", fontsize=11)
    ax.set_title("Functional Group Prevalence: Reference vs GA", fontsize=13)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_ylim(0, min(1.05, max(max(ref_vals), max(ga_vals)) * 1.2))
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    _save_fig(fig, out_dir, "fg_comparison", dpi)


# ======================================================================
# Coverage statistics
# ======================================================================

def print_coverage_stats(coords_2d, labels, ga_df):
    """Print grid-based coverage statistics."""
    ref_mask = labels == "reference"
    exp_mask = labels == "explored"

    ref_coords = coords_2d[ref_mask]
    exp_coords = coords_2d[exp_mask]

    x_range = (coords_2d[:, 0].min(), coords_2d[:, 0].max())
    y_range = (coords_2d[:, 1].min(), coords_2d[:, 1].max())
    n_bins = 50
    total_cells = n_bins * n_bins

    def grid_cells(coords):
        if len(coords) == 0:
            return set()
        x_bins = np.linspace(x_range[0], x_range[1], n_bins + 1)
        y_bins = np.linspace(y_range[0], y_range[1], n_bins + 1)
        xi = np.clip(np.digitize(coords[:, 0], x_bins) - 1, 0, n_bins - 1)
        yi = np.clip(np.digitize(coords[:, 1], y_bins) - 1, 0, n_bins - 1)
        return set(zip(xi, yi))

    ref_set = grid_cells(ref_coords)
    exp_set = grid_cells(exp_coords)
    overlap = ref_set & exp_set

    smiles_col = "SMILES"
    eval_unique = ga_df[smiles_col].nunique() if smiles_col in ga_df.columns else "N/A"

    print(f"\n{'='*60}")
    print(f"  COVERAGE STATISTICS")
    print(f"{'='*60}")
    print(f"\n  Grid-based coverage ({n_bins}x{n_bins} = {total_cells} cells):")
    print(f"    Reference occupies:    {len(ref_set):>5} cells ({100*len(ref_set)/total_cells:.1f}%)")
    print(f"    GA explored:           {len(exp_set):>5} cells ({100*len(exp_set)/total_cells:.1f}%)")
    if ref_set:
        print(f"    Overlap with ref:      {len(overlap):>5} cells ({100*len(overlap)/len(ref_set):.1f}% of reference)")
    print(f"    GA-only (novel) cells: {len(exp_set) - len(overlap):>5}")
    print(f"\n  Unique molecules evaluated: {eval_unique}")
    print(f"  Total evaluations:         {len(ga_df)}")


# ======================================================================
# Figure generation orchestrator
# ======================================================================

def _generate_all_figures(coords_2d, labels, generations, fitness, is_valid,
                          valid_smiles, ga_df, out_dir, dominant_fgs=None,
                          mahal_csv=None, dpi=300):
    """Generate all figures from pre-computed UMAP data."""
    print(f"\n{'='*60}")
    print("  Generating figures")
    print("=" * 60)

    # Standalone FG panels + combined FG grid
    print("\n  --- FG reference + GA + combined ---")
    plot_panel_a(coords_2d, labels, valid_smiles, out_dir,
                 dominant_fgs=dominant_fgs, dpi=dpi)
    exp_mask = labels == "explored"
    if exp_mask.sum() > 0:
        _, _, _, label_to_colour = _prepare_panel_a_data(
            coords_2d, labels, valid_smiles, dominant_fgs)
        plot_fg_ga(coords_2d, labels, valid_smiles, label_to_colour, out_dir,
                   dominant_fgs=dominant_fgs, dpi=dpi)
        plot_fg_combined(coords_2d, labels, valid_smiles, out_dir,
                         dominant_fgs=dominant_fgs, dpi=dpi)

    # Standalone panels
    print("\n  --- Coverage ---")
    plot_panel_b(coords_2d, labels, out_dir, dpi=dpi)

    print("\n  --- Validity ---")
    plot_panel_validity(coords_2d, labels, is_valid, out_dir, dpi=dpi)

    print("\n  --- Generations ---")
    plot_panel_c(coords_2d, labels, generations, out_dir, dpi=dpi)

    print("\n  --- Fitness landscape ---")
    plot_panel_d(coords_2d, labels, fitness, out_dir, dpi=dpi)

    # Combined 2x2 grid (coverage + validity + generations + fitness)
    print("\n  --- Combined 2x2 grid ---")
    plot_combined_grid(coords_2d, labels, generations, fitness, is_valid,
                       out_dir, dpi=dpi)

    # FG over generations (requires GA dataframe)
    if ga_df is not None:
        print("\n  --- FG over generations ---")
        plot_fg_over_generations(ga_df, out_dir, dpi=dpi)
    else:
        print("\n  Skipping FG-over-generations (no GA dataframe available)")

    # Mahalanobis / OOD panels
    if mahal_csv is not None:
        if not os.path.exists(mahal_csv):
            print(f"\n  WARNING: {mahal_csv} not found, skipping OOD panels")
        else:
            print(f"\n  --- Loading Mahalanobis data from {mahal_csv} ---")
            ood_any, mahal_mean = load_mahal_data_single(
                mahal_csv, valid_smiles, labels)

            print("\n  --- Mahalanobis distance ---")
            plot_panel_mahal(coords_2d, labels, mahal_mean, out_dir, dpi=dpi)

            print("\n  --- OOD classification ---")
            plot_panel_ood(coords_2d, labels, ood_any, out_dir, dpi=dpi)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chemical space analysis for the top hyperparameter configuration"
    )
    parser.add_argument("--sweep_dir", type=str, default=None,
                        help="Sweep directory (e.g. ../outputs/hyperparam_sweep)")
    parser.add_argument("--npz", type=str, default=None,
                        help="Path to existing .npz UMAP cache — skip steps 1-4, replot only")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory (auto-detected)")
    parser.add_argument("--ref_csv", type=str, default=None,
                        help="Path to reference set CSV (default: data/ngram_reference_10k.csv)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: {sweep_dir}/analysis/chemical_space)")
    parser.add_argument("--n_top", type=int, default=50,
                        help="Number of top molecules to highlight (default: 50)")
    parser.add_argument("--sample_ga", type=int, default=50000,
                        help="Max GA molecules to load (0=all, default: 50000)")
    parser.add_argument("--cache", type=str, default=None,
                        help="Path to .npz UMAP cache (default: {out_dir}/umap_cache.npz)")
    parser.add_argument("--no_cache", action="store_true",
                        help="Force recomputation of UMAP even if cache exists")
    parser.add_argument("--ga_csv", type=str, default=None,
                        help="Path to GA molecules CSV (for --npz mode FG-over-gen plot)")
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

    if args.out_dir is None:
        if args.npz is not None:
            args.out_dir = os.path.dirname(os.path.abspath(args.npz))
        elif args.sweep_dir is not None:
            args.out_dir = os.path.join(args.sweep_dir, "analysis", "chemical_space")
        else:
            args.out_dir = "."

    if args.cache is None:
        args.cache = os.path.join(args.out_dir, "umap_cache.npz")

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Mode A: --npz supplied → load cache and replot only
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
        is_valid = data["is_valid"] if "is_valid" in data else None
        valid_smiles = data["smiles"].tolist()
        dominant_fgs = data["dominant_fgs"] if "dominant_fgs" in data else None

        print(f"  Loaded {len(labels)} points from {args.npz}")
        print(f"    Reference: {(labels == 'reference').sum()}")
        print(f"    Explored:  {(labels == 'explored').sum()}")
        print(f"    Top:       {(labels == 'top').sum()}")
        if dominant_fgs is not None:
            print(f"    FG labels:  cached ({len(dominant_fgs)})")
        else:
            print(f"    FG labels:  not cached (will recompute via SMARTS)")
        if is_valid is not None:
            exp_mask = labels == "explored"
            exp_valid = is_valid[exp_mask]
            print(f"    Validity:   cached (valid={int((exp_valid==1).sum())}, "
                  f"invalid={int((exp_valid==0).sum())})")
        else:
            print(f"    Validity:   not cached (rerun with --no_cache to add)")

        # Load GA dataframe if available (for FG-over-generations)
        ga_df = None
        if args.ga_csv is not None and os.path.exists(args.ga_csv):
            ga_df = pd.read_csv(args.ga_csv)
            if "CanonicalSMILES" in ga_df.columns and "SMILES" not in ga_df.columns:
                ga_df.rename(columns={"CanonicalSMILES": "SMILES"}, inplace=True)
            print(f"  Loaded {len(ga_df)} GA molecules from {args.ga_csv}")
        elif args.sweep_dir is not None:
            # Try to load from sweep dir (best-effort — don't crash if missing)
            try:
                params, label, mean_fom = identify_top_config(args.sweep_dir)
                seed_dirs = find_seed_dirs(args.sweep_dir, params)
                if seed_dirs:
                    ga_df = load_ga_molecules(seed_dirs, sample_ga=args.sample_ga)
                    print(f"  Loaded {len(ga_df)} GA molecules from sweep dir")
            except SystemExit:
                print("  WARNING: Could not load GA data from sweep dir "
                      "(configs_aggregated.csv missing?). Skipping FG-over-generations.")
                ga_df = None

        _generate_all_figures(coords_2d, labels, generations, fitness, is_valid,
                              valid_smiles, ga_df, args.out_dir,
                              dominant_fgs=dominant_fgs,
                              mahal_csv=args.mahal_csv, dpi=args.dpi)
        return

    # ------------------------------------------------------------------
    # Mode B: Full pipeline (--sweep_dir required)
    # ------------------------------------------------------------------
    if args.sweep_dir is None:
        print("ERROR: Either --sweep_dir or --npz is required.")
        sys.exit(1)

    # ==================================================================
    # 1. Identify top config
    # ==================================================================
    print("=" * 60)
    print("  STEP 1: Identify top configuration")
    print("=" * 60)

    params, label, mean_fom = identify_top_config(args.sweep_dir)
    print(f"  Top config: {label}")
    print(f"  Mean best FOM1_avg: {mean_fom:.4f}")

    # ==================================================================
    # 2. Find seed directories
    # ==================================================================
    print(f"\n{'='*60}")
    print("  STEP 2: Find seed directories")
    print("=" * 60)

    seed_dirs = find_seed_dirs(args.sweep_dir, params)
    if not seed_dirs:
        print("ERROR: No run directories found matching top config.")
        sys.exit(1)
    print(f"  Found {len(seed_dirs)} seed directories:")
    for sd in seed_dirs:
        print(f"    {os.path.basename(sd)}")

    # ==================================================================
    # 3. Load data
    # ==================================================================
    print(f"\n{'='*60}")
    print("  STEP 3: Load data")
    print("=" * 60)

    ref_smiles = load_reference_set(args.ref_csv, args.data_dir)
    ga_df = load_ga_molecules(seed_dirs, sample_ga=args.sample_ga)
    print(f"  GA molecules loaded: {len(ga_df)}")

    # ==================================================================
    # 4. Compute UMAP
    # ==================================================================
    print(f"\n{'='*60}")
    print("  STEP 4: Compute UMAP embedding")
    print("=" * 60)

    coords_2d, labels, generations, fitness, is_valid, valid_smiles, dominant_fgs = \
        compute_or_load_umap(
            ref_smiles, ga_df, args.cache, no_cache=args.no_cache, n_top=args.n_top,
        )

    # ==================================================================
    # 5-6. Generate figures + statistics
    # ==================================================================
    _generate_all_figures(coords_2d, labels, generations, fitness, is_valid,
                          valid_smiles, ga_df, args.out_dir,
                          dominant_fgs=dominant_fgs,
                          mahal_csv=args.mahal_csv, dpi=args.dpi)

    print_coverage_stats(coords_2d, labels, ga_df)

    print(f"\nAll figures saved to: {args.out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
