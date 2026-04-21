"""
UMAP Chemical Space Comparison — NIST 8100 × WSGA × REINVENT.

Creates a shared UMAP embedding across three molecular populations at
multiple flash-point constraint levels, produces a rich metadata CSV,
and generates publication-quality comparison figures.

Figures produced:
  constraint_progression.png  — 1×4 source overlay (key narrative figure)
  fom1_landscape.png          — 1×4 FOM1 colourmap
  ood_heatmap.png             — 1×4 OOD count heatmap
  temporal_progression.png    — 1×4 generation / discovery order
  validity_overlay.png        — 1×4 valid vs invalid
  functional_groups.png       — single panel, all molecules by FG

Usage:
    cd scripts/
    python plot_umap_comparison.py \\
      --nist_csv ../training/data/nist_8100/fom1_cho_cleaned.csv \\
      --wsga_fp100 ../outputs/molprice_sweep_nist8100/nocost_nonbio_stable_seed42 \\
      --wsga_fp125 ../outputs/fp_sweep_nist8100/fp125_nonbio_stable_seed42 \\
      --wsga_fp150 ../outputs/fp_sweep_nist8100/fp150_nonbio_stable_seed42 \\
      --wsga_fp175 ../outputs/fp_sweep_nist8100/fp175_nonbio_stable_seed42 \\
      --reinvent_fp100 ../outputs/reinvent_fp_molprice_production/fp100_nocost_seed42 \\
      --reinvent_fp125 ../outputs/reinvent_fp_molprice_production/fp125_nocost_seed42 \\
      --reinvent_fp150 ../outputs/reinvent_fp_molprice_production/fp150_nocost_seed42 \\
      --reinvent_fp175 ../outputs/reinvent_fp_molprice_production/fp175_nocost_seed42 \\
      --out_dir ../outputs/umap_comparison
"""

import os
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
# Constants
# ======================================================================

FP_LEVELS = ["fp100", "fp125", "fp150", "fp175"]
FP_LABELS = {
    "fp100": r"FP $\geq$ 100 °C",
    "fp125": r"FP $\geq$ 125 °C",
    "fp150": r"FP $\geq$ 150 °C",
    "fp175": r"FP $\geq$ 175 °C",
}

SOURCE_COLORS = {
    "NIST 8100": "#BBBBBB",
    "WSGA":      "#4C72B0",
    "REINVENT":  "#E07B54",
}

MAHAL_COLS = [
    "Mahal_bp", "OOD_bp", "Mahal_mp", "OOD_mp",
    "Mahal_fp", "OOD_fp", "Mahal_dc", "OOD_dc",
    "Mahal_density_40C", "OOD_density_40C",
    "Mahal_viscosity_40C", "OOD_viscosity_40C",
    "Mahal_tc_40C", "OOD_tc_40C",
    "Mahal_cpsat_40C", "OOD_cpsat_40C",
    "Mahal_beta_40C", "OOD_beta_40C",
    "Mahal_fom1_40C", "OOD_fom1_40C",
]

PROPERTY_COLS = [
    "density_40C", "viscosity_40C", "tc_40C", "cpsat_40C", "beta_40C",
    "fom1_40C", "MW", "Cp_40", "alpha_40", "nu_40", "FOM1_40",
    "bp", "mp", "fp", "dc",
    "SCScore", "MolPrice", "Tox21_Score", "Biodegradable",
]

# Functional groups (consistent with plot_chemical_space.py)
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

FIXED_FG_COLOURS = {
    "Ether":           "#1f77b4",
    "Vinyl ether":     "#aec7e8",
    "Alcohol":         "#ff7f0e",
    "Aldehyde":        "#ffbb78",
    "Ketone":          "#2ca02c",
    "Ester":           "#98df8a",
    "Carboxylic acid": "#d62728",
    "Epoxide":         "#ff9896",
    "Alkene":          "#9467bd",
    "Terminal alkene": "#c5b0d5",
    "Acetal":          "#8c564b",
    "Hydrocarbon only": "#BBBBBB",
}


# ======================================================================
# Utility functions
# ======================================================================

def smiles_to_fingerprint(smi, radius=2, n_bits=2048):
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
    arr = np.zeros((len(fps), fps[0].GetNumBits()), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


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


def assign_dominant_fg(smiles_list):
    fg_names = list(FUNCTIONAL_GROUPS.keys())
    labels = []
    for smi in tqdm(smiles_list, desc="Detecting FGs"):
        counts = detect_functional_groups(smi)
        max_count = 0
        dominant = "Hydrocarbon only"
        for name in fg_names:
            c = counts.get(name, 0)
            if c > max_count:
                max_count = c
                dominant = FG_SHORT_NAMES.get(name, name)
        labels.append(dominant)
    return np.array(labels)


def robust_axis_limits(coords_2d, margin=0.08, percentile=0.5):
    lo, hi = percentile, 100 - percentile
    xmin, xmax = np.percentile(coords_2d[:, 0], [lo, hi])
    ymin, ymax = np.percentile(coords_2d[:, 1], [lo, hi])
    x_pad = (xmax - xmin) * margin
    y_pad = (ymax - ymin) * margin
    return (xmin - x_pad, xmax + x_pad), (ymin - y_pad, ymax + y_pad)


def apply_axis_limits(ax, coords_2d):
    xlim, ylim = robust_axis_limits(coords_2d)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def add_side_colorbar(fig, ax, mappable, label):
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02, shrink=1.0)
    cbar.set_label(label, fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    return cbar


def save_fig(fig, out_dir, basename, dpi=150):
    path = os.path.join(out_dir, f"{basename}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


# ======================================================================
# Data loading
# ======================================================================

def load_nist(nist_csv, baseline_csv=None):
    print(f"Loading NIST baseline from {nist_csv}...")
    df = pd.read_csv(nist_csv)
    df["source"] = "NIST 8100"
    df["generation"] = np.nan
    df["discovery_order"] = np.nan
    df["FitnessScore"] = np.nan
    df["FOM1_40"] = df["fom1_40C"]
    for col in MAHAL_COLS:
        if col not in df.columns:
            df[col] = 0.0 if col.startswith("OOD_") else np.nan
    df["OOD_any"] = 0
    df["OOD_count"] = 0

    # Assign validity from baseline constraint analysis
    if baseline_csv and os.path.exists(baseline_csv):
        bl = pd.read_csv(baseline_csv)
        bl["_valid"] = (
            ~bl["has_banned_fragments"]
            & (bl["SCScore"] <= 3)
            & (bl["Tox21_Score"] <= 3)
            & (bl["bp"] >= 100)
            & (bl["dc"] <= 7)
            & (bl["mp"] < -30)
            & (bl["fp"] >= 373.15)
        ).astype(int)
        validity_map = dict(zip(bl["SMILES"], bl["_valid"]))
        df["is_valid"] = df["SMILES"].map(validity_map).fillna(0).astype(int)
        n_valid = (df["is_valid"] == 1).sum()
        print(f"  Validity from baseline: {n_valid} pass / {len(df) - n_valid} fail")
    else:
        df["is_valid"] = np.nan
        print("  No baseline CSV — validity not assigned")

    print(f"  Loaded {len(df)} NIST molecules")
    return df


def load_multiseed(run_dirs, source_name, sample_n=None):
    """Load and deduplicate molecules from multiple seed directories.

    For WSGA: keeps all valid + stratified sample of invalid.
    For REINVENT: keeps all (small enough).
    Deduplicates by SMILES, keeping first occurrence (preserves generation info).
    """
    print(f"\nLoading {source_name} from {len(run_dirs)} seed(s)...")
    all_dfs = []
    for run_dir in run_dirs:
        csv_path = os.path.join(run_dir, "all_evaluated_molecules.csv")
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path, low_memory=False)
        seed = os.path.basename(run_dir).split("seed")[-1]
        df["seed"] = int(seed)
        all_dfs.append(df)
        n_valid = (df["is_valid"] == 1).sum()
        print(f"  seed{seed}: {len(df):,} total, {n_valid:,} valid")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["source"] = source_name

    if source_name == "REINVENT":
        # Assign discovery_order per-seed BEFORE dedup so it reflects
        # true temporal ordering within each seed's RL trajectory
        combined["discovery_order"] = combined.groupby("seed").cumcount()
        combined["generation"] = np.nan
    else:
        combined["discovery_order"] = np.nan

    # Ensure CanonicalSMILES exists
    if "CanonicalSMILES" not in combined.columns:
        combined["CanonicalSMILES"] = combined["SMILES"]

    # Deduplicate by SMILES — keep row with highest FOM1 (best occurrence)
    fom_col = "fom1_40C" if "fom1_40C" in combined.columns else "FOM1_40"
    combined = combined.sort_values(fom_col, ascending=False).drop_duplicates(
        "SMILES", keep="first"
    ).reset_index(drop=True)
    n_unique_valid = (combined["is_valid"] == 1).sum()
    print(f"  Unique SMILES: {len(combined):,} ({n_unique_valid:,} valid)")

    # FOM1-stratified sampling (same strategy for all sources)
    if sample_n and len(combined) > sample_n:
        fom_col = "fom1_40C" if "fom1_40C" in combined.columns else "FOM1_40"
        combined = combined.copy()
        combined["_fom_bin"] = pd.qcut(
            combined[fom_col].fillna(0), q=10, labels=False, duplicates="drop"
        )
        sampled = combined.groupby("_fom_bin", group_keys=False).apply(
            lambda g: g.sample(
                n=min(len(g), max(1, int(len(g) / len(combined) * sample_n))),
                random_state=42,
            )
        )
        if len(sampled) > sample_n:
            sampled = sampled.sample(n=sample_n, random_state=42)
        sampled = sampled.drop(columns=["_fom_bin"])
        combined = sampled.reset_index(drop=True)
        n_valid = (combined["is_valid"] == 1).sum()
        print(f"  Sampled to {len(combined):,} ({n_valid:,} valid, {len(combined) - n_valid:,} invalid)")

    # Fill missing columns
    for col in ["AvgTanimotoSimilarity", "NichedFitnessScore"]:
        if col not in combined.columns:
            combined[col] = np.nan

    # Recompute temporal_progress
    if source_name == "WSGA":
        gen = combined["generation"]
        if gen.notna().any():
            gmin, gmax = gen.min(), gen.max()
            if gmax > gmin:
                combined["temporal_progress"] = (gen - gmin) / (gmax - gmin)
            else:
                combined["temporal_progress"] = np.nan
        else:
            combined["temporal_progress"] = np.nan
    elif source_name == "REINVENT":
        do = combined["discovery_order"]
        dmin, dmax = do.min(), do.max()
        if dmax > dmin:
            combined["temporal_progress"] = (do - dmin) / (dmax - dmin)
        else:
            combined["temporal_progress"] = np.nan
    else:
        combined["temporal_progress"] = np.nan

    return combined


def combine_all(nist_df, wsga_df, reinvent_df):
    all_dfs = [nist_df, wsga_df, reinvent_df]

    # Get union of all columns
    all_cols = set()
    for df in all_dfs:
        all_cols.update(df.columns)

    # Build missing columns as a single concat to avoid fragmentation
    for df in all_dfs:
        missing = {col: np.nan for col in all_cols if col not in df.columns}
        if missing:
            df_missing = pd.DataFrame(missing, index=df.index)
            for col in missing:
                df[col] = df_missing[col]

    combined = pd.concat(all_dfs, ignore_index=True)

    print(f"\nCombined dataset: {len(combined):,} molecules")
    for src in ["NIST 8100", "WSGA", "REINVENT"]:
        n = (combined["source"] == src).sum()
        n_valid = ((combined["source"] == src) & (combined["is_valid"] == 1)).sum()
        print(f"  {src}: {n:,} ({n_valid:,} valid)")
    return combined


# ======================================================================
# UMAP computation
# ======================================================================

def compute_or_load_umap(combined_df, cache_path, no_cache=False):
    if not no_cache and os.path.exists(cache_path):
        print(f"\nLoading cached UMAP from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["coords_2d"],
            data["valid_indices"],
            data["dominant_fgs"],
        )

    smiles_list = combined_df["SMILES"].tolist()

    # Compute fingerprints
    print(f"\nComputing fingerprints for {len(smiles_list):,} molecules...")
    fps = []
    valid_indices = []
    for i, smi in enumerate(tqdm(smiles_list, desc="Fingerprinting")):
        fp = smiles_to_fingerprint(smi)
        if fp is not None:
            fps.append(fp)
            valid_indices.append(i)

    valid_indices = np.array(valid_indices)
    print(f"  Valid fingerprints: {len(fps):,} / {len(smiles_list):,}")

    fp_matrix = fps_to_numpy(fps)

    # Fit UMAP
    print(f"  Fitting UMAP on {fp_matrix.shape[0]:,} molecules (this may take 15-20 min)...")
    reducer = UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric="jaccard",
        random_state=42,
        n_jobs=1,
    )
    coords_2d = reducer.fit_transform(fp_matrix)
    print(f"  UMAP complete: {coords_2d.shape}")

    # Compute dominant functional groups
    print("  Computing functional groups...")
    valid_smiles = [smiles_list[i] for i in valid_indices]
    dominant_fgs = assign_dominant_fg(valid_smiles)

    # Cache
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez_compressed(
        cache_path,
        coords_2d=coords_2d,
        valid_indices=valid_indices,
        dominant_fgs=dominant_fgs,
    )
    print(f"  Cached UMAP to {cache_path}")

    return coords_2d, valid_indices, dominant_fgs


# ======================================================================
# Metadata CSV
# ======================================================================

def save_metadata_csv(combined_df, coords_2d, valid_indices, dominant_fgs, out_path):
    # Build output dataframe from valid rows only
    meta = combined_df.iloc[valid_indices].copy()
    meta = meta.reset_index(drop=True)
    meta.insert(0, "UMAP_x", coords_2d[:, 0])
    meta.insert(1, "UMAP_y", coords_2d[:, 1])
    meta["dominant_functional_group"] = dominant_fgs

    # Reorder key columns to front
    front_cols = [
        "UMAP_x", "UMAP_y", "source", "fp_level",
        "SMILES", "CanonicalSMILES",
        "generation", "discovery_order", "temporal_progress",
        "FOM1_40", "fom1_40C", "FitnessScore", "is_valid",
    ]
    front_cols = [c for c in front_cols if c in meta.columns]
    other_cols = [c for c in meta.columns if c not in front_cols]
    meta = meta[front_cols + other_cols]

    meta.to_csv(out_path, index=False)
    print(f"\nSaved metadata CSV: {out_path} ({len(meta):,} rows, {len(meta.columns)} columns)")
    return meta


# ======================================================================
# Plotting functions
# ======================================================================

def _get_fp_mask(meta, fp_level):
    """Get mask for molecules at a given FP level (including NIST which is 'all')."""
    return (meta["fp_level"] == fp_level) | (meta["source"] == "NIST 8100")


def plot_source_overlay(meta, coords_all, out_dir, dpi=150):
    """1×4 grid: source overlay per FP level."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    for ax, fp in zip(axes, FP_LEVELS):
        mask = _get_fp_mask(meta, fp)
        coords = coords_all[mask.values]
        src = meta.loc[mask, "source"]

        # Draw in z-order: NIST, WSGA, REINVENT
        for source, color, s, alpha, zorder in [
            ("NIST 8100", SOURCE_COLORS["NIST 8100"], 3, 0.3, 1),
            ("WSGA", SOURCE_COLORS["WSGA"], 4, 0.3, 2),
            ("REINVENT", SOURCE_COLORS["REINVENT"], 4, 0.3, 3),
        ]:
            m = src == source
            n = m.sum()
            if n > 0:
                ax.scatter(
                    coords[m.values, 0], coords[m.values, 1],
                    c=color, s=s, alpha=alpha, zorder=zorder,
                    label=f"{source} ({n:,})", rasterized=True,
                )

        ax.legend(fontsize=9, markerscale=3, frameon=False, loc="upper right")
        apply_axis_limits(ax, coords_all)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(FP_LABELS[fp], fontsize=13)

    # Panel labels
    for i, ax in enumerate(axes):
        ax.text(
            0.02, 0.98, f"({chr(97 + i)})",
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            va="top", ha="left",
        )

    fig.tight_layout()
    save_fig(fig, out_dir, "constraint_progression", dpi=dpi)


def plot_fom1_landscape(meta, coords_all, out_dir, dpi=150):
    """1×4 grid: FOM1 colourmap per FP level."""
    fig, axes = plt.subplots(1, 4, figsize=(26, 6))

    for ax, fp in zip(axes, FP_LEVELS):
        mask = _get_fp_mask(meta, fp)
        coords = coords_all[mask.values]
        src = meta.loc[mask, "source"]
        fom1 = meta.loc[mask, "FOM1_40"]

        # NIST as grey background
        nist_m = src == "NIST 8100"
        ax.scatter(
            coords[nist_m.values, 0], coords[nist_m.values, 1],
            c="#EEEEEE", s=2, alpha=0.2, rasterized=True,
        )

        # Generative molecules coloured by FOM1
        gen_m = src != "NIST 8100"
        fom1_gen = fom1[gen_m]
        valid_fom = fom1_gen.dropna()
        if len(valid_fom) > 0:
            vmin, vmax = np.percentile(valid_fom, [5, 95])
            sc = ax.scatter(
                coords[gen_m.values, 0], coords[gen_m.values, 1],
                c=fom1_gen, cmap="viridis", s=3, alpha=0.3,
                vmin=vmin, vmax=vmax, rasterized=True,
            )
            add_side_colorbar(fig, ax, sc, "FOM1")

        apply_axis_limits(ax, coords_all)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(FP_LABELS[fp], fontsize=13)
        ax.text(
            0.02, 0.98, f"({chr(97 + FP_LEVELS.index(fp))})",
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            va="top", ha="left",
        )

    fig.tight_layout()
    save_fig(fig, out_dir, "fom1_landscape", dpi=dpi)


def plot_ood_heatmap(meta, coords_all, out_dir, dpi=150):
    """1×4 grid: OOD count heatmap per FP level."""
    fig, axes = plt.subplots(1, 4, figsize=(26, 6))

    for ax, fp in zip(axes, FP_LEVELS):
        mask = _get_fp_mask(meta, fp)
        coords = coords_all[mask.values]
        src = meta.loc[mask, "source"]
        ood = meta.loc[mask, "OOD_count"]

        # NIST as grey background
        nist_m = src == "NIST 8100"
        ax.scatter(
            coords[nist_m.values, 0], coords[nist_m.values, 1],
            c="#EEEEEE", s=2, alpha=0.2, rasterized=True,
        )

        # Generative molecules coloured by OOD_count
        gen_m = src != "NIST 8100"
        ood_gen = ood[gen_m]
        if len(ood_gen.dropna()) > 0:
            sc = ax.scatter(
                coords[gen_m.values, 0], coords[gen_m.values, 1],
                c=ood_gen, cmap="inferno", s=3, alpha=0.3,
                vmin=0, vmax=10, rasterized=True,
            )
            add_side_colorbar(fig, ax, sc, "OOD count")

        apply_axis_limits(ax, coords_all)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(FP_LABELS[fp], fontsize=13)
        ax.text(
            0.02, 0.98, f"({chr(97 + FP_LEVELS.index(fp))})",
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            va="top", ha="left",
        )

    fig.tight_layout()
    save_fig(fig, out_dir, "ood_heatmap", dpi=dpi)


def plot_temporal_progression(meta, coords_all, out_dir, dpi=150):
    """1×4 grid: temporal progression per FP level, split by method."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    for col_idx, fp in enumerate(FP_LEVELS):
        mask = _get_fp_mask(meta, fp)

        for row_idx, method in enumerate(["WSGA", "REINVENT"]):
            ax = axes[row_idx, col_idx]
            src = meta.loc[mask, "source"]

            # NIST background
            nist_m = mask & (meta["source"] == "NIST 8100")
            nist_coords = coords_all[nist_m.values]
            ax.scatter(
                nist_coords[:, 0], nist_coords[:, 1],
                c="#EEEEEE", s=2, alpha=0.2, rasterized=True,
            )

            # Method-specific temporal colouring
            method_m = mask & (meta["source"] == method)
            method_coords = coords_all[method_m.values]
            tp = meta.loc[method_m, "temporal_progress"]

            if len(tp.dropna()) > 0:
                sc = ax.scatter(
                    method_coords[:, 0], method_coords[:, 1],
                    c=tp, cmap="plasma", s=3, alpha=0.3, rasterized=True,
                    vmin=0, vmax=1,
                )
                label = "Generation" if method == "WSGA" else "Discovery order"
                add_side_colorbar(fig, ax, sc, label)

            apply_axis_limits(ax, coords_all)
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_xlabel(FP_LABELS[fp], fontsize=13)
            if col_idx == 0:
                ax.set_ylabel(method, fontsize=13)

            label_idx = row_idx * 4 + col_idx
            ax.text(
                0.02, 0.98, f"({chr(97 + label_idx)})",
                transform=ax.transAxes, fontsize=14, fontweight="bold",
                va="top", ha="left",
            )

    fig.tight_layout()
    save_fig(fig, out_dir, "temporal_progression", dpi=dpi)


def plot_validity_overlay(meta, coords_all, out_dir, dpi=150):
    """1×4 grid: valid vs invalid per FP level."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    for ax, fp in zip(axes, FP_LEVELS):
        mask = _get_fp_mask(meta, fp)
        coords = coords_all[mask.values]
        src = meta.loc[mask, "source"]
        valid = meta.loc[mask, "is_valid"]

        # NIST background
        nist_m = src == "NIST 8100"
        ax.scatter(
            coords[nist_m.values, 0], coords[nist_m.values, 1],
            c="#EEEEEE", s=2, alpha=0.2, rasterized=True,
        )

        # Invalid molecules (red, behind)
        gen_m = src != "NIST 8100"
        invalid_m = gen_m & (valid != 1)
        n_invalid = invalid_m.sum()
        if n_invalid > 0:
            ax.scatter(
                coords[invalid_m.values, 0], coords[invalid_m.values, 1],
                c="#d62728", s=3, alpha=0.15, rasterized=True,
                label=f"Invalid ({n_invalid:,})",
            )

        # Valid molecules (green, on top)
        valid_m = gen_m & (valid == 1)
        n_valid = valid_m.sum()
        if n_valid > 0:
            ax.scatter(
                coords[valid_m.values, 0], coords[valid_m.values, 1],
                c="#2ca02c", s=4, alpha=0.3, rasterized=True, zorder=3,
                label=f"Valid ({n_valid:,})",
            )

        ax.legend(fontsize=9, markerscale=3, frameon=False, loc="upper right")
        apply_axis_limits(ax, coords_all)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(FP_LABELS[fp], fontsize=13)
        ax.text(
            0.02, 0.98, f"({chr(97 + FP_LEVELS.index(fp))})",
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            va="top", ha="left",
        )

    fig.tight_layout()
    save_fig(fig, out_dir, "validity_overlay", dpi=dpi)


def plot_functional_groups(meta, coords_all, out_dir, dpi=150):
    """Single panel: all molecules coloured by dominant functional group."""
    fig, ax = plt.subplots(figsize=(10, 6))

    fgs = meta["dominant_functional_group"]
    label_counts = Counter(fgs)

    # Draw by FG, most common first (least common on top)
    for fg, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        m = fgs == fg
        color = FIXED_FG_COLOURS.get(fg, "#999999")
        ax.scatter(
            coords_all[m.values, 0], coords_all[m.values, 1],
            c=color, s=3, alpha=0.3, rasterized=True,
            label=f"{fg} ({count:,})",
        )

    ax.legend(
        fontsize=8, markerscale=3, frameon=False,
        loc="upper right", ncol=2,
    )
    apply_axis_limits(ax, coords_all)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    save_fig(fig, out_dir, "functional_groups", dpi=dpi)


# ======================================================================
# Pareto front loading
# ======================================================================

def load_pareto_csvs(wsga_pareto_csv, reinvent_pareto_csv):
    """Load pre-computed global 3D Pareto front SMILES from CSVs."""
    pareto = {"WSGA": set(), "REINVENT": set()}

    if wsga_pareto_csv and os.path.exists(wsga_pareto_csv):
        df = pd.read_csv(wsga_pareto_csv)
        pareto["WSGA"] = set(df["SMILES"].tolist())
        print(f"  WSGA Pareto: {len(pareto['WSGA'])} molecules (max FOM1={df['fom1_40C'].max():.2f})")

    if reinvent_pareto_csv and os.path.exists(reinvent_pareto_csv):
        df = pd.read_csv(reinvent_pareto_csv)
        pareto["REINVENT"] = set(df["SMILES"].tolist())
        print(f"  REINVENT Pareto: {len(pareto['REINVENT'])} molecules (max FOM1={df['fom1_40C'].max():.2f})")

    return pareto


# ======================================================================
# Main 3×4 grid figure
# ======================================================================

def _add_invisible_colorbar(fig, ax):
    """Add invisible colorbar spacer so axes width matches panels with colorbars."""
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, shrink=1.0)
    cbar.ax.set_visible(False)


def plot_main_grid(meta, coords_all, out_dir, pareto, dpi=150):
    """3-column (NIST / WSGA / REINVENT) × 4-row (FG / Generation / FOM1 / Valid) grid.

    Colorbars on rightmost column only; invisible spacers on other columns.
    Pareto front molecules are highlighted with star markers.
    """

    # Masks for each source
    nist_mask = meta["source"] == "NIST 8100"
    wsga_mask = meta["source"] == "WSGA"
    reinvent_mask = meta["source"] == "REINVENT"

    col_masks = [nist_mask, wsga_mask, reinvent_mask]
    col_labels = ["NIST 8100", "WSGA", "REINVENT"]
    row_labels = ["Functional group", "Generation", "FOM1", "Validity"]

    # Pre-compute shared FOM1 range across all sources
    all_fom1 = meta["FOM1_40"].dropna()
    fom1_vmin, fom1_vmax = np.percentile(all_fom1, [2, 98])

    fig, axes = plt.subplots(4, 3, figsize=(18, 22))

    # Pareto overlay: enlarged points with black border showing underlying data
    def _draw_pareto_rings(ax, coords, smiles_series, method,
                           c=None, cmap=None, vmin=None, vmax=None,
                           add_legend=False):
        """Draw Pareto molecules as larger filled circles with black edge.

        If c/cmap are provided, fills match the row's colourmap.
        Otherwise uses a neutral grey fill.
        """
        if method not in pareto or len(pareto[method]) == 0:
            return
        p_mask = smiles_series.isin(pareto[method])
        n_pareto = p_mask.sum()
        if n_pareto == 0:
            return
        px = coords[p_mask.values, 0]
        py = coords[p_mask.values, 1]
        kwargs = dict(s=90, edgecolors="black", linewidths=2.0,
                      alpha=1.0, zorder=20,
                      label=f"Pareto ({n_pareto})" if add_legend else None)
        if c is not None and cmap is not None:
            # Numeric colour data with a colormap — fill NaN to avoid invisible points
            c_arr = pd.Series(c).fillna(vmax if vmax is not None else 1.0).values
            ax.scatter(px, py, c=c_arr, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        elif c is not None:
            ax.scatter(px, py, c=c, **kwargs)
        else:
            ax.scatter(px, py, c="#888888", **kwargs)

    for col_idx, (mask, col_label) in enumerate(zip(col_masks, col_labels)):
        coords = coords_all[mask.values]
        sub = meta.loc[mask]
        method = col_label if col_label in ("WSGA", "REINVENT") else None
        is_last_col = col_idx == 2

        # ---- Row 0: Functional Groups ----
        ax = axes[0, col_idx]
        fgs = sub["dominant_functional_group"]
        fg_counts = Counter(fgs)
        for fg, count in sorted(fg_counts.items(), key=lambda x: -x[1]):
            m = fgs == fg
            color = FIXED_FG_COLOURS.get(fg, "#999999")
            ax.scatter(
                coords[m.values, 0], coords[m.values, 1],
                c=color, s=3, alpha=0.3, rasterized=True,
                label=fg,
            )
        # FG colours for Pareto molecules
        if method:
            p_mask = sub["SMILES"].isin(pareto.get(method, set()))
            p_fgs = sub.loc[p_mask, "dominant_functional_group"]
            p_colors = [FIXED_FG_COLOURS.get(fg, "#999999") for fg in p_fgs]
            _draw_pareto_rings(ax, coords, sub["SMILES"], method,
                               c=p_colors if p_colors else None,
                               add_legend=True)
        leg = ax.legend(fontsize=7, markerscale=1.5, frameon=True,
                        loc="upper right", ncol=2,
                        facecolor="white", edgecolor="#cccccc", framealpha=0.85)
        leg.set_zorder(30)
        for handle in leg.legend_handles:
            handle.set_alpha(1.0)
            if handle.get_label().startswith("Pareto"):
                handle.set_sizes([30])
                handle.set_facecolor("none")
                handle.set_edgecolor("black")
                handle.set_linewidth(1.5)
        _add_invisible_colorbar(fig, ax)

        # ---- Row 1: Generation / temporal ----
        ax = axes[1, col_idx]
        if col_label == "NIST 8100":
            # NIST = starting knowledge (generation 0)
            sc = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=np.zeros(len(coords)), cmap="plasma", s=3, alpha=0.3,
                rasterized=True, vmin=0, vmax=1,
            )
            _add_invisible_colorbar(fig, ax)
        else:
            tp = sub["temporal_progress"]
            if tp.notna().any():
                sc = ax.scatter(
                    coords[:, 0], coords[:, 1],
                    c=tp, cmap="plasma", s=3, alpha=0.3, rasterized=True,
                    vmin=0, vmax=1,
                )
                if is_last_col:
                    cbar_label = "Discovery order"
                    add_side_colorbar(fig, ax, sc, cbar_label)
                else:
                    _add_invisible_colorbar(fig, ax)
            else:
                _add_invisible_colorbar(fig, ax)
            if method:
                p_mask = sub["SMILES"].isin(pareto.get(method, set()))
                p_tp = sub.loc[p_mask, "temporal_progress"]
                _draw_pareto_rings(ax, coords, sub["SMILES"], method,
                                   c=p_tp, cmap="plasma", vmin=0, vmax=1)

        # ---- Row 2: FOM1 (shared colorbar range) ----
        ax = axes[2, col_idx]
        fom1 = sub["FOM1_40"] if "FOM1_40" in sub.columns else (
            sub["fom1_40C"] if "fom1_40C" in sub.columns else None
        )
        if fom1 is not None and fom1.notna().any():
            sc = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=fom1, cmap="viridis", s=3, alpha=0.3, rasterized=True,
                vmin=fom1_vmin, vmax=fom1_vmax,
            )
            if is_last_col:
                add_side_colorbar(fig, ax, sc, "FOM1")
            else:
                _add_invisible_colorbar(fig, ax)
        else:
            _add_invisible_colorbar(fig, ax)
        if method:
            p_mask = sub["SMILES"].isin(pareto.get(method, set()))
            p_fom1 = sub.loc[p_mask, "FOM1_40"]
            _draw_pareto_rings(ax, coords, sub["SMILES"], method,
                               c=p_fom1, cmap="viridis",
                               vmin=fom1_vmin, vmax=fom1_vmax)

        # ---- Row 3: Validity ----
        ax = axes[3, col_idx]
        valid = sub["is_valid"]
        inv_m = valid != 1
        val_m = valid == 1
        n_inv = inv_m.sum()
        n_val = val_m.sum()
        if n_inv > 0:
            ax.scatter(
                coords[inv_m.values, 0], coords[inv_m.values, 1],
                c="#d62728", s=3, alpha=0.15, rasterized=True,
                label=f"Invalid ({n_inv:,})",
            )
        if n_val > 0:
            ax.scatter(
                coords[val_m.values, 0], coords[val_m.values, 1],
                c="#2ca02c", s=4, alpha=0.3, rasterized=True, zorder=3,
                label=f"Valid ({n_val:,})",
            )
        ax.legend(fontsize=8, markerscale=2.5, frameon=True, loc="upper right",
                  facecolor="white", edgecolor="#cccccc", framealpha=0.85)
        if method:
            # Pareto molecules are all valid — show as green with black border
            _draw_pareto_rings(ax, coords, sub["SMILES"], method, c="#2ca02c")
        _add_invisible_colorbar(fig, ax)

    # Style all axes
    for i in range(4):
        for j in range(3):
            ax = axes[i, j]
            apply_axis_limits(ax, coords_all)
            ax.set_xticks([])
            ax.set_yticks([])

    # Column headers
    for j, label in enumerate(col_labels):
        axes[0, j].text(
            0.5, 1.08, label, transform=axes[0, j].transAxes,
            fontsize=15, fontweight="bold", ha="center", va="bottom",
        )

    # Row labels
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, fontsize=13, labelpad=10)

    # Panel labels (a)-(l)
    for i in range(4):
        for j in range(3):
            idx = i * 3 + j
            axes[i, j].text(
                0.02, 0.98, f"({chr(97 + idx)})",
                transform=axes[i, j].transAxes, fontsize=12,
                fontweight="bold", va="top", ha="left",
            )

    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    save_fig(fig, out_dir, "main_grid", dpi=dpi)


# ======================================================================
# SI constraint progression (1×4 overlay across FP levels)
# ======================================================================

def plot_si_constraint_progression(meta, coords_all, out_dir, dpi=150):
    """SI figure: 1×4 source overlay across FP levels."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    for ax, fp in zip(axes, FP_LEVELS):
        mask = _get_fp_mask(meta, fp)
        coords = coords_all[mask.values]
        src = meta.loc[mask, "source"]

        for source, color, s, alpha, zorder in [
            ("NIST 8100", SOURCE_COLORS["NIST 8100"], 3, 0.3, 1),
            ("WSGA", SOURCE_COLORS["WSGA"], 4, 0.3, 2),
            ("REINVENT", SOURCE_COLORS["REINVENT"], 4, 0.3, 3),
        ]:
            m = src == source
            n = m.sum()
            if n > 0:
                ax.scatter(
                    coords[m.values, 0], coords[m.values, 1],
                    c=color, s=s, alpha=alpha, zorder=zorder,
                    label=f"{source} ({n:,})", rasterized=True,
                )

        ax.legend(fontsize=9, markerscale=3, frameon=False, loc="upper right")
        apply_axis_limits(ax, coords_all)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(FP_LABELS[fp], fontsize=13)

    for i, ax in enumerate(axes):
        ax.text(
            0.02, 0.98, f"({chr(97 + i)})",
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            va="top", ha="left",
        )

    fig.tight_layout()
    save_fig(fig, out_dir, "si_constraint_progression", dpi=dpi)


# ======================================================================
# Main
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="UMAP comparison: NIST 8100 × WSGA × REINVENT"
    )
    p.add_argument("--nist_csv", required=True)
    p.add_argument("--baseline_csv", default=None,
                   help="BaselineFOM1Eval categories CSV for NIST validity assignment")
    p.add_argument("--wsga_dirs", nargs="+", required=True,
                   help="WSGA run directories (multiple seeds)")
    p.add_argument("--reinvent_dirs", nargs="+", required=True,
                   help="REINVENT run directories (multiple seeds)")
    p.add_argument("--wsga_pareto_csv", default=None,
                   help="Pre-computed WSGA 3D Pareto front CSV (SMILES,fom1_40C,fp,MolPrice)")
    p.add_argument("--reinvent_pareto_csv", default=None,
                   help="Pre-computed REINVENT 3D Pareto front CSV")
    p.add_argument("--wsga_pareto_discovery", default=None,
                   help="CSV mapping Pareto SMILES to earliest_generation")
    p.add_argument("--reinvent_pareto_discovery", default=None,
                   help="CSV mapping Pareto SMILES to earliest_step")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--sample_wsga", type=int, default=100000,
                   help="Max WSGA molecules after dedup (0=all)")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Phase 1: Data Loading ----
    nist_df = load_nist(args.nist_csv, args.baseline_csv)

    sample_n = args.sample_wsga if args.sample_wsga > 0 else None
    wsga_df = load_multiseed(args.wsga_dirs, "WSGA", sample_n=sample_n)
    reinvent_df = load_multiseed(args.reinvent_dirs, "REINVENT", sample_n=None)

    # Load Pareto front SMILES and add any missing Pareto molecules to the data
    print("\nLoading Pareto fronts...")
    pareto = load_pareto_csvs(args.wsga_pareto_csv, args.reinvent_pareto_csv)

    # Load discovery lookups for Pareto molecules (earliest generation/step)
    wsga_disc = {}
    if args.wsga_pareto_discovery and os.path.exists(args.wsga_pareto_discovery):
        disc_df = pd.read_csv(args.wsga_pareto_discovery)
        wsga_disc = dict(zip(disc_df["SMILES"], disc_df["earliest_generation"]))
        print(f"  WSGA discovery lookup: {len(wsga_disc)} molecules")

    ri_disc = {}
    if args.reinvent_pareto_discovery and os.path.exists(args.reinvent_pareto_discovery):
        disc_df = pd.read_csv(args.reinvent_pareto_discovery)
        ri_disc = dict(zip(disc_df["SMILES"], disc_df["earliest_step"]))
        print(f"  REINVENT discovery lookup: {len(ri_disc)} molecules")

    # Ensure all Pareto molecules are in the dataset (add if missing)
    for method, df in [("WSGA", wsga_df), ("REINVENT", reinvent_df)]:
        existing = set(df["SMILES"].tolist())
        missing = pareto[method] - existing
        if missing:
            print(f"  Adding {len(missing)} missing {method} Pareto molecules to dataset")
            csv_path = args.wsga_pareto_csv if method == "WSGA" else args.reinvent_pareto_csv
            if csv_path:
                pf_df = pd.read_csv(csv_path)
                pf_missing = pf_df[pf_df["SMILES"].isin(missing)].copy()
                pf_missing["source"] = method
                pf_missing["is_valid"] = 1
                # Ensure FOM1_40 is populated from fom1_40C
                if "FOM1_40" not in pf_missing.columns and "fom1_40C" in pf_missing.columns:
                    pf_missing["FOM1_40"] = pf_missing["fom1_40C"]
                # Assign real discovery time from cross-reference lookup
                disc = wsga_disc if method == "WSGA" else ri_disc
                if method == "WSGA":
                    pf_missing["generation"] = pf_missing["SMILES"].map(disc)
                    # Normalize to [0,1] using the same range as the main data
                    gen_max = df["generation"].max() if "generation" in df.columns else 200
                    pf_missing["temporal_progress"] = pf_missing["generation"] / gen_max
                else:
                    pf_missing["discovery_order"] = pf_missing["SMILES"].map(disc)
                    do_max = df["discovery_order"].max() if "discovery_order" in df.columns else 20000
                    pf_missing["temporal_progress"] = pf_missing["discovery_order"] / do_max
                pf_missing["temporal_progress"] = pf_missing["temporal_progress"].clip(0, 1)
                if method == "WSGA":
                    wsga_df = pd.concat([wsga_df, pf_missing], ignore_index=True)
                else:
                    reinvent_df = pd.concat([reinvent_df, pf_missing], ignore_index=True)

    combined = combine_all(nist_df, wsga_df, reinvent_df)

    # ---- Phase 2 + 3: UMAP Embedding ----
    cache_path = os.path.join(args.out_dir, "umap_cache.npz")
    coords_2d, valid_indices, dominant_fgs = compute_or_load_umap(
        combined, cache_path, no_cache=args.no_cache
    )

    # ---- Phase 4: Metadata CSV ----
    csv_path = os.path.join(args.out_dir, "umap_metadata.csv")
    meta = save_metadata_csv(combined, coords_2d, valid_indices, dominant_fgs, csv_path)

    coords_all = coords_2d

    # ---- Phase 5: Plotting ----
    print("\nGenerating figures...")
    plot_main_grid(meta, coords_all, args.out_dir, pareto, dpi=args.dpi)
    plot_functional_groups(meta, coords_all, args.out_dir, args.dpi)
    # plot_ood_heatmap requires fp_level — skipped in single-FP mode

    print("\nDone!")


if __name__ == "__main__":
    main()
