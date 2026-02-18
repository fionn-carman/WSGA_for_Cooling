"""
Local Chemical Space Visualisation — Functional Group Colouring

Generates 5000 molecules from the 8-gram model, computes Morgan
fingerprints, runs UMAP, and colours each molecule by its dominant
functional group. Designed to run locally (no PBS required).

Usage:
    python plot_local_fg_umap.py
    python plot_local_fg_umap.py --n_mol 10000 --out_dir ../outputs/local_fg_umap
"""

import os
import sys
import argparse
import warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
except ImportError:
    print("ERROR: RDKit required. Install: conda install -c conda-forge rdkit")
    sys.exit(1)

RDLogger.DisableLog("rdApp.*")

# Block TensorFlow before importing UMAP (prevents crash on some systems)
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
        print("ERROR: umap-learn required. Install: pip install umap-learn")
        sys.exit(1)


# ======================================================================
# Functional groups (consistent with compare_chemical_space.py)
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


# ======================================================================
# Fingerprints
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


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate molecules and plot UMAP coloured by dominant functional group"
    )
    parser.add_argument("--n_mol", type=int, default=5000,
                        help="Number of molecules to generate (default: 5000)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory for training SMILES (auto-detected if omitted)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: ../outputs/local_fg_umap)")
    args = parser.parse_args()

    # Auto-detect data directory
    if args.data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for candidate in [
            os.path.join(script_dir, "..", "data"),
            os.path.join(script_dir, "..", "..", "data"),
        ]:
            if os.path.isdir(candidate):
                args.data_dir = os.path.abspath(candidate)
                break
        if args.data_dir is None:
            print("ERROR: Could not find data directory. Use --data_dir")
            sys.exit(1)

    if args.out_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.out_dir = os.path.abspath(os.path.join(script_dir, "..", "outputs", "local_fg_umap"))

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Generate molecules via n-gram model
    # ------------------------------------------------------------------
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    sys.path.insert(0, os.path.abspath(src_dir))

    from generate_molecules import generate_initial_population, load_combined_training_data

    print(f"Loading training data from {args.data_dir}...")
    training_smiles = load_combined_training_data(args.data_dir)
    print(f"  {len(training_smiles)} training SMILES loaded")

    print(f"\nGenerating {args.n_mol} molecules via 8-gram model...")
    ref_df = generate_initial_population(
        n=args.n_mol,
        max_heavy_atoms=30,
        min_heavy_atoms=5,
        max_carbons=30,
        max_oxygens=6,
        training_smiles=training_smiles,
        ngram_order=8,
    )
    smiles_list = ref_df["SMILES"].tolist()
    print(f"  Generated {len(smiles_list)} molecules")

    # ------------------------------------------------------------------
    # 2. Compute fingerprints
    # ------------------------------------------------------------------
    print(f"\nComputing Morgan fingerprints...")
    fps = []
    valid_smiles = []
    for i, smi in enumerate(smiles_list):
        if i % 1000 == 0:
            print(f"\r  Fingerprinting: {i}/{len(smiles_list)}", end="", flush=True)
        fp = smiles_to_fingerprint(smi)
        if fp is not None:
            fps.append(fp)
            valid_smiles.append(smi)
    print(f"\r  Fingerprinting: {len(smiles_list)}/{len(smiles_list)} — {len(fps)} valid")

    if len(fps) < 50:
        print("ERROR: Too few valid molecules for UMAP")
        sys.exit(1)

    fp_matrix = fps_to_numpy(fps)
    print(f"  Fingerprint matrix: {fp_matrix.shape}")

    # ------------------------------------------------------------------
    # 3. Fit UMAP
    # ------------------------------------------------------------------
    print(f"\nFitting UMAP on {len(fps)} molecules...")
    reducer = UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric="jaccard",
        random_state=42,
        n_jobs=1,
    )
    coords_2d = reducer.fit_transform(fp_matrix)
    print(f"  UMAP complete: {coords_2d.shape}")

    # ------------------------------------------------------------------
    # 4. Detect functional groups
    # ------------------------------------------------------------------
    print(f"\nDetecting functional groups...")
    fg_names = list(FUNCTIONAL_GROUPS.keys())
    n_mol = len(valid_smiles)
    fg_counts = np.zeros((n_mol, len(fg_names)), dtype=np.int32)

    for i, smi in enumerate(valid_smiles):
        if i % 1000 == 0:
            print(f"\r  FG detection: {i}/{n_mol}", end="", flush=True)
        counts = detect_functional_groups(smi)
        for fi, name in enumerate(fg_names):
            fg_counts[i, fi] = counts.get(name, 0)
    print(f"\r  FG detection: {n_mol}/{n_mol} — done")

    # Assign dominant FG per molecule
    dominant_labels = []
    for i in range(n_mol):
        max_count = 0
        dominant = "Hydrocarbon only"
        for fi, name in enumerate(fg_names):
            if fg_counts[i, fi] > max_count:
                max_count = fg_counts[i, fi]
                dominant = FG_SHORT_NAMES.get(name, name)
        dominant_labels.append(dominant)
    dominant_labels = np.array(dominant_labels)

    # Print summary
    label_counts = Counter(dominant_labels)
    print(f"\n  Functional group distribution:")
    for lab, cnt in label_counts.most_common():
        print(f"    {lab:<20s}: {cnt:>5d} ({100*cnt/n_mol:.1f}%)")

    # ------------------------------------------------------------------
    # 5. Plot: dominant FG colouring
    # ------------------------------------------------------------------
    unique_labels = [lab for lab, _ in label_counts.most_common()]

    cmap = plt.cm.tab20
    label_to_colour = {}
    colour_idx = 0
    for lab in unique_labels:
        if lab == "Hydrocarbon only":
            label_to_colour[lab] = "#BBBBBB"
        else:
            label_to_colour[lab] = cmap(colour_idx / max(len(unique_labels) - 1, 1))
            colour_idx += 1

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot hydrocarbons first (background)
    if "Hydrocarbon only" in label_to_colour:
        mask = dominant_labels == "Hydrocarbon only"
        if mask.sum() > 0:
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                       c=label_to_colour["Hydrocarbon only"], s=8, alpha=0.3,
                       label=f"Hydrocarbon only ({mask.sum()})", rasterized=True)

    # Plot each FG category
    for lab in unique_labels:
        if lab == "Hydrocarbon only":
            continue
        mask = dominant_labels == lab
        if mask.sum() < 3:
            continue
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                   c=[label_to_colour[lab]], s=10, alpha=0.5,
                   label=f"{lab} ({mask.sum()})", rasterized=True)

    ax.set_title(f"N-gram Generated Molecules — Dominant Functional Group\n"
                 f"({n_mol} molecules, 8-gram model)", fontsize=14)
    ax.legend(fontsize=9, markerscale=3, bbox_to_anchor=(1.02, 1),
              loc="upper left", title="Dominant FG (count)")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "ngram_fg_umap.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {os.path.abspath(out_png)}")

    # ------------------------------------------------------------------
    # 6. Plot: per-FG panels
    # ------------------------------------------------------------------
    fg_present = fg_counts > 0
    active_fgs = [(fi, name) for fi, name in enumerate(fg_names)
                  if fg_present[:, fi].mean() > 0.01]

    if active_fgs:
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

            # Molecules without this FG
            ax.scatter(coords_2d[~has_fg, 0], coords_2d[~has_fg, 1],
                       c="#DDDDDD", s=5, alpha=0.3, rasterized=True)

            # Molecules with this FG, coloured by count
            if has_fg.sum() > 0:
                counts_fg = fg_counts[has_fg, fi]
                sc = ax.scatter(coords_2d[has_fg, 0], coords_2d[has_fg, 1],
                                c=counts_fg, cmap="YlOrRd", s=8, alpha=0.6,
                                vmin=0, vmax=max(counts_fg.max(), 1), rasterized=True)
                plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Count")

            ax.set_title(f"{FG_SHORT_NAMES.get(name, name)} ({pct:.0f}%)",
                         fontsize=10, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

        for idx in range(n_panels, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle(f"N-gram Molecules — Functional Group Distribution\n"
                     f"({n_mol} molecules, 8-gram model)", fontsize=14, y=1.02)
        fig.tight_layout()
        out_panels_png = os.path.join(args.out_dir, "ngram_fg_panels.png")
        fig.savefig(out_panels_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {os.path.abspath(out_panels_png)}")

    print(f"\nAll outputs in: {args.out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
