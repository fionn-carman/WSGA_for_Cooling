#!/usr/bin/env python3
"""
Side-by-side parity plot: old FOM1 models vs new temp-feature models at 40C.
Glymes highlighted (>=3 C-O-C ether linkages, no carbonyl).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from sklearn.metrics import r2_score, mean_absolute_error


def detect_glymes(smiles_list, min_ether=3):
    """>=min_ether C-O-C ether linkages AND no carbonyl (C=O)."""
    ether_patt = Chem.MolFromSmarts("[#6]~[#8]~[#6]")
    carbonyl_patt = Chem.MolFromSmarts("[#6]=[#8]")
    mask = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mask.append(False)
            continue
        n_ether = len(mol.GetSubstructMatches(ether_patt))
        has_carbonyl = mol.HasSubstructMatch(carbonyl_patt)
        mask.append(n_ether >= min_ether and not has_carbonyl)
    return np.array(mask)


def plot_panel(ax, true, pred, glyme, label, lims):
    """Plot a single parity panel."""
    non_glyme = ~glyme

    r2_all = r2_score(true, pred)
    mae_all = mean_absolute_error(true, pred)
    r2_ng = r2_score(true[non_glyme], pred[non_glyme])

    if glyme.sum() > 2:
        r2_g = r2_score(true[glyme], pred[glyme])
        mae_g = mean_absolute_error(true[glyme], pred[glyme])
        bias_g = (pred[glyme] - true[glyme]).mean()
    else:
        r2_g = mae_g = bias_g = float("nan")

    ax.scatter(true[non_glyme], pred[non_glyme],
               alpha=0.15, s=8, c="grey", rasterized=True)
    ax.scatter(true[glyme], pred[glyme],
               alpha=0.85, s=45, c="red", marker="D",
               edgecolors="black", linewidths=0.4, zorder=5,
               label=f"Glymes (n={glyme.sum()})")

    ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("NIST FOM1 (40\u00b0C)", fontsize=11)
    ax.set_ylabel("Predicted FOM1 (40\u00b0C)", fontsize=11)
    ax.set_aspect("equal")

    text = (f"All: R\u00b2={r2_all:.3f}, MAE={mae_all:.1f} (n={len(true)})\n"
            f"Glymes: R\u00b2={r2_g:.3f}, MAE={mae_g:.1f}, bias={bias_g:+.1f}\n"
            f"Non-glyme: R\u00b2={r2_ng:.3f}")
    ax.text(0.03, 0.97, text, transform=ax.transAxes, fontsize=9,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      alpha=0.85, edgecolor="grey"))
    ax.legend(fontsize=9, frameon=False, loc="lower right")

    return r2_all, r2_g, mae_g, bias_g


def main():
    # --- Old model: 5-fold OOF predictions at 40C (~806 molecules) ---
    old_dfs = []
    for fold in range(5):
        df = pd.read_csv(f"../models/FOM1_direct_5fold_40C/"
                         f"xgboost_descriptors_fold{fold}_predictions.csv")
        old_dfs.append(df)
    old_df = pd.concat(old_dfs, ignore_index=True)

    old_true = old_df["y_true"].values
    old_pred = old_df["y_pred"].values
    old_glyme = detect_glymes(old_df["SMILES"].values)

    # --- New model: temp-feature predictions at 40C (7381 molecules) ---
    new_df = pd.read_csv("FOM1_temp_feature/predictions.csv")
    new_40 = new_df[new_df["T_K"] == 313.15].copy()

    if "value_real" in new_40.columns:
        new_true = new_40["value_real"].values
        new_pred = new_40["pred_real"].values
    else:
        new_true = new_40["value"].values
        new_pred = new_40["pred"].values
    new_glyme = detect_glymes(new_40["SMILES"].values)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    lims = [10, 200]

    # Shared axis limits
    all_vals = np.concatenate([old_true, old_pred, new_true, new_pred])
    shared_lims = [10, max(all_vals) + 5]

    print("OLD MODEL (5-fold, ~806 molecules, 40C):")
    r2a, r2g, maeg, biasg = plot_panel(ax1, old_true, old_pred, old_glyme,
                                        "Old", shared_lims)
    print(f"  All R²={r2a:.4f}, Glyme R²={r2g:.4f}, MAE={maeg:.2f}, bias={biasg:+.2f}")
    ax1.text(0.5, 1.02, "Old model (806 molecules)",
             transform=ax1.transAxes, ha="center", fontsize=12, fontweight="bold")

    print("\nNEW MODEL (temp-feature, 7381 molecules, 40C):")
    r2a, r2g, maeg, biasg = plot_panel(ax2, new_true, new_pred, new_glyme,
                                        "New", shared_lims)
    print(f"  All R²={r2a:.4f}, Glyme R²={r2g:.4f}, MAE={maeg:.2f}, bias={biasg:+.2f}")
    ax2.text(0.5, 1.02, "New model (7,381 molecules, temp-as-feature)",
             transform=ax2.transAxes, ha="center", fontsize=12, fontweight="bold")

    fig.tight_layout()
    out_path = "FOM1_temp_feature/fom1_parity_old_vs_new.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
