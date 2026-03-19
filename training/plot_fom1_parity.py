#!/usr/bin/env python3
"""
Parity plot for FOM1 temp-as-feature model with glymes highlighted.
Compares R² for glymes vs non-glymes at 40C and 100C.

Usage:
    cd TrainingFullNist/
    python plot_fom1_parity.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from sklearn.metrics import r2_score, mean_absolute_error


def detect_glymes(smiles_list):
    """Return boolean mask: True if molecule is a glyme/polyether.
    Requires >=4 C-O-C ether linkages AND no carbonyl (C=O).
    """
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
        mask.append(n_ether >= 3 and not has_carbonyl)
    return np.array(mask)


def main():
    # Load predictions from temp-feature model
    pred_df = pd.read_csv("FOM1_temp_feature/predictions.csv")

    # Use pre-computed real-space columns if available, else raw
    if "value_real" in pred_df.columns:
        pred_df["true_real"] = pred_df["value_real"]
        # pred_real already exists
    else:
        pred_df["true_real"] = pred_df["value"]
        pred_df["pred_real"] = pred_df["pred"]

    # Classify glymes
    unique_smi = pred_df["SMILES"].unique()
    is_glyme_map = dict(zip(unique_smi, detect_glymes(unique_smi)))
    pred_df["is_glyme"] = pred_df["SMILES"].map(is_glyme_map)

    n_glymes = sum(is_glyme_map.values())
    print(f"Total molecules: {len(unique_smi)}")
    print(f"Glymes/polyethers: {n_glymes}")

    # Plot: 2 panels (40C and 100C)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, temp_c in zip(axes, [40, 100]):
        temp_k = temp_c + 273.15
        mask_t = pred_df["T_K"] == temp_k
        df_t = pred_df[mask_t].copy()

        if len(df_t) == 0:
            continue

        true = df_t["true_real"].values
        pred = df_t["pred_real"].values
        glyme = df_t["is_glyme"].values

        # Overall metrics
        r2_all = r2_score(true, pred)
        mae_all = mean_absolute_error(true, pred)

        # Glyme metrics
        if glyme.sum() > 2:
            r2_g = r2_score(true[glyme], pred[glyme])
            mae_g = mean_absolute_error(true[glyme], pred[glyme])
            bias_g = (pred[glyme] - true[glyme]).mean()
        else:
            r2_g = mae_g = bias_g = float("nan")

        # Non-glyme metrics
        non_glyme = ~glyme
        r2_ng = r2_score(true[non_glyme], pred[non_glyme])

        # Plot non-glymes
        ax.scatter(true[non_glyme], pred[non_glyme],
                   alpha=0.15, s=8, c="grey", rasterized=True)

        # Plot glymes
        ax.scatter(true[glyme], pred[glyme],
                   alpha=0.85, s=45, c="red", marker="D",
                   edgecolors="black", linewidths=0.4, zorder=5,
                   label=f"Glymes (n={glyme.sum()})")

        # Diagonal
        lims = [min(true.min(), pred.min()) - 5,
                max(true.max(), pred.max()) + 5]
        ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"NIST FOM1 ({temp_c}\u00b0C)", fontsize=11)
        ax.set_ylabel(f"Predicted FOM1 ({temp_c}\u00b0C)", fontsize=11)
        ax.set_aspect("equal")

        # Annotation box
        text = (f"All: R\u00b2={r2_all:.3f}, MAE={mae_all:.1f}\n"
                f"Glymes: R\u00b2={r2_g:.3f}, MAE={mae_g:.1f}, bias={bias_g:+.1f}\n"
                f"Non-glyme: R\u00b2={r2_ng:.3f}")
        ax.text(0.03, 0.97, text, transform=ax.transAxes, fontsize=9,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          alpha=0.85, edgecolor="grey"))

        ax.legend(fontsize=9, frameon=False, loc="lower right")

        print(f"\n{temp_c}C:")
        print(f"  All:     R²={r2_all:.4f}  MAE={mae_all:.2f}  N={len(df_t)}")
        print(f"  Glymes:  R²={r2_g:.4f}  MAE={mae_g:.2f}  bias={bias_g:+.2f}  N={glyme.sum()}")
        print(f"  Other:   R²={r2_ng:.4f}  N={non_glyme.sum()}")

    fig.tight_layout()
    fig.savefig("FOM1_temp_feature/fom1_parity_glymes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved: FOM1_temp_feature/fom1_parity_glymes.png")


if __name__ == "__main__":
    main()
