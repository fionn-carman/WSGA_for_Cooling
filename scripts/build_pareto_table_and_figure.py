#!/usr/bin/env python3
"""
Build full Pareto reliability table (LaTeX) and updated MAE vs MD figure
using the complete 3D Pareto fronts (138 WSGA + 50 REINVENT).

Outputs:
  - pareto_tier_table.tex           (LaTeX longtable, Tier 1+2 main, Tier 3 SI)
  - mae_vs_md_full_pareto.png/pdf   (updated Figure 4 with full PF histograms)
  - full_pareto_with_md.csv         (full assessment data)
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from rdkit import Chem

# ── Paths ───────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
MANUSCRIPT = Path.home() / "Desktop" / "ML-for-Immersion-Cooling"
OOD_DIR = REPO / "OOD_testing" / "r2_vs_ood_percentile_hpc"
OUT_DIR = REPO / "outputs" / "pareto_reliability_analysis"
FIG_OUT = MANUSCRIPT / "Figures" / "OOD"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

PUBCHEM_CORPUS = REPO / "data" / "pubchem_cho_5_30ha.csv"

# ── Thresholds ──────────────────────────────────────────────────────────
P90 = {"fom1_40C": 7.933, "dc": 4.909, "fp": 7.655, "mp": 12.952}
P95 = {"fom1_40C": 9.877, "dc": 6.132, "fp": 9.448, "mp": 15.214}

MD_COL = {
    "fom1_40C": "Mahal_fom1_40C", "dc": "Mahal_dc",
    "fp": "Mahal_fp", "mp": "Mahal_mp",
}

# ── Publication style (match plot_mae_vs_md_wsga.py) ────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 7.5,
    "axes.labelsize": 8.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": False, "ytick.right": True,
    "axes.spines.top": True, "axes.spines.right": True,
})

COLOR_MAE = "#2176AE"
COLOR_WSGA = "#E63946"
COLOR_REINVENT = "#2D9B27"
MS = 3.5
LW = 1.0
MEW = 0.6


# ═══════════════════════════════════════════════════════════════════════
# PARETO + DATA LOADING (mirrors plot_three_way_pareto.py)
# ═══════════════════════════════════════════════════════════════════════

def pareto_3d(data, xc, yc, cc):
    """3D Pareto front: maximise xc and yc, minimise cc."""
    fp_bins = pd.cut(data[xc], bins=50)
    cand = data.groupby(fp_bins, observed=True).apply(
        lambda g: pd.concat([
            g.nlargest(min(30, len(g)), yc),
            g.nsmallest(min(30, len(g)), cc),
        ]).drop_duplicates(),
        include_groups=False,
    ).reset_index(drop=True)
    pts = cand[[xc, yc, cc]].values
    n = len(pts)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dom = ((pts[:, 0] >= pts[i, 0]) & (pts[:, 1] >= pts[i, 1])
               & (pts[:, 2] <= pts[i, 2]))
        strict = ((pts[:, 0] > pts[i, 0]) | (pts[:, 1] > pts[i, 1])
                  | (pts[:, 2] < pts[i, 2]))
        dominated = dom & strict
        dominated[i] = False
        if dominated.any():
            is_pareto[i] = False
    return cand[is_pareto]


def load_wsga():
    path = REPO / "outputs" / "fp_molprice_sweep_nist8100" / "valid_molecules_summary_full.csv"
    df = pd.read_csv(path)
    df["fp_C"] = df["fp"] - 273.15
    return df


def load_reinvent():
    ri_dir = REPO / "outputs" / "reinvent_fp_molprice_production"
    frames = []
    for d in sorted(ri_dir.iterdir()):
        csv = d / "all_evaluated_molecules.csv"
        if not csv.exists():
            continue
        chunk = pd.read_csv(csv)
        if "is_valid" in chunk.columns:
            chunk = chunk[chunk["is_valid"] == True]
        frames.append(chunk)
    ri = pd.concat(frames).drop_duplicates(subset="SMILES").copy()
    ri["fp_C"] = ri["fp"] - 273.15
    return ri


def canonicalise(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None


def load_pubchem_corpus():
    corpus = pd.read_csv(PUBCHEM_CORPUS)["SMILES"]
    canon = set()
    for smi in corpus:
        c = canonicalise(str(smi))
        if c:
            canon.add(c)
    return canon


def classify_zone(md_val, prop):
    if md_val <= P90[prop]:
        return "green"
    if md_val <= P95[prop]:
        return "amber"
    return "red"


def assign_tier(row):
    z_fom1 = row.get("zone_fom1_40C", "red")
    z_fp = row.get("zone_fp", "red")
    z_dc = row.get("zone_dc", "red")
    novel = row.get("is_novel", False)
    mp = row.get("MolPrice", 99)
    if (z_fom1 == "green" and z_fp == "green"
            and z_dc in ("green", "amber") and novel and mp < 6):
        return 1
    n_bad = sum(1 for p in ["fom1_40C", "dc", "fp", "mp"]
                if row.get(f"zone_{p}", "red") != "green")
    if z_fom1 == "green" and n_bad <= 1 and novel:
        return 2
    return 3


# ═══════════════════════════════════════════════════════════════════════
# MAE INTERPOLATION
# ═══════════════════════════════════════════════════════════════════════

def load_mae_curve(prop_key):
    coarse = pd.read_csv(OOD_DIR / prop_key / "coarse_md.csv")
    tail = pd.read_csv(OOD_DIR / prop_key / "tail_md.csv")
    coarse = coarse.iloc[:-1]
    x = list(coarse["metric_mean"].values) + list(tail["metric_mean"].values)
    mae = list(coarse["MAE"].values) + list(tail["MAE"].values)
    nc = len(coarse)
    return np.array(x), np.array(mae), nc


def build_mae_interpolator(prop_key):
    x, mae, _ = load_mae_curve(prop_key)
    return lambda md_vals: np.interp(md_vals, x, mae, left=mae[0], right=mae[-1])


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("Loading data...")
    wsga_all = load_wsga()
    ri_all = load_reinvent()

    print("Computing 3D Pareto fronts...")
    wsga_pf = pareto_3d(wsga_all, "fp_C", "fom1_40C", "MolPrice")
    ri_pf = pareto_3d(ri_all, "fp_C", "fom1_40C", "MolPrice")
    print(f"  WSGA PF: {len(wsga_pf)}, REINVENT PF: {len(ri_pf)}")

    # Identify shared molecules
    wsga_smiles = set(wsga_pf.SMILES)
    ri_smiles = set(ri_pf.SMILES)
    both = wsga_smiles & ri_smiles
    print(f"  Shared: {len(both)}")

    # Tag source
    wsga_pf = wsga_pf.copy()
    wsga_pf["source"] = wsga_pf.SMILES.apply(
        lambda s: "both" if s in both else "WSGA")
    ri_pf = ri_pf.copy()
    ri_pf["source"] = ri_pf.SMILES.apply(
        lambda s: "both" if s in both else "REINVENT")

    # Combine, deduplicate (keep WSGA row for shared)
    combined = pd.concat([wsga_pf, ri_pf], ignore_index=True)
    combined = combined.sort_values("source")  # REINVENT first, then WSGA/both
    deduped = combined.drop_duplicates("SMILES", keep="last").copy()
    deduped.loc[deduped.SMILES.isin(both), "source"] = "both"

    # Novelty check
    print("Loading PubChem corpus...")
    corpus = load_pubchem_corpus()
    deduped["is_novel"] = [canonicalise(s) not in corpus for s in deduped.SMILES]
    print(f"  Novel: {deduped.is_novel.sum()}/{len(deduped)}")

    # Zone classification + expected MAE
    for prop in ["fom1_40C", "dc", "fp", "mp"]:
        col = MD_COL[prop]
        interp = build_mae_interpolator(prop)
        deduped[f"zone_{prop}"] = deduped[col].apply(lambda x: classify_zone(x, prop))
        deduped[f"expMAE_{prop}"] = interp(deduped[col].values)

    # Tier assignment
    deduped["tier"] = deduped.apply(assign_tier, axis=1)
    deduped = deduped.sort_values(["tier", "fom1_40C"], ascending=[True, False])

    print(f"\nTier distribution:")
    for t in [1, 2, 3]:
        n = (deduped.tier == t).sum()
        print(f"  Tier {t}: {n}")

    # ── Save full CSV ───────────────────────────────────────────
    save_cols = ["SMILES", "fom1_40C", "fp_C", "MolPrice", "source",
                 "is_novel", "tier"]
    for prop in ["fom1_40C", "dc", "fp", "mp"]:
        save_cols += [MD_COL[prop], f"zone_{prop}", f"expMAE_{prop}"]
    deduped[save_cols].to_csv(OUT_DIR / "full_pareto_with_md.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'full_pareto_with_md.csv'}")

    # ── Generate LaTeX table ────────────────────────────────────
    generate_latex_table(deduped)

    # ── Generate MAE vs MD figure ───────────────────────────────
    generate_mae_vs_md_figure(wsga_pf, ri_pf)

    # ── Print tier comparison ───────────────────────────────────
    print("\n── WSGA vs REINVENT tier comparison ──")
    for label, smiles_set in [("WSGA", wsga_smiles), ("REINVENT", ri_smiles)]:
        sub = deduped[deduped.SMILES.isin(smiles_set)]
        counts = sub.tier.value_counts().sort_index()
        print(f"  {label} ({len(sub)} molecules): {counts.to_dict()}")


# ═══════════════════════════════════════════════════════════════════════
# LATEX TABLE
# ═══════════════════════════════════════════════════════════════════════

def generate_latex_table(df):
    """Generate LaTeX longtable for Tier 1+2 (main) and Tier 3 (SI)."""

    def fmt_zone(zone):
        if zone == "green":
            return r"\cellcolor{green!20}"
        elif zone == "amber":
            return r"\cellcolor{orange!25}"
        return r"\cellcolor{red!20}"

    def fmt_source(src):
        if src == "both":
            return "W+R"
        return "W" if src == "WSGA" else "R"

    def write_table(sub, path, caption, label):
        with open(path, "w") as f:
            f.write(r"\begin{longtable}{clrrrccccc}" + "\n")
            f.write(r"\caption{" + caption + r"}" + "\n")
            f.write(r"\label{" + label + r"} \\" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"Tier & SMILES & FOM1 & FP / \si{\degreeCelsius} & "
                    r"MolPrice & Source & "
                    r"$\mathrm{MD_{FOM1}}$ & $\mathrm{MD_{DC}}$ & "
                    r"$\mathrm{MD_{FP}}$ & $\mathrm{MD_{MP}}$ \\" + "\n")
            f.write(r"\midrule" + "\n")
            f.write(r"\endfirsthead" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"Tier & SMILES & FOM1 & FP / \si{\degreeCelsius} & "
                    r"MolPrice & Source & "
                    r"$\mathrm{MD_{FOM1}}$ & $\mathrm{MD_{DC}}$ & "
                    r"$\mathrm{MD_{FP}}$ & $\mathrm{MD_{MP}}$ \\" + "\n")
            f.write(r"\midrule" + "\n")
            f.write(r"\endhead" + "\n")
            f.write(r"\bottomrule" + "\n")
            f.write(r"\endfoot" + "\n")

            for _, row in sub.iterrows():
                smi = row.SMILES.replace("_", r"\_")
                smi_tex = r"\texttt{" + smi + "}"
                z_fom1 = fmt_zone(row.zone_fom1_40C)
                z_dc = fmt_zone(row.zone_dc)
                z_fp = fmt_zone(row.zone_fp)
                z_mp = fmt_zone(row.zone_mp)
                src = fmt_source(row.source)
                f.write(
                    f"  {row.tier} & {smi_tex} & {row.fom1_40C:.1f} & "
                    f"{row.fp_C:.0f} & {row.MolPrice:.1f} & {src} & "
                    f"{z_fom1}{row[MD_COL['fom1_40C']]:.1f} & "
                    f"{z_dc}{row[MD_COL['dc']]:.1f} & "
                    f"{z_fp}{row[MD_COL['fp']]:.1f} & "
                    f"{z_mp}{row[MD_COL['mp']]:.1f} "
                    r"\\" + "\n"
                )

            f.write(r"\end{longtable}" + "\n")

    tier12 = df[df.tier <= 2]
    tier3 = df[df.tier == 3]

    write_table(
        tier12,
        OUT_DIR / "pareto_tier12_table.tex",
        "Tier~1 and Tier~2 Pareto-optimal molecules from WSGA and REINVENT "
        "production sweeps, ranked by prediction reliability. Mahalanobis distance "
        "cells are coloured by zone: green (MD~$<$~$p_{90}$), amber "
        "($p_{90}$--$p_{95}$), red ($>$~$p_{95}$). Source: W~=~WSGA only, "
        "R~=~REINVENT only, W+R~=~both methods.",
        "tab:pareto_tier12",
    )
    write_table(
        tier3,
        OUT_DIR / "pareto_tier3_table.tex",
        "Tier~3 Pareto-optimal molecules (high prediction uncertainty). "
        "Column definitions as in the main text.",
        "tab:pareto_tier3",
    )
    print(f"\nLaTeX tables: Tier 1+2 ({len(tier12)} rows), Tier 3 ({len(tier3)} rows)")
    print(f"  {OUT_DIR / 'pareto_tier12_table.tex'}")
    print(f"  {OUT_DIR / 'pareto_tier3_table.tex'}")


# ═══════════════════════════════════════════════════════════════════════
# MAE vs MD FIGURE (updated version of plot_mae_vs_md_wsga.py)
# ═══════════════════════════════════════════════════════════════════════

def generate_mae_vs_md_figure(wsga_pf, ri_pf):
    """4-panel MAE vs MD with full 3D Pareto histograms, no candidate vlines."""
    PROPS = [
        {"key": "fom1_40C", "name": "FOM1", "md_col": "Mahal_fom1_40C",
         "ylabel": "MAE", "panel": "(a)"},
        {"key": "dc", "name": "Dielectric constant", "md_col": "Mahal_dc",
         "ylabel": "MAE", "panel": "(b)"},
        {"key": "fp", "name": "Flash point", "md_col": "Mahal_fp",
         "ylabel": r"MAE / $\degree$C", "panel": "(c)"},
        {"key": "mp", "name": "Melting point", "md_col": "Mahal_mp",
         "ylabel": r"MAE / $\degree$C", "panel": "(d)"},
    ]

    fig = plt.figure(figsize=(6.5, 6.2))
    outer = GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.40)

    for idx, cfg in enumerate(PROPS):
        row, col = divmod(idx, 2)
        inner = GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer[row, col],
            height_ratios=[1, 1, 5], hspace=0.08,
        )
        ax_reinv = fig.add_subplot(inner[0])
        ax_wsga = fig.add_subplot(inner[1], sharex=ax_reinv)
        ax_main = fig.add_subplot(inner[2], sharex=ax_reinv)

        # MAE curve data
        x_md, mae, nc = load_mae_curve(cfg["key"])

        # MD values from full Pareto fronts
        md_col = cfg["md_col"]
        wsga_md = wsga_pf[md_col].dropna().values
        reinv_md = ri_pf[md_col].dropna().values

        # Shared x-limits
        all_md = np.concatenate([x_md, wsga_md, reinv_md])
        xlim_lo = all_md.min() * 0.9
        xlim_hi = all_md.max() * 1.05
        bins = np.linspace(xlim_lo, xlim_hi, 18)

        # MAE curve (blue)
        ax_main.plot(x_md[:nc], mae[:nc], "o-", color=COLOR_MAE,
                     markersize=MS, linewidth=LW, zorder=3)
        if len(x_md) > nc:
            ax_main.plot(x_md[nc-1:nc+1], mae[nc-1:nc+1], "-",
                         color=COLOR_MAE, linewidth=LW, zorder=3)
            ax_main.plot(x_md[nc:], mae[nc:], "o-", color=COLOR_MAE,
                         markersize=MS, linewidth=LW,
                         markerfacecolor="white", markeredgecolor=COLOR_MAE,
                         markeredgewidth=MEW, zorder=3)

        ax_main.set_ylim(bottom=0)
        ax_main.set_xlabel("Mahalanobis distance")
        ax_main.set_ylabel(cfg["ylabel"])
        ax_main.set_xlim(xlim_lo, xlim_hi)

        # Panel label
        ax_reinv.text(-0.02, 1.15, cfg["panel"], transform=ax_reinv.transAxes,
                      fontsize=9, fontweight="bold", va="bottom", ha="left")
        # Property name
        ax_main.text(0.97, 0.95, cfg["name"], transform=ax_main.transAxes,
                     fontsize=7, va="top", ha="right")

        # REINVENT histogram (green, top)
        ax_reinv.hist(reinv_md, bins=bins, color=COLOR_REINVENT, alpha=0.7,
                      edgecolor="white", linewidth=0.4)
        ax_reinv.tick_params(labelbottom=False, labelsize=6)
        ax_reinv.tick_params(axis="x", which="both", top=True)
        ax_reinv.set_ylabel("", fontsize=6)
        ax_reinv.text(0.97, 0.82, "REINVENT", transform=ax_reinv.transAxes,
                      fontsize=5.5, va="top", ha="right", color=COLOR_REINVENT,
                      fontstyle="italic")

        # WSGA histogram (red, middle)
        ax_wsga.hist(wsga_md, bins=bins, color=COLOR_WSGA, alpha=0.7,
                     edgecolor="white", linewidth=0.4)
        ax_wsga.tick_params(labelbottom=False, labelsize=6)
        ax_wsga.set_ylabel("", fontsize=6)
        ax_wsga.text(0.97, 0.82, "WSGA", transform=ax_wsga.transAxes,
                     fontsize=5.5, va="top", ha="right", color=COLOR_WSGA,
                     fontstyle="italic")

        # Shared y-limit for histograms
        ymax = max(ax_reinv.get_ylim()[1], ax_wsga.get_ylim()[1])
        ax_reinv.set_ylim(0, ymax)
        ax_wsga.set_ylim(0, ymax)
        ax_reinv.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
        ax_wsga.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))

    # Legend (no candidate lines)
    handles = [
        Line2D([0], [0], color=COLOR_MAE, marker="o", markersize=MS,
               linewidth=LW, label="Decile bin (0\u201390th percentile)"),
        Line2D([0], [0], color=COLOR_MAE, marker="o", markersize=MS,
               linewidth=LW, markerfacecolor="white", markeredgecolor=COLOR_MAE,
               markeredgewidth=MEW, label="Tail sub-bin (90\u2013100th percentile)"),
        Patch(facecolor=COLOR_WSGA, alpha=0.7, edgecolor="white",
              label="WSGA Pareto"),
        Patch(facecolor=COLOR_REINVENT, alpha=0.7, edgecolor="white",
              label="REINVENT Pareto"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=6.5, bbox_to_anchor=(0.5, -0.04))

    for ext in ("png", "pdf"):
        fig.savefig(FIG_OUT / f"mae_vs_md_wsga.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {FIG_OUT / 'mae_vs_md_wsga.png'}")


if __name__ == "__main__":
    main()
