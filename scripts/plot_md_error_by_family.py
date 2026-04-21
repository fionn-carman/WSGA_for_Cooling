#!/usr/bin/env python3
"""
4-panel scatter: Mahalanobis distance vs absolute prediction error for
FOM1, DC, FP, MP training molecules, colored by structural family.

Goal: show whether the high-error molecules are systematically different
structural types from the low-error ones (esters/ethers vs unsaturated/
aromatic), and whether this pattern holds across all 4 key properties.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from rdkit import Chem

REPO = Path(__file__).resolve().parent.parent
OOD_DIR = REPO / "OOD_testing" / "r2_vs_ood_percentile_hpc"
OUT_DIR = REPO / "outputs" / "pareto_reliability_analysis"
FIG_DIR = Path.home() / "Desktop" / "WSGA_Notes" / "Figures" / "pareto_reliability"
MANUSCRIPT_FIG = Path.home() / "Desktop" / "ML-for-Immersion-Cooling" / "Figures" / "Reliability"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_FIG.mkdir(parents=True, exist_ok=True)

# ── Publication style ───────────────────────────────────────────────────
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

# ── Structural family classification ────────────────────────────────────

_PAT_CACHE = {}

def _pat(smarts: str):
    if smarts not in _PAT_CACHE:
        _PAT_CACHE[smarts] = Chem.MolFromSmarts(smarts)
    return _PAT_CACHE[smarts]


def classify_family(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "other"

    has_aromatic = mol.HasSubstructMatch(_pat("c"))
    n_ether = len(mol.GetSubstructMatches(_pat("[OD2]([CX4])[CX4]")))
    has_ester = mol.HasSubstructMatch(_pat("[CX3](=O)[OX2]"))
    has_allene = mol.HasSubstructMatch(_pat("[C]=[C]=[C]"))
    has_alkyne = mol.HasSubstructMatch(_pat("[C]#[C]"))

    # Count C=C double bonds (non-aromatic)
    n_cc_double = 0
    for bond in mol.GetBonds():
        if (bond.GetBondTypeAsDouble() == 2.0
                and not bond.GetIsAromatic()
                and bond.GetBeginAtom().GetAtomicNum() == 6
                and bond.GetEndAtom().GetAtomicNum() == 6):
            n_cc_double += 1

    has_ring = mol.GetRingInfo().NumRings() > 0

    # Classification priority
    if has_allene:
        return "cumulated diene / allene"
    if has_alkyne:
        return "alkyne"
    if has_aromatic:
        return "aromatic"
    if n_cc_double >= 1:
        return "unsaturated (C=C)"
    if has_ester and n_ether >= 2:
        return "ester-ether"
    if has_ester:
        return "ester"
    if n_ether >= 1:
        return "ether"
    if has_ring:
        return "saturated cyclic"
    return "saturated acyclic"


FAMILY_COLORS = {
    "ether":                   "#1f77b4",  # blue
    "ester":                   "#2ca02c",  # green
    "ester-ether":             "#17becf",  # teal
    "aromatic":                "#9467bd",  # purple
    "unsaturated (C=C)":       "#ff7f0e",  # orange
    "cumulated diene / allene":"#d62728",  # red
    "alkyne":                  "#e377c2",  # pink
    "saturated cyclic":        "#8c564b",  # brown
    "saturated acyclic":       "#7f7f7f",  # grey
    "other":                   "#bcbd22",  # olive
}

# Families in legend order (most to least relevant for discussion)
FAMILY_ORDER = [
    "ether", "ester", "ester-ether",
    "aromatic", "unsaturated (C=C)",
    "cumulated diene / allene", "alkyne",
    "saturated cyclic", "saturated acyclic",
]


def load_and_classify(prop_key: str) -> pd.DataFrame:
    path = OOD_DIR / prop_key / "molecule_level.csv"
    df = pd.read_csv(path)
    df["family"] = df.SMILES.apply(classify_family)
    return df


# ── Build figure ────────────────────────────────────────────────────────
PROPS = [
    ("fom1_40C", "FOM1",               "Absolute error"),
    ("dc",       "Dielectric constant", "Absolute error"),
    ("fp",       "Flash point",         r"Absolute error / $\degree$C"),
    ("mp",       "Melting point",       r"Absolute error / $\degree$C"),
]
PANELS = ["(a)", "(b)", "(c)", "(d)"]

# Collapse rare families for visual clarity
MERGE_MAP = {
    "ether":                   "ether / ester",
    "ester":                   "ether / ester",
    "ester-ether":             "ether / ester",
    "aromatic":                "aromatic",
    "unsaturated (C=C)":       "unsaturated",
    "cumulated diene / allene":"unsaturated",
    "alkyne":                  "unsaturated",
    "saturated cyclic":        "saturated (no O-link)",
    "saturated acyclic":       "saturated (no O-link)",
    "other":                   "saturated (no O-link)",
}

MERGED_COLORS = {
    "ether / ester":        "#1f77b4",  # blue
    "aromatic":             "#9467bd",  # purple
    "unsaturated":          "#d62728",  # red
    "saturated (no O-link)":"#7f7f7f",  # grey
}

# Draw order: background families first, interesting ones on top
MERGED_ORDER = ["saturated (no O-link)", "aromatic", "unsaturated", "ether / ester"]

# Per-panel size: 4x4 square → 2x2 grid = 8x8 total
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

for idx, (prop_key, name, ylabel) in enumerate(PROPS):
    ax = axes[idx // 2, idx % 2]
    df = load_and_classify(prop_key)
    df["merged"] = df.family.map(MERGE_MAP)

    # Keep only top 10% by MD
    p90_md = df.md.quantile(0.9)
    df = df[df.md >= p90_md].copy()

    # Clip upper percentile for cleaner viz
    md_clip = df.md.quantile(0.99)
    err_clip = df.abs_error.quantile(0.99)

    for merged_fam in MERGED_ORDER:
        sub = df[df.merged == merged_fam]
        if len(sub) == 0:
            continue
        color = MERGED_COLORS[merged_fam]
        # Fewer points in top 10% — use bigger markers
        if merged_fam == "unsaturated":
            s, alpha = 14, 0.7
        elif merged_fam == "ether / ester":
            s, alpha = 12, 0.65
        else:
            s, alpha = 8, 0.4
        ax.scatter(
            sub.md.clip(upper=md_clip),
            sub.abs_error.clip(upper=err_clip),
            s=s, alpha=alpha, c=color, edgecolors="none",
            zorder=2 if merged_fam in ("saturated (no O-link)", "aromatic") else 3,
            rasterized=True,
        )

    ax.set_xlabel("Mahalanobis distance")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, md_clip * 1.05)
    ax.set_ylim(0, err_clip * 1.05)

    ax.text(0.02, 0.96, PANELS[idx], transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top")
    ax.text(0.97, 0.96, name, transform=ax.transAxes, fontsize=7,
            va="top", ha="right")

# ── Legend ──────────────────────────────────────────────────────────────
handles = []
for fam in ["ether / ester", "aromatic", "unsaturated", "saturated (no O-link)"]:
    handles.append(
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=MERGED_COLORS[fam],
               markersize=5, markeredgewidth=0, label=fam)
    )

fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           fontsize=7, bbox_to_anchor=(0.5, -0.02),
           handletextpad=0.3, columnspacing=1.2)

fig.tight_layout(rect=[0, 0.04, 1, 1])
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"md_error_by_family.{ext}", dpi=300,
                bbox_inches="tight")
    fig.savefig(FIG_DIR / f"md_error_by_family.{ext}", dpi=300,
                bbox_inches="tight")
    fig.savefig(MANUSCRIPT_FIG / f"md_error_by_family.{ext}", dpi=300,
                bbox_inches="tight")
plt.close(fig)
print(f"Saved: md_error_by_family.png/pdf")

# ── Summary statistics ──────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STRUCTURAL FAMILY ANALYSIS: MD vs ERROR BY FAMILY")
print("=" * 80)

for prop_key, name, _ in PROPS:
    df = load_and_classify(prop_key)
    print(f"\n── {name} ({len(df)} molecules) ──")
    print(f"{'Family':<28} {'N':>5} {'%':>5}  "
          f"{'MAE':>7} {'MedAE':>7} {'MD_med':>7}  "
          f"{'MAE(MD>p90)':>11} {'N(MD>p90)':>9}")

    p90_md = df.md.quantile(0.9)
    for family in FAMILY_ORDER:
        sub = df[df.family == family]
        if len(sub) == 0:
            continue
        ood = sub[sub.md > p90_md]
        print(f"{family:<28} {len(sub):5d} {100*len(sub)/len(df):5.1f}  "
              f"{sub.abs_error.mean():7.2f} {sub.abs_error.median():7.2f} "
              f"{sub.md.median():7.2f}  "
              f"{ood.abs_error.mean():11.2f} {len(ood):9d}" if len(ood) > 0
              else f"{family:<28} {len(sub):5d} {100*len(sub)/len(df):5.1f}  "
              f"{sub.abs_error.mean():7.2f} {sub.abs_error.median():7.2f} "
              f"{sub.md.median():7.2f}  {'—':>11} {0:9d}")

    # Top 10 highest-error molecules
    top_err = df.nlargest(10, "abs_error")
    print(f"\n  Top 10 highest-error molecules:")
    for _, row in top_err.iterrows():
        print(f"    {row.SMILES:<40} err={row.abs_error:7.2f}  "
              f"MD={row.md:6.2f}  family={row.family}")
