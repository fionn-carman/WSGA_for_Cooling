#!/usr/bin/env python3
"""
Pareto front reliability analysis: assess prediction confidence of candidate
molecules using per-property Mahalanobis distance, interpolated expected MAE,
novelty checking, and structural classification.

Outputs:
  - pareto_reliability_assessment.csv  (full ranked table)
  - tier_comparison.csv               (WSGA vs REINVENT tier distribution)
  - md_vs_error_4panel.png/pdf        (MD vs abs_error scatter, 4 properties)
  - md_heatmap_pareto.png/pdf         (per-property MD zone heatmap)
  - pareto_reliability.png/pdf        (Pareto front colored by reliability)
"""
import hashlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd
from rdkit import Chem

# ── Paths ───────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
MANUSCRIPT = Path.home() / "Desktop" / "ML-for-Immersion-Cooling"
OOD_DIR = REPO / "OOD_testing" / "r2_vs_ood_percentile_hpc"
OUT_DIR = REPO / "outputs" / "pareto_reliability_analysis"
FIG_DIR = Path.home() / "Desktop" / "WSGA_Notes" / "Figures" / "pareto_reliability"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

PARETO_WSGA = MANUSCRIPT / "Figures" / "BaselineAnalysis" / "pareto_md_raw_wsga.csv"
PARETO_REINVENT = MANUSCRIPT / "Figures" / "BaselineAnalysis" / "pareto_md_raw_reinvent.csv"
WSGA_FULL = REPO / "outputs" / "fp_molprice_sweep_nist8100" / "valid_molecules_summary_full.csv"
PUBCHEM_CORPUS = REPO / "data" / "pubchem_cho_5_30ha.csv"

# ── Constants ───────────────────────────────────────────────────────────
# 90th / 95th percentile MD boundaries from coarse_md.csv and tail_md.csv
# p90 = metric_max of decile 8;  p95 = metric_max of tail sub_bin 1
P90 = {
    "fom1_40C": 7.933, "dc": 4.909, "fp": 7.655, "mp": 12.952,
}
P95 = {
    "fom1_40C": 9.877, "dc": 6.132, "fp": 9.448, "mp": 15.214,
}

CRITICAL_PROPS = ["fom1_40C", "dc", "fp", "mp"]
MD_COL_MAP = {
    "fom1_40C": "MD_fom1_40C", "dc": "MD_dc", "fp": "MD_fp", "mp": "MD_mp",
}

# ── Publication style ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": False, "ytick.right": True,
    "axes.spines.top": True, "axes.spines.right": True,
})

ZONE_COLORS = {"green": "#2ca02c", "amber": "#ff7f0e", "red": "#d62728"}


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def canonicalise(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None


def load_pubchem_corpus() -> set[str]:
    """Load PubChem CHO corpus and return set of canonical SMILES."""
    print("Loading PubChem corpus...")
    corpus = pd.read_csv(PUBCHEM_CORPUS)["SMILES"]
    canon = set()
    for smi in corpus:
        c = canonicalise(str(smi))
        if c:
            canon.add(c)
    print(f"  {len(canon):,} canonical SMILES loaded")
    return canon


def check_novelty(smiles_list: list[str], corpus: set[str]) -> list[bool]:
    return [canonicalise(s) not in corpus for s in smiles_list]


def build_mae_interpolator(prop_key: str):
    """Build linear interpolator: MD → expected MAE from binned data."""
    coarse = pd.read_csv(OOD_DIR / prop_key / "coarse_md.csv")
    tail = pd.read_csv(OOD_DIR / prop_key / "tail_md.csv")
    coarse = coarse.iloc[:-1]  # drop last decile (replaced by tail)
    x = np.concatenate([coarse["metric_mean"].values, tail["metric_mean"].values])
    mae = np.concatenate([coarse["MAE"].values, tail["MAE"].values])
    return lambda md_vals: np.interp(md_vals, x, mae,
                                      left=mae[0], right=mae[-1])


def classify_zone(md_val: float, prop: str) -> str:
    if md_val <= P90[prop]:
        return "green"
    elif md_val <= P95[prop]:
        return "amber"
    return "red"


def classify_structure(smi: str) -> str:
    """Classify molecule into structural family using SMARTS."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "unknown"
    has_aromatic = mol.HasSubstructMatch(Chem.MolFromSmarts("c"))
    n_ether = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[OD2]([CX4])[CX4]")))
    has_ester = mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2][CX4]"))
    has_carbonate = mol.HasSubstructMatch(
        Chem.MolFromSmarts("[OX2][CX3](=O)[OX2]"))
    has_ring = any(not a.GetIsAromatic() for r in mol.GetRingInfo().AtomRings()
                   for a in [mol.GetAtomWithIdx(r[0])])
    if has_aromatic:
        return "aromatic"
    if has_carbonate and n_ether >= 2:
        return "glycol ether carbonate"
    if has_ester and n_ether >= 2:
        return "mixed ester-ether"
    if has_ester:
        return "ester"
    if n_ether >= 2:
        return "polyether"
    if has_ring:
        return "cyclic"
    return "other"


def assign_tier(row: pd.Series) -> int:
    """Assign reliability tier (1=best, 3=worst)."""
    zone_fom1 = row.get("zone_fom1_40C", "red")
    zone_fp = row.get("zone_fp", "red")
    zone_dc = row.get("zone_dc", "red")
    is_novel = row.get("is_novel", False)
    molprice = row.get("MolPrice", 99)

    # Tier 1: FOM1 green, FP green, DC green or amber, novel, cheap
    if (zone_fom1 == "green" and zone_fp == "green"
            and zone_dc in ("green", "amber")
            and is_novel and molprice < 6):
        return 1
    # Tier 2: FOM1 green, at most 1 critical in amber, novel
    n_amber_or_red = sum(
        1 for p in CRITICAL_PROPS if row.get(f"zone_{p}", "red") != "green")
    if zone_fom1 == "green" and n_amber_or_red <= 1 and is_novel:
        return 2
    # Tier 3: anything else
    return 3


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_pareto_data():
    """Load WSGA and REINVENT Pareto fronts, add MolPrice, merge."""
    print("\nLoading Pareto front data...")

    # WSGA Pareto (39 molecules)
    wsga = pd.read_csv(PARETO_WSGA)
    wsga["source"] = "WSGA"
    wsga.rename(columns={"fp_C": "fp_C_raw"}, inplace=True)

    # REINVENT Pareto (31 molecules)
    reinv = pd.read_csv(PARETO_REINVENT)
    reinv["source"] = "REINVENT"
    reinv.rename(columns={"fp_C": "fp_C_raw"}, inplace=True)

    # Look up MolPrice from WSGA full summary
    print("  Looking up MolPrice...")
    wsga_full = pd.read_csv(WSGA_FULL, usecols=["SMILES", "MolPrice"])
    wsga_full = wsga_full.drop_duplicates("SMILES")
    price_map = dict(zip(wsga_full.SMILES, wsga_full.MolPrice))

    # Also scan REINVENT production runs
    for parent_name in ["reinvent_fp_molprice_production",
                        "reinvent_fp_molprice_nosphere"]:
        parent = REPO / "outputs" / parent_name
        if not parent.exists():
            continue
        for run_dir in parent.iterdir():
            csv_path = run_dir / "all_evaluated_molecules.csv"
            if not csv_path.exists():
                continue
            chunk = pd.read_csv(csv_path, usecols=["SMILES", "MolPrice"])
            for _, r in chunk.iterrows():
                if r.SMILES not in price_map:
                    price_map[r.SMILES] = r.MolPrice

    # Compute MolPrice for any remaining missing molecules
    all_smiles = set(wsga.SMILES) | set(reinv.SMILES)
    missing = [s for s in all_smiles if s not in price_map]
    if missing:
        print(f"  Computing MolPrice for {len(missing)} missing molecules...")
        sys.path.insert(0, str(REPO / "src"))
        from molprice import MolPriceModel
        mp_model = MolPriceModel(
            str(REPO / "models" / "molprice" / "MP_Morgan_hybrid.pkl"),
            scaler_path=str(REPO / "models" / "molprice"))
        for smi in missing:
            price_map[smi] = mp_model.predict(smi)

    wsga["MolPrice"] = wsga.SMILES.map(price_map)
    reinv["MolPrice"] = reinv.SMILES.map(price_map)

    # Combine and mark shared molecules
    combined = pd.concat([wsga, reinv], ignore_index=True)

    # Mark molecules found by both methods
    both = set(wsga.SMILES) & set(reinv.SMILES)
    combined["found_by_both"] = combined.SMILES.isin(both)

    # Deduplicate: keep one row per SMILES, prefer WSGA source for shared
    combined = combined.sort_values("source")  # REINVENT before WSGA
    deduped = combined.drop_duplicates("SMILES", keep="last")

    # For shared molecules, mark source as "both"
    deduped.loc[deduped.SMILES.isin(both), "source"] = "both"

    print(f"  WSGA: {len(wsga)}, REINVENT: {len(reinv)}, "
          f"shared: {len(both)}, unique: {len(deduped)}")
    return deduped, wsga, reinv


# ═══════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def main():
    pareto, wsga_raw, reinv_raw = load_pareto_data()

    # ── Novelty check ───────────────────────────────────────────────
    corpus = load_pubchem_corpus()
    pareto["is_novel"] = check_novelty(pareto.SMILES.tolist(), corpus)
    n_novel = pareto.is_novel.sum()
    print(f"\n  Novel: {n_novel}/{len(pareto)}")
    non_novel = pareto[~pareto.is_novel].SMILES.tolist()
    if non_novel:
        print(f"  Known molecules: {non_novel}")

    # ── MAE interpolation ───────────────────────────────────────────
    print("\nInterpolating expected MAE...")
    for prop in CRITICAL_PROPS:
        interp_fn = build_mae_interpolator(prop)
        md_col = MD_COL_MAP[prop]
        pareto[f"expected_MAE_{prop}"] = interp_fn(pareto[md_col].values)
        pareto[f"zone_{prop}"] = pareto[md_col].apply(
            lambda x: classify_zone(x, prop))
        print(f"  {prop}: zones = {pareto[f'zone_{prop}'].value_counts().to_dict()}")

    # ── Structural classification ───────────────────────────────────
    pareto["structural_family"] = pareto.SMILES.apply(classify_structure)
    print(f"\nStructural families:\n{pareto.structural_family.value_counts()}")

    # ── Tier assignment ─────────────────────────────────────────────
    pareto["tier"] = pareto.apply(assign_tier, axis=1)
    pareto = pareto.sort_values(["tier", "fom1_40C"], ascending=[True, False])
    print(f"\nTier distribution (combined):")
    print(pareto.tier.value_counts().sort_index())

    # ── WSGA vs REINVENT tier comparison ────────────────────────────
    print("\n── WSGA vs REINVENT tier comparison ──")

    # Assign tiers to raw WSGA and REINVENT individually
    for raw_df, label in [(wsga_raw, "WSGA"), (reinv_raw, "REINVENT")]:
        raw_df["is_novel"] = check_novelty(raw_df.SMILES.tolist(), corpus)
        for prop in CRITICAL_PROPS:
            interp_fn = build_mae_interpolator(prop)
            md_col = MD_COL_MAP[prop]
            raw_df[f"zone_{prop}"] = raw_df[md_col].apply(
                lambda x, p=prop: classify_zone(x, p))
        # Need MolPrice
        if "MolPrice" not in raw_df.columns:
            raw_df["MolPrice"] = raw_df.SMILES.map(
                dict(zip(pareto.SMILES, pareto.MolPrice)))
        raw_df["tier"] = raw_df.apply(assign_tier, axis=1)
        print(f"\n  {label} ({len(raw_df)} Pareto molecules):")
        print(f"    {raw_df.tier.value_counts().sort_index().to_dict()}")

    tier_comp = pd.DataFrame({
        "WSGA": wsga_raw.tier.value_counts().sort_index(),
        "REINVENT": reinv_raw.tier.value_counts().sort_index(),
    }).fillna(0).astype(int)
    tier_comp.index.name = "tier"
    tier_comp.to_csv(OUT_DIR / "tier_comparison.csv")
    print(f"\n{tier_comp}")

    # ── Save assessment CSV ─────────────────────────────────────────
    out_cols = ["SMILES", "fom1_40C", "fp_C_raw", "MolPrice", "source",
                "is_novel", "structural_family", "tier"]
    for prop in CRITICAL_PROPS:
        out_cols += [MD_COL_MAP[prop], f"zone_{prop}", f"expected_MAE_{prop}"]
    # Add all MD columns
    all_md = [c for c in pareto.columns if c.startswith("MD_")]
    for c in all_md:
        if c not in out_cols:
            out_cols.append(c)
    available = [c for c in out_cols if c in pareto.columns]
    pareto[available].to_csv(OUT_DIR / "pareto_reliability_assessment.csv",
                             index=False)
    print(f"\nSaved: {OUT_DIR / 'pareto_reliability_assessment.csv'}")

    # ── Print summary table ─────────────────────────────────────────
    print("\n" + "="*100)
    print("PARETO RELIABILITY ASSESSMENT (sorted by tier, then FOM1)")
    print("="*100)
    print(f"{'SMILES':<45} {'FOM1':>6} {'FP°C':>6} {'Price':>6} "
          f"{'Src':>5} {'Novel':>5} {'Tier':>4}  "
          f"{'z_FOM1':>6} {'z_DC':>6} {'z_FP':>6} {'z_MP':>6}  {'Family'}")
    print("-"*140)
    for _, row in pareto.iterrows():
        smi = row.SMILES[:43]
        print(f"{smi:<45} {row.fom1_40C:6.1f} {row.fp_C_raw:6.1f} "
              f"{row.MolPrice:6.2f} {row.source:>5} "
              f"{'Y' if row.is_novel else 'N':>5} {row.tier:>4}  "
              f"{row.zone_fom1_40C:>6} {row.zone_dc:>6} {row.zone_fp:>6} "
              f"{row.zone_mp:>6}  {row.structural_family}")

    # ═══════════════════════════════════════════════════════════════
    # FIGURES
    # ═══════════════════════════════════════════════════════════════

    # ── Figure 1: 4-panel MD vs abs_error scatter ───────────────────
    print("\n\nGenerating MD vs abs_error scatter (4 panels)...")
    generate_scatter_4panel(pareto)

    # ── Figure 2: MD heatmap ────────────────────────────────────────
    print("Generating MD heatmap...")
    generate_md_heatmap(pareto)

    # ── Figure 3: Pareto front colored by reliability ───────────────
    print("Generating reliability Pareto plot...")
    generate_pareto_reliability(pareto)

    print(f"\nAll outputs in: {OUT_DIR}")
    print(f"Figures also in: {FIG_DIR}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE GENERATORS
# ═══════════════════════════════════════════════════════════════════════

def generate_scatter_4panel(pareto: pd.DataFrame):
    """4-panel MD vs abs_error scatter with Pareto MD positions."""
    prop_configs = [
        ("fom1_40C", "FOM1", "MAE", "(a)"),
        ("dc", "Dielectric constant", "MAE", "(b)"),
        ("fp", "Flash point", r"MAE / $\degree$C", "(c)"),
        ("mp", "Melting point", r"MAE / $\degree$C", "(d)"),
    ]

    # Structural family colors
    family_colors = {
        "polyether": "#1f77b4",
        "glycol ether carbonate": "#2ca02c",
        "mixed ester-ether": "#ff7f0e",
        "ester": "#d62728",
        "aromatic": "#9467bd",
        "cyclic": "#8c564b",
        "other": "#7f7f7f",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for idx, (prop, name, ylabel, panel) in enumerate(prop_configs):
        ax = axes[idx // 2, idx % 2]

        # Load molecule-level data
        ml_path = OOD_DIR / prop / "molecule_level.csv"
        if not ml_path.exists():
            ax.text(0.5, 0.5, f"No data for {prop}", transform=ax.transAxes,
                    ha="center")
            continue
        ml = pd.read_csv(ml_path)

        # Background scatter (training molecules)
        ax.scatter(ml.md, ml.abs_error, s=1, alpha=0.15, c="#888888",
                   zorder=1, rasterized=True)

        # Binned MAE curve
        interp_fn = build_mae_interpolator(prop)
        md_range = np.linspace(ml.md.min(), ml.md.quantile(0.995), 200)
        ax.plot(md_range, interp_fn(md_range), color="#2176AE", linewidth=1.5,
                zorder=4, label="Binned MAE")

        # Zone shading
        xlim = (ml.md.min() * 0.9, ml.md.quantile(0.995) * 1.05)
        ax.axvspan(P90[prop], P95[prop], alpha=0.12, color=ZONE_COLORS["amber"],
                   zorder=0)
        ax.axvspan(P95[prop], xlim[1] * 1.1, alpha=0.12,
                   color=ZONE_COLORS["red"], zorder=0)
        ax.axvline(P90[prop], color=ZONE_COLORS["amber"], linestyle="--",
                   linewidth=0.6, alpha=0.7, zorder=2)
        ax.axvline(P95[prop], color=ZONE_COLORS["red"], linestyle="--",
                   linewidth=0.6, alpha=0.7, zorder=2)

        # Pareto molecule MD positions (rug plot + expected MAE)
        md_col = MD_COL_MAP[prop]
        for _, row in pareto.iterrows():
            md_val = row[md_col]
            if md_val < xlim[0] or md_val > xlim[1]:
                continue
            family = row.structural_family
            color = family_colors.get(family, "#7f7f7f")
            expected_mae = row[f"expected_MAE_{prop}"]
            ax.scatter(md_val, expected_mae, s=25, marker="v", c=color,
                       edgecolors="black", linewidths=0.3, zorder=5, alpha=0.8)

        ax.set_xlabel("Mahalanobis distance")
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.set_ylim(bottom=0)
        ax.text(0.02, 0.95, panel, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top")
        ax.text(0.97, 0.95, name, transform=ax.transAxes, fontsize=7,
                va="top", ha="right")

    # Legend
    handles = [
        plt.scatter([], [], s=1, c="#888888", label="Training molecules"),
        Line2D([0], [0], color="#2176AE", linewidth=1.5, label="Binned MAE"),
        Patch(facecolor=ZONE_COLORS["amber"], alpha=0.3, label="90-95th %ile"),
        Patch(facecolor=ZONE_COLORS["red"], alpha=0.3, label=">95th %ile"),
    ]
    for fam, col in list(family_colors.items())[:5]:
        handles.append(plt.scatter([], [], s=25, marker="v", c=col,
                                   edgecolors="black", linewidths=0.3,
                                   label=fam))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=6.5, bbox_to_anchor=(0.5, -0.06))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"md_vs_error_4panel.{ext}", dpi=300,
                    bbox_inches="tight")
        fig.savefig(FIG_DIR / f"md_vs_error_4panel.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: md_vs_error_4panel.png/pdf")


def generate_md_heatmap(pareto: pd.DataFrame):
    """Heatmap of per-property MD zones for all Pareto molecules."""
    props = CRITICAL_PROPS
    zone_to_num = {"green": 0, "amber": 1, "red": 2}

    # Build matrix
    labels = []
    matrix = []
    md_text = []
    for _, row in pareto.iterrows():
        smi_short = row.SMILES[:30] + ("..." if len(row.SMILES) > 30 else "")
        labels.append(f"{smi_short}  (T{row.tier})")
        zone_row = []
        md_row = []
        for prop in props:
            zone_row.append(zone_to_num[row[f"zone_{prop}"]])
            md_row.append(row[MD_COL_MAP[prop]])
        matrix.append(zone_row)
        md_text.append(md_row)

    matrix = np.array(matrix)
    md_text = np.array(md_text)

    # Plot
    cmap = ListedColormap([ZONE_COLORS["green"], ZONE_COLORS["amber"],
                           ZONE_COLORS["red"]])
    norm = BoundaryNorm([0, 0.5, 1.5, 2.5], cmap.N)

    n_rows = len(labels)
    fig_h = max(6, n_rows * 0.3 + 1.5)
    fig, ax = plt.subplots(figsize=(6, fig_h))
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    # Cell text (MD values)
    for i in range(n_rows):
        for j in range(len(props)):
            color = "white" if matrix[i, j] == 2 else "black"
            ax.text(j, i, f"{md_text[i, j]:.1f}", ha="center", va="center",
                    fontsize=5.5, color=color)

    ax.set_xticks(range(len(props)))
    ax.set_xticklabels(["FOM1", "DC", "FP", "MP"], fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=5)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True,
                   labelbottom=False)

    # Legend
    handles = [Patch(facecolor=ZONE_COLORS["green"], label="< p90 (safe)"),
               Patch(facecolor=ZONE_COLORS["amber"], label="p90-p95 (caution)"),
               Patch(facecolor=ZONE_COLORS["red"], label="> p95 (high risk)")]
    ax.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
              fontsize=6, bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"md_heatmap_pareto.{ext}", dpi=300,
                    bbox_inches="tight")
        fig.savefig(FIG_DIR / f"md_heatmap_pareto.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: md_heatmap_pareto.png/pdf")


def generate_pareto_reliability(pareto: pd.DataFrame):
    """Pareto front (FP vs FOM1) colored by FOM1 zone, sized by 1/MolPrice."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for zone, color in ZONE_COLORS.items():
        mask = pareto.zone_fom1_40C == zone
        if mask.sum() == 0:
            continue
        sub = pareto[mask]
        sizes = 200 / sub.MolPrice.clip(lower=1)  # cheaper = bigger
        ax.scatter(sub.fp_C_raw, sub.fom1_40C, s=sizes, c=color,
                   edgecolors="black", linewidths=0.4, alpha=0.8, zorder=3,
                   label=f"{zone} (n={mask.sum()})")

    # Annotate source
    for _, row in pareto.iterrows():
        if row.source == "both":
            ax.annotate("*", (row.fp_C_raw, row.fom1_40C),
                         fontsize=6, ha="center", va="bottom", color="black")

    ax.set_xlabel("Flash point / °C")
    ax.set_ylabel("FOM1")
    ax.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"pareto_reliability.{ext}", dpi=300,
                    bbox_inches="tight")
        fig.savefig(FIG_DIR / f"pareto_reliability.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: pareto_reliability.png/pdf")


if __name__ == "__main__":
    main()
