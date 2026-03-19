#!/usr/bin/env python3
"""
Comprehensive pipeline comparison figures:
1. FOM1 parity: linear beta vs quadratic beta (ground truth comparison)
2. FOM1 parity: direct model vs pipeline (direct beta) model
3. Glyme MW analysis for direct and pipeline approaches

Usage:
    python plot_pipeline_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from xgboost import XGBRegressor

DESCRIPTOR_NAMES = [desc[0] for desc in Descriptors._descList]
FIXED_PARAMS = {
    "n_estimators": 600, "max_depth": 4, "learning_rate": 0.1,
    "subsample": 1.0, "colsample_bytree": 0.7,
    "min_child_weight": 10, "reg_alpha": 0, "reg_lambda": 0.1,
}
OUT = "pipeline_comparison"
TEMPS_C = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_glyme(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]=[#8]")):
        return False
    ether = Chem.MolFromSmarts("[OD2]([#6])[#6]")
    return len(mol.GetSubstructMatches(ether)) >= 3


KEY_LINEAR_GLYMES = {
    "CCOCCOCCOCC",             # diethylene glycol diethyl ether
    "CCOCCOCCOCCOCC",          # triethylene glycol diethyl ether
    "COCCOCCOCCOCCOCCOC",      # pentaglyme (PEG-DME n=5)
    "CCCCOCCOCCOCCOCCCC",      # tetraethylene glycol dibutyl ether
    "CCCCOCCOCCOCCO",          # triethylene glycol monobutyl ether
}


def is_linear_peg_dme(smi):
    """Detect the 5 key linear glymes specified for analysis."""
    return smi in KEY_LINEAR_GLYMES


def compute_fom1(rho_gcm3, kv_cSt, k_WmK, cp_molar, beta, mw):
    cp_si = cp_molar / mw * 1000  # J/(kg·K)
    rho_si = rho_gcm3 * 1000       # kg/m³
    nu_si = kv_cSt * 1e-6          # m²/s
    inner = (beta * cp_si * rho_si) / (nu_si * k_WmK)
    return k_WmK * np.power(inner, 0.2813)


def metrics_str(y_true, y_pred, label="All", n=None):
    if n is None:
        n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    return f"{label}: R²={r2:.3f}, MAE={mae:.1f}, bias={bias:+.1f}" + (f" (n={n})" if label == "All" else "")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading data...")
fom1_improved = pd.read_csv("data/FOM1_improved_40C.csv")
fom1_old = pd.read_csv("data/FOM1_cho_cleaned.csv")[["SMILES", "FOM1_40C"]]
density = pd.read_csv("data/density_cho_cleaned.csv")
pred_df = pd.read_csv(f"{OUT}/predictions.csv")

# Merge old FOM1 with improved
merged = fom1_improved.merge(fom1_old, on="SMILES", how="inner")

# Compute linear beta: 2-point finite difference (ρ(100) - ρ(40)) / 60
dens_cols = {f"density_{t}C_g_cm3": t for t in TEMPS_C}
dens_40_100 = density[["SMILES", "density_40C_g_cm3", "density_100C_g_cm3"]].copy()
dens_40_100 = dens_40_100.dropna()
dens_40_100["beta_linear"] = -(1 / dens_40_100["density_40C_g_cm3"]) * \
    (dens_40_100["density_100C_g_cm3"] - dens_40_100["density_40C_g_cm3"]) / 60.0
merged = merged.merge(dens_40_100[["SMILES", "beta_linear"]], on="SMILES", how="inner")

# Compute FOM1 with linear beta
merged["FOM1_linear_40C"] = compute_fom1(
    merged["density_40C_g_cm3"].values,
    merged["kv_40C_cSt"].values,
    merged["tc_40C_W_mK"].values,
    merged["cpsat_40C_J_K_mol"].values,
    merged["beta_linear"].values,
    merged["MolWt"].values,
)

# Detect glymes
merged["is_glyme"] = merged["SMILES"].apply(is_glyme)
merged["is_key_glyme"] = merged["SMILES"].apply(is_linear_peg_dme)

glyme_mask = merged["is_glyme"].values
key_mask = merged["is_key_glyme"].values
other_mask = ~glyme_mask

print(f"Total molecules: {len(merged)}")
print(f"Glymes: {glyme_mask.sum()}, Key linear PEG-DMEs: {key_mask.sum()}")

# ---------------------------------------------------------------------------
# Retrain direct beta model for pipeline predictions
# ---------------------------------------------------------------------------

print("Training direct beta model...")
desc_cols = [c for c in DESCRIPTOR_NAMES if c in fom1_improved.columns]
fom1_sorted = fom1_improved.sort_values("SMILES").reset_index(drop=True)

# Fold assignment (same as train_pipeline_comparison.py)
sorted_smiles = sorted(fom1_improved["SMILES"].unique())
gkf = GroupKFold(n_splits=5)
dummy_X = np.arange(len(sorted_smiles)).reshape(-1, 1)
fold_map = {}
for fi, (_, test_idx) in enumerate(gkf.split(dummy_X, np.zeros(len(sorted_smiles)), sorted_smiles)):
    for idx in test_idx:
        fold_map[sorted_smiles[idx]] = fi

X_desc = fom1_improved[desc_cols].values.astype(float)
finite_mask = np.all(np.isfinite(X_desc), axis=0)
X_desc = X_desc[:, finite_mask]
y_beta = fom1_improved["beta_40C"].values
smiles_all = fom1_improved["SMILES"].values
folds_all = np.array([fold_map[s] for s in smiles_all])

oof_beta_direct = np.full(len(y_beta), np.nan)
for fi in range(5):
    tr, te = folds_all != fi, folds_all == fi
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_desc[tr])
    X_te = sc.transform(X_desc[te])
    m = XGBRegressor(objective="reg:squarederror", n_jobs=-1,
                     random_state=fi, verbosity=0, **FIXED_PARAMS)
    m.fit(X_tr, y_beta[tr])
    oof_beta_direct[te] = m.predict(X_te)

beta_direct_map = dict(zip(smiles_all, oof_beta_direct))
print(f"  Direct beta R²={r2_score(y_beta, oof_beta_direct):.4f}")

# Build pipeline FOM1 (direct beta)
rho_map = dict(zip(pred_df["SMILES"], pred_df["density_pred"]))
visc_map = dict(zip(pred_df["SMILES"], pred_df["viscosity_pred"]))
tc_map = dict(zip(pred_df["SMILES"], pred_df["tc_pred"]))
cp_map = dict(zip(pred_df["SMILES"], pred_df["cpsat_pred"]))
direct_fom1_map = dict(zip(pred_df["SMILES"], pred_df["FOM1_direct_pred"]))
mw_map = dict(zip(fom1_improved["SMILES"], fom1_improved["MolWt"]))

# Compute pipeline FOM1 with direct beta for merged molecules
pipe_fom1 = []
direct_fom1 = []
for smi in merged["SMILES"]:
    if smi in rho_map and smi in beta_direct_map:
        b = beta_direct_map[smi]
        if b > 0:
            pipe_fom1.append(compute_fom1(
                rho_map[smi], visc_map[smi], tc_map[smi],
                cp_map[smi], b, mw_map[smi]))
        else:
            pipe_fom1.append(np.nan)
        direct_fom1.append(direct_fom1_map.get(smi, np.nan))
    else:
        pipe_fom1.append(np.nan)
        direct_fom1.append(np.nan)

merged["FOM1_pipeline_pred"] = pipe_fom1
merged["FOM1_direct_pred"] = direct_fom1
merged = merged.dropna(subset=["FOM1_pipeline_pred", "FOM1_direct_pred"]).reset_index(drop=True)

# Recompute masks after dropna
glyme_mask = merged["is_glyme"].values
key_mask = merged["is_key_glyme"].values
other_mask = ~glyme_mask

fom1_true = merged["FOM1_improved_40C"].values
fom1_direct = merged["FOM1_direct_pred"].values
fom1_pipeline = merged["FOM1_pipeline_pred"].values
mw_arr = merged["MolWt"].values

print(f"Final dataset: {len(merged)} molecules, {glyme_mask.sum()} glymes, {key_mask.sum()} key PEG-DMEs")

# ---------------------------------------------------------------------------
# Figure 1: FOM1 parity — linear beta vs quadratic beta
# ---------------------------------------------------------------------------

print("Plotting Figure 1: beta method comparison...")
fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

fom1_lin = merged["FOM1_linear_40C"].values
fom1_quad = merged["FOM1_improved_40C"].values

ax.scatter(fom1_lin[other_mask], fom1_quad[other_mask],
           s=8, alpha=0.3, color="0.6", edgecolors="none", rasterized=True)
ax.scatter(fom1_lin[glyme_mask & ~key_mask], fom1_quad[glyme_mask & ~key_mask],
           s=40, alpha=0.8, color="crimson", marker="D", edgecolors="darkred",
           linewidths=0.5, zorder=5)
ax.scatter(fom1_lin[key_mask], fom1_quad[key_mask],
           s=100, alpha=0.9, color="darkred", marker="*", edgecolors="black",
           linewidths=0.5, zorder=6, label=f"Key linear glymes (n={key_mask.sum()})")

lims = [min(fom1_lin.min(), fom1_quad.min()) * 0.9,
        max(fom1_lin.max(), fom1_quad.max()) * 1.05]
ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("FOM1 (linear β, 2-point)")
ax.set_ylabel("FOM1 (quadratic β, poly fit)")
ax.set_aspect("equal")

r2_all = r2_score(fom1_lin, fom1_quad)
mae_all = mean_absolute_error(fom1_lin, fom1_quad)
stats = (f"All: R²={r2_all:.4f}, MAE={mae_all:.2f} (n={len(merged)})\n"
         f"Glymes (n={glyme_mask.sum()})")
if glyme_mask.sum() > 1:
    r2_g = r2_score(fom1_lin[glyme_mask], fom1_quad[glyme_mask])
    mae_g = mean_absolute_error(fom1_lin[glyme_mask], fom1_quad[glyme_mask])
    stats += f": R²={r2_g:.4f}, MAE={mae_g:.2f}"

ax.text(0.03, 0.97, stats, transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="0.7"))

ax.scatter([], [], s=40, color="crimson", marker="D", edgecolors="darkred",
           linewidths=0.5, label=f"Glymes (n={glyme_mask.sum()})")
ax.legend(frameon=False, loc="lower right", fontsize=8)

fig.tight_layout()
fig.savefig(f"{OUT}/fom1_beta_methods.png")
plt.close()

# ---------------------------------------------------------------------------
# Figure 2: FOM1 model parity — direct vs pipeline
# ---------------------------------------------------------------------------

print("Plotting Figure 2: model parity comparison...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

for ax, pred, title in [
    (axes[0], fom1_direct, f"Direct FOM1 model ({len(merged):,} molecules)"),
    (axes[1], fom1_pipeline, f"Pipeline with direct β ({len(merged):,} molecules)"),
]:
    ax.scatter(fom1_true[other_mask], pred[other_mask],
               s=8, alpha=0.3, color="0.6", edgecolors="none", rasterized=True)
    ax.scatter(fom1_true[glyme_mask & ~key_mask], pred[glyme_mask & ~key_mask],
               s=40, alpha=0.8, color="crimson", marker="D", edgecolors="darkred",
               linewidths=0.5, zorder=5)
    ax.scatter(fom1_true[key_mask], pred[key_mask],
               s=100, alpha=0.9, color="darkred", marker="*", edgecolors="black",
               linewidths=0.5, zorder=6)

    lims = [fom1_true.min() * 0.85, fom1_true.max() * 1.08]
    ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("NIST FOM1 (40°C)")
    ax.set_ylabel("Predicted FOM1 (40°C)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_aspect("equal")

    # Stats
    s_all = metrics_str(fom1_true, pred, "All", len(merged))
    s_gly = metrics_str(fom1_true[glyme_mask], pred[glyme_mask], "Glymes") if glyme_mask.sum() > 1 else ""
    s_non = metrics_str(fom1_true[other_mask], pred[other_mask], "Non-glyme") if other_mask.sum() > 1 else ""
    stats = "\n".join(filter(None, [s_all, s_gly, s_non]))
    ax.text(0.03, 0.97, stats, transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="0.7"))

    # Legend
    ax.scatter([], [], s=40, color="crimson", marker="D", edgecolors="darkred",
               linewidths=0.5, label=f"Glymes (n={glyme_mask.sum()})")
    ax.legend(frameon=False, loc="lower right", fontsize=8)

fig.tight_layout()
fig.savefig(f"{OUT}/fom1_model_parity.png")
plt.close()

# ---------------------------------------------------------------------------
# Figure 3 & 4: Glyme MW analysis for direct and pipeline
# ---------------------------------------------------------------------------

def plot_glyme_mw_analysis(fom1_true, fom1_pred, glyme_mask, key_mask, mw,
                            smiles, approach_label, filename):
    """Two-panel glyme MW analysis matching reference style."""
    residual = fom1_pred - fom1_true

    # Fit polynomial to glyme FOM1 vs MW (using true FOM1)
    glyme_mw = mw[glyme_mask]
    glyme_fom1 = fom1_true[glyme_mask]

    # Fit: log(FOM1) ~ a + b*MW + c*MW^2  (power-law-like via log)
    # Or just a degree-2 polynomial in MW space
    mw_sort = np.linspace(glyme_mw.min() * 0.9, glyme_mw.max() * 1.1, 200)
    coeffs = np.polyfit(glyme_mw, glyme_fom1, 2)
    fit_curve = np.polyval(coeffs, mw_sort)
    glyme_fit_vals = np.polyval(coeffs, glyme_mw)
    delta_from_fit = glyme_fom1 - glyme_fit_vals  # true - fit

    glyme_resid = residual[glyme_mask]
    key_resid = residual[key_mask]
    key_mw = mw[key_mask]
    key_fom1 = fom1_true[key_mask]
    key_delta = key_fom1 - np.polyval(coeffs, key_mw)

    # Colormap for residuals (diverging)
    vmax = max(abs(glyme_resid.min()), abs(glyme_resid.max()))
    vmax = min(vmax, 30)  # cap for readability
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), dpi=150)

    # --- Top panel: FOM1 vs MW ---
    ax = axes[0]

    # All molecules (gray)
    ax.scatter(mw[~glyme_mask], fom1_true[~glyme_mask],
               s=6, alpha=0.15, color="0.75", edgecolors="none", rasterized=True)

    # Glyme MW fit curve
    ax.plot(mw_sort, fit_curve, "k-", alpha=0.6, linewidth=1.5, label="Glyme MW fit")

    # Glymes colored by residual
    sc = ax.scatter(glyme_mw[~key_mask[glyme_mask]], glyme_fom1[~key_mask[glyme_mask]],
                    c=glyme_resid[~key_mask[glyme_mask]],
                    s=50, alpha=0.85, marker="D", edgecolors="0.3", linewidths=0.5,
                    cmap="RdYlGn_r", norm=norm, zorder=5)

    # Key linear glymes as stars
    if key_mask.sum() > 0:
        ax.scatter(key_mw, key_fom1,
                   c=key_resid, s=150, alpha=0.95, marker="*",
                   edgecolors="0.2", linewidths=0.8,
                   cmap="RdYlGn_r", norm=norm, zorder=6)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(f"Residual (pred − true)", fontsize=9)

    ax.set_xlabel("Molecular Weight / Da")
    ax.set_ylabel("NIST FOM1 (40°C)")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_title(f"{approach_label}", fontsize=11, fontweight="bold")

    # --- Bottom panel: Residual vs Δ from MW fit ---
    ax = axes[1]

    ax.scatter(delta_from_fit[~key_mask[glyme_mask]],
               glyme_resid[~key_mask[glyme_mask]],
               s=50, alpha=0.8, color="crimson", marker="D",
               edgecolors="darkred", linewidths=0.5, zorder=4,
               label=f"Glymes (n={glyme_mask.sum()})")

    if key_mask.sum() > 0:
        ax.scatter(key_delta, key_resid,
                   s=150, alpha=0.95, color="darkred", marker="*",
                   edgecolors="black", linewidths=0.8, zorder=5,
                   label=f"Key linear glymes (n={key_mask.sum()})")

    # Trend line
    r_val, p_val = pearsonr(delta_from_fit, glyme_resid)
    z = np.polyfit(delta_from_fit, glyme_resid, 1)
    x_line = np.linspace(delta_from_fit.min(), delta_from_fit.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.4, linewidth=1)

    ax.axhline(0, color="0.7", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="0.7", linestyle=":", linewidth=0.8)

    ax.set_xlabel(r"$\Delta$ from glyme MW fit (true − fit)")
    ax.set_ylabel("Prediction residual (pred − true)")
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    ax.text(0.95, 0.05, f"r = {r_val:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.7"))

    fig.tight_layout()
    fig.savefig(f"{OUT}/{filename}")
    plt.close()
    print(f"  r(Δ, residual) = {r_val:.3f} (p={p_val:.4f})")


print("Plotting Figure 3: glyme MW analysis (direct)...")
plot_glyme_mw_analysis(
    fom1_true, fom1_direct, glyme_mask, key_mask, mw_arr,
    merged["SMILES"].values, "Direct FOM1 model", "glyme_mw_direct.png",
)

print("Plotting Figure 4: glyme MW analysis (pipeline)...")
plot_glyme_mw_analysis(
    fom1_true, fom1_pipeline, glyme_mask, key_mask, mw_arr,
    merged["SMILES"].values, "Pipeline (direct β model)", "glyme_mw_pipeline.png",
)

# Print key glyme details
print("\nKey linear PEG-DME glymes:")
key_rows = merged[merged["is_key_glyme"]].sort_values("MolWt")
for _, r in key_rows.iterrows():
    print(f"  {r['SMILES']:<35} MW={r['MolWt']:.1f}  "
          f"true={r['FOM1_improved_40C']:.1f}  "
          f"direct={r['FOM1_direct_pred']:.1f} ({r['FOM1_direct_pred']-r['FOM1_improved_40C']:+.1f})  "
          f"pipe={r['FOM1_pipeline_pred']:.1f} ({r['FOM1_pipeline_pred']-r['FOM1_improved_40C']:+.1f})")

print("\nDone.")
