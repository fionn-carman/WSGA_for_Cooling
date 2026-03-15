#!/usr/bin/env python3
"""
Compare XGBoost, Ridge, and MLP (all using RDKit descriptors) as direct FOM1
surrogates. Uses the same 5-fold CV indices as the architecture comparison
for a fair comparison.

Produces:
  - surrogate_comparison/fom1_surrogate_parity.png  (3 panels)
  - surrogate_comparison/fom1_surrogate_results.csv
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = (
    Path(__file__).resolve().parent
    / "FOM1_architecture_comparison" / "results" / "fom1_40"
)
OUT_DIR = Path(__file__).resolve().parent / "surrogate_comparison"


# ── Glyme / polyether detection ─────────────────────────────────────────────
def detect_glymes(smiles_list):
    """Return boolean mask: True if molecule is a true glyme/polyether.

    Requires ≥4 C-O-C ether linkages AND no carbonyl (C=O) groups, to
    exclude esters, ketones, and pentaerythritol derivatives.
    """
    from rdkit import Chem
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
        mask.append(n_ether >= 4 and not has_carbonyl)
    return np.array(mask)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training; load saved OOF predictions and replot")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ───────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "fom1_dataset.csv")
    y = df["FOM1_avg"].values.astype(np.float32)
    smiles = df["SMILES"].values

    logger.info("Dataset: %d molecules", len(df))
    is_glyme = detect_glymes(smiles)
    logger.info("Glymes/polyethers (no carbonyls): %d", is_glyme.sum())

    oof_path = OUT_DIR / "oof_predictions.npz"

    if args.plot_only and oof_path.exists():
        logger.info("Loading saved OOF predictions from %s", oof_path)
        data = np.load(oof_path)
        oof = {k: data[k] for k in data.files}
    else:
        # ── Train 3 models with 5-fold CV ──────────────────────────────────
        with open(DATA_DIR / "fold_indices.json") as f:
            fold_indices = json.load(f)
        with open(DATA_DIR / "descriptor_columns.json") as f:
            descriptor_columns = json.load(f)

        X = df[descriptor_columns].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        oof = {
            "XGBoost + Descriptors": np.full(len(y), np.nan),
            "Ridge + Descriptors": np.full(len(y), np.nan),
            "MLP + Descriptors": np.full(len(y), np.nan),
        }

        xgb_param_dist = {
            "n_estimators": [100, 200, 500, 1000, 2000],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 6, 8, 10],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        }

        for fold_id in range(5):
            train_idx = fold_indices[str(fold_id)]["train"]
            test_idx = fold_indices[str(fold_id)]["test"]

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # ── XGBoost ────────────────────────────────────────────────────
            t0 = time.time()
            search = RandomizedSearchCV(
                XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
                xgb_param_dist, n_iter=200, cv=5, scoring="r2",
                random_state=42, n_jobs=-1, verbose=0,
            )
            search.fit(X_train_s, y_train)
            oof["XGBoost + Descriptors"][test_idx] = search.best_estimator_.predict(X_test_s)
            logger.info("Fold %d XGBoost: R²=%.4f (%.0fs)", fold_id,
                         r2_score(y_test, oof["XGBoost + Descriptors"][test_idx]),
                         time.time() - t0)

            # ── Ridge ──────────────────────────────────────────────────────
            t0 = time.time()
            ridge = RidgeCV(alphas=np.logspace(-3, 3, 50))
            ridge.fit(X_train_s, y_train)
            oof["Ridge + Descriptors"][test_idx] = ridge.predict(X_test_s)
            logger.info("Fold %d Ridge:   R²=%.4f (%.0fs)", fold_id,
                         r2_score(y_test, oof["Ridge + Descriptors"][test_idx]),
                         time.time() - t0)

            # ── MLP (sklearn) ──────────────────────────────────────────────
            t0 = time.time()
            mlp = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-5,
                batch_size=64,
                learning_rate_init=1e-3,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=42 + fold_id,
            )
            mlp.fit(X_train_s, y_train)
            oof["MLP + Descriptors"][test_idx] = mlp.predict(X_test_s)
            logger.info("Fold %d MLP:     R²=%.4f (%.0fs)", fold_id,
                         r2_score(y_test, oof["MLP + Descriptors"][test_idx]),
                         time.time() - t0)

        # Save OOF predictions for future --plot-only runs
        np.savez(oof_path, **oof)
        logger.info("Saved OOF predictions to %s", oof_path)

    # ── Overall metrics ─────────────────────────────────────────────────────
    results = {}
    for name, preds in oof.items():
        valid = ~np.isnan(preds)
        r2 = r2_score(y[valid], preds[valid])
        mae = mean_absolute_error(y[valid], preds[valid])

        # Glyme subset
        glyme_valid = valid & is_glyme
        if glyme_valid.sum() > 0:
            glyme_r2 = r2_score(y[glyme_valid], preds[glyme_valid])
            glyme_mae = mean_absolute_error(y[glyme_valid], preds[glyme_valid])
            glyme_bias = (preds[glyme_valid] - y[glyme_valid]).mean()
        else:
            glyme_r2 = glyme_mae = glyme_bias = float("nan")

        results[name] = {
            "r2": r2, "mae": mae,
            "glyme_r2": glyme_r2, "glyme_mae": glyme_mae, "glyme_bias": glyme_bias,
        }
        logger.info("%s: R²=%.4f MAE=%.2f | Glyme R²=%.4f MAE=%.2f bias=%.2f",
                     name, r2, mae, glyme_r2, glyme_mae, glyme_bias)

    # ── Parity plots ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (name, preds) in zip(axes, oof.items()):
        valid = ~np.isnan(preds)
        res = results[name]

        # All points grey
        other = valid & ~is_glyme
        ax.scatter(y[other], preds[other],
                   alpha=0.3, s=15, c="grey", label="Other")
        # Glymes highlighted
        glyme_mask = valid & is_glyme
        ax.scatter(y[glyme_mask], preds[glyme_mask],
                   alpha=0.8, s=40, c="red", marker="D",
                   label=f"Glymes/polyethers (n={glyme_mask.sum()})",
                   edgecolors="black", linewidths=0.3)

        lims = [20, 120]
        ax.plot(lims, lims, "k--", alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Experimental FOM1 (avg)")
        ax.set_ylabel("Predicted FOM1 (avg)")
        ax.set_title(f"{name}\nR²={res['r2']:.4f}  MAE={res['mae']:.2f}")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_aspect("equal")

    fig.suptitle("Direct FOM1 Surrogate Comparison (5-fold CV)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fom1_surrogate_parity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", OUT_DIR / "fom1_surrogate_parity.png")

    # ── Summary CSV ─────────────────────────────────────────────────────────
    rows = []
    for name, res in results.items():
        rows.append({
            "Model": name,
            "R²": f"{res['r2']:.4f}",
            "MAE": f"{res['mae']:.2f}",
            "Glyme R²": f"{res['glyme_r2']:.4f}",
            "Glyme MAE": f"{res['glyme_mae']:.2f}",
            "Glyme Bias": f"{res['glyme_bias']:.2f}",
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_DIR / "fom1_surrogate_results.csv", index=False)
    logger.info("\n%s", summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
