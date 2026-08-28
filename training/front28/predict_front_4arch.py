#!/usr/bin/env python3
"""
Predict the 10 NIST 8100 regression properties for the 28 pooled Pareto-front
molecules with all four benchmarked architectures (XGBoost, MLP, D-MPNN/Chemprop
tuned, Uni-Mol), using the saved five-fold checkpoints from the architecture
benchmark.

Each architecture contributes one value per property = mean over its 5 fold
models (real units, matching src/wsga_helper.predict_fold_ensemble: inverse
log1p is applied per fold before averaging).

Writes a long CSV: SMILES, prop, arch, fold, pred   (real units)

Usage:
    python predict_front_4arch.py --smiles_csv front28.csv --out preds_long.csv
    python predict_front_4arch.py ... --archs xgb mlp        # subset
"""
# ---------------------------------------------------------------------------
# Compatibility patches required before importing chemprop (copied verbatim
# from training/Chemprop/nist8100/train_single_temp.py)
# ---------------------------------------------------------------------------
import sklearn.metrics as _skm
_orig_mse = _skm.mean_squared_error


def _patched_mse(y_true, y_pred, *, squared=True, **kwargs):
    mse = _orig_mse(y_true, y_pred, **kwargs)
    return mse if squared else mse ** 0.5


_skm.mean_squared_error = _patched_mse

import torch as _torch
_orig_torch_load = _torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


_torch.load = _patched_torch_load
# ---------------------------------------------------------------------------

import argparse
import csv
import os
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

TRAINING = Path("/rds/general/user/fc4018/projects/fionn2023/live/"
                "WSGA_for_Cooling/training")

# property -> (log1p transform?, which XGBoost tree the folds live under)
PROPS = {
    "density":   {"log": False, "xgb": "nist8100"},
    "viscosity": {"log": True,  "xgb": "nist8100"},
    "tc":        {"log": False, "xgb": "nist8100"},
    "cpsat":     {"log": False, "xgb": "nist8100"},
    "beta":      {"log": False, "xgb": "nist8100"},
    "fom1":      {"log": True,  "xgb": "nist8100"},
    "bp":        {"log": False, "xgb": "constraints"},
    "fp":        {"log": False, "xgb": "constraints"},
    "mp":        {"log": False, "xgb": "constraints"},
    "dc":        {"log": False, "xgb": "constraints"},
}


def xgb_dir(prop):
    if PROPS[prop]["xgb"] == "nist8100":
        return TRAINING / "XGBoost/nist8100/output/single_temp" / prop / "40C"
    return TRAINING / "XGBoost/constraints/output/single_temp" / prop


def mlp_dir(prop):
    return TRAINING / "MLP/nist8100/output/single_temp" / prop / "40C"


def chemprop_fold_dir(prop, k):
    return (TRAINING / "Chemprop/nist8100/output/single_temp_tuned" / prop
            / "40C" / f"fold{k}_model")


def unimol_fold_dir(prop, k):
    return (TRAINING / "UniMol/nist8100/output/single_temp" / prop / "40C"
            / f"fold{k}_model")


# ---------------------------------------------------------------------------
# Descriptors (identical recipe to training/XGBoost/nist8100/compute_descriptors.py)
# ---------------------------------------------------------------------------

def compute_descriptors(smiles_list):
    from mordred import Calculator, descriptors as mordred_descriptors

    rdkit_names = [d[0] for d in Descriptors._descList]
    rdkit_fns = {d[0]: d[1] for d in Descriptors._descList}

    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    if any(m is None for m in mols):
        bad = [s for s, m in zip(smiles_list, mols) if m is None]
        raise SystemExit(f"unparseable SMILES: {bad}")

    rows = []
    for mol in mols:
        row = {}
        for n in rdkit_names:
            try:
                row[n] = rdkit_fns[n](mol)
            except Exception:
                row[n] = np.nan
        rows.append(row)
    rdf = pd.DataFrame(rows)
    rdf.columns = [f"rdkit_{c}" for c in rdf.columns]

    calc = Calculator(mordred_descriptors, ignore_3D=True)
    mdf = calc.pandas(mols, quiet=True)
    for c in mdf.columns:
        mdf[c] = pd.to_numeric(mdf[c], errors="coerce")
    mdf.columns = [f"mordred_{c}" for c in mdf.columns]

    out = pd.concat([pd.DataFrame({"SMILES": smiles_list}), rdf, mdf], axis=1)
    return out


def design_matrix(desc_df, features):
    """Select `features` from desc_df, filling absent columns with 0 (the
    training pipeline nan_to_num's missing/NaN descriptors to 0.0)."""
    missing = [f for f in features if f not in desc_df.columns]
    X = np.zeros((len(desc_df), len(features)), dtype=np.float64)
    for j, f in enumerate(features):
        if f in desc_df.columns:
            X[:, j] = pd.to_numeric(desc_df[f], errors="coerce").values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, missing


# ---------------------------------------------------------------------------
# Per-architecture prediction
# ---------------------------------------------------------------------------

def predict_xgb(prop, desc_df, records):
    d = xgb_dir(prop)
    for k in range(5):
        p = d / f"fold{k}_model.joblib"
        blob = joblib.load(p)
        X, missing = design_matrix(desc_df, blob["selected_features"])
        if missing:
            print(f"    [xgb {prop} fold{k}] {len(missing)} features absent "
                  f"from descriptor frame -> 0: {missing[:5]}", flush=True)
        y = blob["model"].predict(X).astype(np.float64)
        if blob.get("log_transform", PROPS[prop]["log"]):
            y = np.expm1(y)
        for smi, v in zip(desc_df.SMILES, y):
            records.append((smi, prop, "XGBoost", k, float(v)))


def predict_mlp(prop, desc_df, records):
    d = mlp_dir(prop)
    for k in range(5):
        blob = joblib.load(d / f"fold{k}_model.joblib")
        X, missing = design_matrix(desc_df, blob["selected_features"])
        if missing:
            print(f"    [mlp {prop} fold{k}] {len(missing)} features absent "
                  f"-> 0: {missing[:5]}", flush=True)
        Xs = blob["scaler_X"].transform(X)
        ys = blob["model"].predict(Xs)
        y = blob["scaler_y"].inverse_transform(ys.reshape(-1, 1)).ravel()
        if blob.get("log_transform", PROPS[prop]["log"]):
            y = np.expm1(y)
        for smi, v in zip(desc_df.SMILES, y):
            records.append((smi, prop, "MLP", k, float(v)))


def predict_chemprop(prop, smiles, records):
    from chemprop.train import chemprop_predict

    gpu_args = ["--gpu", "0"] if _torch.cuda.is_available() else \
               ["--no_cuda", "--num_workers", "0"]

    for k in range(5):
        ckpt = chemprop_fold_dir(prop, k)
        with tempfile.TemporaryDirectory() as td:
            test_csv = os.path.join(td, "test.csv")
            pred_csv = os.path.join(td, "preds.csv")
            with open(test_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["smiles", "target"])
                for s in smiles:
                    w.writerow([s, ""])
            saved_argv = sys.argv
            sys.argv = ["chemprop_predict",
                        "--test_path", test_csv,
                        "--checkpoint_dir", str(ckpt),
                        "--preds_path", pred_csv] + gpu_args
            try:
                chemprop_predict()
            finally:
                sys.argv = saved_argv
            y = pd.read_csv(pred_csv).iloc[:, 1].values.astype(np.float64)
        if PROPS[prop]["log"]:
            y = np.expm1(y)
        for smi, v in zip(smiles, y):
            records.append((smi, prop, "D-MPNN", k, float(v)))


def predict_unimol(prop, smiles, records):
    from unimol_tools import MolPredict

    for k in range(5):
        d = unimol_fold_dir(prop, k)
        p = MolPredict(load_model=str(d)).predict(data=list(smiles))
        if isinstance(p, dict):
            p = list(p.values())[0]
        y = np.asarray(p, dtype=np.float64).flatten()
        if PROPS[prop]["log"]:
            y = np.expm1(y)
        for smi, v in zip(smiles, y):
            records.append((smi, prop, "Uni-Mol", k, float(v)))


ARCH_FNS = {
    "xgb":      ("XGBoost", predict_xgb,      True),   # True -> needs descriptors
    "mlp":      ("MLP",     predict_mlp,      True),
    "dmpnn":    ("D-MPNN",  predict_chemprop, False),
    "unimol":   ("Uni-Mol", predict_unimol,   False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--archs", nargs="+", default=list(ARCH_FNS),
                    choices=list(ARCH_FNS))
    ap.add_argument("--props", nargs="+", default=list(PROPS),
                    choices=list(PROPS))
    args = ap.parse_args()

    smiles = pd.read_csv(args.smiles_csv)["SMILES"].tolist()
    print(f"{len(smiles)} molecules, {len(args.props)} properties, "
          f"architectures: {args.archs}", flush=True)

    need_desc = any(ARCH_FNS[a][2] for a in args.archs)
    desc_df = None
    if need_desc:
        t0 = time.time()
        desc_df = compute_descriptors(smiles)
        print(f"descriptors: {desc_df.shape[1] - 1} columns "
              f"({time.time() - t0:.1f}s)", flush=True)

    records = []
    failures = []
    for prop in args.props:
        for a in args.archs:
            name, fn, uses_desc = ARCH_FNS[a]
            t0 = time.time()
            try:
                if uses_desc:
                    fn(prop, desc_df, records)
                else:
                    fn(prop, smiles, records)
                print(f"  {prop:10s} {name:8s} ok  ({time.time() - t0:.1f}s)",
                      flush=True)
            except Exception as e:
                failures.append((prop, name, repr(e)))
                print(f"  {prop:10s} {name:8s} FAILED: {e}", flush=True)
                traceback.print_exc()

    out = pd.DataFrame(records, columns=["SMILES", "prop", "arch", "fold", "pred"])
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out)} rows)", flush=True)
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures:
            print("  ", f)


if __name__ == "__main__":
    main()
