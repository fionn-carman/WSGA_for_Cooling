#!/usr/bin/env python3
"""
Control: is the melting-point shift on the 28 front molecules a property of the
front, or does the XGBoost five-fold ensemble disagree with the deployed
full-data refit on any molecule from the same pool?

Predicts T_m with (a) the deployed models/mp/model/xgb_model.joblib and (b) the
five-fold ensemble models/mp/model/fold{0..4}_model.joblib, for
  - the 28 Pareto-front molecules
  - 1200 molecules drawn at random from the same feasible pool
and reports how many still clear the -30 degC gate under each.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors as mordred_descriptors

RDLogger.DisableLog("rdApp.*")
WD = Path("/Users/fionncarman/Desktop/WSGA_for_Cooling")
OUT = WD / "outputs/front28_4arch"


def descs(smiles):
    rn = [d[0] for d in Descriptors._descList]
    rf = {d[0]: d[1] for d in Descriptors._descList}
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    ok = [m is not None for m in mols]
    mols = [m for m in mols if m is not None]
    smiles = [s for s, k in zip(smiles, ok) if k]
    rows = []
    for m in mols:
        r = {}
        for n in rn:
            try:
                r[n] = rf[n](m)
            except Exception:
                r[n] = np.nan
        rows.append(r)
    rdf = pd.DataFrame(rows)
    rdf.columns = [f"rdkit_{c}" for c in rdf.columns]
    calc = Calculator(mordred_descriptors, ignore_3D=True)
    mdf = calc.pandas(mols, quiet=True)
    for c in mdf.columns:
        mdf[c] = pd.to_numeric(mdf[c], errors="coerce")
    mdf.columns = [f"mordred_{c}" for c in mdf.columns]
    return pd.concat([pd.DataFrame({"SMILES": smiles}), rdf, mdf], axis=1)


def X_of(df, feats):
    X = np.zeros((len(df), len(feats)))
    for j, f in enumerate(feats):
        if f in df.columns:
            X[:, j] = pd.to_numeric(df[f], errors="coerce").values
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def predict(df, model_dir, prop_log=False):
    full = joblib.load(model_dir / "xgb_model.joblib")
    yf = full["model"].predict(X_of(df, full["selected_features"]))
    if full.get("log_transform"):
        yf = np.expm1(yf)
    folds = []
    for k in range(5):
        b = joblib.load(model_dir / f"fold{k}_model.joblib")
        y = b["model"].predict(X_of(df, b["selected_features"]))
        if b.get("log_transform"):
            y = np.expm1(y)
        folds.append(y)
    return yf, np.mean(folds, axis=0), np.std(folds, axis=0, ddof=1)


def main():
    front = pd.read_csv(WD / "training/front28/front28.csv").SMILES.tolist()
    pool = pd.read_csv(OUT / "pool_sample2000.csv")

    print(f"computing descriptors for {len(front)} front + {len(pool)} pool "
          f"molecules ...", flush=True)
    d_front = descs(front)
    d_pool = descs(pool.SMILES.tolist())
    print("done", flush=True)

    rows = []
    for name, dd, ref in [("front", d_front, None), ("pool", d_pool, pool)]:
        for prop, gate, op in [("mp", -30.0, "<="), ("dc", 7.0, "<="),
                               ("fp", 373.15, ">=")]:
            full, fold, sd = predict(dd, WD / "models" / prop / "model")
            pf = (full <= gate) if op == "<=" else (full >= gate)
            pk = (fold <= gate) if op == "<=" else (fold >= gate)
            rows.append(dict(
                set=name, prop=prop, n=len(dd),
                full_pass=int(pf.sum()), fold_pass=int(pk.sum()),
                full_pass_pct=100 * pf.mean(), fold_pass_pct=100 * pk.mean(),
                median_shift=float(np.median(fold - full)),
                mean_shift=float(np.mean(fold - full)),
                median_fold_sd=float(np.median(sd)),
            ))
            print(f"  {name:6s} {prop:3s}  full {pf.sum():5d}/{len(dd)} "
                  f"({100*pf.mean():5.1f}%)  fold {pk.sum():5d}/{len(dd)} "
                  f"({100*pk.mean():5.1f}%)  median shift "
                  f"{np.median(fold-full):+7.2f}", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "control_fold_vs_full.csv", index=False)
    print("\n" + r.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
