#!/usr/bin/env python3
"""Evaluate reference lubricant base oils under each weight profile.

Reference set matches Egheosas's draft: PAO 4 (representative isomer),
hexadecane, squalane, dioctyl sebacate (DOS), diisodecyl adipate (DIDA).

Run from repo root:
    python LubeOil/scripts/baseline_eval.py --model_dir models --training_data_dir training/data
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "LubeOil" / "src"))

from lube_wsga_helper import (  # noqa: E402
    evaluate_molecules,
    assign_validity,
    load_regression_models_with_aux,
    load_tox21_predictor,
    load_biodeg_model,
    compute_mahalanobis_params,
)
from SCScorer import SCScorer  # noqa: E402
from lube_fitness import add_dvi_and_lube_fitness, WEIGHT_PROFILES  # noqa: E402


REFERENCE_OILS = {
    # Representative C30 PAO 4 isomer (tri-decene trimer, 2,4,6-tri-octyl-decane skeleton simplified)
    "PAO 4":              "CCCCCCCCC(CCCCCCCC)CC(CCCCCCCC)CCCCCCCC",
    "Hexadecane":          "CCCCCCCCCCCCCCCC",
    "Squalane":            "CC(C)CCCC(C)CCCC(C)CCCCC(C)CCCC(C)CCCC(C)C",
    "Dioctyl sebacate":    "CCCCCCCCOC(=O)CCCCCCCCC(=O)OCCCCCCCC",
    "Diisodecyl adipate":  "CC(C)CCCCCCCOC(=O)CCCCC(=O)OCCCCCCCC(C)C",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(REPO / "models"))
    ap.add_argument("--training_data_dir", default=str(REPO / "training" / "data"))
    ap.add_argument("--output", default=str(REPO / "LubeOil" / "outputs" / "baselines.csv"))
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    thermo_targets = [
        "bp", "mp", "fp",
        "density_40C", "viscosity_40C", "viscosity_100C",
        "tc_40C", "cpsat_40C", "beta_40C",
    ]
    thermo_models = load_regression_models_with_aux(thermo_targets, args.model_dir)
    sc_model = SCScorer()
    sc_model.restore(os.path.join(
        args.model_dir,
        "SCScorer/scscore/models/full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz"))
    tox21_models = load_tox21_predictor(os.path.join(args.model_dir, "tox21_gt4sd"))
    biodeg_model = load_biodeg_model(os.path.join(args.model_dir, "biodeg"))

    mahal = {}
    for t, data in thermo_models.items():
        p = compute_mahalanobis_params(t, data["features"], args.training_data_dir)
        if p is not None:
            mahal[t] = p
    mahal = mahal or None

    df = pd.DataFrame({
        "name": list(REFERENCE_OILS.keys()),
        "SMILES": list(REFERENCE_OILS.values()),
    })

    evaluated = evaluate_molecules(
        df,
        thermo_models=thermo_models,
        sc_model=sc_model,
        tox21_models=tox21_models,
        biodeg_model=biodeg_model,
        molprice_model=None,
        mahal_params=mahal,
        drop_descriptors=True,
    )
    evaluated = assign_validity(evaluated, dc_max=999)

    rows = []
    for profile in WEIGHT_PROFILES:
        scored = add_dvi_and_lube_fitness(evaluated, weight_profile=profile)
        for i, r in scored.iterrows():
            name = df.loc[df["SMILES"] == r["SMILES"], "name"].iloc[0]
            rows.append({
                "name": name,
                "SMILES": r["SMILES"],
                "profile": profile,
                "viscosity_40C": r.get("viscosity_40C"),
                "viscosity_100C": r.get("viscosity_100C"),
                "density_40C": r.get("density_40C"),
                "tc_40C": r.get("tc_40C"),
                "cpsat_40C": r.get("cpsat_40C"),
                "DVI": r.get("DVI"),
                "SCScore": r.get("SCScore"),
                "Tox21_Score": r.get("Tox21_Score"),
                "FOM_LUBE_raw": r.get("FOM_LUBE_raw"),
                "FOM_LUBE": r.get("FOM_LUBE"),
                "Biodegradable": r.get("Biodegradable"),
                "is_valid": r.get("is_valid"),
            })

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"Saved baselines to {args.output}")
    for name, grp in out.groupby("name"):
        row = grp.iloc[0]
        print(f"  {name:20s} nu40={row['viscosity_40C']:.2f} nu100={row['viscosity_100C']:.2f} "
              f"DVI={row['DVI']:.1f}  even_FOM_LUBE={grp.loc[grp['profile']=='even','FOM_LUBE'].iloc[0]:.3f}")


if __name__ == "__main__":
    main()
