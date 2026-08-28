#!/usr/bin/env python3
"""
Four-architecture consensus properties for the 28 pooled Pareto-front molecules.

Takes the long prediction table written by training/front28/predict_front_4arch.py
(SMILES, prop, arch, fold, pred) and

  1. collapses each architecture to one value per property (mean of its 5 fold
     models, real units — the same ensembling src/wsga_helper.py uses),
  2. forms the consensus value as the mean over the four architectures, with the
     error bar taken as the sample standard deviation between architectures,
  3. re-applies the production validity gates to the consensus values,
  4. recomputes the FOM1 / flash-point Pareto front and reports how it moves
     relative to the deployed single-XGBoost front the molecules were found on.

Usage:
    python analyse_front28_4arch.py \
        --preds ../outputs/front28_4arch/front28_predictions_long.csv \
        --reference <TierReview>/cache/pooled.csv \
        --outdir ../outputs/front28_4arch
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARCHS = ["XGBoost", "MLP", "D-MPNN", "Uni-Mol"]

# Production validity gates (scripts/fp_molprice_sweep_nist8100.sh)
GATES = {
    "mp": ("<=", -30.0,   "Melting point / degC"),
    "bp": (">=", 100.0,   "Boiling point / degC"),
    "dc": ("<=", 7.0,     "Dielectric constant"),
    "fp": (">=", 373.15,  "Flash point / K"),
}

PROP_LABEL = {
    "density":   "Density / g cm$^{-3}$",
    "viscosity": "Kinematic viscosity / cSt",
    "tc":        "Thermal conductivity / W m$^{-1}$ K$^{-1}$",
    "cpsat":     "$C_{p,sat}$ / J K$^{-1}$ mol$^{-1}$",
    "beta":      "Thermal expansion coeff. / K$^{-1}$",
    "fom1":      "FOM1",
    "bp":        "Boiling point / $^\\circ$C",
    "fp":        "Flash point / K",
    "mp":        "Melting point / $^\\circ$C",
    "dc":        "Dielectric constant",
}
PROP_ORDER = ["fom1", "fp", "mp", "bp", "dc",
              "density", "viscosity", "tc", "cpsat", "beta"]


def deployed_props(path, index):
    """The deployed full-data-refit XGBoost values the GA filtered on."""
    if not path or not Path(path).exists():
        return None
    d = pd.read_csv(path).drop_duplicates("SMILES").set_index("SMILES")
    return d.reindex(index)


def pareto_front(df, xcol, ycol):
    """Indices of the non-dominated set, both objectives maximised."""
    pts = df[[xcol, ycol]].values
    keep = []
    for i, p in enumerate(pts):
        dominated = np.any(
            (pts[:, 0] >= p[0]) & (pts[:, 1] >= p[1]) &
            ((pts[:, 0] > p[0]) | (pts[:, 1] > p[1]))
        )
        if not dominated:
            keep.append(i)
    return df.index[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--reference", required=True,
                    help="pooled.csv: the deployed-model front the 28 came from")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--deployed", default=None,
                    help="front_props.csv: the deployed full-refit XGBoost "
                         "values the GA actually filtered on")
    ap.add_argument("--metrics", default=None,
                    help="benchmark_metrics.csv: prop,arch,r2,mae,rmse")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    long = pd.read_csv(args.preds)
    ref = pd.read_csv(args.reference)

    n_arch = long.arch.nunique()
    print(f"{long.SMILES.nunique()} molecules x {long['prop'].nunique()} properties "
          f"x {n_arch} architectures x {long.fold.nunique()} folds "
          f"= {len(long)} predictions")
    if n_arch != 4:
        print(f"WARNING: expected 4 architectures, found {sorted(long.arch.unique())}")

    # ---- 1. architecture value = mean of its 5 folds -----------------------
    arch_val = (long.groupby(["SMILES", "prop", "arch"])
                    .pred.agg(["mean", "std"])
                    .rename(columns={"mean": "arch_mean", "std": "fold_std"})
                    .reset_index())
    arch_val.to_csv(outdir / "front28_by_architecture.csv", index=False)

    # ---- 2. consensus = mean +/- sample SD across architectures ------------
    cons = (arch_val.groupby(["SMILES", "prop"]).arch_mean
                    .agg(mean="mean", std=lambda s: s.std(ddof=1),
                         lo="min", hi="max", n="count")
                    .reset_index())
    cons.to_csv(outdir / "front28_consensus.csv", index=False)

    mean_w = cons.pivot(index="SMILES", columns="prop", values="mean")
    std_w = cons.pivot(index="SMILES", columns="prop", values="std")
    arch_w = arch_val.pivot_table(index="SMILES", columns=["prop", "arch"],
                                  values="arch_mean")

    # ---- 2b. human-readable wide table -------------------------------------
    dp = {"density": 3, "viscosity": 2, "tc": 4, "cpsat": 1, "beta": 5,
          "fom1": 1, "bp": 1, "fp": 1, "mp": 1, "dc": 2}
    rep = pd.DataFrame(index=mean_w.index)
    for p in [c for c in PROP_ORDER if c in mean_w.columns]:
        d = dp.get(p, 3)
        rep[p] = [f"{m:.{d}f} +/- {s:.{d}f}"
                  for m, s in zip(mean_w[p], std_w[p])]
    rep.to_csv(outdir / "front28_consensus_readable.csv")

    # per-architecture values, one row per molecule/property
    wide = arch_val.pivot_table(index=["SMILES", "prop"], columns="arch",
                                values="arch_mean")
    wide = wide.reindex(columns=[a for a in ARCHS if a in wide.columns])
    wide["consensus_mean"] = wide.mean(axis=1)
    wide["consensus_sd"] = wide[[a for a in ARCHS if a in wide.columns]].std(
        axis=1, ddof=1)
    wide.to_csv(outdir / "front28_arch_wide.csv")

    # ---- 3. validity gates on the consensus values -------------------------
    gate_rows = []
    for smi in mean_w.index:
        row = {"SMILES": smi}
        all_pass, any_marginal = True, False
        for p, (op, thr, _) in GATES.items():
            m, s = mean_w.loc[smi, p], std_w.loc[smi, p]
            ok = m <= thr if op == "<=" else m >= thr
            # marginal = the threshold sits inside +/- 1 SD of the consensus
            marginal = abs(m - thr) <= s
            row[f"{p}_mean"] = m
            row[f"{p}_std"] = s
            row[f"{p}_pass"] = ok
            row[f"{p}_marginal"] = marginal
            # per-architecture unanimity
            row[f"{p}_n_arch_pass"] = int(sum(
                (arch_w.loc[smi, (p, a)] <= thr) if op == "<="
                else (arch_w.loc[smi, (p, a)] >= thr)
                for a in ARCHS if (p, a) in arch_w.columns))
            all_pass &= ok
            any_marginal |= marginal
        row["all_pass"] = all_pass
        row["any_marginal"] = any_marginal
        gate_rows.append(row)
    gates = pd.DataFrame(gate_rows).set_index("SMILES")
    gates.to_csv(outdir / "front28_constraint_check.csv")

    n_fail = int((~gates.all_pass).sum())
    print(f"\nConstraint check on consensus values: "
          f"{len(gates) - n_fail}/{len(gates)} pass, {n_fail} fail")
    for p in GATES:
        f = gates.index[~gates[f"{p}_pass"]]
        if len(f):
            op, thr, _ = GATES[p]
            print(f"  {p} {op} {thr}: {len(f)} fail")

    # ---- 4. Pareto front on consensus FOM1 / flash point -------------------
    fr = pd.DataFrame({
        "fom1": mean_w.fom1, "fom1_std": std_w.fom1,
        "fp_K": mean_w.fp, "fp_K_std": std_w.fp,
    })
    fr["fp_C"] = fr.fp_K - 273.15
    fr["fp_C_std"] = fr.fp_K_std
    fr["all_pass"] = gates.all_pass
    fr = fr.join(ref.set_index("SMILES")[["fom1_40C", "fp_C", "route", "found_by"]]
                    .rename(columns={"fom1_40C": "fom1_deployed",
                                     "fp_C": "fp_C_deployed"}))

    fr["on_front_all"] = False
    fr.loc[pareto_front(fr, "fom1", "fp_C"), "on_front_all"] = True

    feas = fr[fr.all_pass]
    fr["on_front_feasible"] = False
    if len(feas):
        fr.loc[pareto_front(feas, "fom1", "fp_C"), "on_front_feasible"] = True

    fr = fr.sort_values("fom1", ascending=False)
    fr.to_csv(outdir / "front28_new_front.csv")

    print(f"\nRecomputed front (all 28, ignoring gates): "
          f"{int(fr.on_front_all.sum())}/28 non-dominated")
    print(f"Recomputed front (feasible only):          "
          f"{int(fr.on_front_feasible.sum())}/{len(feas)} non-dominated")

    print("\nBetween-architecture spread, median over the 28 molecules:")
    sp = (cons.assign(rel=100 * cons["std"] / cons["mean"].abs())
              .groupby("prop")[["std", "rel"]].median())
    for p in [x for x in PROP_ORDER if x in sp.index]:
        print(f"  {p:10s} SD {sp.loc[p, 'std']:>10.4g}   "
              f"({sp.loc[p, 'rel']:.1f}% of mean)")

    print("\nFOM1 and flash point, consensus vs deployed (sorted by FOM1):")
    hdr = fr.assign(
        dfom1=fr.fom1 - fr.fom1_deployed,
        dfp=fr.fp_C - fr.fp_C_deployed)[
        ["fom1", "fom1_std", "fom1_deployed", "dfom1",
         "fp_C", "fp_C_std", "fp_C_deployed", "dfp",
         "all_pass", "on_front_feasible"]]
    print(hdr.round(2).to_string())

    # ---- 4b. where the attrition comes from --------------------------------
    # Three distinct things can move a molecule across a gate:
    #   (i)   full-data refit -> five-fold ensemble, same architecture
    #   (ii)  one architecture -> another
    #   (iii) averaging the four
    # Reporting only (iii) would blame architecture disagreement for an
    # attrition that is mostly (i).
    dep = ref.set_index("SMILES")
    dep_props = deployed_props(args.deployed, mean_w.index)

    print("\nGate pass counts out of 28, by model:")
    hdr = ["deployed"] + ARCHS + ["consensus"]
    print(f"  {'gate':<6}" + "".join(f"{h:>11}" for h in hdr))
    for p, (op, thr, _) in GATES.items():
        cells = []
        if dep_props is not None and p in dep_props:
            v = dep_props[p]
            cells.append((v <= thr).sum() if op == "<=" else (v >= thr).sum())
        else:
            cells.append("-")
        for a in ARCHS:
            if (p, a) not in arch_w.columns:
                cells.append("-")
                continue
            v = arch_w[(p, a)]
            cells.append((v <= thr).sum() if op == "<=" else (v >= thr).sum())
        v = mean_w[p]
        cells.append((v <= thr).sum() if op == "<=" else (v >= thr).sum())
        print(f"  {p:<6}" + "".join(f"{str(c):>11}" for c in cells))

    rows = []
    for label, get in ([("deployed", lambda p: dep_props[p])]
                       if dep_props is not None else []) + \
                      [(a, (lambda p, a=a: arch_w[(p, a)])) for a in ARCHS] + \
                      [("consensus", lambda p: mean_w[p])]:
        ok = np.ones(len(mean_w), bool)
        for p, (op, thr, _) in GATES.items():
            v = get(p).values
            ok &= (v <= thr) if op == "<=" else (v >= thr)
        rows.append({"model": label, "passes_all_four_gates": int(ok.sum()),
                     "of": len(mean_w)})
    dec = pd.DataFrame(rows)
    dec.to_csv(outdir / "front28_gate_attrition.csv", index=False)
    print("\nPasses all four gates simultaneously:")
    for r in rows:
        print(f"  {r['model']:<10} {r['passes_all_four_gates']:>3}/{r['of']}")

    # ---- 5. figures --------------------------------------------------------
    mae = None
    if args.metrics and Path(args.metrics).exists():
        m = pd.read_csv(args.metrics)
        mae = m.groupby("prop").mae.mean().to_dict()
    make_front_figure(fr, outdir)
    make_spread_figure(arch_val, cons, outdir, mae=mae)
    make_gate_figure(gates, outdir)

    print(f"\nWrote to {outdir}:")
    for f in sorted(outdir.iterdir()):
        print("  ", f.name)


def make_front_figure(fr, outdir):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

    ax.errorbar(fr.fp_C, fr.fom1, xerr=fr.fp_C_std, yerr=fr.fom1_std,
                fmt="none", ecolor="0.75", elinewidth=0.8, capsize=0, zorder=1)

    drop = fr[~fr.all_pass]
    keep = fr[fr.all_pass]
    ax.scatter(drop.fp_C, drop.fom1, s=26, facecolors="none",
               edgecolors="#c44e52", linewidths=1.2, zorder=3,
               label=f"fails a gate ({len(drop)})")
    ax.scatter(keep.fp_C, keep.fom1, s=26, color="#4c72b0", zorder=3,
               label=f"passes all gates ({len(keep)})")

    # the front the 28 sit on when the gates are ignored — isolates how the
    # FOM1/flash-point trade-off itself moves, separately from gate attrition
    na = fr[fr.on_front_all].sort_values("fp_C")
    ax.step(na.fp_C, na.fom1, where="post", color="#4c72b0", lw=1.2, ls=":",
            zorder=2, label=f"consensus front, gates ignored ({len(na)})")

    nf = fr[fr.on_front_feasible].sort_values("fp_C")
    ax.step(nf.fp_C, nf.fom1, where="post", color="#4c72b0", lw=1.4,
            zorder=2, label=f"consensus front, gates applied ({len(nf)})")

    od = fr.sort_values("fp_C_deployed")
    ax.plot(od.fp_C_deployed, od.fom1_deployed, color="0.4", lw=1.0, ls="--",
            zorder=2, label="deployed XGBoost front (28)")

    ax.axvline(100.0, color="0.3", lw=0.8, ls=":")
    ax.text(101, ax.get_ylim()[0] + 1, "flash point gate", fontsize=7,
            color="0.3", rotation=90, va="bottom")

    ax.set_xlabel("Flash point / $^\\circ$C")
    ax.set_ylabel("FOM1")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"front28_consensus_front.{ext}",
                    bbox_inches="tight")
    plt.close(fig)


def make_spread_figure(arch_val, cons, outdir, mae=None):
    """Between-architecture spread per property, in units of the property's own
    out-of-fold MAE. A value above 1 means the four architectures disagree with
    each other by more than any one of them is typically wrong on held-out
    data — i.e. the benchmark R2 does not transfer to these molecules.

    Normalising by the consensus mean is wrong here: melting point straddles
    zero, so a percentage is meaningless for it.
    """
    c = cons.copy()
    if mae is None:
        c["rel"] = 100 * c["std"] / c["mean"].abs()
        ylabel = "Between-architecture SD / % of consensus mean"
    else:
        c["rel"] = c["std"] / c["prop"].map(mae)
        ylabel = "Between-architecture SD / out-of-fold MAE"
    order = [p for p in PROP_ORDER if p in set(c["prop"])]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    data = [c.loc[c["prop"] == p, "rel"].values for p in order]
    bp = ax.boxplot(data, vert=True, widths=0.6, patch_artist=True,
                    medianprops=dict(color="black", lw=1.2),
                    flierprops=dict(marker="o", ms=2.5, mfc="0.4",
                                    mec="none"))
    for patch in bp["boxes"]:
        patch.set_facecolor("#cdd8e8")
        patch.set_edgecolor("0.35")
        patch.set_linewidth(0.9)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color("0.35")
        w.set_linewidth(0.9)

    if mae is not None:
        ax.axhline(1.0, color="#c44e52", lw=1.0, ls="--", zorder=0)
        ax.text(len(order) + 0.45, 1.0, "parity with\nsingle-model MAE",
                fontsize=6.5, color="#c44e52", va="center", ha="right")

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"front28_arch_spread.{ext}", bbox_inches="tight")
    plt.close(fig)


def make_gate_figure(gates, outdir):
    props = list(GATES)
    M = np.zeros((len(gates), len(props)))
    for j, p in enumerate(props):
        M[:, j] = np.where(gates[f"{p}_pass"],
                           np.where(gates[f"{p}_marginal"], 1, 2), 0)

    order = np.lexsort((M.sum(1), ~gates.all_pass.values))
    M = M[order]
    labels = gates.index.values[order]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    cmap = matplotlib.colors.ListedColormap(
        ["#c44e52", "#e8c76a", "#4c72b0"])
    ax.imshow(np.clip(M, 0, 2), cmap=cmap, aspect="auto", vmin=0, vmax=2)

    ax.set_xticks(range(len(props)))
    ax.set_xticklabels([f"{p}\n{GATES[p][0]} {GATES[p][1]:g}" for p in props],
                       fontsize=7.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=5.5, family="monospace")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    handles = [plt.Line2D([], [], marker="s", ls="none", ms=7, mec="none",
                          mfc=c, label=l)
               for c, l in zip(["#4c72b0", "#e8c76a", "#c44e52"],
                               ["passes", "passes, within 1 SD of gate",
                                "fails"])]
    ax.legend(handles=handles, frameon=False, fontsize=7.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"front28_gates.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
