"""
Analyse hyperparameter sweep results.

Parses output directories from hyperparam_sweep.sh, extracts the top
molecules and generation stats from each run, and identifies the best
hyperparameter configurations.

Ranking approach:
  - For each run, read the final top-25 CSV and take the #1 molecule
    (by FitnessScore) and the mean FOM1_avg of the top 10.
  - Group the 3 repeat seeds per hyperparameter config and compute
    the mean and std of both metrics across repeats.
  - Rank configurations by mean #1 FOM1_avg across repeats (most
    robust indicator — a config that flukes one good molecule but
    has poor top-10 average is less useful than one that reliably
    produces a strong population).
  - Also report FOM1_40 and FOM1_100 breakdowns for the top configs.
  - Collect the best unique molecules across all runs.

Usage:
    python analyse_hyperparam_sweep.py --sweep_dir ../outputs/hyperparam_sweep
    python analyse_hyperparam_sweep.py --sweep_dir ../outputs/hyperparam_sweep --top_configs 20
"""

import os
import re
import argparse
import pandas as pd
import numpy as np


def parse_run_dir(dirname):
    """Extract hyperparameters from directory name.

    Expected format: pop{}_mr{}_er{}_tau{}_seed{}
    """
    pattern = r"pop(\d+)_mr([\d.]+)_er([\d.]+)_tau([\d.]+)_seed(\d+)"
    m = re.match(pattern, dirname)
    if not m:
        return None
    return {
        "pop": int(m.group(1)),
        "mr": float(m.group(2)),
        "er": float(m.group(3)),
        "tau": float(m.group(4)),
        "seed": int(m.group(5)),
    }


def load_top_molecules(run_path):
    """Load the final top-25 CSV from a run directory."""
    candidates = [f for f in os.listdir(run_path)
                  if f.startswith("top_") and f.endswith(".csv")]
    if not candidates:
        return None
    return pd.read_csv(os.path.join(run_path, candidates[0]))


def load_generation_stats(run_path):
    """Load generation_stats.csv from a run directory."""
    path = os.path.join(run_path, "generation_stats.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def analyse_sweep(sweep_dir, top_n_configs=10, top_n_molecules=50):

    # ------------------------------------------------------------------
    # 1. Scan all run directories
    # ------------------------------------------------------------------
    runs = []
    for dirname in sorted(os.listdir(sweep_dir)):
        run_path = os.path.join(sweep_dir, dirname)
        if not os.path.isdir(run_path):
            continue
        params = parse_run_dir(dirname)
        if params is None:
            continue

        top_df = load_top_molecules(run_path)
        if top_df is None or top_df.empty:
            continue

        gen_stats = load_generation_stats(run_path)

        best = top_df.iloc[0]
        top10 = top_df.head(10)

        run_info = {
            **params,
            "dir": dirname,
            "best_FOM1_avg": best.get("FOM1_avg", np.nan),
            "best_FOM1_40": best.get("FOM1_40", np.nan),
            "best_FOM1_100": best.get("FOM1_100", np.nan),
            "best_fitness": best.get("FitnessScore", np.nan),
            "best_smiles": best.get("SMILES", best.get("CanonicalSMILES", "")),
            "top10_mean_FOM1_avg": top10["FOM1_avg"].mean() if "FOM1_avg" in top10.columns else np.nan,
            "top10_mean_FOM1_40": top10["FOM1_40"].mean() if "FOM1_40" in top10.columns else np.nan,
            "top10_mean_FOM1_100": top10["FOM1_100"].mean() if "FOM1_100" in top10.columns else np.nan,
            "n_valid_top25": int(top_df["is_valid"].sum()) if "is_valid" in top_df.columns else len(top_df),
        }

        if gen_stats is not None and not gen_stats.empty:
            run_info["final_gen"] = int(gen_stats["generation"].max())
            if "FitnessScore_max" in gen_stats.columns:
                run_info["final_best_fitness"] = gen_stats["FitnessScore_max"].iloc[-1]
        else:
            run_info["final_gen"] = np.nan
            run_info["final_best_fitness"] = np.nan

        runs.append(run_info)

    if not runs:
        print("No completed runs found.")
        return

    runs_df = pd.DataFrame(runs)
    print(f"Found {len(runs_df)} completed runs\n")

    # ------------------------------------------------------------------
    # 2. Aggregate across seeds per config
    # ------------------------------------------------------------------
    group_cols = ["pop", "mr", "er", "tau"]
    metric_cols = [
        "best_FOM1_avg", "best_FOM1_40", "best_FOM1_100",
        "top10_mean_FOM1_avg", "top10_mean_FOM1_40", "top10_mean_FOM1_100",
        "best_fitness",
    ]

    agg_dict = {col: ["mean", "std", "count"] for col in metric_cols}
    configs = runs_df.groupby(group_cols).agg(agg_dict).reset_index()

    # Flatten multi-level columns
    configs.columns = [
        f"{a}_{b}" if b else a
        for a, b in configs.columns
    ]

    # ------------------------------------------------------------------
    # 3. Report best configs
    # ------------------------------------------------------------------
    configs = configs.sort_values("best_FOM1_avg_mean", ascending=False)
    show = configs.head(top_n_configs)

    n_seeds = int(show["best_FOM1_avg_count"].iloc[0])

    print(f"{'='*90}")
    print(f"  TOP {top_n_configs} HYPERPARAMETER CONFIGS (ranked by mean #1 FOM1_avg across {n_seeds} seeds)")
    print(f"{'='*90}\n")

    print(f"{'Rank':<5} {'Pop':<6} {'MR':<5} {'ER':<5} {'Tau':<6} "
          f"{'#1 FOM1_avg':>16}   {'Top10 FOM1_avg':>18}   "
          f"{'#1 FOM1_40':>10} {'#1 FOM1_100':>11}")
    print("-" * 105)

    for i, (_, row) in enumerate(show.iterrows(), 1):
        std_1 = row["best_FOM1_avg_std"]
        std_10 = row["top10_mean_FOM1_avg_std"]
        # handle NaN std (single seed completed)
        s1 = f"{std_1:.4f}" if not np.isnan(std_1) else "  n/a"
        s10 = f"{std_10:.4f}" if not np.isnan(std_10) else "  n/a"
        print(
            f"{i:<5} {row['pop']:<6.0f} {row['mr']:<5.1f} {row['er']:<5.1f} {row['tau']:<6.2f} "
            f"{row['best_FOM1_avg_mean']:>10.4f} +/- {s1}"
            f"   {row['top10_mean_FOM1_avg_mean']:>10.4f} +/- {s10}"
            f"   {row['best_FOM1_40_mean']:>10.4f}"
            f"   {row['best_FOM1_100_mean']:>10.4f}"
        )

    # ------------------------------------------------------------------
    # 4. Per-parameter marginal analysis
    # ------------------------------------------------------------------
    print(f"\n{'='*90}")
    print(f"  MARGINAL EFFECT OF EACH PARAMETER (mean #1 FOM1_avg)")
    print(f"{'='*90}\n")

    for param in group_cols:
        marginal = runs_df.groupby(param)["best_FOM1_avg"].agg(["mean", "std", "count"])
        marginal = marginal.sort_values("mean", ascending=False)
        print(f"  {param}:")
        for val, row in marginal.iterrows():
            print(f"    {val:<8}  mean={row['mean']:.4f}  std={row['std']:.4f}  n={int(row['count'])}")
        print()

    # ------------------------------------------------------------------
    # 5. Collect best unique molecules across all runs
    # ------------------------------------------------------------------
    all_top_mols = []
    for _, run in runs_df.iterrows():
        run_path = os.path.join(sweep_dir, run["dir"])
        top_df = load_top_molecules(run_path)
        if top_df is None:
            continue
        top_df = top_df.copy()
        top_df["source_config"] = (
            f"pop{run['pop']}_mr{run['mr']}_er{run['er']}_tau{run['tau']}"
        )
        top_df["source_seed"] = run["seed"]
        all_top_mols.append(top_df)

    if all_top_mols:
        combined = pd.concat(all_top_mols, ignore_index=True)
        smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in combined.columns else "SMILES"
        sort_col = "FOM1_avg" if "FOM1_avg" in combined.columns else "FitnessScore"
        combined = combined.sort_values(sort_col, ascending=False)
        unique = combined.drop_duplicates(subset=[smiles_col], keep="first")
        top_unique = unique.head(top_n_molecules)

        print(f"{'='*130}")
        print(f"  TOP {top_n_molecules} UNIQUE MOLECULES ACROSS ALL RUNS ({len(unique)} unique total)")
        print(f"{'='*130}\n")

        print(f"{'Rank':<5} {'SMILES':<45} {'FOM1_avg':>10} {'FOM1_40':>10} "
              f"{'FOM1_100':>10} {'MP':>7} {'FP':>7} {'SC':>5}  {'Config'}")
        print("-" * 130)

        for i, (_, mol) in enumerate(top_unique.iterrows(), 1):
            smi = str(mol.get(smiles_col, ""))
            if len(smi) > 42:
                smi = smi[:39] + "..."
            print(
                f"{i:<5} {smi:<45} "
                f"{mol.get('FOM1_avg', np.nan):>10.4f} "
                f"{mol.get('FOM1_40', np.nan):>10.4f} "
                f"{mol.get('FOM1_100', np.nan):>10.4f} "
                f"{mol.get('MP-Measured', np.nan):>7.1f} "
                f"{mol.get('flashpoint', np.nan):>7.1f} "
                f"{mol.get('SCScore', np.nan):>5.2f}  "
                f"{mol.get('source_config', '')}"
            )

    # ------------------------------------------------------------------
    # 6. Save results to CSV
    # ------------------------------------------------------------------
    out_dir = os.path.join(sweep_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    runs_df.to_csv(os.path.join(out_dir, "all_runs.csv"), index=False)
    configs.to_csv(os.path.join(out_dir, "configs_aggregated.csv"), index=False)

    if all_top_mols:
        top_unique.to_csv(
            os.path.join(out_dir, "top_molecules.csv"), index=False
        )

    print(f"\nResults saved to: {out_dir}/")
    print(f"  all_runs.csv            - per-run metrics ({len(runs_df)} runs)")
    print(f"  configs_aggregated.csv  - configs averaged over seeds ({len(configs)} configs)")
    print(f"  top_molecules.csv       - best {top_n_molecules} unique molecules")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse WSGA hyperparameter sweep results"
    )
    parser.add_argument("--sweep_dir", type=str,
                        default="../outputs/hyperparam_sweep")
    parser.add_argument("--top_configs", type=int, default=10)
    parser.add_argument("--top_molecules", type=int, default=50)
    args = parser.parse_args()

    analyse_sweep(args.sweep_dir, args.top_configs, args.top_molecules)
