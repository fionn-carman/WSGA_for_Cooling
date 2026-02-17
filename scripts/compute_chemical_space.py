"""
Chemical Space — Step 1: Compute shared UMAP embedding.

Generates reference molecules, loads evaluated molecules from selected
hyperparameter configurations (pooling all seeds per config), fits a
single UMAP on the combined fingerprint matrix, and saves the results
to a .npz file for downstream plotting.

Because UMAP coordinates are only comparable when fit on the same data,
we must embed everything in one go. This script handles the heavy
computation; the companion compare_chemical_space.py generates figures.

Usage:
    # Auto-select extreme configs from old sweep (legacy format)
    python compute_chemical_space.py \\
        --sweep_dir ../outputs/hyperparam_sweep \\
        --data_dir ../data \\
        --output chemical_space_data.npz

    # Manually specify run directories (pools seeds automatically)
    python compute_chemical_space.py \\
        --run_dirs ../outputs/hyperparam_sweep/pop500_mr0.5_er0.1_tau0.015_seed42 \\
                   ../outputs/hyperparam_sweep/pop2000_mr0.8_er0.3_tau0.15_seed42 \\
        --data_dir ../data \\
        --output chemical_space_data.npz
"""

import os
import re
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
except ImportError:
    print("ERROR: RDKit required. Install: conda install -c conda-forge rdkit")
    sys.exit(1)

RDLogger.DisableLog("rdApp.*")

try:
    # Block TensorFlow import *before* importing umap to prevent
    # umap/__init__.py -> parametric_umap -> tensorflow crash on some systems.
    import types as _types
    _tf_stub = _types.ModuleType("tensorflow")
    _tf_stub.__version__ = "0.0.0"
    sys.modules.setdefault("tensorflow", _tf_stub)

    import umap.umap_ as umap_module
    UMAP = umap_module.UMAP
except (ImportError, RuntimeError):
    try:
        import umap
        UMAP = umap.UMAP
    except (ImportError, RuntimeError):
        print("ERROR: umap-learn required. Install: pip install umap-learn")
        sys.exit(1)


# ======================================================================
# Fingerprints
# ======================================================================

def smiles_to_fingerprint(smi, radius=2, n_bits=2048):
    if not isinstance(smi, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None


def fps_to_numpy(fps):
    arr = np.zeros((len(fps), fps[0].GetNumBits()), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


# ======================================================================
# Reference set generation
# ======================================================================

def generate_reference_molecules(data_dir, n_molecules=10000):
    """Generate random reference molecules via n-gram model, with fallback."""
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    sys.path.insert(0, src_dir)

    try:
        from generate_molecules import generate_initial_population, load_combined_training_data
        print(f"  Generating {n_molecules} reference molecules via n-gram model...")
        training_smiles = load_combined_training_data(data_dir)
        ref_df = generate_initial_population(
            n=n_molecules,
            max_heavy_atoms=30,
            min_heavy_atoms=5,
            max_carbons=30,
            max_oxygens=6,
            training_smiles=training_smiles,
        )
        print(f"  Generated {len(ref_df)} reference molecules")
        return ref_df["SMILES"].tolist()
    except Exception as e:
        print(f"  n-gram generator failed ({e}), falling back to training SMILES...")
        all_smiles = []
        data_files = [
            "processed_full_hydrocarbon_dataset.csv",
            "biodegradability_cleaned.csv",
            "BP-Measured_cleaned.csv",
            "DC_exp_cleaned.csv",
            "flashpoint_cleaned.csv",
            "MP-Measured_cleaned.csv",
        ]
        for fname in data_files:
            fpath = os.path.join(data_dir, fname)
            if os.path.exists(fpath):
                try:
                    df = pd.read_csv(fpath)
                    for col in df.columns:
                        if col.lower() in ("smiles", "canonical_smiles", "canonicalsmiles"):
                            all_smiles.extend(df[col].dropna().tolist())
                            break
                except Exception:
                    pass
        unique = list(set(all_smiles))
        if len(unique) > n_molecules:
            np.random.seed(42)
            unique = list(np.random.choice(unique, size=n_molecules, replace=False))
        print(f"  Loaded {len(unique)} training SMILES as reference")
        return unique


# ======================================================================
# Config selection
# ======================================================================

def parse_legacy_dir(dirname):
    """Parse pop{}_mr{}_er{}_tau{}_seed{} directory name."""
    m = re.match(r"pop(\d+)_mr([\d.]+)_er([\d.]+)_tau([\d.]+)_seed(\d+)", dirname)
    if m:
        return {
            "pop": int(m.group(1)),
            "mr": float(m.group(2)),
            "er": float(m.group(3)),
            "tau": float(m.group(4)),
            "seed": int(m.group(5)),
        }
    # Stage 1 format: pop{}_mr{}_er{}_k{}_seed{}
    m = re.match(r"pop(\d+)_mr([\d.]+)_er([\d.]+)_k(\d+)_seed(\d+)", dirname)
    if m:
        return {
            "pop": int(m.group(1)),
            "mr": float(m.group(2)),
            "er": float(m.group(3)),
            "k": int(m.group(4)),
            "seed": int(m.group(5)),
        }
    return None


def config_key(params):
    """Create a hashable key for a config (excluding seed)."""
    keys = sorted(k for k in params if k != "seed")
    return tuple((k, params[k]) for k in keys)


def config_label(params):
    """Human-readable label for a config (excluding seed)."""
    parts = []
    for k in ("pop", "mr", "er", "tau", "k"):
        if k in params:
            parts.append(f"{k}={params[k]}")
    return ", ".join(parts)


def select_extreme_configs(sweep_dir, max_configs=8):
    """Select a diverse set of configs from a sweep directory.

    Strategy: pick configs that represent extremes of each parameter axis
    plus the overall best and worst performers.
    """
    print(f"\nScanning {sweep_dir} for completed runs...")

    # Parse all directories and group by config
    configs = {}  # config_key -> {params, seeds: [seed_ints], best_fom: float}
    for dirname in sorted(os.listdir(sweep_dir)):
        dirpath = os.path.join(sweep_dir, dirname)
        if not os.path.isdir(dirpath):
            continue
        params = parse_legacy_dir(dirname)
        if params is None:
            continue
        # Check run completed
        eval_path = os.path.join(dirpath, "all_evaluated_molecules.csv")
        if not os.path.exists(eval_path):
            continue

        ck = config_key(params)
        if ck not in configs:
            configs[ck] = {"params": {k: v for k, v in params.items() if k != "seed"},
                           "seeds": [], "best_foms": []}
        configs[ck]["seeds"].append(params["seed"])

        # Read best FOM from top file
        top_files = [f for f in os.listdir(dirpath)
                     if f.startswith("top_") and f.endswith(".csv")]
        if top_files:
            try:
                top_df = pd.read_csv(os.path.join(dirpath, top_files[0]))
                if not top_df.empty:
                    fom = top_df.iloc[0].get("FOM1_avg", 0)
                    configs[ck]["best_foms"].append(fom)
            except Exception:
                pass

    print(f"  Found {len(configs)} unique configs ({sum(len(c['seeds']) for c in configs.values())} total runs)")

    if not configs:
        return []

    # Compute mean FOM per config
    for ck, info in configs.items():
        info["mean_fom"] = np.mean(info["best_foms"]) if info["best_foms"] else 0

    ranked = sorted(configs.items(), key=lambda x: x[1]["mean_fom"], reverse=True)

    # Selection strategy: best, worst, and extremes of each varying parameter
    selected_keys = set()

    # 1. Overall best
    selected_keys.add(ranked[0][0])
    # 2. Overall worst
    selected_keys.add(ranked[-1][0])

    # 3. For each parameter, pick the config with the highest and lowest value
    #    (using the best-performing config at that extreme)
    param_names = set()
    for ck, info in configs.items():
        param_names.update(info["params"].keys())

    for param in param_names:
        # Sort configs by this parameter value
        configs_with_param = [(ck, info) for ck, info in configs.items()
                              if param in info["params"]]
        if not configs_with_param:
            continue

        # Lowest value of param -> pick the one with best FOM at that value
        min_val = min(info["params"][param] for _, info in configs_with_param)
        max_val = max(info["params"][param] for _, info in configs_with_param)

        min_configs = [(ck, info) for ck, info in configs_with_param
                       if info["params"][param] == min_val]
        max_configs = [(ck, info) for ck, info in configs_with_param
                       if info["params"][param] == max_val]

        # Best performer at each extreme
        best_at_min = max(min_configs, key=lambda x: x[1]["mean_fom"])[0]
        best_at_max = max(max_configs, key=lambda x: x[1]["mean_fom"])[0]

        selected_keys.add(best_at_min)
        selected_keys.add(best_at_max)

    # 4. If we have room, add median performer
    if len(selected_keys) < max_configs:
        mid_idx = len(ranked) // 2
        selected_keys.add(ranked[mid_idx][0])

    # Trim to max
    selected_keys = list(selected_keys)[:max_configs]

    # Build output
    selected = []
    for ck in selected_keys:
        info = configs[ck]
        selected.append({
            "params": info["params"],
            "seeds": info["seeds"],
            "mean_fom": info["mean_fom"],
            "label": config_label(info["params"]),
        })

    # Sort by mean FOM descending for consistent ordering
    selected.sort(key=lambda x: x["mean_fom"], reverse=True)

    print(f"\n  Selected {len(selected)} configs for comparison:")
    for i, s in enumerate(selected):
        print(f"    {i+1}. {s['label']}  (mean FOM1={s['mean_fom']:.4f}, {len(s['seeds'])} seeds)")

    return selected


def collect_run_dirs_for_config(sweep_dir, params, seeds):
    """Get all run directories matching a config across its seeds."""
    dirs = []
    for dirname in os.listdir(sweep_dir):
        parsed = parse_legacy_dir(dirname)
        if parsed is None:
            continue
        match = all(parsed.get(k) == v for k, v in params.items())
        if match and parsed["seed"] in seeds:
            dirs.append(os.path.join(sweep_dir, dirname))
    return sorted(dirs)


# ======================================================================
# Main computation
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute shared UMAP embedding for chemical space comparison"
    )
    parser.add_argument("--sweep_dir", type=str, default=None,
                        help="Sweep directory (auto-selects extreme configs)")
    parser.add_argument("--run_dirs", type=str, nargs="+", default=None,
                        help="Explicit run directories to include")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory for reference molecules")
    parser.add_argument("--output", type=str, default="chemical_space_data.npz",
                        help="Output .npz file path")
    parser.add_argument("--n_ref", type=int, default=5000,
                        help="Number of reference molecules")
    parser.add_argument("--max_configs", type=int, default=8,
                        help="Maximum number of configs to compare (sweep_dir mode)")
    parser.add_argument("--sample_per_seed", type=int, default=0,
                        help="Max molecules to sample per seed run (0=all)")
    args = parser.parse_args()

    if not args.sweep_dir and not args.run_dirs:
        print("ERROR: Provide either --sweep_dir or --run_dirs")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Determine configs and run directories
    # ------------------------------------------------------------------
    if args.sweep_dir:
        selected = select_extreme_configs(args.sweep_dir, max_configs=args.max_configs)
        if not selected:
            print("ERROR: No completed runs found")
            sys.exit(1)

        # Collect all run dirs per config
        config_runs = []
        for s in selected:
            dirs = collect_run_dirs_for_config(args.sweep_dir, s["params"], s["seeds"])
            config_runs.append({
                "label": s["label"],
                "params": s["params"],
                "dirs": dirs,
                "mean_fom": s["mean_fom"],
            })
    else:
        # Manual mode: each run_dir is its own "config"
        config_runs = []
        for rd in args.run_dirs:
            dirname = os.path.basename(rd)
            params = parse_legacy_dir(dirname)
            label = config_label(params) if params else dirname
            config_runs.append({
                "label": label,
                "params": params or {},
                "dirs": [rd],
                "mean_fom": 0,
            })

    # ------------------------------------------------------------------
    # 2. Generate reference set
    # ------------------------------------------------------------------
    if args.data_dir is None:
        for candidate in ["../data", "../../data"]:
            if os.path.isdir(candidate):
                args.data_dir = os.path.abspath(candidate)
                break

    if args.data_dir:
        print(f"\nGenerating reference molecules...")
        ref_smiles = generate_reference_molecules(args.data_dir, n_molecules=args.n_ref)
    else:
        print("WARNING: No data directory found, skipping reference set")
        ref_smiles = []

    # ------------------------------------------------------------------
    # 3. Load evaluated molecules per config
    # ------------------------------------------------------------------
    print(f"\nLoading evaluated molecules...")

    all_smiles = []
    all_source = []      # "reference" / config index (int)
    all_generation = []
    all_fitness = []

    # Reference molecules
    for smi in ref_smiles:
        all_smiles.append(smi)
        all_source.append(-1)  # -1 = reference
        all_generation.append(-1)
        all_fitness.append(0.0)

    config_labels = []
    for ci, cr in enumerate(config_runs):
        config_labels.append(cr["label"])
        n_loaded = 0
        for rd in cr["dirs"]:
            eval_path = os.path.join(rd, "all_evaluated_molecules.csv")
            if not os.path.exists(eval_path):
                print(f"  WARNING: {eval_path} not found, skipping")
                continue
            try:
                df = pd.read_csv(eval_path)
            except Exception as e:
                print(f"  WARNING: Could not read {eval_path}: {e}")
                continue

            if df.empty:
                continue

            # Sample if requested
            if args.sample_per_seed > 0 and len(df) > args.sample_per_seed:
                df = df.sample(n=args.sample_per_seed, random_state=42)

            smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in df.columns else "SMILES"
            fom_col = "FOM1_avg" if "FOM1_avg" in df.columns else "FitnessScore"

            for _, row in df.iterrows():
                smi = row.get(smiles_col, None)
                if not isinstance(smi, str):
                    continue
                all_smiles.append(smi)
                all_source.append(ci)
                all_generation.append(row.get("generation", 0))
                all_fitness.append(row.get(fom_col, 0.0))
                n_loaded += 1

        print(f"  Config {ci}: {cr['label']}  — {n_loaded} molecules from {len(cr['dirs'])} runs")

    print(f"\n  Total SMILES collected: {len(all_smiles)}")
    print(f"    Reference: {sum(1 for s in all_source if s == -1)}")
    for ci, lab in enumerate(config_labels):
        n = sum(1 for s in all_source if s == ci)
        print(f"    Config {ci} ({lab}): {n}")

    # ------------------------------------------------------------------
    # 4. Compute fingerprints
    # ------------------------------------------------------------------
    print(f"\nComputing fingerprints...")
    fps = []
    valid_indices = []
    total = len(all_smiles)

    for i, smi in enumerate(all_smiles):
        if i % 10000 == 0:
            print(f"\r  Fingerprinting: {i}/{total} ({100*i/total:.0f}%)", end="", flush=True)
        fp = smiles_to_fingerprint(smi)
        if fp is not None:
            fps.append(fp)
            valid_indices.append(i)
    print(f"\r  Fingerprinting: {total}/{total} (100%)      ")
    print(f"  Valid: {len(fps)} / {total}")

    source = np.array([all_source[i] for i in valid_indices], dtype=np.int32)
    generation = np.array([all_generation[i] for i in valid_indices], dtype=np.float32)
    fitness = np.array([all_fitness[i] for i in valid_indices], dtype=np.float32)
    smiles_valid = np.array([all_smiles[i] for i in valid_indices], dtype=object)

    fp_matrix = fps_to_numpy(fps)
    print(f"  Fingerprint matrix: {fp_matrix.shape}")

    # ------------------------------------------------------------------
    # 5. Fit UMAP on entire dataset
    # ------------------------------------------------------------------
    print(f"\nFitting UMAP on {fp_matrix.shape[0]} molecules (this may take a while)...")
    reducer = UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric="jaccard",
        random_state=42,
        n_jobs=1,
    )
    coords_2d = reducer.fit_transform(fp_matrix)
    print(f"  UMAP complete: {coords_2d.shape}")

    # ------------------------------------------------------------------
    # 6. Save everything
    # ------------------------------------------------------------------
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    np.savez_compressed(
        out_path,
        coords_2d=coords_2d,
        source=source,
        generation=generation,
        fitness=fitness,
        smiles=smiles_valid,
        config_labels=np.array(config_labels, dtype=object),
        config_params=np.array([str(cr["params"]) for cr in config_runs], dtype=object),
        config_mean_foms=np.array([cr["mean_fom"] for cr in config_runs], dtype=np.float32),
    )

    file_size = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nSaved: {out_path} ({file_size:.1f} MB)")
    print(f"  {len(config_labels)} configs, {len(fps)} molecules, 2D coords")
    print(f"\nNext step: python compare_chemical_space.py --input {out_path}")


if __name__ == "__main__":
    main()
