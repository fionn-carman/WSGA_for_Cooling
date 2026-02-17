# Scripts Reference

## PBS Submission Scripts (HPC)

### `hyperparam_sweep.sh`
Stage 1 GA mechanics sweep. Sweeps population size, mutation rate, elitism rate, and tournament k with tau fixed at 0.15. 640 PBS array jobs (5 pop x 2 mr x 4 er x 4 k x 4 seeds), 150 generations each.

```bash
qsub hyperparam_sweep.sh          # submit all 640 jobs
qsub -J 0-0 hyperparam_sweep.sh   # single test job
bash hyperparam_sweep.sh 0         # local test (index 0)
```

### `tau_sweep.sh`
Stage 2 niching sweep. Uses the best GA parameters from Stage 1 (must be updated manually) and sweeps tau across 8 levels with 5 seeds. 40 PBS array jobs.

**Important:** Update the "Best GA parameters" section with Stage 1 results before submitting.

```bash
qsub tau_sweep.sh
bash tau_sweep.sh 0
```

### `fp_mp_sweep.sh`
Flash point / melting point constraint sweep. Sweeps 13 flash point thresholds x 15 melting point thresholds x 2 biodegradability options. 390 PBS array jobs.

```bash
qsub fp_mp_sweep.sh
```

### `chemical_space_sweep.sh`
Single PBS job (not an array) that runs the two-step chemical space comparison pipeline. Auto-selects 8 extreme hyperparameter configs from the sweep, generates 5000 reference molecules, fits a shared UMAP embedding on all molecules, then generates comparison figures. Requires 96 GB memory.

```bash
qsub chemical_space_sweep.sh
bash chemical_space_sweep.sh       # local run
```

---

## Analysis Scripts (Python)

### `analyse_hyperparam_sweep.py`
Parses completed runs from `hyperparam_sweep.sh`, ranks configs by mean FOM1, and generates figures: pairwise heatmaps, marginal box plots, convergence curves, parallel coordinates, Tanimoto similarity heatmap of top molecules, and Murcko scaffold distribution.

```bash
python analyse_hyperparam_sweep.py --sweep_dir ../outputs/hyperparam_sweep
```

### `analyse_tau_sweep.py`
Parses completed runs from `tau_sweep.sh` and generates: FOM1 vs tau curve, convergence curves per tau, structural similarity heatmaps, and scaffold diversity vs tau.

```bash
python analyse_tau_sweep.py --sweep_dir ../outputs/tau_sweep
```

### `analyse_chemical_space.py`
Single-run or multi-run chemical space analysis. Plots UMAP of evaluated molecules against a random reference set, coloured by generation and fitness. Also shows functional group frequencies and exploration trajectories.

```bash
python analyse_chemical_space.py --run_dir ../outputs/hyperparam_sweep/pop2000_mr0.8_er0.3_k3_seed42
python analyse_chemical_space.py --sweep_dir ../outputs/hyperparam_sweep --n_runs 5
```

### `compute_chemical_space.py`
Step 1 of the cross-config comparison pipeline. Auto-selects extreme configs from the sweep, pools all seeds per config, generates a shared reference set via the n-gram model, computes Morgan fingerprints, fits a single UMAP on the combined data, and saves everything to a `.npz` file. Called by `chemical_space_sweep.sh`.

```bash
python compute_chemical_space.py --sweep_dir ../outputs/hyperparam_sweep --data_dir ../data --output out.npz
```

### `compare_chemical_space.py`
Step 2 of the cross-config comparison pipeline. Loads the `.npz` from `compute_chemical_space.py` and generates: grid comparison panels, density contour overlay, coverage bar charts, search trajectories, functional group heatmap, reference FG panels, reference dominant FG map, and Murcko scaffold comparison.

```bash
python compare_chemical_space.py --input ../outputs/analysis/chemical_space_comparison/chemical_space_data.npz
```

### `plot_local_fg_umap.py`
Standalone local script (no HPC required). Generates 5000 molecules from the 8-gram model, runs UMAP, and produces two figures: a single plot coloured by dominant functional group, and a multi-panel plot with one panel per FG coloured by count.

```bash
python plot_local_fg_umap.py
python plot_local_fg_umap.py --n_mol 10000 --out_dir ../outputs/custom
```

---

## Pipeline Summary

```
Stage 1: hyperparam_sweep.sh  -->  analyse_hyperparam_sweep.py
Stage 2: tau_sweep.sh         -->  analyse_tau_sweep.py
Chem space: chemical_space_sweep.sh
              |-> compute_chemical_space.py (Step 1: UMAP fitting)
              |-> compare_chemical_space.py (Step 2: figures)
Local FG vis: plot_local_fg_umap.py (standalone)
```
