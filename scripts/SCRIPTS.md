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
Single PBS job that runs the chemical space analysis pipeline. Generates a 10k n-gram reference set (if not cached), identifies the top hyperparameter config from `configs_aggregated.csv`, loads all seeds for that config, fits a shared UMAP, and produces publication-quality figures. Requires 64 GB memory.

**Prerequisite:** `analyse_hyperparam_sweep.py` must have been run first.

```bash
qsub chemical_space_sweep.sh
bash chemical_space_sweep.sh       # local run
```

### `tau_chem_space_sweep.sh`
Chemical space analysis for the tau sweep. Runs `analyse_tau_sweep.py`, generates the n-gram reference set (if not cached), then fits a single shared UMAP across all 8 tau values and produces a 2x4 coverage grid plus FG comparison figure. Requires 64 GB memory, ~6 hours.

**Prerequisite:** `tau_sweep.sh` must have completed (all 40 jobs).

```bash
qsub tau_chem_space_sweep.sh
bash tau_chem_space_sweep.sh       # local run
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

### `generate_reference_set.py`
Standalone script to generate 10k molecules from the 8-gram model and save to `data/ngram_reference_10k.csv`. Uses fixed seed 42 for reproducibility. Skips generation if the output file already exists.

```bash
python generate_reference_set.py                  # generate with defaults
python generate_reference_set.py --force           # regenerate even if exists
python generate_reference_set.py --n_mol 5000      # custom count
```

### `plot_chemical_space.py`
Main chemical space visualisation script. Identifies the top config from `configs_aggregated.csv`, loads all seed runs, computes a shared UMAP embedding with the n-gram reference set, and produces figures:
- (a) Reference population coloured by dominant functional group (standalone)
- (b) GA coverage overlay on reference
- (c) Exploration over time (generation colourmap)
- (d) FOM1 fitness landscape
- 1x3 combined grid of (b)-(d)
- FG prevalence comparison: reference vs GA
- FG prevalence vs generation line chart

Supports `--npz` mode for fast replotting from cached UMAP.

```bash
python plot_chemical_space.py --sweep_dir ../outputs/hyperparam_sweep
python plot_chemical_space.py --npz path/to/umap_cache.npz --out_dir ./figs
```

### `plot_tau_chemical_space.py`
Tau sweep chemical space visualisation. Fits a single shared UMAP across all tau values and the n-gram reference set, then produces a 2x4 grid figure (one panel per tau) and a FG comparison bar chart. Supports `--npz` mode for replotting from cached UMAP.

```bash
python plot_tau_chemical_space.py --sweep_dir ../outputs/tau_sweep
python plot_tau_chemical_space.py --npz path/to/tau_umap_cache.npz --out_dir ./figs
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
                                   -->  tau_chem_space_sweep.sh
                                          |-> analyse_tau_sweep.py
                                          |-> generate_reference_set.py
                                          |-> plot_tau_chemical_space.py (shared UMAP + 2x4 grid)
Chem space: chemical_space_sweep.sh
              |-> generate_reference_set.py  (Step 1: 10k reference set)
              |-> plot_chemical_space.py      (Step 2: UMAP + 6 figures)
Local FG vis: plot_local_fg_umap.py (standalone)
```
