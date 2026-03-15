# WSGA: Weighted Sum Genetic Algorithm for Cooling Fluid Discovery

A genetic algorithm for de novo molecular design, optimised for discovering cooling fluids with superior thermophysical properties. Molecules are evolved from SMILES representations using domain-specific mutation operators, with fitness evaluated across 12 XGBoost property models and multi-constraint validity filtering.

## Overview

WSGA searches chemical space for molecules that maximise a figure of merit (FOM1) for single-phase liquid cooling performance. FOM1 combines thermal conductivity, volumetric heat capacity, thermal expansion, and kinematic viscosity into a single scalar objective derived from natural convection correlations.

The algorithm incorporates:
- **Hybrid elite selection** balancing exploitation (raw fitness) and exploration (niched fitness via Tanimoto similarity)
- **Adaptive mutation rate** that increases during stagnation to escape local optima
- **Stagnation detection with restart** replacing a portion of the population with fresh n-gram-generated molecules
- **Soft melting point penalty** providing a continuous fitness gradient rather than a hard threshold cutoff
- **Biodegradability and toxicity filtering** via PaccMann MCA Tox21 predictor ([GT4SD](https://github.com/PaccMann/chemical_representation_learning_for_toxicity_prediction)) and RB-QSAR biodegradability classifier

Property predictions are made by XGBoost models trained on curated experimental datasets with RDKit Morgan fingerprint descriptors and RFE feature selection.

## Repository Structure

```
wsga_clean/
├── src/                        # Core algorithm
│   ├── wsga.py                 # Main GA engine (entry point)
│   ├── wsga_helper.py          # Fitness evaluation, niching, elite selection
│   ├── mutations.py            # 10 molecular mutation operators
│   ├── evaluation.py           # Property prediction and SCScore caching
│   ├── generate_molecules.py   # N-gram SMILES model for initial populations
│   ├── fragment_utils.py       # Fragment-based crossover
│   ├── SCScorer.py             # Synthetic complexity scoring
│   ├── tox21_gt4sd.py          # PaccMann MCA Tox21 toxicity predictor
│   └── descriptors.py          # RDKit descriptor computation
│
├── scripts/                    # HPC submission and analysis
│   ├── hyperparam_sweep.sh         # Stage 1: GA parameter sweep (640 jobs)
│   ├── hyperparam_sweep_stage2.sh  # Stage 2: Full grid with best_elite_ratio (960 jobs)
│   ├── tau_sweep.sh                # Niching parameter sweep (40 jobs)
│   ├── fp_mp_sweep.sh              # Constraint threshold sweep (390 jobs)
│   ├── mahal_tau_comparison.sh     # Mahalanobis OOD detection
│   ├── mahal_tau_plot.sh           # OOD visualisation pipeline
│   ├── chemical_space_sweep.sh     # UMAP chemical space analysis
│   ├── tau_chem_space_sweep.sh     # Tau sweep chemical space analysis
│   ├── analyse_hyperparam_sweep.py # Sweep ranking and visualisation
│   ├── analyse_tau_sweep.py        # Tau sweep analysis
│   ├── plot_chemical_space.py      # Publication figures: UMAP, FG distributions
│   ├── plot_tau_chemical_space.py  # 2x4 tau comparison grid
│   ├── compute_mahalanobis.py      # Per-model OOD distance computation
│   ├── generate_reference_set.py   # N-gram reference population (10k molecules)
│   └── SCRIPTS.md                  # Detailed script documentation
│
├── models/                     # Pre-trained XGBoost property models
│   ├── Thermal_Conductivity_{40,100}C/
│   ├── Heat_Capacity_Constant_Pressure_{40,100}C_J_K_Mol/
│   ├── Density_{40,100}C_g_cm^3/
│   ├── Kinematic_Viscosity_{40,100}C/
│   ├── BP-Measured/
│   ├── MP-Measured/
│   ├── DC_exp/
│   ├── flashpoint/
│   ├── biodegradability/
│   ├── tox21_gt4sd/            # PaccMann MCA Tox21 model (download separately)
│   └── SCScorer/               # Synthetic complexity model
│
├── training/                   # Model training scripts
│   ├── train_regression.py     # XGBoost regression with RFE
│   ├── train_classification.py # Biodegradability classifier
│   └── data/                   # Curated training datasets
│
├── data/                       # Cleaned experimental datasets
│   ├── processed_full_hydrocarbon_dataset.csv
│   ├── flashpoint_cleaned.csv
│   ├── MP-Measured_cleaned.csv
│   ├── BP-Measured_cleaned.csv
│   ├── DC_exp_cleaned.csv
│   └── biodegradability_cleaned.csv
│
└── outputs/                    # Generated results (gitignored)
```

## Algorithm

### Fitness Evaluation

Each candidate molecule is converted to Morgan fingerprints and evaluated across 12 XGBoost regression models predicting thermophysical properties at 40 and 100 degrees C: thermal conductivity, heat capacity, density, kinematic viscosity, boiling point, melting point, dielectric constant, and flash point.

These predictions are combined into FOM1:

```
FOM1 = k * (beta * Cp * rho / (nu * k))^0.2813
```

where k is thermal conductivity, beta is volumetric thermal expansion coefficient, Cp is molar heat capacity, rho is density, and nu is kinematic viscosity. The exponent 0.2813 derives from natural convection heat transfer correlations for laminar flow over a vertical plate.

### Validity Constraints

Molecules must satisfy safety, practicality, and synthesisability thresholds:

| Constraint | Default | Description |
|------------|---------|-------------|
| Melting point | < -10 degC (hard), soft penalty from -30 degC | Liquid at low temperatures |
| Boiling point | > 70 degC | Sufficient operating range |
| Flash point | > 373 K | Fire safety |
| Dielectric constant | < 8 | Electrical insulation |
| SCScore | < 3 | Synthetic accessibility |
| Tox21 | < 3 | Low predicted toxicity |
| Biodegradable | Yes | Environmental persistence |

### Niching

Diversity is maintained through Tanimoto-similarity-based niching. Molecules in crowded regions of chemical space receive a fitness penalty:

```
NichedFitness = Fitness * exp(-alpha * max(0, AvgSimilarity - tau)^2) * (1 - AvgSimilarity)
```

The niching threshold tau controls the trade-off between convergence and diversity.

### Elite Selection

Each generation, elites are selected via a hybrid strategy controlled by `best_elite_ratio`:
- A fraction selected by raw FitnessScore (exploitation: preserves the best performers)
- The remainder selected by NichedFitnessScore (exploration: preserves diverse high-quality molecules)

### Mutation Operators

Ten domain-specific mutation operators modify molecular graphs:

| Operator | Description |
|----------|-------------|
| AddAtom | Attach C or O atom to a random position |
| RemoveAtom | Remove a terminal atom |
| ReplaceAtom | Swap C for O or vice versa |
| ReplaceBond | Change bond order at a random bond |
| AddFragment | Attach a small fragment from the training set |
| RemoveFragment | Remove a terminal fragment |
| InsertAromatic | Insert a benzene ring |
| Napthalenate | Fuse a naphthalene ring onto an aromatic system |
| Glycolate | Attach an ethylene glycol unit |
| Esterify | Form an ester linkage |

Mutation probabilities are weighted by molecular size: smaller molecules receive more additive mutations, larger molecules receive more subtractive mutations.

### Initial Population

The initial population is generated using a character-level 8-gram model trained on SMILES strings from all available experimental datasets. This provides chemically plausible starting molecules with functional group distributions similar to the training data while maintaining approximately 50% structural novelty.

## Hyperparameter Sweeps

The optimisation pipeline runs in stages on HPC (PBS Pro):

**Stage 1** (`hyperparam_sweep.sh`): Sweeps population size, mutation rate, elitism rate, and tournament size (640 configurations x 3 seeds). Identifies which GA mechanics parameters matter most.

**Stage 2** (`hyperparam_sweep_stage2.sh`): Full factorial grid over population size, elitism rate, best elite ratio, and tournament size (960 configurations). Mutation rate is fixed since adaptive mutation overrides the base rate during stagnation.

**Tau sweep** (`tau_sweep.sh`): Sweeps the niching threshold across 8 values using the best GA parameters from Stage 1.

**Constraint sweep** (`fp_mp_sweep.sh`): Explores the flash point and melting point constraint trade-off space.

Analysis scripts parse completed sweep results and produce publication-quality figures: pairwise heatmaps, marginal distributions, convergence curves, parallel coordinates, and chemical space UMAP projections.

## Out-of-Distribution Detection

Mahalanobis distance is computed per property model to flag molecules whose descriptor profiles fall outside the training data distribution. The pipeline uses Ledoit-Wolf covariance estimation with correlation-based feature filtering to handle high-dimensional fingerprint spaces. Results are visualised as OOD heatmaps overlaid with molecular structures.

## Usage

### Single Run

```bash
cd src/
python wsga.py \
    --target FOM1 \
    --population_size 2000 \
    --num_generations 150 \
    --Tau 0.15 \
    --output_dir ../outputs/my_run
```

### HPC Sweep

```bash
cd scripts/
qsub hyperparam_sweep_stage2.sh          # submit all 960 jobs
qsub -J 0-0 hyperparam_sweep_stage2.sh   # single test job
```

### Analysis

```bash
cd scripts/
python analyse_hyperparam_sweep.py --sweep_dir ../outputs/hyperparam_sweep_stage2 --format stage2
```

## Dependencies

- Python 3.10
- RDKit
- XGBoost
- scikit-learn
- PyTorch
- NumPy, Pandas, Matplotlib, Seaborn
- UMAP-learn (for chemical space analysis)
- pytoda, paccmann_predictor, toxsmi (for Tox21 toxicity prediction)
- tqdm

Install via conda:
```bash
conda env create -f environment.yml
conda activate mol-rl
# paccmann_generator must be installed separately (pytoda version conflict):
pip install --no-deps "paccmann_generator @ git+https://github.com/PaccMann/paccmann_generator.git"
```

### Downloading Tox21 Model Weights

The pretrained PaccMann MCA Tox21 weights are not tracked in git. Download them from the GT4SD model hub:

```python
from minio import Minio
import os

client = Minio(
    's3.par01.cloud-object-storage.appdomain.cloud',
    access_key='b087e6810a5d4246a64e07e36ace338f',
    secret_key='ba4a1db5647a32c6109b58714befb7ea7145b983143e0836',
)
bucket = 'gt4sd-cos-properties-artifacts'
prefix = 'molecules/MCA/Tox21/v0/'
dest = 'models/tox21_gt4sd'
for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
    rel = obj.object_name[len(prefix):]
    if not rel:
        continue
    local_path = os.path.join(dest, rel)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    client.fget_object(bucket, obj.object_name, local_path)
```

## Property Models

XGBoost models are trained on curated experimental datasets using RDKit Morgan fingerprints (radius 2, 2048 bits) with recursive feature elimination. Each model directory contains the trained model, selected feature indices, and diagnostic plots (parity plots, SHAP feature importance).

To retrain property models:
```bash
cd training/
bash train_all_regression.sh      # all 12 regression models
bash train_all_classification.sh  # biodegradability classifier
```

The Tox21 toxicity model is a pretrained PaccMann MCA neural network from the [GT4SD model hub](https://github.com/PaccMann/chemical_representation_learning_for_toxicity_prediction) and does not need retraining.
