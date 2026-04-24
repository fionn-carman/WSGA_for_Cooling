# LubeOil — Lubricant base-oil design

Same WSGA and REINVENT as the cooling-fluid paper, with one thing changed: the
fitness function. No other behavioural differences — mutation set, crossover,
initial population (GRU prior), niching, hybrid elite selection, adaptive
mutation boost on stagnation, heavy-atom limits are all as in `src/`.

## What's new

- `src/dvi.py` — Kajita et al. 2020 Dynamic Viscosity Index (eq. 6).
- `src/lube_fitness.py` — `FOM_LUBE`: weighted sum over six normalised
  objectives (viscosity low, thermal conductivity high, heat capacity high,
  DVI high, Tox21 low, SCScore low). Six weight profiles (one objective at 3×,
  others at 1×; plus `even`).
- `src/lube_wsga.py` — verbatim copy of `src/wsga.py` with the fitness column
  swapped for `FOM_LUBE`.
- `src/lube_wsga_helper.py` — copy of `src/wsga_helper.py`. Only tweak: the
  `dc` threshold in `assign_validity` tolerates a missing `dc` column since
  dielectric constant is not a lubricant property.
- `src/reinvent/lube_reward.py`, `src/reinvent/run_lube_reinvent.py` — REINVENT
  counterparts that call the same lubricant fitness.

## Model dependency (already trained locally)

- `models/viscosity_100C/` — XGBoost for DVI. Trained via
  `LubeOil/scripts/train_viscosity_100C.py`, reusing the 40 °C feature basis
  and hyperparameters. OOF 5-fold R² = 0.892 on 8,082 NIST molecules.
  `*.joblib` is git-ignored; SCP to HPC when running there.

## Running

```bash
# Baseline comparison against PAO 4, C16, squalane, DOS, DIDA
python LubeOil/scripts/baseline_eval.py

# WSGA (production HPs — 3000 pop, 150 gens, mut=0.3, elite=0.25, k=5, tau=0.25)
bash LubeOil/scripts/run_lube_wsga.sh even       # local, single profile
qsub -J 0-5 LubeOil/scripts/run_lube_wsga.sh     # HPC, six profiles in parallel

# REINVENT (production HPs — lr=5e-5, sigma=1.0, batch=128)
bash LubeOil/scripts/run_lube_reinvent.sh even
qsub -J 0-5 LubeOil/scripts/run_lube_reinvent.sh

# Post-hoc biodegradability filter
python LubeOil/scripts/postfilter_biodeg.py --input <path>/all_evaluated_molecules.csv
```

## Fitness details

`FOM_LUBE` is a min–max-normalised weighted sum over the batch:

```
F = (w_v · n_visc + w_t · n_tc + w_h · n_hc + w_d · n_dvi + w_x · n_tox + w_s · n_sc) / Σw
```

`n_visc`, `n_tox`, `n_sc` are inverted (lower raw value → higher normalised
score). DVI is computed from predicted viscosity at 40 and 100 °C plus
density; ρ₁₀₀ is estimated from ρ₄₀ with the trained β model.

## Note on Egheosas's draft

His DVI formula has two transcription errors — the `+1.2` belongs *outside*
`log10`, and the denominator is `(135+40)/(135+100) = 175/235`, not `27/35`.
`dvi.py` follows the Kajita paper verbatim; `compute_dvi` gives PAO 4 ≈ 132
(lit ~130) as a sanity check.
