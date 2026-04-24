"""Lubricant base-oil fitness: weighted sum over normalised objectives.

Same six-run protocol as Egheosas's draft: one objective at 3x, others at 1x;
plus an even-weighted run. Our WSGA is otherwise unchanged — only the
fitness column the GA optimises is replaced.
"""

import numpy as np
import pandas as pd

from dvi import compute_dvi, estimate_density_100C


# Weights for (visc, tc, hc, dvi, tox, sc).
WEIGHT_PROFILES = {
    "visc": (3, 1, 1, 1, 1, 1),
    "tc":   (1, 3, 1, 1, 1, 1),
    "hc":   (1, 1, 3, 1, 1, 1),
    "dvi":  (1, 1, 1, 3, 1, 1),
    "tox":  (1, 1, 1, 1, 3, 1),
    "even": (1, 1, 1, 1, 1, 1),
}


def _minmax(series, invert=False):
    x = np.asarray(series, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.zeros_like(x)
    lo = np.nanmin(x[finite])
    hi = np.nanmax(x[finite])
    if hi - lo < 1e-12:
        out = np.full_like(x, 0.5)
    else:
        out = (x - lo) / (hi - lo)
    if invert:
        out = 1.0 - out
    out = np.where(np.isfinite(x), out, 0.0)
    return out


def add_dvi_and_lube_fitness(df: pd.DataFrame, weight_profile: str = "even") -> pd.DataFrame:
    """Add DVI and FOM_LUBE columns. Expects df to contain:
    viscosity_40C, viscosity_100C, density_40C, beta_40C, tc_40C, cpsat_40C,
    SCScore, Tox21_Score.
    """
    weights = WEIGHT_PROFILES[weight_profile]
    w_visc, w_tc, w_hc, w_dvi, w_tox, w_sc = weights

    df = df.copy()

    rho100 = estimate_density_100C(df["density_40C"].values, df["beta_40C"].values)
    df["density_100C_est"] = rho100
    df["DVI"] = compute_dvi(
        df["viscosity_40C"].values,
        df["viscosity_100C"].values,
        df["density_40C"].values,
        density_100C=rho100,
    )

    n_visc = _minmax(df["viscosity_40C"], invert=True)
    n_tc = _minmax(df["tc_40C"], invert=False)
    n_hc = _minmax(df["cpsat_40C"], invert=False)
    n_dvi = _minmax(df["DVI"], invert=False)
    n_tox = _minmax(df["Tox21_Score"], invert=True)
    n_sc = _minmax(df["SCScore"], invert=True)

    weighted = (
        w_visc * n_visc + w_tc * n_tc + w_hc * n_hc
        + w_dvi * n_dvi + w_tox * n_tox + w_sc * n_sc
    ) / float(sum(weights))

    df["FOM_LUBE"] = np.where(np.isfinite(weighted), weighted, 0.0)
    return df
