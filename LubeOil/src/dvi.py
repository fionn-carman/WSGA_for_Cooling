"""Dynamic Viscosity Index (Kajita et al. 2020, Commun. Phys. 3, 77, eq. 6).

DVI = 220 - 7 * 10^S
S   = -log10( (log10(eta40) + 1.2) / (log10(eta100) + 1.2) )
      / log10( (135 + 40) / (135 + 100) )

eta is dynamic viscosity in cP = kinematic viscosity (cSt) * density (g/cm^3).

Density at 100C is estimated from density at 40C using volumetric thermal
expansion coefficient beta:  rho_100 ~= rho_40 / (1 + beta * dT),  dT = 60 K.
"""

import numpy as np


S_DENOM = np.log10((135.0 + 40.0) / (135.0 + 100.0))


def estimate_density_100C(density_40C, beta_40C, dT=60.0):
    """Linear extrapolation of density from 40C to 100C using beta."""
    return density_40C / (1.0 + beta_40C * dT)


def compute_dvi(nu40_cSt, nu100_cSt, density_40C, density_100C=None, beta_40C=None):
    """Kajita Dynamic Viscosity Index.

    Args:
        nu40_cSt, nu100_cSt: kinematic viscosity at 40 and 100C, cSt (mm^2/s)
        density_40C: g/cm^3 at 40C
        density_100C: g/cm^3 at 100C; if None, estimated from beta_40C
        beta_40C: thermal expansion coefficient (K^-1); used only when
            density_100C is None

    Returns:
        DVI (array-like, same shape as inputs). Values outside the valid
        Walther regime (nu <= 0, or log10(nu+1.2) <= 0) are returned as NaN.
    """
    nu40 = np.asarray(nu40_cSt, dtype=float)
    nu100 = np.asarray(nu100_cSt, dtype=float)
    rho40 = np.asarray(density_40C, dtype=float)

    if density_100C is None:
        if beta_40C is None:
            raise ValueError("density_100C or beta_40C must be provided")
        rho100 = estimate_density_100C(rho40, np.asarray(beta_40C, dtype=float))
    else:
        rho100 = np.asarray(density_100C, dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        eta40 = nu40 * rho40
        eta100 = nu100 * rho100
        arg40 = np.log10(eta40) + 1.2
        arg100 = np.log10(eta100) + 1.2
        S = -np.log10(arg40 / arg100) / S_DENOM
        dvi = 220.0 - 7.0 * np.power(10.0, S)

    bad = (
        ~np.isfinite(dvi)
        | (eta40 <= 0)
        | (eta100 <= 0)
        | (arg40 <= 0)
        | (arg100 <= 0)
    )
    dvi = np.where(bad, np.nan, dvi)
    return dvi
