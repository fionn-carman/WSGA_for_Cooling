#!/usr/bin/env python
"""Plot niching penalty as a function of Tanimoto similarity for two tau values."""

import numpy as np
import matplotlib.pyplot as plt
import os

# ── Niching penalty formula (from wsga_helper.py) ──
# penalty = exp(-alpha * max(0, similarity - tau)^p)
# NichedFitness = Fitness * penalty * (1 - similarity)
# So total multiplier on fitness = penalty * (1 - similarity)

ALPHA = 1000
P = 2

def niching_multiplier(sim, tau):
    """Total fitness multiplier from niching: exp(-α·max(0, s-τ)²) · (1-s)."""
    penalty = np.exp(-ALPHA * np.maximum(0, sim - tau) ** P)
    return penalty * (1 - sim)


def make_plot(tau_values, highlight_idx, out_path, dpi=300):
    sim = np.linspace(0, 1, 1000)

    fig, ax = plt.subplots(figsize=(6, 4))

    colours = ["#E74C3C", "#3498DB"]  # red, blue
    labels = [f"τ = {t}" for t in tau_values]

    for i, tau in enumerate(tau_values):
        mult = niching_multiplier(sim, tau)
        if i == highlight_idx:
            ax.plot(sim, mult, color=colours[i], lw=2.5, label=labels[i], zorder=3)
        else:
            ax.plot(sim, mult, color=colours[i], lw=1.2, alpha=0.35,
                    label=labels[i], zorder=2)

    # Mark tau thresholds with vertical dashed lines
    for i, tau in enumerate(tau_values):
        if i == highlight_idx:
            ax.axvline(tau, color=colours[i], ls="--", lw=1.0, alpha=0.6)

    ax.set_xlabel("Average Tanimoto Similarity", fontsize=12)
    ax.set_ylabel("Fitness Multiplier", fontsize=12)
    ax.legend(fontsize=11, frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "niching_penalty")
    os.makedirs(out_dir, exist_ok=True)

    tau_values = [0.05, 0.20]

    # Version 1: highlight tau=0.05
    make_plot(tau_values, highlight_idx=0,
              out_path=os.path.join(out_dir, "niching_penalty_highlight_tau005.png"))

    # Version 2: highlight tau=0.20
    make_plot(tau_values, highlight_idx=1,
              out_path=os.path.join(out_dir, "niching_penalty_highlight_tau020.png"))
