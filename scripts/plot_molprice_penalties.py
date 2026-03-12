#!/usr/bin/env python
"""
Plot the MolPrice cosine penalty curves used in the molprice sweep.
"""

import numpy as np
import matplotlib.pyplot as plt

# Sweep configurations: (label, soft_threshold, hard_threshold)
CONFIGS = [
    ("Gentle",     3.5, 6.0),
    ("Moderate",   3.0, 5.0),
    ("Firm",       2.5, 4.5),
    ("Tight",      2.5, 4.0),
    ("Aggressive", 2.0, 3.5),
]

# Match the viridis-like palette from the boxplot
COLORS = ["#443983", "#31688E", "#21918C", "#90D743", "#FDE725"]


def cosine_penalty(x, soft, hard):
    """Cosine S-curve penalty: 1 below soft, 0 above hard."""
    p = np.ones_like(x)
    mask = (x > soft) & (x < hard)
    t = (x[mask] - soft) / (hard - soft)
    p[mask] = (1 + np.cos(np.pi * t)) / 2
    p[x >= hard] = 0.0
    return p


def main():
    x = np.linspace(0, 8, 1000)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for (label, soft, hard), color in zip(CONFIGS, COLORS):
        y = cosine_penalty(x, soft, hard)
        ax.plot(x, y, linewidth=2.2, color=color,
                label=f"{label} (soft={soft}, hard={hard})")

    ax.set_xlabel("MolPrice — ln(USD / mmol)", fontsize=12)
    ax.set_ylabel("Penalty factor", fontsize=12)
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9, loc="upper right")
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = "../outputs/molprice_reference/molprice_penalty_curves.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
