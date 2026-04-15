"""SI figure: calibration of the Tanimoto similarity grouping used by the
REINVENT diversity filter.

Three panels:
  (a) Number of groups vs threshold on NIST 8100 CHO corpus
  (b) UMAP projection coloured by group membership at tau_sim = 0.45
  (c) Within-group vs between-group pairwise similarity histograms
"""

import glob
import os
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
import umap

RDLogger.DisableLog("rdApp.*")

# Match the manuscript figure style guide
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 7.5,
    "axes.labelsize": 8.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})


def fp_to_vec(fp):
    arr = np.zeros((fp.GetNumBits(),), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def sphere_cluster(fps, threshold, record_sims=False):
    """Greedy Tanimoto grouping. Returns (group_ids, n_groups) or
    (group_ids, n_groups, join_sims, rep_mask) if record_sims=True.

    join_sims[i] is the Tanimoto of molecule i to the representative of the
    group it joined (0.0 if molecule i started its own group).
    rep_mask[i] is True if molecule i became a new representative.
    """
    reps = []
    raw_ids = [-1] * len(fps)
    join_sims = [0.0] * len(fps)
    rep_mask = [False] * len(fps)
    for i, fp in enumerate(fps):
        if reps:
            sims = DataStructs.BulkTanimotoSimilarity(fp, reps)
            best_rep = int(np.argmax(sims))
            best_sim = sims[best_rep]
        else:
            best_sim, best_rep = 0.0, -1
        if best_sim >= threshold and best_rep >= 0:
            raw_ids[i] = best_rep
            join_sims[i] = float(best_sim)
        else:
            reps.append(fp)
            raw_ids[i] = len(reps) - 1
            rep_mask[i] = True
    counts = Counter(raw_ids)
    ordered = [cid for cid, _ in counts.most_common()]
    remap = {old: new for new, old in enumerate(ordered)}
    group_ids = [remap[g] for g in raw_ids]
    if record_sims:
        return group_ids, len(ordered), join_sims, rep_mask
    return group_ids, len(ordered)


def load_nist_corpus(training_dir):
    all_smi = set()
    for f in glob.glob(os.path.join(training_dir, "nist_8100", "*_cho_cleaned.csv")):
        df = pd.read_csv(f, usecols=["SMILES"])
        all_smi.update(df["SMILES"].dropna())
    return sorted(all_smi)


def main():
    wsga_root = os.path.expanduser("~/Desktop/WSGA_for_Cooling")
    training_dir = os.path.join(wsga_root, "training/data")
    out_dir = os.path.join(wsga_root,
                           "../ML-for-Immersion-Cooling/Figures/NichingSweep")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading NIST 8100 CHO corpus...")
    smiles = load_nist_corpus(training_dir)
    print(f"  {len(smiles)} unique SMILES")

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    mols = [m for m in mols if m is not None]
    print(f"  {len(mols)} parseable")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    print("Computing Morgan radius-2 fingerprints...")
    fps = [gen.GetFingerprint(m) for m in mols]
    fp_matrix = np.array([fp_to_vec(fp) for fp in fps])

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    print("Computing group counts at each threshold...")
    curve = []
    for t in thresholds:
        _, n_g = sphere_cluster(fps, t)
        curve.append((t, n_g))
        print(f"  tau={t:.2f}: {n_g} groups")

    TAU = 0.35
    print(f"Sphere grouping at tau = {TAU} for panels (b) and (c)...")
    group_ids_45, n_groups_45, join_sims_45, rep_mask_45 = sphere_cluster(
        fps, TAU, record_sims=True)

    print("Running UMAP (Jaccard metric)...")
    reducer = umap.UMAP(n_neighbors=25, min_dist=0.12,
                        metric="jaccard", random_state=42)
    coords = reducer.fit_transform(fp_matrix)

    # Panel (c): similarity of each non-representative molecule to the
    # representative of the group it joined + a random sample of pairwise
    # similarities to any non-matched representative (showing what "below
    # threshold" pairs look like).
    print("Collecting within-group and between-group similarities...")
    within_sims = [s for s, is_rep in zip(join_sims_45, rep_mask_45)
                   if not is_rep]
    print(f"  within-group (non-representative → rep): {len(within_sims)}, "
          f"median {np.median(within_sims) if within_sims else 0:.2f}")

    # Between-group: each non-rep molecule's similarity to its NEAREST
    # non-assigned representative (strictly below threshold by construction).
    rng = np.random.default_rng(0)
    rep_indices = [i for i, is_rep in enumerate(rep_mask_45) if is_rep]
    rep_fps_arr = [fps[i] for i in rep_indices]
    id_arr = np.array(group_ids_45)
    between_sims = []
    # Sample 3000 non-rep molecules; for each, find similarity to a random
    # representative that is NOT its assigned one.
    non_rep_indices = [i for i, is_rep in enumerate(rep_mask_45) if not is_rep]
    sample_size = min(3000, len(non_rep_indices))
    for i in rng.choice(non_rep_indices, size=sample_size, replace=False):
        my_group = id_arr[i]
        # Pick another representative
        other_reps = [k for k in range(len(rep_indices))
                      if id_arr[rep_indices[k]] != my_group]
        if not other_reps:
            continue
        k = rng.choice(other_reps)
        s = DataStructs.TanimotoSimilarity(fps[i], rep_fps_arr[k])
        between_sims.append(s)
    print(f"  between-group (non-rep → other rep): {len(between_sims)}, "
          f"median {np.median(between_sims) if between_sims else 0:.2f}")

    # ── figure ──
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(10.5, 3.2), dpi=150)

    # (a) number of groups vs threshold
    xs = [t for t, _ in curve]
    ys = [n for _, n in curve]
    ax_a.plot(xs, ys, "o-", color="#4c72b0", markersize=4, linewidth=1.0)
    ax_a.set_xlabel(r"Similarity threshold $\tau_{\mathrm{sim}}$")
    ax_a.set_ylabel("Number of groups")
    ax_a.text(-0.18, 1.05, "(a)", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold", va="bottom")

    # (b) UMAP coloured by group — top 20 groups shown.
    # First 10 groups: filled circles (tab10 palette).
    # Next 10 groups:  unfilled circles (same colours, ring only).
    top_k = 20
    counts = Counter(group_ids_45)
    top_groups = [g for g, _ in counts.most_common(top_k)]
    top_set = set(top_groups)
    tab10 = plt.cm.tab10(np.linspace(0, 1, 10))

    mask_grey = np.array([g not in top_set for g in group_ids_45])
    ax_b.scatter(coords[mask_grey, 0], coords[mask_grey, 1],
                 s=2.5, c="lightgrey", alpha=0.30,
                 edgecolor="none", rasterized=True)

    for rank, gid in enumerate(top_groups):
        mask = np.array([g == gid for g in group_ids_45])
        color = tab10[rank % 10]
        if rank < 10:
            # filled markers
            ax_b.scatter(coords[mask, 0], coords[mask, 1],
                         s=10, color=color,
                         alpha=0.9, edgecolor="none", rasterized=True)
        else:
            # unfilled markers (ring only)
            ax_b.scatter(coords[mask, 0], coords[mask, 1],
                         s=14, facecolors="none", edgecolors=color,
                         alpha=0.9, linewidths=0.7, rasterized=True)

    ax_b.set_xticks([])
    ax_b.set_yticks([])
    ax_b.set_xlabel("UMAP 1")
    ax_b.set_ylabel("UMAP 2")
    ax_b.text(-0.18, 1.05, "(b)", transform=ax_b.transAxes,
              fontsize=10, fontweight="bold", va="bottom")
    ax_b.text(0.98, 0.02,
              f"{n_groups_45} groups\n(top 20 shown)",
              transform=ax_b.transAxes, fontsize=7, color="grey",
              ha="right", va="bottom")

    # (c) within vs between histograms
    bins = np.linspace(0, 1, 41)
    ax_c.hist(between_sims, bins=bins, alpha=0.7, color="#c44e52",
              label="To a different group",
              edgecolor="none", density=True)
    ax_c.hist(within_sims, bins=bins, alpha=0.8, color="#4c72b0",
              label="To own group's representative",
              edgecolor="none", density=True)
    ax_c.axvline(TAU, color="grey", linestyle="--", linewidth=0.7)
    ax_c.set_xlabel("Tanimoto similarity")
    ax_c.set_ylabel("Density")
    ax_c.set_xlim(0, 1)
    ax_c.legend(frameon=False, fontsize=6.5, loc="upper right",
                handlelength=1.2, handletextpad=0.4)
    ax_c.text(-0.18, 1.05, "(c)", transform=ax_c.transAxes,
              fontsize=10, fontweight="bold", va="bottom")

    fig.tight_layout()
    out_pdf = os.path.join(out_dir, "sphere_threshold_si.pdf")
    out_png = os.path.join(out_dir, "sphere_threshold_si.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
