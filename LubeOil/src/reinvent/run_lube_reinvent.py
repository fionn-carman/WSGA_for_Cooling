"""REINVENT entry point for de novo cooling fluid design.

Matches the wsga.py interface so the same PBS sweep scripts can drive both
methods.  Output CSVs are format-compatible with the WSGA analysis pipeline.

Usage (from src/):
    python reinvent/run_reinvent.py --target FOM1 --output_dir ../outputs/reinvent_test

Or from repo root:
    python src/reinvent/run_reinvent.py --target FOM1 --output_dir outputs/reinvent_test
"""

import os

# Prevent OMP/MKL segfault when PyTorch + XGBoost coexist
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import copy
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

# Ensure src/ is importable
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from reinvent.vocabulary import SMILESVocabulary
from reinvent.model import GRUModel
from reinvent.reward import load_nist8100_corpus, load_expanded_corpus
from reinvent.lube_reward import LubricantReward
from reinvent.trainer import pretrain, ReinventTrainer
from reinvent.augment import augment_corpus

RDLogger.DisableLog("rdApp.*")


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def setup_logging(output_dir):
    """Redirect stdout/stderr to log file (matches wsga.py behaviour)."""
    jobid_raw = os.environ.get("PBS_JOBID", "local")
    jobid = jobid_raw.split(".")[0].split("[")[0]
    array_idx = os.environ.get("PBS_ARRAY_INDEX", "0")

    logs_dir = os.path.join(_src_dir, "logs", f"reinvent_{jobid}")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"reinvent_{array_idx}.log")
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
    print(f"Logging to: {log_path}")
    return log_file


# ─────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────

# Properties tracked per generation (subset of WSGA TRACKED_PROPERTIES)
TRACKED_PROPERTIES = [
    "FitnessScore", "is_valid",
    "density_40C", "viscosity_40C", "tc_40C", "cpsat_40C", "beta_40C",
    "fom1_40C",
    "MW", "Cp_40", "alpha_40", "nu_40", "FOM1_40",
    "bp", "mp", "fp", "dc",
    "SCScore", "Tox21_Score", "Biodegradable",
    "MolPrice", "MolPrice_Penalty",
    "OOD_any", "OOD_count",
]


def get_top_n(trainer, n=40):
    """Return top-N molecules by FitnessScore from the evaluation cache."""
    items = [
        (csmi, data["FitnessScore"], data["row"])
        for csmi, data in trainer.eval_cache.items()
        if data["FitnessScore"] > 0
    ]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:n]

    rows = []
    for rank, (csmi, fitness, row_dict) in enumerate(items, start=1):
        rec = {"SMILES": csmi, "rank": rank}
        for prop in TRACKED_PROPERTIES:
            rec[prop] = row_dict.get(prop, np.nan) if row_dict else np.nan
        # Ensure FitnessScore from cache overrides row (it includes penalty)
        rec["FitnessScore"] = fitness
        rows.append(rec)
    return pd.DataFrame(rows)


def compute_generation_stats(top_n_df, step):
    """Compute summary statistics for top-N molecules at a given step."""
    stats = {"generation": step}

    stat_props = [
        "FitnessScore", "fom1_40C", "FOM1_40",
        "tc_40C", "viscosity_40C",
        "SCScore", "Tox21_Score", "mp", "OOD_count",
    ]
    for prop in stat_props:
        if prop in top_n_df.columns:
            vals = top_n_df[prop].dropna()
            if len(vals) > 0:
                stats[f"{prop}_mean"] = vals.mean()
                stats[f"{prop}_std"] = vals.std()
                stats[f"{prop}_min"] = vals.min()
                stats[f"{prop}_max"] = vals.max()

    if "is_valid" in top_n_df.columns:
        stats["n_valid"] = int(top_n_df["is_valid"].sum())
        stats["pct_valid"] = 100 * top_n_df["is_valid"].mean()

    return stats


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="REINVENT for de novo cooling fluid design",
    )

    # --- Shared with lube_wsga.py ---
    parser.add_argument("--target", type=str, default="FOM_LUBE",
                        choices=["FOM_LUBE"])
    parser.add_argument("--weight_profile", type=str, default="even",
                        choices=["visc", "tc", "hc", "dvi", "tox", "even"])
    parser.add_argument("--output_dir", type=str, default="../outputs")
    parser.add_argument("--model_dir", type=str, default="../models")
    parser.add_argument("--training_data_dir", type=str,
                        default="../training/data")
    parser.add_argument("--top_n", type=int, default=40)

    # Validity thresholds — mp/bp/dc disabled by default for lubricants
    parser.add_argument("--mp_threshold", type=float, default=1e9)
    parser.add_argument("--bp_threshold", type=float, default=-1e9)
    parser.add_argument("--fp_threshold", type=float, default=373)
    parser.add_argument("--sc_threshold", type=float, default=3)
    parser.add_argument("--tox_threshold", type=float, default=3)
    parser.add_argument("--no_biodeg", action="store_true")
    parser.add_argument("--molprice_soft", type=float, default=0.0)
    parser.add_argument("--molprice_hard", type=float, default=0.0)
    parser.add_argument("--soft_constraints", action="store_true",
                        default=False,
                        help="Use soft sigmoid constraint penalties "
                             "instead of hard gates")
    parser.add_argument("--no_soft_constraints", action="store_false",
                        dest="soft_constraints")

    # --- REINVENT-specific ---
    parser.add_argument("--pretrain_epochs", type=int, default=30)
    parser.add_argument("--lr_pretrain", type=float, default=1e-3)
    parser.add_argument("--rl_steps", type=int, default=5000,
                        help="Maximum RL fine-tuning steps (safety cap)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=0.5,
                        help="Reward scaling in augmented likelihood")
    parser.add_argument("--lr_rl", type=float, default=5e-5)
    parser.add_argument("--convergence_patience", type=int, default=200,
                        help="Stop if best FOM1 doesn't improve for N steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)

    # --- Prior corpus ---
    parser.add_argument("--prior_corpus", type=str, default="nist8100",
                        choices=["nist8100", "pubchem", "pubchem+nist"],
                        help="Pretraining corpus: nist8100 (default), "
                             "pubchem (PubChem CHO only), or "
                             "pubchem+nist (merged)")
    parser.add_argument("--pubchem_path", type=str, default=None,
                        help="Path to curated PubChem CHO CSV "
                             "(required if --prior_corpus is pubchem or "
                             "pubchem+nist)")
    parser.add_argument("--augment", action="store_true", default=False,
                        help="Enable SMILES augmentation for pretraining")
    parser.add_argument("--n_augmentations", type=int, default=None,
                        help="Augmentation multiplier per molecule "
                             "(auto-selected by corpus size if omitted)")

    # --- Pretrained prior reuse ---
    parser.add_argument("--prior_path", type=str, default=None,
                        help="Path to pretrained prior.pt weights. "
                             "Skips Phase 1 pretraining if provided.")
    parser.add_argument("--vocab_path", type=str, default=None,
                        help="Path to vocabulary.json matching the prior. "
                             "Required if --prior_path is set.")
    parser.add_argument("--pretrain_only", action="store_true", default=False,
                        help="Run Phase 1 only (pretrain + save), then exit.")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device for pretraining (cpu or cuda)")

    # --- Diversity mechanisms ---
    parser.add_argument("--diversity_filter", action="store_true",
                        default=True,
                        help="Enable scaffold diversity filter (default: on)")
    parser.add_argument("--no_diversity_filter", action="store_false",
                        dest="diversity_filter",
                        help="Disable scaffold diversity filter")
    parser.add_argument("--max_per_scaffold", type=int, default=25,
                        help="Max times a scaffold is rewarded before zeroing")
    parser.add_argument("--diversity_mode", type=str, default="scaffold",
                        choices=["scaffold", "brics", "sphere"],
                        help="Diversity key: 'scaffold' (Murcko), 'brics' "
                             "(fragment decomposition), or 'sphere' (Tanimoto "
                             "sphere exclusion on Morgan fingerprints)")
    parser.add_argument("--tanimoto_threshold", type=float, default=0.35,
                        help="Sphere mode: Tanimoto similarity threshold; "
                             "molecules above this join the same sphere")
    parser.add_argument("--tanimoto_fp_radius", type=int, default=2,
                        help="Sphere mode: Morgan fingerprint radius (default 2, "
                             "matching WSGA niching production)")
    parser.add_argument("--replay_buffer_size", type=int, default=100,
                        help="Experience replay buffer size (0 to disable)")
    parser.add_argument("--replay_max_per_scaffold", type=int, default=10,
                        help="Max molecules per scaffold in replay buffer")
    parser.add_argument("--replay_fraction", type=float, default=0.5,
                        help="Fraction of batch from replay buffer")
    parser.add_argument("--tanimoto_niching", action="store_true",
                        default=False,
                        help="Enable Tanimoto niching penalty (WSGA-style)")
    parser.add_argument("--niching_tau", type=float, default=0.15,
                        help="Niching threshold (similarity below tau is penalty-free)")
    parser.add_argument("--niching_alpha", type=float, default=1000,
                        help="Niching penalty steepness (lower = gentler)")
    parser.add_argument("--niching_radius", type=int, default=8,
                        help="Morgan fingerprint radius for niching")

    args = parser.parse_args()

    # ─── Validate prior reuse flags ──────────────
    if args.prior_path and not args.vocab_path:
        parser.error("--vocab_path is required when --prior_path is set")
    if args.vocab_path and not args.prior_path:
        parser.error("--prior_path is required when --vocab_path is set")
    if args.prior_path and args.pretrain_only:
        parser.error("--prior_path and --pretrain_only are mutually exclusive")

    # ─── Setup ─────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = setup_logging(args.output_dir)

    # Set seeds
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda but CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Output paths (match wsga.py names)
    all_eval_path = os.path.join(args.output_dir, "all_evaluated_molecules.csv")
    top_n_path = os.path.join(args.output_dir, "top_n_tracking.csv")
    gen_stats_path = os.path.join(args.output_dir, "generation_stats.csv")

    print("=" * 60)
    print("REINVENT Configuration")
    print("=" * 60)
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("=" * 60)

    if args.prior_path:
        # ─── Load pretrained prior ────────────────
        print("\n" + "=" * 60)
        print(f"Loading pretrained prior from {args.prior_path}")
        print("=" * 60)

        vocab = SMILESVocabulary.load(args.vocab_path)
        print(f"Vocabulary: {len(vocab)} tokens, loaded from {args.vocab_path}")

        model = GRUModel(len(vocab), embed_dim=128, hidden_dim=512,
                         num_layers=3).to(device)
        state_dict = torch.load(args.prior_path, map_location=device,
                                weights_only=True)
        model.load_state_dict(state_dict)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"GRU model: {n_params:,} parameters (loaded)")

        # Copy to output_dir for reproducibility
        shutil.copy2(args.vocab_path,
                     os.path.join(args.output_dir, "vocabulary.json"))
        torch.save(model.state_dict(),
                   os.path.join(args.output_dir, "prior.pt"))

    else:
        # ─── Load pre-training corpus ──────────────
        if args.prior_corpus == "nist8100":
            corpus = load_nist8100_corpus(args.training_data_dir)
        elif args.prior_corpus == "pubchem":
            if not args.pubchem_path:
                raise ValueError(
                    "--pubchem_path required when --prior_corpus=pubchem")
            corpus = load_expanded_corpus(args.pubchem_path, nist_path=None)
        elif args.prior_corpus == "pubchem+nist":
            if not args.pubchem_path:
                raise ValueError(
                    "--pubchem_path required when --prior_corpus=pubchem+nist")
            corpus = load_expanded_corpus(args.pubchem_path,
                                          nist_path=args.training_data_dir)
        else:
            raise ValueError(f"Unknown prior_corpus: {args.prior_corpus}")

        if len(corpus) == 0:
            raise RuntimeError("No training SMILES found — check corpus settings")

        # ─── Build vocabulary ──────────────────────
        vocab = SMILESVocabulary().build(corpus)
        vocab_path = os.path.join(args.output_dir, "vocabulary.json")
        vocab.save(vocab_path)
        print(f"Vocabulary: {len(vocab)} tokens, saved to {vocab_path}")

        # ─── Augment corpus if requested ──────────
        if args.augment:
            corpus = augment_corpus(corpus,
                                    n_augmentations=args.n_augmentations)

        # ─── Create model ─────────────────────────
        model = GRUModel(len(vocab), embed_dim=128, hidden_dim=512,
                         num_layers=3).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"GRU model: {n_params:,} parameters")

        # ─── Phase 1: Pre-training ─────────────────
        print("\n" + "=" * 60)
        print(f"PHASE 1: Pre-training ({args.pretrain_epochs} epochs, "
              f"{len(corpus)} SMILES)")
        print("=" * 60)

        model = pretrain(
            model, vocab, corpus,
            epochs=args.pretrain_epochs,
            batch_size=(max(args.batch_size, 512)
                        if len(corpus) > 50000 else args.batch_size),
            lr=args.lr_pretrain,
            device=device,
        )
        torch.save(model.state_dict(),
                   os.path.join(args.output_dir, "prior.pt"))

    # ─── Pretrain-only early exit ─────────────
    if args.pretrain_only:
        print("\n" + "=" * 60)
        print("PRETRAIN ONLY — exiting after Phase 1.")
        print(f"Prior:  {os.path.join(args.output_dir, 'prior.pt')}")
        print(f"Vocab:  {os.path.join(args.output_dir, 'vocabulary.json')}")
        print("=" * 60)
        return

    # ─── Create prior (frozen) + agent (trainable) ────
    prior = copy.deepcopy(model)
    prior.eval()
    for p in prior.parameters():
        p.requires_grad = False

    agent = copy.deepcopy(model)

    # ─── Build reward function ─────────────────
    reward_fn = LubricantReward(
        model_dir=args.model_dir,
        training_data_dir=args.training_data_dir,
        target=args.target,
        weight_profile=args.weight_profile,
        sc_threshold=args.sc_threshold,
        mp_threshold=args.mp_threshold,
        bp_threshold=args.bp_threshold,
        fp_threshold=args.fp_threshold,
        tox_threshold=args.tox_threshold,
        use_biodeg=not args.no_biodeg,
        molprice_soft=args.molprice_soft,
        molprice_hard=args.molprice_hard,
        soft_constraints=args.soft_constraints,
    )

    # ─── Phase 2: REINVENT Fine-tuning ─────────
    print("\n" + "=" * 60)
    print(f"PHASE 2: REINVENT ({args.rl_steps} max steps, "
          f"patience={args.convergence_patience})")
    print("=" * 60)

    trainer = ReinventTrainer(
        prior=prior,
        agent=agent,
        vocab=vocab,
        reward_fn=reward_fn,
        sigma=args.sigma,
        lr=args.lr_rl,
        batch_size=args.batch_size,
        max_len=80,
        device=device,
        diversity_filter=args.diversity_filter,
        max_per_scaffold=args.max_per_scaffold,
        diversity_mode=args.diversity_mode,
        tanimoto_threshold=args.tanimoto_threshold,
        tanimoto_fp_radius=args.tanimoto_fp_radius,
        replay_buffer_size=args.replay_buffer_size,
        replay_max_per_scaffold=args.replay_max_per_scaffold,
        replay_fraction=args.replay_fraction,
        tanimoto_niching=args.tanimoto_niching,
        niching_tau=args.niching_tau,
        niching_alpha=args.niching_alpha,
        niching_radius=args.niching_radius,
    )

    tracking_data = []
    generation_stats = []
    batch_stats = []

    for step in range(args.rl_steps):
        metrics = trainer.step()

        # Track per-batch diversity metrics every step
        batch_stats.append({
            "step": step,
            "n_unique_batch": metrics["n_unique_batch"],
            "n_new": metrics["n_new"],
            "n_parseable": metrics["n_valid"],
            "n_valid_constraint": metrics["n_valid_constraint"],
            "batch_best_valid": metrics["batch_best_valid"],
            "best_valid_total": metrics["best_fitness"],
            "reward_mean": metrics["reward_mean"],
            "n_filtered": metrics["n_filtered"],
            "n_scaffolds": metrics["n_scaffolds"],
            "n_saturated": metrics["n_saturated"],
            "brics_unknown_assigned": metrics["brics_unknown_assigned"],
            "brics_unknown_filtered": metrics["brics_unknown_filtered"],
        })

        # --- Log progress ---
        if step % args.log_interval == 0 or step == args.rl_steps - 1:
            div_str = ""
            if args.diversity_filter or args.replay_buffer_size > 0:
                div_str = (
                    f" filt={metrics['n_filtered']}"
                    f" scaff={metrics['n_scaffolds']}"
                    f"({metrics['n_saturated']}sat)"
                    f" replay={metrics['n_replay']}"
                    f"/{metrics['replay_size']}"
                )
            if args.tanimoto_niching:
                div_str += f" niche={metrics['niching_penalty']:.3f}"
            print(
                f"Step {step:5d}: "
                f"reward={metrics['reward_mean']:.2f} "
                f"max={metrics['reward_max']:.2f} "
                f"loss={metrics['loss']:.4f} "
                f"valid={metrics['n_valid']}/{args.batch_size} "
                f"new={metrics['n_new']} "
                f"unique={metrics['n_unique_total']} "
                f"best={metrics['best_fitness']:.2f} "
                f"stag={metrics['steps_since_improvement']}"
                f"{div_str} "
                f"uniqBatch={metrics['n_unique_batch']}/{args.batch_size} "
                f"({metrics['elapsed']:.1f}s)"
            )

        # --- Track top-N ---
        top_n_df = get_top_n(trainer, n=args.top_n)
        if len(top_n_df) > 0:
            for _, row in top_n_df.iterrows():
                record = {
                    "generation": step,
                    "rank": row.get("rank", 0),
                    "SMILES": row["SMILES"],
                }
                for prop in TRACKED_PROPERTIES:
                    record[prop] = row.get(prop, np.nan)
                tracking_data.append(record)

            generation_stats.append(compute_generation_stats(top_n_df, step))

        # --- Periodic CSV save ---
        if step % 50 == 0 and step > 0:
            pd.DataFrame(tracking_data).to_csv(top_n_path, index=False)
            pd.DataFrame(generation_stats).to_csv(gen_stats_path, index=False)
            pd.DataFrame(batch_stats).to_csv(
                os.path.join(args.output_dir, "batch_stats.csv"), index=False)
            pd.DataFrame(trainer.batch_log).to_csv(
                os.path.join(args.output_dir, "batch_log.csv"), index=False)

        # --- Convergence check ---
        if metrics["steps_since_improvement"] >= args.convergence_patience:
            print(f"\nConverged: no improvement for "
                  f"{args.convergence_patience} steps. Stopping.")
            break

    # ─── Save final results ────────────────────
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)

    # All evaluated molecules
    all_eval_df = trainer.get_all_evaluated()
    all_eval_df.to_csv(all_eval_path, index=False)
    print(f"All evaluated molecules ({len(all_eval_df)}): {all_eval_path}")

    # Top-N tracking
    pd.DataFrame(tracking_data).to_csv(top_n_path, index=False)
    print(f"Top-N tracking: {top_n_path}")

    # Generation stats
    pd.DataFrame(generation_stats).to_csv(gen_stats_path, index=False)
    print(f"Generation stats: {gen_stats_path}")

    # Per-batch diversity stats
    batch_stats_path = os.path.join(args.output_dir, "batch_stats.csv")
    pd.DataFrame(batch_stats).to_csv(batch_stats_path, index=False)
    print(f"Batch stats: {batch_stats_path}")

    # Full batch log (all 128 molecules per step, including duplicates)
    batch_log_path = os.path.join(args.output_dir, "batch_log.csv")
    pd.DataFrame(trainer.batch_log).to_csv(batch_log_path, index=False)
    print(f"Batch log ({len(trainer.batch_log)} rows): {batch_log_path}")

    # Agent weights
    agent_path = os.path.join(args.output_dir, "agent.pt")
    torch.save(agent.state_dict(), agent_path)

    # Top-25 final molecules
    top25 = get_top_n(trainer, n=25)
    top25_path = os.path.join(
        args.output_dir, f"top_25_molecules_{args.target}.csv")
    top25.to_csv(top25_path, index=False)

    print(f"\n{'=' * 60}")
    print("REINVENT COMPLETE")
    print(f"{'=' * 60}")
    print(f"Best molecule: {trainer.best_smiles}")
    print(f"Best FitnessScore: {trainer.best_fitness:.4f}")
    print(f"Unique molecules evaluated: {trainer.n_unique_evaluated}")
    print(f"RL steps completed: {step + 1}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
