#!/usr/bin/env python3
"""
Ridge regression on frozen ChemBERTa-2 [CLS] embeddings for FOM1.

Fast baseline — extracts 384-dim embeddings and fits RidgeCV.
Should run in seconds per fold.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import joblib

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_NAME = str(Path(__file__).resolve().parent / "chemberta_model")


def extract_embeddings(smiles_list, tokenizer, model, device, batch_size=64):
    """Extract [CLS] embeddings from frozen ChemBERTa."""
    model.eval()
    embeddings = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        enc = tokenizer(
            batch.tolist(),
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
    return np.vstack(embeddings)


def main():
    parser = argparse.ArgumentParser(
        description="Ridge on ChemBERTa embeddings for FOM1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--target", type=str, default="FOM1_avg",
                        help="Target column to predict (FOM1_40, FOM1_100, or FOM1_avg)")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cpu")
    logger.info("Using device: %s", device)

    data_dir = Path(args.data_dir)
    df = pd.read_csv(data_dir / "fom1_dataset.csv")
    with open(data_dir / "fold_indices.json") as f:
        fold_indices = json.load(f)

    smiles = df["SMILES"].values
    y = df[args.target].values.astype(np.float32)
    logger.info("Target: %s", args.target)

    # Load model and extract all embeddings at once
    logger.info("Loading ChemBERTa and extracting embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME).to(device)

    t0 = time.time()
    all_embeddings = extract_embeddings(smiles, tokenizer, backbone, device)
    logger.info("Extracted embeddings: %s in %.1fs", all_embeddings.shape, time.time() - t0)

    # Save embeddings for potential reuse
    np.save(data_dir / "chemberta_embeddings.npy", all_embeddings)

    folds = range(5) if args.fold is None else [args.fold]

    for fold_id in folds:
        logger.info("=" * 50)
        logger.info("FOLD %d — Ridge on ChemBERTa Embeddings", fold_id)
        logger.info("=" * 50)
        t0 = time.time()

        train_idx = fold_indices[str(fold_id)]["train"]
        test_idx = fold_indices[str(fold_id)]["test"]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(all_embeddings[train_idx])
        X_test = scaler.transform(all_embeddings[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        # RidgeCV
        alphas = np.logspace(-3, 3, 50)
        model = RidgeCV(alphas=alphas)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        elapsed = time.time() - t0
        logger.info("Test R²: %.4f, RMSE: %.4f, MAE: %.4f (%.1fs)", r2, rmse, mae, elapsed)
        logger.info("Best alpha: %.4f", model.alpha_)

        pd.DataFrame({"SMILES": df.iloc[test_idx]["SMILES"].values, "y_true": y_test, "y_pred": y_pred})\
            .to_csv(data_dir / f"ridge_embeddings_fold{fold_id}_predictions.csv", index=False)

        with open(data_dir / f"ridge_embeddings_fold{fold_id}_metrics.json", "w") as f:
            json.dump({"model": "ridge_embeddings", "target": args.target, "fold": fold_id, "r2": float(r2),
                        "rmse": float(rmse), "mae": float(mae), "training_time_s": float(elapsed),
                        "best_alpha": float(model.alpha_)}, f, indent=2)

        joblib.dump({"model": model, "scaler": scaler},
                     data_dir / f"ridge_embeddings_fold{fold_id}_model.joblib")

    logger.info("Ridge Embeddings training complete.")


if __name__ == "__main__":
    main()
