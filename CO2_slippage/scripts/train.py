#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from co2_slippage.config import DATA_DIR, MODELS_DIR
from co2_slippage.data import generate_synthetic_dataset, split_dataset, save_splits, load_dataset
from co2_slippage.model import train_model, evaluate_model_on_split


def main():
    parser = argparse.ArgumentParser(description="Train model to predict CO2 slippage")
    parser.add_argument("--use-synthetic", action="store_true", help="Generate and use synthetic data")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--model-name", type=str, default="rf_model")
    parser.add_argument("--search", action="store_true", help="Run hyperparameter search")
    args = parser.parse_args()

    if args.use_synthetic:
        df = generate_synthetic_dataset(n_samples=args.n_samples)
        splits = split_dataset(df)
        save_splits(splits, base_dir=DATA_DIR)
    else:
        # Expect existing CSVs in DATA_DIR
        pass

    train_df = load_dataset("train", base_dir=DATA_DIR)
    val_df = load_dataset("val", base_dir=DATA_DIR)
    test_df = load_dataset("test", base_dir=DATA_DIR)

    trained = train_model(train_df, val_df=val_df, do_search=args.search, model_name=args.model_name)

    train_metrics = evaluate_model_on_split(trained.pipeline, train_df)
    val_metrics = evaluate_model_on_split(trained.pipeline, val_df)
    test_metrics = evaluate_model_on_split(trained.pipeline, test_df)

    print("Saved model to:", trained.model_path)
    print("Best params:", trained.best_params)
    print("Train metrics:", train_metrics)
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()