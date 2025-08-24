#!/usr/bin/env python3
from __future__ import annotations

import argparse

from co2_slippage.config import DATA_DIR
from co2_slippage.data import load_dataset
from co2_slippage.model import load_trained_model, evaluate_model_on_split


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on a dataset split")
    parser.add_argument("--dataset", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    df = load_dataset(args.dataset, base_dir=DATA_DIR)
    model = load_trained_model()
    metrics = evaluate_model_on_split(model, df)
    print(metrics)


if __name__ == "__main__":
    main()