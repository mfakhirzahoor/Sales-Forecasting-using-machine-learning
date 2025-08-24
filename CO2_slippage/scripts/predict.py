#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from co2_slippage.config import FEATURE_COLUMNS
from co2_slippage.model import load_trained_model


def main():
    parser = argparse.ArgumentParser(description="Predict CO2 slippage for given features JSON")
    parser.add_argument("--features", type=str, required=True, help="JSON string of features")
    args = parser.parse_args()

    payload = json.loads(args.features)
    x = [[payload[c] for c in FEATURE_COLUMNS]]

    model = load_trained_model()
    yhat = float(model.predict(x)[0])
    print(yhat)


if __name__ == "__main__":
    main()