from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd

from .config import DATA_DIR, FEATURE_COLUMNS, TARGET_COLUMN


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def generate_synthetic_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    pressure_bar = rng.uniform(5.0, 30.0, size=n_samples)
    temperature_c = rng.uniform(10.0, 80.0, size=n_samples)
    gas_flow_rate_nm3h = rng.uniform(100.0, 2000.0, size=n_samples)
    solvent_concentration_pct = rng.uniform(5.0, 40.0, size=n_samples)
    contact_time_s = rng.uniform(0.5, 10.0, size=n_samples)
    packing_height_m = rng.uniform(1.0, 15.0, size=n_samples)

    # Synthetic physics-inspired relationship for CO2 slippage (lower is better)
    optimal_temp_c = 35.0
    temp_penalty = 0.015 * np.abs(temperature_c - optimal_temp_c)
    pressure_effect = 10.0 / np.sqrt(pressure_bar)
    flow_effect = 0.02 * np.log1p(gas_flow_rate_nm3h)
    solvent_effect = 30.0 / np.maximum(solvent_concentration_pct, 1.0)
    contact_effect = 10.0 / np.maximum(contact_time_s, 0.1)
    packing_effect = 12.0 / np.maximum(packing_height_m, 0.5)

    base = 1.5
    slippage = (
        base
        + temp_penalty
        + pressure_effect
        + flow_effect
        + solvent_effect
        + contact_effect
        + packing_effect
    )

    # Interaction effects
    slippage -= 0.004 * pressure_bar * contact_time_s
    slippage -= 0.006 * solvent_concentration_pct * packing_height_m

    # Noise
    noise = np.clip(np.random.default_rng(random_state + 1).normal(0, 0.8, size=n_samples), -2.0, 2.0)
    slippage = np.clip(slippage + noise, 0.1, None)

    data = pd.DataFrame(
        {
            "pressure_bar": pressure_bar,
            "temperature_c": temperature_c,
            "gas_flow_rate_nm3h": gas_flow_rate_nm3h,
            "solvent_concentration_pct": solvent_concentration_pct,
            "contact_time_s": contact_time_s,
            "packing_height_m": packing_height_m,
            TARGET_COLUMN: slippage,
        }
    )
    return data


def split_dataset(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15, random_state: int = 42) -> DatasetSplits:
    assert 0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1
    df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n = len(df_shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df_shuffled.iloc[:n_train]
    val = df_shuffled.iloc[n_train : n_train + n_val]
    test = df_shuffled.iloc[n_train + n_val :]
    return DatasetSplits(train=train, val=val, test=test)


def save_splits(splits: DatasetSplits, base_dir: Path | None = None) -> Dict[str, Path]:
    out_dir = (base_dir or DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": out_dir / "train.csv",
        "val": out_dir / "val.csv",
        "test": out_dir / "test.csv",
    }
    splits.train.to_csv(paths["train"], index=False)
    splits.val.to_csv(paths["val"], index=False)
    splits.test.to_csv(paths["test"], index=False)
    return paths


def load_dataset(split: str = "train", base_dir: Path | None = None) -> pd.DataFrame:
    path = (base_dir or DATA_DIR) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset split '{split}' not found at {path}")
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {split}: {missing}")
    return df