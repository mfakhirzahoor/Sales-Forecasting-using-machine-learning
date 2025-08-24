from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
from scipy.optimize import minimize
from sklearn.pipeline import Pipeline

from .config import FEATURE_COLUMNS


@dataclass
class Bounds:
    lower: Dict[str, float]
    upper: Dict[str, float]


DEFAULT_BOUNDS = Bounds(
    lower={
        "pressure_bar": 5.0,
        "temperature_c": 10.0,
        "gas_flow_rate_nm3h": 100.0,
        "solvent_concentration_pct": 5.0,
        "contact_time_s": 0.5,
        "packing_height_m": 1.0,
    },
    upper={
        "pressure_bar": 30.0,
        "temperature_c": 80.0,
        "gas_flow_rate_nm3h": 2000.0,
        "solvent_concentration_pct": 40.0,
        "contact_time_s": 10.0,
        "packing_height_m": 15.0,
    },
)


def _clip_vector(x: np.ndarray, bounds: Bounds) -> np.ndarray:
    return np.array([
        np.clip(v, bounds.lower[col], bounds.upper[col]) for v, col in zip(x, FEATURE_COLUMNS)
    ], dtype=float)


def suggest_optimal_setpoints(pipeline: Pipeline, bounds: Bounds = DEFAULT_BOUNDS, n_starts: int = 50, random_state: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(random_state)

    def objective(x: np.ndarray) -> float:
        x = _clip_vector(x, bounds)
        pred = float(pipeline.predict([x])[0])
        return pred

    best_x = None
    best_val = float("inf")

    for _ in range(n_starts):
        x0 = np.array([rng.uniform(bounds.lower[c], bounds.upper[c]) for c in FEATURE_COLUMNS])
        res = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=[(bounds.lower[c], bounds.upper[c]) for c in FEATURE_COLUMNS],
            options={"maxiter": 200},
        )
        if res.fun < best_val:
            best_val = float(res.fun)
            best_x = res.x

    assert best_x is not None
    return {c: float(v) for c, v in zip(FEATURE_COLUMNS, best_x)}