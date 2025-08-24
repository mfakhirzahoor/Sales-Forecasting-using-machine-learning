from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    # Compute RMSE without using the 'squared' kwarg for compatibility
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.maximum(1e-6, y_true)))))
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}