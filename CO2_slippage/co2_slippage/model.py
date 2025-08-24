from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .config import FEATURE_COLUMNS, TARGET_COLUMN, MODELS_DIR
from .features import build_preprocessing_pipeline
from .metrics import regression_metrics


@dataclass
class TrainedModel:
    pipeline: Pipeline
    best_params: Dict[str, Any]
    model_path: Path


def build_model() -> Pipeline:
    pre = build_preprocessing_pipeline(with_interactions=True)
    reg = RandomForestRegressor(random_state=42, n_estimators=300, max_depth=None, n_jobs=-1)
    pipe = Pipeline(steps=[("pre", pre), ("reg", reg)])
    return pipe


def hyperparameter_search(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> Tuple[Pipeline, Dict[str, Any]]:
    param_grid = {
        "reg__n_estimators": [200, 400],
        "reg__max_depth": [None, 12, 20],
        "reg__min_samples_split": [2, 5],
        "reg__min_samples_leaf": [1, 2, 4],
    }
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame | None = None, do_search: bool = False, model_name: str = "rf_model") -> TrainedModel:
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    pipe = build_model()
    if do_search:
        pipe, best_params = hyperparameter_search(pipe, X_train, y_train)
    else:
        pipe.fit(X_train, y_train)
        best_params = getattr(pipe.named_steps.get("reg"), "get_params", lambda: {})()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(pipe, model_path)

    return TrainedModel(pipeline=pipe, best_params=best_params, model_path=model_path)


def evaluate_model_on_split(pipe: Pipeline, df: pd.DataFrame) -> Dict[str, float]:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    pred = pipe.predict(X)
    return regression_metrics(y, pred)


def load_trained_model(path: Path | None = None) -> Pipeline:
    model_path = path or (MODELS_DIR / "rf_model.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)