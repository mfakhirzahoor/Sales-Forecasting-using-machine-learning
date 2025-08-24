from __future__ import annotations

from typing import List

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from .config import FEATURE_COLUMNS


def build_preprocessing_pipeline(with_interactions: bool = True) -> Pipeline:
    numeric_features: List[str] = FEATURE_COLUMNS

    steps = [("scaler", StandardScaler())]
    if with_interactions:
        # Lightweight interactions to let linear models capture some curvature
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))

    numeric_transformer = Pipeline(steps=steps)

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="drop",
        sparse_threshold=0.0,
    )

    return preprocessor