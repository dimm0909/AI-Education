from __future__ import annotations

import math

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_FEATURE_COLUMNS = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]

CATEGORICAL_COLUMNS = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
]

CYCLIC_NUMERIC_COLUMNS = [
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "hr_sin",
    "hr_cos",
    "mnth_sin",
    "mnth_cos",
]

MODEL_FEATURE_COLUMNS = CATEGORICAL_COLUMNS + CYCLIC_NUMERIC_COLUMNS


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare model-ready DataFrame with engineered cyclical features."""
    frame = df.copy()

    frame["hr_sin"] = frame["hr"].apply(lambda h: math.sin(2 * math.pi * float(h) / 24.0))
    frame["hr_cos"] = frame["hr"].apply(lambda h: math.cos(2 * math.pi * float(h) / 24.0))
    frame["mnth_sin"] = frame["mnth"].apply(lambda m: math.sin(2 * math.pi * float(m) / 12.0))
    frame["mnth_cos"] = frame["mnth"].apply(lambda m: math.cos(2 * math.pi * float(m) / 12.0))

    return frame[MODEL_FEATURE_COLUMNS]


def _build_one_hot_encoder() -> OneHotEncoder:
    """Create OHE compatible with multiple scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing pipeline for model training/inference."""
    return ColumnTransformer(
        transformers=[
            ("categorical", _build_one_hot_encoder(), CATEGORICAL_COLUMNS),
            ("numeric", StandardScaler(), CYCLIC_NUMERIC_COLUMNS),
        ]
    )
