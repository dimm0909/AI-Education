from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.pipeline import Pipeline

from src.features.transform import BASE_FEATURE_COLUMNS, build_model_frame, build_preprocessor

TARGET_COLUMN = "cnt"


@dataclass
class TrainResult:
    model_name: str
    model: Pipeline
    metrics: dict[str, Any]
    all_metrics: dict[str, dict[str, float]]


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute key metrics for regression."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    y_true_safe = y_true.replace(0, 1e-6)
    mape = float(mean_absolute_percentage_error(y_true_safe, y_pred))

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }


def _build_baseline_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", LinearRegression()),
        ]
    )


def _build_improved_model(random_state: int, n_estimators: int, max_depth: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int,
    n_estimators: int,
    max_depth: int,
) -> TrainResult:
    """Train baseline and improved models, then select the best by RMSE."""
    x_train = build_model_frame(train_df)
    y_train = train_df[TARGET_COLUMN]

    x_test = build_model_frame(test_df)
    y_test = test_df[TARGET_COLUMN]

    candidates: dict[str, Pipeline] = {
        "baseline_linear_regression": _build_baseline_model(),
        "improved_random_forest": _build_improved_model(
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
        ),
    }

    all_metrics: dict[str, dict[str, float]] = {}
    trained_models: dict[str, Pipeline] = {}

    for model_name, model in candidates.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        all_metrics[model_name] = regression_metrics(y_test, pred)
        trained_models[model_name] = model

    best_name = min(all_metrics, key=lambda name: all_metrics[name]["rmse"])

    return TrainResult(
        model_name=best_name,
        model=trained_models[best_name],
        metrics=all_metrics[best_name],
        all_metrics=all_metrics,
    )


def build_demand_thresholds(y_train: pd.Series) -> dict[str, float]:
    """Build low/medium/high demand thresholds based on target quantiles."""
    q33, q66 = np.quantile(y_train, [0.33, 0.66])
    return {
        "low_to_medium": float(q33),
        "medium_to_high": float(q66),
    }


def save_artifacts(
    model: Pipeline,
    model_name: str,
    metrics: dict[str, Any],
    all_metrics: dict[str, dict[str, float]],
    thresholds: dict[str, float],
    model_path: str | Path,
    metrics_path: str | Path,
) -> None:
    """Persist model and metrics on disk."""
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "model_name": model_name,
        "feature_columns": BASE_FEATURE_COLUMNS,
        "thresholds": thresholds,
    }

    joblib.dump(payload, model_path)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_model": model_name,
                "selected_metrics": metrics,
                "all_metrics": all_metrics,
                "thresholds": thresholds,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
