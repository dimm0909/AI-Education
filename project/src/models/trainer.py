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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.features.transform import BASE_FEATURE_COLUMNS, build_model_frame, build_preprocessor

TARGET_COLUMN = "cnt"
REGRESSION_METRICS = ["rmse", "mae", "mape"]
CLASSIFICATION_METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

try:
    from catboost import CatBoostRegressor

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class TrainResult:
    model_name: str
    model: Pipeline
    metrics: dict[str, Any]
    all_metrics: dict[str, dict[str, float]]
    thresholds: dict[str, float]
    y_test: np.ndarray
    predictions_by_model: dict[str, np.ndarray]
    catboost_tuning_results: list[dict[str, float | int]]


def _to_demand_classes(values: np.ndarray | pd.Series, thresholds: dict[str, float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.where(
        arr < thresholds["low_to_medium"],
        0,
        np.where(arr < thresholds["medium_to_high"], 1, 2),
    )


def build_demand_thresholds(y_train: pd.Series) -> dict[str, float]:
    """Build low/medium/high demand thresholds based on target quantiles."""
    q33, q66 = np.quantile(y_train, [0.33, 0.66])
    return {
        "low_to_medium": float(q33),
        "medium_to_high": float(q66),
    }


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray, thresholds: dict[str, float]) -> dict[str, float]:
    """Compute key metrics for regression and demand-level classification."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    y_true_safe = y_true.replace(0, 1e-6)
    mape = float(mean_absolute_percentage_error(y_true_safe, y_pred))

    y_true_cls = _to_demand_classes(y_true, thresholds)
    y_pred_cls = _to_demand_classes(y_pred, thresholds)

    accuracy = float(accuracy_score(y_true_cls, y_pred_cls))
    precision_macro = float(precision_score(y_true_cls, y_pred_cls, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true_cls, y_pred_cls, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0))

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }


def _build_baseline_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", LinearRegression()),
        ]
    )


def _build_random_forest_model(random_state: int, n_estimators: int, max_depth: int) -> Pipeline:
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


def _build_catboost_model(random_state: int, iterations: int, depth: int, learning_rate: float) -> Pipeline:
    if not CATBOOST_AVAILABLE:
        raise RuntimeError("CatBoost is not installed in this environment")

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                CatBoostRegressor(
                    random_seed=random_state,
                    iterations=iterations,
                    depth=depth,
                    learning_rate=learning_rate,
                    loss_function="RMSE",
                    verbose=False,
                    allow_writing_files=False,
                ),
            ),
        ]
    )


def _build_lightgbm_model(
    random_state: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    num_leaves: int,
) -> Pipeline:
    if not LIGHTGBM_AVAILABLE:
        raise RuntimeError("LightGBM is not installed in this environment")

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                LGBMRegressor(
                    random_state=random_state,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    subsample=0.9,
                    colsample_bytree=0.9,
                ),
            ),
        ]
    )


def _build_xgboost_model(
    random_state: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> Pipeline:
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("XGBoost is not installed in this environment")

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=random_state,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _inner_holdout_split(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    holdout_ratio: float,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    split_idx = int(len(x_train) * (1.0 - holdout_ratio))
    split_idx = max(split_idx, 1)
    split_idx = min(split_idx, len(x_train) - 1)

    x_inner_train = x_train.iloc[:split_idx].copy()
    y_inner_train = y_train.iloc[:split_idx].copy()
    x_inner_val = x_train.iloc[split_idx:].copy()
    y_inner_val = y_train.iloc[split_idx:].copy()

    return x_inner_train, y_inner_train, x_inner_val, y_inner_val


def _tune_catboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    thresholds: dict[str, float],
    random_state: int,
    iterations_grid: list[int],
    depth_grid: list[int],
    learning_rate_grid: list[float],
    holdout_ratio: float,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    x_inner_train, y_inner_train, x_inner_val, y_inner_val = _inner_holdout_split(
        x_train,
        y_train,
        holdout_ratio,
    )

    tuning_results: list[dict[str, float | int]] = []
    best_params: dict[str, float | int] = {
        "iterations": int(iterations_grid[0]),
        "depth": int(depth_grid[0]),
        "learning_rate": float(learning_rate_grid[0]),
    }
    best_rmse = float("inf")

    for iterations in iterations_grid:
        for depth in depth_grid:
            for learning_rate in learning_rate_grid:
                model = _build_catboost_model(
                    random_state=random_state,
                    iterations=int(iterations),
                    depth=int(depth),
                    learning_rate=float(learning_rate),
                )
                model.fit(x_inner_train, y_inner_train)
                pred_val = model.predict(x_inner_val)
                metrics = regression_metrics(y_inner_val, pred_val, thresholds=thresholds)

                row = {
                    "iterations": int(iterations),
                    "depth": int(depth),
                    "learning_rate": float(learning_rate),
                    "rmse": float(metrics["rmse"]),
                    "mae": float(metrics["mae"]),
                }
                tuning_results.append(row)

                if row["rmse"] < best_rmse:
                    best_rmse = row["rmse"]
                    best_params = {
                        "iterations": int(iterations),
                        "depth": int(depth),
                        "learning_rate": float(learning_rate),
                    }

    return best_params, tuning_results


def train_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    catboost_iterations: int = 500,
    catboost_depth: int = 8,
    catboost_learning_rate: float = 0.05,
    catboost_tuning_iterations: list[int] | None = None,
    catboost_tuning_depths: list[int] | None = None,
    catboost_tuning_learning_rates: list[float] | None = None,
    lightgbm_n_estimators: int = 500,
    lightgbm_max_depth: int = 8,
    lightgbm_learning_rate: float = 0.05,
    lightgbm_num_leaves: int = 31,
    xgboost_n_estimators: int = 500,
    xgboost_max_depth: int = 8,
    xgboost_learning_rate: float = 0.05,
) -> TrainResult:
    """Train baseline and improved models, then select the best by RMSE."""
    x_train = build_model_frame(train_df)
    y_train = train_df[TARGET_COLUMN]

    x_test = build_model_frame(test_df)
    y_test = test_df[TARGET_COLUMN]
    thresholds = build_demand_thresholds(y_train)

    candidates: dict[str, Pipeline] = {
        "baseline_linear_regression": _build_baseline_model(),
        "improved_random_forest": _build_random_forest_model(
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
        ),
    }

    catboost_tuning_results: list[dict[str, float | int]] = []
    if CATBOOST_AVAILABLE:
        iterations_grid = catboost_tuning_iterations or [catboost_iterations]
        depth_grid = catboost_tuning_depths or [catboost_depth]
        learning_rate_grid = catboost_tuning_learning_rates or [catboost_learning_rate]

        best_catboost_params, catboost_tuning_results = _tune_catboost(
            x_train=x_train,
            y_train=y_train,
            thresholds=thresholds,
            random_state=random_state,
            iterations_grid=iterations_grid,
            depth_grid=depth_grid,
            learning_rate_grid=learning_rate_grid,
            holdout_ratio=0.2,
        )

        candidates["improved_catboost_regressor"] = _build_catboost_model(
            random_state=random_state,
            iterations=int(best_catboost_params["iterations"]),
            depth=int(best_catboost_params["depth"]),
            learning_rate=float(best_catboost_params["learning_rate"]),
        )

    if LIGHTGBM_AVAILABLE:
        candidates["improved_lightgbm_regressor"] = _build_lightgbm_model(
            random_state=random_state,
            n_estimators=lightgbm_n_estimators,
            max_depth=lightgbm_max_depth,
            learning_rate=lightgbm_learning_rate,
            num_leaves=lightgbm_num_leaves,
        )

    if XGBOOST_AVAILABLE:
        candidates["improved_xgboost_regressor"] = _build_xgboost_model(
            random_state=random_state,
            n_estimators=xgboost_n_estimators,
            max_depth=xgboost_max_depth,
            learning_rate=xgboost_learning_rate,
        )

    all_metrics: dict[str, dict[str, float]] = {}
    trained_models: dict[str, Pipeline] = {}
    predictions_by_model: dict[str, np.ndarray] = {}

    for model_name, model in candidates.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        all_metrics[model_name] = regression_metrics(y_test, pred, thresholds=thresholds)
        trained_models[model_name] = model
        predictions_by_model[model_name] = pred

    best_name = min(all_metrics, key=lambda name: all_metrics[name]["rmse"])

    return TrainResult(
        model_name=best_name,
        model=trained_models[best_name],
        metrics=all_metrics[best_name],
        all_metrics=all_metrics,
        thresholds=thresholds,
        y_test=np.asarray(y_test, dtype=float),
        predictions_by_model=predictions_by_model,
        catboost_tuning_results=catboost_tuning_results,
    )


def save_training_figures(
    all_metrics: dict[str, dict[str, float]],
    y_true: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    best_model_name: str,
    catboost_tuning_results: list[dict[str, float | int]],
    figures_dir: str | Path,
) -> list[str]:
    """Save core model-training figures and return file paths."""
    if not MATPLOTLIB_AVAILABLE:
        return []

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    model_names = list(all_metrics.keys())

    # 1) Regression metrics comparison
    fig, axes = plt.subplots(1, len(REGRESSION_METRICS), figsize=(15, 4), constrained_layout=True)
    if len(REGRESSION_METRICS) == 1:
        axes = [axes]
    for idx, metric_name in enumerate(REGRESSION_METRICS):
        values = [all_metrics[name][metric_name] for name in model_names]
        axes[idx].bar(model_names, values, color="#4C78A8")
        axes[idx].set_title(metric_name.upper())
        axes[idx].tick_params(axis="x", rotation=20)
    reg_metrics_path = figures_dir / "metrics_regression.png"
    fig.savefig(reg_metrics_path, dpi=150)
    plt.close(fig)
    saved_paths.append(str(reg_metrics_path))

    # 2) Classification-style metrics comparison
    fig, axes = plt.subplots(1, len(CLASSIFICATION_METRICS), figsize=(20, 4), constrained_layout=True)
    if len(CLASSIFICATION_METRICS) == 1:
        axes = [axes]
    for idx, metric_name in enumerate(CLASSIFICATION_METRICS):
        values = [all_metrics[name][metric_name] for name in model_names]
        axes[idx].bar(model_names, values, color="#F58518")
        axes[idx].set_title(metric_name)
        axes[idx].set_ylim(0.0, 1.0)
        axes[idx].tick_params(axis="x", rotation=20)
    cls_metrics_path = figures_dir / "metrics_classification.png"
    fig.savefig(cls_metrics_path, dpi=150)
    plt.close(fig)
    saved_paths.append(str(cls_metrics_path))

    # 3) Best model: actual vs predicted
    best_pred = predictions_by_model[best_model_name]
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.scatter(y_true, best_pred, alpha=0.35, s=10)
    diag_min = float(min(np.min(y_true), np.min(best_pred)))
    diag_max = float(max(np.max(y_true), np.max(best_pred)))
    ax.plot([diag_min, diag_max], [diag_min, diag_max], color="red", linestyle="--")
    ax.set_title(f"Actual vs Predicted ({best_model_name})")
    ax.set_xlabel("Actual cnt")
    ax.set_ylabel("Predicted cnt")
    scatter_path = figures_dir / "best_model_actual_vs_pred.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    saved_paths.append(str(scatter_path))

    # 4) Best model residual distribution
    residuals = y_true - best_pred
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.hist(residuals, bins=40, color="#54A24B", alpha=0.85)
    ax.set_title(f"Residuals ({best_model_name})")
    ax.set_xlabel("Actual - Predicted")
    ax.set_ylabel("Frequency")
    residual_path = figures_dir / "best_model_residuals.png"
    fig.savefig(residual_path, dpi=150)
    plt.close(fig)
    saved_paths.append(str(residual_path))

    # 5) CatBoost tuning curve (if available)
    if catboost_tuning_results:
        tuning_frame = pd.DataFrame(catboost_tuning_results)
        tuning_frame = tuning_frame.sort_values("rmse").reset_index(drop=True)
        labels = [
            f"it={int(r['iterations'])},d={int(r['depth'])},lr={float(r['learning_rate']):.3f}"
            for _, r in tuning_frame.iterrows()
        ]

        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        ax.plot(range(len(tuning_frame)), tuning_frame["rmse"].values, marker="o")
        ax.set_xticks(range(len(tuning_frame)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title("CatBoost tuning (RMSE on inner holdout)")
        ax.set_ylabel("RMSE")
        tuning_path = figures_dir / "catboost_tuning_rmse.png"
        fig.savefig(tuning_path, dpi=150)
        plt.close(fig)
        saved_paths.append(str(tuning_path))

    return saved_paths


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
