from __future__ import annotations

import argparse
import logging

from src.data.dataset import get_train_test_split, load_dataset, pick_training_data
from src.models.trainer import save_artifacts, save_training_figures, train_models
from src.utils.config import load_yaml, resolve_path
from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BikeFlow demand model")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    artifacts_cfg = cfg.get("artifacts", {})

    full_path = resolve_path(data_cfg.get("full_path", "data/raw/hour.csv"))
    sample_path = resolve_path(data_cfg.get("sample_path", "data/sample/hour_sample.csv"))

    training_path = pick_training_data(full_path, sample_path)
    df = load_dataset(training_path)
    train_df, test_df = get_train_test_split(df, test_size=float(model_cfg.get("test_size", 0.2)))

    train_result = train_models(
        train_df=train_df,
        test_df=test_df,
        random_state=int(model_cfg.get("random_state", 42)),
        n_estimators=int(model_cfg.get("random_forest", {}).get("n_estimators", 300)),
        max_depth=int(model_cfg.get("random_forest", {}).get("max_depth", 20)),
        catboost_iterations=int(model_cfg.get("catboost", {}).get("iterations", 500)),
        catboost_depth=int(model_cfg.get("catboost", {}).get("depth", 8)),
        catboost_learning_rate=float(model_cfg.get("catboost", {}).get("learning_rate", 0.05)),
        catboost_tuning_iterations=[int(v) for v in model_cfg.get("catboost", {}).get("tuning", {}).get("iterations", [500, 700, 900])],
        catboost_tuning_depths=[int(v) for v in model_cfg.get("catboost", {}).get("tuning", {}).get("depths", [6, 8, 10])],
        catboost_tuning_learning_rates=[
            float(v)
            for v in model_cfg.get("catboost", {}).get("tuning", {}).get("learning_rates", [0.03, 0.05, 0.08])
        ],
        lightgbm_n_estimators=int(model_cfg.get("lightgbm", {}).get("n_estimators", 500)),
        lightgbm_max_depth=int(model_cfg.get("lightgbm", {}).get("max_depth", 8)),
        lightgbm_learning_rate=float(model_cfg.get("lightgbm", {}).get("learning_rate", 0.05)),
        lightgbm_num_leaves=int(model_cfg.get("lightgbm", {}).get("num_leaves", 31)),
        xgboost_n_estimators=int(model_cfg.get("xgboost", {}).get("n_estimators", 500)),
        xgboost_max_depth=int(model_cfg.get("xgboost", {}).get("max_depth", 8)),
        xgboost_learning_rate=float(model_cfg.get("xgboost", {}).get("learning_rate", 0.05)),
    )

    model_path = resolve_path(artifacts_cfg.get("model_path", "artifacts/model.joblib"))
    metrics_path = resolve_path(artifacts_cfg.get("metrics_path", "artifacts/metrics.json"))

    save_artifacts(
        model=train_result.model,
        model_name=train_result.model_name,
        metrics=train_result.metrics,
        all_metrics=train_result.all_metrics,
        thresholds=train_result.thresholds,
        model_path=model_path,
        metrics_path=metrics_path,
    )

    figures_dir = resolve_path(artifacts_cfg.get("figures_dir", "artifacts/figures"))
    saved_figures = save_training_figures(
        all_metrics=train_result.all_metrics,
        y_true=train_result.y_test,
        predictions_by_model=train_result.predictions_by_model,
        best_model_name=train_result.model_name,
        catboost_tuning_results=train_result.catboost_tuning_results,
        figures_dir=figures_dir,
    )

    LOGGER.info("Training complete")
    LOGGER.info("Selected model: %s", train_result.model_name)
    LOGGER.info("Selected metrics: %s", train_result.metrics)
    LOGGER.info("Model saved to: %s", model_path)
    LOGGER.info("Metrics saved to: %s", metrics_path)
    if saved_figures:
        LOGGER.info("Saved figures: %s", saved_figures)


if __name__ == "__main__":
    main()
