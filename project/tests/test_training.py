from __future__ import annotations

import json
from pathlib import Path

from src.data.dataset import get_train_test_split, load_dataset
from src.models.trainer import build_demand_thresholds, save_artifacts, train_models


def test_training_and_artifact_saving(tmp_path: Path) -> None:
    df = load_dataset("data/sample/hour_sample.csv").head(500)
    train_df, test_df = get_train_test_split(df, test_size=0.2)

    result = train_models(
        train_df=train_df,
        test_df=test_df,
        random_state=42,
        n_estimators=40,
        max_depth=8,
        catboost_iterations=60,
        catboost_depth=6,
        catboost_learning_rate=0.1,
    )

    assert result.model_name in {
        "baseline_linear_regression",
        "improved_random_forest",
        "improved_catboost_regressor",
    }
    assert result.metrics["rmse"] > 0
    assert result.metrics["accuracy"] >= 0
    assert result.metrics["precision_macro"] >= 0
    assert result.metrics["recall_macro"] >= 0
    assert result.metrics["f1_macro"] >= 0

    thresholds = build_demand_thresholds(train_df["cnt"])

    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    save_artifacts(
        model=result.model,
        model_name=result.model_name,
        metrics=result.metrics,
        all_metrics=result.all_metrics,
        thresholds=thresholds,
        model_path=model_path,
        metrics_path=metrics_path,
    )

    assert model_path.exists()
    assert metrics_path.exists()

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "selected_model" in metrics_payload
