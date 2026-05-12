from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.routing import APIRoute

from src.data.dataset import get_train_test_split, load_dataset
from src.models.trainer import build_demand_thresholds, save_artifacts, train_models
from src.service.app import create_app
from src.service.schemas import PredictRequest


def test_service_health_and_predict(tmp_path: Path) -> None:
    df = load_dataset("data/sample/hour_sample.csv").head(700)
    train_df, test_df = get_train_test_split(df, test_size=0.2)

    result = train_models(
        train_df=train_df,
        test_df=test_df,
        random_state=42,
        n_estimators=60,
        max_depth=10,
        catboost_iterations=80,
        catboost_depth=6,
        catboost_learning_rate=0.1,
    )
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

    config_path = tmp_path / "service.yaml"
    config_path.write_text(
        "\n".join(
            [
                "logging:",
                "  level: INFO",
                "service:",
                "  host: 127.0.0.1",
                "  port: 8001",
                "artifacts:",
                f"  model_path: {model_path}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    app = create_app(config_path)
    sample_row = pd.read_csv("data/sample/hour_sample.csv").iloc[0]

    payload = {
        "season": int(sample_row["season"]),
        "yr": int(sample_row["yr"]),
        "mnth": int(sample_row["mnth"]),
        "hr": int(sample_row["hr"]),
        "holiday": int(sample_row["holiday"]),
        "weekday": int(sample_row["weekday"]),
        "workingday": int(sample_row["workingday"]),
        "weathersit": int(sample_row["weathersit"]),
        "temp": float(sample_row["temp"]),
        "atemp": float(sample_row["atemp"]),
        "hum": float(sample_row["hum"]),
        "windspeed": float(sample_row["windspeed"]),
    }

    health_endpoint = None
    predict_endpoint = None
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if route.path == "/health":
            health_endpoint = route.endpoint
        if route.path == "/predict":
            predict_endpoint = route.endpoint

    assert health_endpoint is not None
    assert predict_endpoint is not None

    health_response = health_endpoint()
    assert health_response.model_loaded is True

    predict_response = predict_endpoint(PredictRequest(**payload))
    assert predict_response.prediction >= 0
    assert predict_response.demand_level in {"low", "medium", "high"}
