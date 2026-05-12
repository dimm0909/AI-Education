from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.features.transform import build_model_frame
from src.service.schemas import HealthResponse, PredictRequest, PredictResponse
from src.utils.config import load_yaml, resolve_path
from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


def _to_payload(request: PredictRequest) -> dict[str, Any]:
    if hasattr(request, "model_dump"):
        return request.model_dump()  # pydantic v2
    return request.dict()  # pydantic v1


def _demand_level(prediction: float, thresholds: dict[str, float]) -> str:
    if prediction < thresholds["low_to_medium"]:
        return "low"
    if prediction < thresholds["medium_to_high"]:
        return "medium"
    return "high"


def create_app(config_path: str | Path | None = None) -> FastAPI:
    cfg_path = config_path or os.getenv("BIKEFLOW_SERVICE_CONFIG", "configs/service.yaml")
    cfg = load_yaml(cfg_path)

    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    model_path = resolve_path(cfg.get("artifacts", {}).get("model_path", "artifacts/model.joblib"))

    app = FastAPI(
        title="BikeFlow API",
        description="Predict next-hour bike rental demand",
        version="1.0.0",
    )

    app.state.model_bundle = None
    app.state.model_path = model_path

    if model_path.exists():
        # Loading at app creation avoids startup-thread deadlocks with some joblib/sklearn bundles.
        app.state.model_bundle = joblib.load(model_path)
        LOGGER.info("Loaded model bundle from %s", model_path)
    else:
        LOGGER.warning("Model artifact not found: %s", model_path)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        bundle = app.state.model_bundle
        if bundle is None:
            return HealthResponse(status="degraded", model_loaded=False, model_name=None)

        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=bundle.get("model_name"),
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        bundle = app.state.model_bundle
        if bundle is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model is not loaded. Run training first: python -m src.train "
                    "or place artifacts/model.joblib"
                ),
            )

        payload = _to_payload(request)
        frame = pd.DataFrame([payload])
        model_input = build_model_frame(frame)

        prediction = float(bundle["model"].predict(model_input)[0])
        prediction = max(prediction, 0.0)

        demand_level = _demand_level(prediction, bundle["thresholds"])

        return PredictResponse(
            prediction=round(prediction, 3),
            demand_level=demand_level,
            model_name=bundle["model_name"],
        )

    return app


app = create_app()
