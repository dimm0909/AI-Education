from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    season: int = Field(..., ge=1, le=4)
    yr: int = Field(..., ge=0, le=1)
    mnth: int = Field(..., ge=1, le=12)
    hr: int = Field(..., ge=0, le=23)
    holiday: int = Field(..., ge=0, le=1)
    weekday: int = Field(..., ge=0, le=6)
    workingday: int = Field(..., ge=0, le=1)
    weathersit: int = Field(..., ge=1, le=4)
    temp: float = Field(..., ge=0.0, le=1.0)
    atemp: float = Field(..., ge=0.0, le=1.0)
    hum: float = Field(..., ge=0.0, le=1.0)
    windspeed: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    prediction: float
    demand_level: str
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
