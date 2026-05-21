from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from eth_cp3.model import load_metadata, predict_next_close

app = FastAPI(
    title="ETHUSD CP3 Price Prediction API",
    version="0.3.0",
    description="FastAPI service for one-minute ETHUSD close-price prediction.",
)


class Candle(BaseModel):
    open_time: str = Field(..., examples=["2025-10-11 10:04:00"])
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_asset_volume: float
    number_of_trades: float
    taker_buy_base_asset_volume: float
    taker_buy_quote_asset_volume: float


class PredictRequest(BaseModel):
    candles: Annotated[
        list[Candle],
        Field(min_length=61, description="At least 61 consecutive one-minute candles"),
    ]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    metadata = load_metadata()
    return {
        "model_type": metadata.get("model_type"),
        "target": metadata.get("target"),
        "feature_count": metadata.get("feature_count"),
        "split": metadata.get("split"),
        "test_metrics": metadata.get("test_metrics"),
        "baseline_test_metrics": metadata.get("baseline_test_metrics"),
        "source_file": metadata.get("source_file"),
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    try:
        return predict_next_close([c.model_dump() for c in request.candles])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
