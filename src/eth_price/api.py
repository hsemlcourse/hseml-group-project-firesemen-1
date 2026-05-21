"""FastAPI app for next-minute ETH close prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from eth_price.config import DEFAULT_MODEL_PATH
from eth_price.predict import load_artifact, predict_from_raw_candles

app = FastAPI(
    title="ETH/USD next-minute close predictor",
    description="Predicts ETH/USD close price for the next minute from recent OHLCV candles.",
    version="1.0.0",
)


class Candle(BaseModel):
    open_time: str = Field(..., description="Candle open time, ISO format")
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_asset_volume: float
    number_of_trades: float
    taker_buy_base_asset_volume: float
    taker_buy_quote_asset_volume: float


class PredictionRequest(BaseModel):
    candles: list[Candle] = Field(
        ...,
        min_length=70,
        description="Recent candles in chronological order",
    )


def _request_to_frame(request: PredictionRequest) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candle in request.candles:
        rows.append(
            {
                "Open time": candle.open_time,
                "Open": candle.open,
                "High": candle.high,
                "Low": candle.low,
                "Close": candle.close,
                "Volume": candle.volume,
                "Quote asset volume": candle.quote_asset_volume,
                "Number of trades": candle.number_of_trades,
                "Taker buy base asset volume": candle.taker_buy_base_asset_volume,
                "Taker buy quote asset volume": candle.taker_buy_quote_asset_volume,
            }
        )
    return pd.DataFrame(rows)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float | str]:
    """Predict next-minute close from a list of recent candles."""
    model_path = Path(DEFAULT_MODEL_PATH)
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Model file is missing. Run scripts/train.py first.",
        )
    try:
        artifact = load_artifact(model_path)
        predictions = predict_from_raw_candles(_request_to_frame(request), artifact)
        latest = predictions.iloc[-1]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "last_candle_time": str(latest["Open time"]),
        "last_close": float(latest["Close"]),
        "prediction_next_close": float(latest["prediction_next_close"]),
    }
