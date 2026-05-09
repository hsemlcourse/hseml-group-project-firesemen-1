"""Prediction utilities for saved model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from eth_price.config import DEFAULT_MODEL_PATH, TARGET_COLUMN, TIME_COLUMN
from eth_price.data import clean_raw_data
from eth_price.features import make_features
from eth_price.modeling import predict_blend


def load_artifact(path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load saved model artifact."""
    return joblib.load(Path(path))


def predict_from_raw_candles(raw_df: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    """Predict next close for the last available candle in a raw candle frame."""
    clean_df, _ = clean_raw_data(raw_df)
    features = make_features(clean_df, horizon=artifact.get("horizon", 1))
    feature_cols = artifact["feature_columns"]
    if features.empty:
        msg = "Not enough candles to build lag/rolling features. Provide at least 70 recent rows."
        raise ValueError(msg)
    X = features[feature_cols]
    model = artifact["model"]
    naive_weight = float(artifact.get("naive_weight", 0.0))
    pred = predict_blend(model, X, naive_weight=naive_weight)
    output = features[[TIME_COLUMN, "Close", TARGET_COLUMN]].copy()
    output["prediction_next_close"] = pred
    return output


def predict_latest(raw_df: pd.DataFrame, model_path: str | Path = DEFAULT_MODEL_PATH) -> float:
    """Return one next-close prediction for the latest candle."""
    artifact = load_artifact(model_path)
    predictions = predict_from_raw_candles(raw_df, artifact)
    return float(predictions["prediction_next_close"].iloc[-1])
