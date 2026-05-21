"""Feature engineering for next-minute ETH price prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eth_price.config import DEFAULT_HORIZON, TARGET_COLUMN, TIME_COLUMN

BASE_NUMERIC_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Quote asset volume",
    "Number of trades",
    "Taker buy base asset volume",
    "Taker buy quote asset volume",
]

LAG_WINDOWS = [1, 2, 3, 5, 10, 15, 30, 60]
ROLLING_WINDOWS = [5, 15, 30, 60]


def add_target(df: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    """Add next-close target with positive forecast horizon."""
    if horizon <= 0:
        msg = "horizon must be positive"
        raise ValueError(msg)
    result = df.copy()
    result[TARGET_COLUMN] = result["Close"].shift(-horizon)
    return result


def make_features(df: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    """Create leakage-safe features from current and past candles only.

    All rolling features use values available at timestamp t. The target is
    Close(t + horizon), so the future value is stored only in TARGET_COLUMN.
    """
    data = df.sort_values(TIME_COLUMN).reset_index(drop=True).copy()
    eps = 1e-9
    feature_values: dict[str, pd.Series | np.ndarray] = {}

    feature_values["high_low_spread"] = data["High"] - data["Low"]
    feature_values["close_open_diff"] = data["Close"] - data["Open"]
    feature_values["typical_price"] = (data["High"] + data["Low"] + data["Close"]) / 3.0
    feature_values["vwap_proxy"] = data["Quote asset volume"] / (data["Volume"] + eps)
    feature_values["quote_per_trade"] = data["Quote asset volume"] / (
        data["Number of trades"] + eps
    )
    feature_values["volume_per_trade"] = data["Volume"] / (data["Number of trades"] + eps)
    feature_values["taker_buy_ratio_base"] = data["Taker buy base asset volume"] / (
        data["Volume"] + eps
    )
    feature_values["taker_buy_ratio_quote"] = data["Taker buy quote asset volume"] / (
        data["Quote asset volume"] + eps
    )
    feature_values["log_volume"] = np.log1p(data["Volume"])
    feature_values["log_quote_volume"] = np.log1p(data["Quote asset volume"])
    feature_values["log_trades"] = np.log1p(data["Number of trades"])

    feature_values["return_1"] = data["Close"].pct_change(1)
    feature_values["log_return_1"] = np.log(data["Close"]).diff(1)
    for period in [5, 15, 30, 60]:
        feature_values[f"return_{period}"] = data["Close"].pct_change(period)
        feature_values[f"log_return_{period}"] = np.log(data["Close"]).diff(period)

    for lag in LAG_WINDOWS:
        feature_values[f"close_lag_{lag}"] = data["Close"].shift(lag)
        feature_values[f"volume_lag_{lag}"] = data["Volume"].shift(lag)
        feature_values[f"trades_lag_{lag}"] = data["Number of trades"].shift(lag)
        feature_values[f"return_lag_{lag}"] = feature_values["return_1"].shift(lag)

    for window in ROLLING_WINDOWS:
        close_roll = data["Close"].rolling(window=window, min_periods=window)
        volume_roll = data["Volume"].rolling(window=window, min_periods=window)
        trade_roll = data["Number of trades"].rolling(window=window, min_periods=window)
        close_mean = close_roll.mean()
        close_std = close_roll.std()
        volume_mean = volume_roll.mean()
        volume_std = volume_roll.std()
        feature_values[f"close_roll_mean_{window}"] = close_mean
        feature_values[f"close_roll_std_{window}"] = close_std
        feature_values[f"close_roll_min_{window}"] = close_roll.min()
        feature_values[f"close_roll_max_{window}"] = close_roll.max()
        feature_values[f"volume_roll_mean_{window}"] = volume_mean
        feature_values[f"volume_roll_std_{window}"] = volume_std
        feature_values[f"trades_roll_mean_{window}"] = trade_roll.mean()
        feature_values[f"trades_roll_std_{window}"] = trade_roll.std()
        feature_values[f"close_zscore_{window}"] = (data["Close"] - close_mean) / (close_std + eps)
        feature_values[f"volume_zscore_{window}"] = (data["Volume"] - volume_mean) / (
            volume_std + eps
        )

    for span in [5, 15, 60]:
        feature_values[f"close_ema_{span}"] = data["Close"].ewm(span=span, adjust=False).mean()
        feature_values[f"volume_ema_{span}"] = data["Volume"].ewm(span=span, adjust=False).mean()

    # RSI-like momentum feature computed from past/current returns only.
    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=14).mean()
    relative_strength = gain / (loss + eps)
    feature_values["rsi_14"] = 100 - (100 / (1 + relative_strength))

    minute = data[TIME_COLUMN].dt.minute
    hour = data[TIME_COLUMN].dt.hour
    dayofweek = data[TIME_COLUMN].dt.dayofweek
    feature_values["minute"] = minute
    feature_values["hour"] = hour
    feature_values["dayofweek"] = dayofweek
    feature_values["minute_sin"] = np.sin(2 * np.pi * minute / 60)
    feature_values["minute_cos"] = np.cos(2 * np.pi * minute / 60)
    feature_values["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feature_values["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feature_values["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    feature_values["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7)

    engineered = pd.DataFrame(feature_values, index=data.index)
    data = pd.concat([data, engineered], axis=1).copy()
    data = add_target(data, horizon=horizon)
    data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return data


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all model features, excluding time and target columns."""
    excluded = {TIME_COLUMN, TARGET_COLUMN}
    return [column for column in df.columns if column not in excluded]


def raw_baseline_frame(df: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    """Dataset for a simple linear baseline without engineered features."""
    result = df[[TIME_COLUMN, *BASE_NUMERIC_COLUMNS]].copy()
    result = add_target(result, horizon=horizon)
    return result.dropna().reset_index(drop=True)
