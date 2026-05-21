from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
]

BINANCE_TO_API_COLUMNS = {
    "Open time": "open_time",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Quote asset volume": "quote_asset_volume",
    "Number of trades": "number_of_trades",
    "Taker buy base asset volume": "taker_buy_base_asset_volume",
    "Taker buy quote asset volume": "taker_buy_quote_asset_volume",
}

API_TO_TRAINING_COLUMNS = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "quote_asset_volume": "Quote asset volume",
    "number_of_trades": "Number of trades",
    "taker_buy_base_asset_volume": "Taker buy base asset volume",
    "taker_buy_quote_asset_volume": "Taker buy quote asset volume",
}


def normalize_candles(candles: Iterable[dict] | pd.DataFrame) -> pd.DataFrame:
    """Return a normalized dataframe with API column names.

    The function accepts either snake_case API columns or raw Binance CSV columns.
    """
    df = pd.DataFrame(candles).copy()
    df = df.rename(columns=BINANCE_TO_API_COLUMNS)
    missing = [col for col in RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required candle columns: {missing}")
    df = df[RAW_COLUMNS]
    df["open_time"] = pd.to_datetime(df["open_time"])
    numeric_columns = [c for c in RAW_COLUMNS if c != "open_time"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[numeric_columns].isna().any().any():
        bad = df[numeric_columns].columns[df[numeric_columns].isna().any()].tolist()
        raise ValueError(f"Non-numeric values in columns: {bad}")
    return df.sort_values("open_time").reset_index(drop=True)


def build_feature_frame(
    candles: Iterable[dict] | pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Build the same features that were used for the CP3 model artifact."""
    api_df = normalize_candles(candles)
    if len(api_df) < 61:
        raise ValueError(
            "At least 61 consecutive one-minute candles are required for lag/rolling features.",
        )

    x = api_df.rename(columns=API_TO_TRAINING_COLUMNS).copy()
    x["return_1m"] = x["Close"].pct_change()
    x["hl_spread_rel"] = (x["High"] - x["Low"]) / x["Close"].replace(0, np.nan)
    x["oc_change_rel"] = (x["Close"] - x["Open"]) / x["Open"].replace(0, np.nan)
    x["volume_log"] = np.log1p(x["Volume"].clip(lower=0))
    x["quote_volume_log"] = np.log1p(x["Quote asset volume"].clip(lower=0))
    x["trades_log"] = np.log1p(x["Number of trades"].clip(lower=0))
    x["taker_buy_base_ratio"] = x["Taker buy base asset volume"] / x["Volume"].replace(
        0,
        np.nan,
    )
    x["taker_buy_quote_ratio"] = x["Taker buy quote asset volume"] / x[
        "Quote asset volume"
    ].replace(0, np.nan)

    for lag in [1, 5, 10, 30, 60]:
        x[f"close_lag_{lag}"] = x["Close"].shift(lag)
        x[f"return_lag_{lag}"] = x["return_1m"].shift(lag)

    for win in [5, 15, 60]:
        rolling_mean = x["Close"].rolling(win).mean()
        x[f"close_ma_{win}_rel"] = rolling_mean / x["Close"].replace(0, np.nan) - 1
        x[f"return_std_{win}"] = x["return_1m"].rolling(win).std()

    hour = x["open_time"].dt.hour + x["open_time"].dt.minute / 60
    x["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    x = x.replace([np.inf, -np.inf], np.nan)
    last = x.tail(1)[feature_names]
    return last


def current_close(candles: Iterable[dict] | pd.DataFrame) -> float:
    df = normalize_candles(candles)
    return float(df["close"].iloc[-1])
