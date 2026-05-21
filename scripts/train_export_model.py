"""Recreate the lightweight CP3 deployment model.

Expected input:
    data/raw/ETHUSD_1m_Binance.zip

The script streams the ZIP/CSV, keeps the latest 50k rows, trains a Ridge model,
and rewrites models/final_model.joblib plus metrics in reports/.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_ZIP = ROOT / "data" / "raw" / "ETHUSD_1m_Binance.zip"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
TAIL_ROWS = 50_000


def read_tail_from_zip(path: Path, tail_rows: int = TAIL_ROWS) -> tuple[pd.DataFrame, int]:
    if not path.exists():
        raise FileNotFoundError(f"Put ETHUSD_1m_Binance.zip here: {path}")
    chunks: list[pd.DataFrame] = []
    total_rows = 0
    for chunk in pd.read_csv(path, compression="zip", chunksize=100_000):
        total_rows += len(chunk)
        chunks.append(chunk)
        current = sum(len(c) for c in chunks)
        while len(chunks) > 1 and current - len(chunks[0]) >= tail_rows:
            current -= len(chunks.pop(0))
    tail = pd.concat(chunks, ignore_index=True).tail(tail_rows)
    return tail.reset_index(drop=True), total_rows


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    x = df.copy()
    x["Open time"] = pd.to_datetime(x["Open time"])
    num = [
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
    x["target_next_close"] = x["Close"].shift(-1)
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
    hour = x["Open time"].dt.hour + x["Open time"].dt.minute / 60
    x["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features = (
        num
        + [
            "return_1m",
            "hl_spread_rel",
            "oc_change_rel",
            "volume_log",
            "quote_volume_log",
            "trades_log",
            "taker_buy_base_ratio",
            "taker_buy_quote_ratio",
        ]
        + [f"close_lag_{lag}" for lag in [1, 5, 10, 30, 60]]
        + [f"return_lag_{lag}" for lag in [1, 5, 10, 30, 60]]
        + [f"close_ma_{win}_rel" for win in [5, 15, 60]]
        + [f"return_std_{win}" for win in [5, 15, 60]]
        + ["hour_sin", "hour_cos"]
    )
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(subset=["target_next_close"]).reset_index(drop=True)
    return x[features], x["target_next_close"], features


def metrics(y_true: pd.Series, pred: np.ndarray, prev_close: pd.Series) -> dict[str, float]:
    y = np.asarray(y_true)
    prev = np.asarray(prev_close)
    y_safe = np.where(y == 0, np.nan, y)
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAPE_%": float(np.mean(np.abs((y - pred) / y_safe)) * 100),
        "R2": float(r2_score(y, pred)),
        "direction_accuracy_%": float(
            (np.sign(y - prev) == np.sign(pred - prev)).mean() * 100,
        ),
    }


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    df, total_rows = read_tail_from_zip(DATA_ZIP)
    x, y, features = build_features(df)
    val_end = int(len(x) * 0.85)
    x_train_val = x.iloc[:val_end]
    y_train_val = y.iloc[:val_end]
    x_test = x.iloc[val_end:]
    y_test = y.iloc[val_end:]

    model = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, solver="lsqr")),
        ]
    )
    model.fit(x_train_val, y_train_val)
    pred = model.predict(x_test)
    baseline_pred = x_test["Close"].values
    metadata = {
        "model_type": "Ridge(alpha=1.0, solver=lsqr)",
        "target": "next one-minute ETHUSD close price",
        "random_state": RANDOM_STATE,
        "features": features,
        "feature_count": len(features),
        "data_rows_total": int(total_rows),
        "modeling_tail_rows": len(df),
        "feature_rows": len(x),
        "split": {"type": "chronological 70/15/15 on latest 50k sample"},
        "test_metrics": metrics(y_test, pred, x_test["Close"]),
        "baseline_test_metrics": metrics(y_test, baseline_pred, x_test["Close"]),
        "source_file": "ETHUSD_1m_Binance.csv inside ETHUSD_1m_Binance.zip",
    }
    joblib.dump({"pipeline": model, "metadata": metadata}, MODEL_DIR / "final_model.joblib")
    metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2)
    (MODEL_DIR / "model_metadata.json").write_text(metadata_text, encoding="utf-8")
    print(json.dumps(metadata["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
