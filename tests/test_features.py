from __future__ import annotations

import pandas as pd

from eth_price.config import TARGET_COLUMN
from eth_price.features import make_features


def _toy_frame(n_rows: int = 90) -> pd.DataFrame:
    close = [100 + i * 0.1 for i in range(n_rows)]
    return pd.DataFrame(
        {
            "Open time": pd.date_range("2026-01-01", periods=n_rows, freq="min"),
            "Open": close,
            "High": [x + 0.2 for x in close],
            "Low": [x - 0.2 for x in close],
            "Close": close,
            "Volume": [10 + i for i in range(n_rows)],
            "Quote asset volume": [(10 + i) * close[i] for i in range(n_rows)],
            "Number of trades": [100 + i for i in range(n_rows)],
            "Taker buy base asset volume": [5 + i * 0.1 for i in range(n_rows)],
            "Taker buy quote asset volume": [(5 + i * 0.1) * close[i] for i in range(n_rows)],
        }
    )


def test_target_is_next_close() -> None:
    raw = _toy_frame()
    features = make_features(raw)
    first_source_idx = raw.index[raw["Open time"] == features.iloc[0]["Open time"]][0]
    assert features.iloc[0][TARGET_COLUMN] == raw.iloc[first_source_idx + 1]["Close"]


def test_no_future_close_feature_is_created() -> None:
    features = make_features(_toy_frame())
    suspicious = [
        column
        for column in features.columns
        if "future" in column.lower() or "lead" in column.lower()
    ]
    assert suspicious == []
