from __future__ import annotations

import pandas as pd

from eth_price.data import assert_no_temporal_overlap, train_val_test_split_by_time


def test_chronological_split_has_no_overlap() -> None:
    df = pd.DataFrame(
        {
            "Open time": pd.date_range("2026-01-01", periods=100, freq="min"),
            "value": range(100),
        }
    )
    train, val, test = train_val_test_split_by_time(df)
    assert_no_temporal_overlap([train, val, test])
    assert train["Open time"].max() < val["Open time"].min() < test["Open time"].min()
