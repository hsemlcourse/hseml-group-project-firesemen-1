from __future__ import annotations

import pandas as pd

from eth_price.data import clean_raw_data
from eth_price.features import make_features


def test_clean_and_feature_pipeline_on_sample() -> None:
    raw = pd.read_csv("data/sample/ethusd_1m_sample.csv", nrows=500)
    clean, report = clean_raw_data(raw)
    features = make_features(clean)
    assert report["rows_final"] > 0
    assert len(features) > 300
    assert "target_close_next" in features.columns
